"""High-performance vectorized Finite Element Engine for topology optimization.

Provides the FEMEngine class capable of assembling global stiffness and mass matrices,
evaluating vectorized element compliances, and computing Cauchy and von Mises stresses
across StructuredMesh2D, StructuredMesh3D, and UnstructuredMesh.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

from .mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from .numba_kernels import (
    HAS_NUMBA,
    build_csc_assembly_map,
    numba_assemble_csc_batch,
    numba_assemble_csc_uniform,
    numba_element_kg0_uniform,
    numba_element_stresses_uniform,
    numba_unit_compliance_batch,
    numba_unit_compliance_uniform,
)
from .fe_builders import (
    isotropic_elasticity_2d,
    isotropic_elasticity_3d,
    tri3_ke0_me0,
    tri3_kg0_tensor,
    q4_ke0_me0,
    q4_kg0_tensor,
    tet4_ke0_me0,
    tet4_kg0_tensor,
    tet10_ke0_me0,
    hex8_ke0_me0,
    hex8_kg0_tensor,
    hex20_ke0_me0,
    mitc3_ke0_me0,
    mitc4_ke0_me0,
)


class FEMEngine:
    """Vectorized Finite Element Engine for topology optimization and sensitivity analysis.

    Precomputes unit-modulus reference matrices (Ke0, Me0), centroid B-matrices,
    and COO assembly indices. Works seamlessly with 2D, 3D solid, and shell meshes.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        nu: float = 0.3,
        thickness: float = 1.0,
        plane: str = "plane_stress",
        formulation: str = "FI",
        e0: float = 1.0,
        emin: float = 1e-9,
        use_numba: bool = True,
    ) -> None:
        """Initialize FEMEngine on a given mesh.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh instance.
        nu : float
            Poisson's ratio (-1.0 < nu < 0.5).
        thickness : float
            Plate/shell or 2D plane thickness (default 1.0).
        plane : str
            2D plane assumption ('plane_stress' or 'plane_strain').
        formulation : str
            Formulation option ('FI' for full integration or 'BBAR' for Hughes B-bar).
        """
        self.mesh = mesh
        self.nu: float = float(nu)
        self.thickness: float = float(thickness)
        self.plane: str = plane
        self.formulation: str = formulation.upper()
        self.e0: float = float(e0)
        self.emin: float = float(emin)
        self.use_numba: bool = bool(use_numba and HAS_NUMBA)

        self.n_elem: int = self.mesh.n_elem
        self.ndof: int = self.mesh.ndof
        self.n_elem_dof: int = self.mesh.edofMat.shape[1]
        self.edof_mat: np.ndarray = self.mesh.edofMat

        # Determine element family & constitutive matrix
        if isinstance(self.mesh, StructuredMesh2D):
            self.etype = "Q4"
            self.ndim = 2
            self.is_structured = True
            self.C0 = isotropic_elasticity_2d(self.nu, plane=self.plane)
        elif isinstance(self.mesh, StructuredMesh3D):
            self.etype = "HEX8"
            self.ndim = 3
            self.is_structured = True
            self.C0 = isotropic_elasticity_3d(self.nu)
        elif isinstance(self.mesh, UnstructuredMesh):
            self.etype = self.mesh.element_type
            self.ndim = self.mesh.ndim
            self.is_structured = False
            if self.etype in {"TRI3", "Q4"}:
                self.C0 = isotropic_elasticity_2d(self.nu, plane=self.plane)
            elif self.etype in {"TET4", "TET10", "HEX8", "HEX20"}:
                self.C0 = isotropic_elasticity_3d(self.nu)
            elif self.etype in {"MITC3", "MITC4"}:
                self.C0 = isotropic_elasticity_2d(self.nu, plane="plane_stress")
            else:
                raise ValueError(f"Unsupported element type: {self.etype}")
        else:
            raise TypeError(f"Unknown mesh type: {type(mesh)}")

        # Build reference stiffness, mass, and centroid B-matrix
        self._build_reference_matrices()

        # Build COO assembly indices
        # For each element, edof is of length d = n_elem_dof
        # Row indices: repeat each edof d times along cols
        # Col indices: tile each edof d times along cols
        d = self.n_elem_dof
        edof = self.mesh.edofMat  # shape (n_elem, d)
        self.iK: np.ndarray = np.repeat(edof, d, axis=1).ravel()
        self.jK: np.ndarray = np.tile(edof, (1, d)).ravel()

        # Precompute unique CSC sparsity structure and gather map (top99neo style)
        (
            self._csc_indptr,
            self._csc_indices,
            self._gather_ptr,
            self._gather_elem,
            self._gather_local,
        ) = build_csc_assembly_map(self.iK, self.jK, self.ndof, d * d)

        if self.is_uniform:
            self.ke0: np.ndarray = self.Ke0
            self._Ke0_flat: np.ndarray = np.ascontiguousarray(self.Ke0.ravel(), dtype=np.float64)
            self._Me0_flat: np.ndarray = np.ascontiguousarray(self.Me0.ravel(), dtype=np.float64)
            self._DB0: np.ndarray = np.ascontiguousarray(self.C0 @ self.B0, dtype=np.float64)
        else:
            self.ke0 = self.Ke0_batch[0]
            self._Ke0_batch_flat: np.ndarray = np.ascontiguousarray(
                self.Ke0_batch.reshape(self.n_elem, d * d), dtype=np.float64
            )
            self._Me0_batch_flat: np.ndarray = np.ascontiguousarray(
                self.Me0_batch.reshape(self.n_elem, d * d), dtype=np.float64
            )

        # Build von Mises stress quadratic form matrix V (sigma_vM = sqrt(sigma.T @ V @ sigma))
        self._build_von_mises_matrix()

    def _build_reference_matrices(self) -> None:
        """Precompute reference Ke0, Me0, and B0 matrices."""
        d = self.n_elem_dof

        if self.is_structured and isinstance(self.mesh, StructuredMesh2D):
            # Regular rectangular Q4 with dimensions dx, dy
            dx, dy = self.mesh.dx, self.mesh.dy
            coords = np.array([
                [0.0, 0.0],
                [dx, 0.0],
                [dx, dy],
                [0.0, dy],
            ], dtype=np.float64)
            ke0, me0, b0 = q4_ke0_me0(
                coords,
                self.nu,
                thickness=self.thickness,
                plane=self.plane,
                formulation=self.formulation,
            )
            self.Ke0: np.ndarray = ke0  # (8, 8)
            self.Me0: np.ndarray = me0  # (8, 8)
            self.B0: np.ndarray = b0    # (3, 8)
            self.KG_tensor: np.ndarray = q4_kg0_tensor(
                coords, self.nu, thickness=self.thickness, plane=self.plane
            )  # (8, 8, 8)
            self.is_uniform: bool = True

        elif self.is_structured and isinstance(self.mesh, StructuredMesh3D):
            # Regular cuboidal HEX8 with dimensions dx, dy, dz
            dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
            coords = np.array([
                [0.0, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [dx, dy, 0.0],
                [0.0, dy, 0.0],
                [0.0, 0.0, dz],
                [dx, 0.0, dz],
                [dx, dy, dz],
                [0.0, dy, dz],
            ], dtype=np.float64)
            ke0, me0, b0 = hex8_ke0_me0(
                coords,
                self.nu,
                formulation=self.formulation,
            )
            self.Ke0 = ke0  # (24, 24)
            self.Me0 = me0  # (24, 24)
            self.B0 = b0    # (6, 24)
            self.KG_tensor = hex8_kg0_tensor(coords, self.nu)  # (24, 24, 24)
            self.is_uniform = True

        else:
            # Unstructured mesh: compute per-element or cached congruent matrices
            self.is_uniform = False
            self.Ke0_batch = np.zeros((self.n_elem, d, d), dtype=np.float64)
            self.Me0_batch = np.zeros((self.n_elem, d, d), dtype=np.float64)
            self.KG_tensor_batch = np.zeros((self.n_elem, d, d, d), dtype=np.float64)

            # Determine B-matrix shape
            n_stress_comp = 3 if self.etype in {"TRI3", "Q4"} else 6
            self.B0_batch = np.zeros((self.n_elem, n_stress_comp, d), dtype=np.float64)

            mesh = self.mesh
            nodes = mesh.node_coords
            elems = mesh.elements

            for e in range(self.n_elem):
                c = nodes[elems[e]]
                kg_t = np.zeros((d, d, d), dtype=np.float64)
                if self.etype == "TRI3":
                    ke, me, b = tri3_ke0_me0(c, self.nu, self.thickness, self.plane)
                    kg_t = tri3_kg0_tensor(c, self.nu, self.thickness, self.plane)
                elif self.etype == "Q4":
                    ke, me, b = q4_ke0_me0(c, self.nu, self.thickness, self.plane, self.formulation)
                    kg_t = q4_kg0_tensor(c, self.nu, self.thickness, self.plane)
                elif self.etype == "TET4":
                    ke, me, b = tet4_ke0_me0(c, self.nu)
                    kg_t = tet4_kg0_tensor(c, self.nu)
                elif self.etype == "TET10":
                    ke, me, b = tet10_ke0_me0(c, self.nu)
                elif self.etype == "HEX8":
                    ke, me, b = hex8_ke0_me0(c, self.nu, self.formulation)
                    kg_t = hex8_kg0_tensor(c, self.nu)
                elif self.etype == "HEX20":
                    ke, me, b = hex20_ke0_me0(c, self.nu)
                elif self.etype == "MITC3":
                    ke, me = mitc3_ke0_me0(c, self.nu, self.thickness)
                    b = np.zeros((n_stress_comp, d), dtype=np.float64)
                elif self.etype == "MITC4":
                    ke, me = mitc4_ke0_me0(c, self.nu, self.thickness)
                    b = np.zeros((n_stress_comp, d), dtype=np.float64)
                else:
                    raise ValueError(f"Unsupported element: {self.etype}")

                self.Ke0_batch[e] = ke
                self.Me0_batch[e] = me
                self.B0_batch[e] = b
                self.KG_tensor_batch[e] = kg_t

    def _build_von_mises_matrix(self) -> None:
        """Construct the von Mises quadratic form matrix V for Voigt stress components."""
        if self.ndim == 2 or self.etype in {"TRI3", "Q4"}:
            # Stress order: [sigma_xx, sigma_yy, tau_xy]
            # sigma_vM^2 = sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*tau_xy^2
            self.V_vm = np.array([
                [1.0, -0.5, 0.0],
                [-0.5, 1.0, 0.0],
                [0.0, 0.0, 3.0],
            ], dtype=np.float64)
        else:
            # Stress order: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]
            self.V_vm = np.array([
                [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
                [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            ], dtype=np.float64)

    def assemble_K(self, E_e: np.ndarray, use_numba: Optional[bool] = None) -> sp.csc_matrix:
        """Fast assembly of global stiffness matrix K = sum_e E_e * Ke0.

        Uses Numba parallel pre-indexed CSC assembly (zero COO sorting overhead) by default.
        """
        E_e = np.ascontiguousarray(E_e, dtype=np.float64)
        if E_e.shape[0] != self.n_elem:
            raise ValueError(f"E_e length {E_e.shape[0]} does not match n_elem {self.n_elem}")

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel:
            if self.is_uniform:
                data = numba_assemble_csc_uniform(
                    E_e, self._Ke0_flat, self._gather_ptr, self._gather_elem, self._gather_local
                )
            else:
                data = numba_assemble_csc_batch(
                    E_e, self._Ke0_batch_flat, self._gather_ptr, self._gather_elem, self._gather_local
                )
            return sp.csc_matrix(
                (data, self._csc_indices, self._csc_indptr),
                shape=(self.ndof, self.ndof),
                copy=False,
            )

        if self.is_uniform:
            vals = (E_e[:, None, None] * self.Ke0[None, :, :]).ravel()
        else:
            vals = (E_e[:, None, None] * self.Ke0_batch).ravel()

        K_coo = sp.coo_matrix((vals, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        return K_coo.tocsc()

    def assemble_stiffness_matrix(
        self,
        rho_phys: np.ndarray,
        penal: float = 3.0,
        e0: Optional[float] = None,
        emin: Optional[float] = None,
        interpolation: Optional[Any] = None,
        use_numba: Optional[bool] = None,
    ) -> sp.csc_matrix:
        """Assemble global stiffness matrix from physical densities rho_phys."""
        if interpolation is not None:
            E_e = interpolation.evaluate(rho_phys)
        else:
            e0_val = self.e0 if e0 is None else float(e0)
            emin_val = self.emin if emin is None else float(emin)
            E_e = emin_val + (np.asarray(rho_phys, dtype=np.float64) ** penal) * (e0_val - emin_val)
        return self.assemble_K(E_e, use_numba=use_numba)

    def assemble_M(self, rho_mass_e: np.ndarray, use_numba: Optional[bool] = None) -> sp.csc_matrix:
        """Fast assembly of global mass matrix M = sum_e rho_mass_e * Me0."""
        rho_mass_e = np.ascontiguousarray(rho_mass_e, dtype=np.float64)
        if rho_mass_e.shape[0] != self.n_elem:
            raise ValueError(f"rho_mass_e length {rho_mass_e.shape[0]} does not match n_elem {self.n_elem}")

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel:
            if self.is_uniform:
                data = numba_assemble_csc_uniform(
                    rho_mass_e, self._Me0_flat, self._gather_ptr, self._gather_elem, self._gather_local
                )
            else:
                data = numba_assemble_csc_batch(
                    rho_mass_e, self._Me0_batch_flat, self._gather_ptr, self._gather_elem, self._gather_local
                )
            return sp.csc_matrix(
                (data, self._csc_indices, self._csc_indptr),
                shape=(self.ndof, self.ndof),
                copy=False,
            )

        if self.is_uniform:
            vals = (rho_mass_e[:, None, None] * self.Me0[None, :, :]).ravel()
        else:
            vals = (rho_mass_e[:, None, None] * self.Me0_batch).ravel()

        M_coo = sp.coo_matrix((vals, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        return M_coo.tocsc()

    def assemble_KG(self, E_G_e: np.ndarray, u: np.ndarray, use_numba: Optional[bool] = None) -> sp.csc_matrix:
        """Fast assembly of global geometric stiffness matrix KG."""
        E_G_e = np.ascontiguousarray(E_G_e, dtype=np.float64)
        if E_G_e.shape[0] != self.n_elem:
            raise ValueError(f"E_G_e length {E_G_e.shape[0]} does not match n_elem {self.n_elem}")

        KG0_e = self.compute_element_kg0(u, use_numba=use_numba)  # (n_elem, d, d)
        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel:
            d = self.n_elem_dof
            kg0_flat = np.ascontiguousarray(KG0_e.reshape(self.n_elem, d * d), dtype=np.float64)
            data = numba_assemble_csc_batch(
                E_G_e, kg0_flat, self._gather_ptr, self._gather_elem, self._gather_local
            )
            return sp.csc_matrix(
                (data, self._csc_indices, self._csc_indptr),
                shape=(self.ndof, self.ndof),
                copy=False,
            )

        vals = (E_G_e[:, None, None] * KG0_e).ravel()
        KG_coo = sp.coo_matrix((vals, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        return KG_coo.tocsc()

    def compute_element_kg0(self, u: np.ndarray, use_numba: Optional[bool] = None) -> np.ndarray:
        """Compute base unit geometric stiffness matrices KG_e^0(u_e) of shape (n_elem, d, d)."""
        u = np.ascontiguousarray(u, dtype=np.float64)
        if u.ndim != 1 or u.shape[0] != self.ndof:
            raise ValueError(f"Displacement vector u must have shape ({self.ndof},), got {u.shape}")

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel and self.is_uniform:
            return numba_element_kg0_uniform(u, self.mesh.edofMat, self.KG_tensor)

        ue = u[self.mesh.edofMat]  # (n_elem, d)
        if self.is_uniform:
            return np.einsum("ek,kij->eij", ue, self.KG_tensor)
        else:
            return np.einsum("ek,ekij->eij", ue, self.KG_tensor_batch)

    def compute_buckling_adjoint_forces(
        self,
        phi: np.ndarray,
        E_G_e: np.ndarray,
    ) -> np.ndarray:
        """Vectorized computation of adjoint load psi = sum_e E_G_e * (phi_e^T dKG_e^0/du_e phi_e)."""
        phi_arr = np.asarray(phi, dtype=np.float64)
        E_G_e = np.asarray(E_G_e, dtype=np.float64)
        is_1d = (phi_arr.ndim == 1)
        phis = phi_arr[None, :] if is_1d else phi_arr
        n_modes = phis.shape[0]

        psi_out = np.zeros((n_modes, self.ndof), dtype=np.float64)
        edof_flat = self.mesh.edofMat.ravel()

        for m in range(n_modes):
            phi_e = phis[m][self.mesh.edofMat]  # (n_elem, d)
            if self.is_uniform:
                elem_psi = E_G_e[:, None] * np.einsum("ei,kij,ej->ek", phi_e, self.KG_tensor, phi_e)
            else:
                elem_psi = E_G_e[:, None] * np.einsum("ei,ekij,ej->ek", phi_e, self.KG_tensor_batch, phi_e)

            np.add.at(psi_out[m], edof_flat, elem_psi.ravel())

        return psi_out[0] if is_1d else psi_out

    def compute_unit_element_compliance(self, u: np.ndarray, use_numba: Optional[bool] = None) -> np.ndarray:
        """Fast computation of unit-modulus element compliance c_e0 = u_e^T Ke0 u_e."""
        u = np.ascontiguousarray(u, dtype=np.float64)
        is_1d = (u.ndim == 1)
        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)

        if is_1d:
            if u.shape[0] != self.ndof:
                raise ValueError(f"Displacement vector length {u.shape[0]} != ndof {self.ndof}")
            if accel:
                if self.is_uniform:
                    return numba_unit_compliance_uniform(u, self.mesh.edofMat, self.Ke0)
                return numba_unit_compliance_batch(u, self.mesh.edofMat, self.Ke0_batch)
            ue = u[self.mesh.edofMat]
            if self.is_uniform:
                Ke_u = ue @ self.Ke0
                c_e0 = np.sum(Ke_u * ue, axis=1)
            else:
                Ke_u = np.einsum("eij,ej->ei", self.Ke0_batch, ue)
                c_e0 = np.sum(Ke_u * ue, axis=1)
            return c_e0
        else:
            n_lc = u.shape[1]
            c_e0 = np.empty((self.n_elem, n_lc), dtype=np.float64)
            for lc in range(n_lc):
                u_lc = np.ascontiguousarray(u[:, lc], dtype=np.float64)
                if accel:
                    if self.is_uniform:
                        c_e0[:, lc] = numba_unit_compliance_uniform(u_lc, self.mesh.edofMat, self.Ke0)
                    else:
                        c_e0[:, lc] = numba_unit_compliance_batch(u_lc, self.mesh.edofMat, self.Ke0_batch)
                else:
                    ue = u_lc[self.mesh.edofMat]
                    if self.is_uniform:
                        Ke_u = ue @ self.Ke0
                        c_e0[:, lc] = np.sum(Ke_u * ue, axis=1)
                    else:
                        Ke_u = np.einsum("eij,ej->ei", self.Ke0_batch, ue)
                        c_e0[:, lc] = np.sum(Ke_u * ue, axis=1)
            return c_e0

    def compute_element_stresses(
        self,
        u: np.ndarray,
        E0: float = 1.0,
        use_numba: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute microscopic Cauchy stress and scalar von Mises stress at element centroids."""
        u = np.ascontiguousarray(u, dtype=np.float64)
        if u.ndim != 1 or u.shape[0] != self.ndof:
            raise ValueError(f"Displacement vector u must have shape ({self.ndof},), got {u.shape}")

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel and self.is_uniform:
            DB0_scaled = self._DB0 if abs(E0 - 1.0) < 1e-15 else (float(E0) * self._DB0)
            return numba_element_stresses_uniform(u, self.mesh.edofMat, DB0_scaled, self.V_vm)

        ue = u[self.mesh.edofMat]  # (n_elem, d)
        if self.is_uniform:
            strains = ue @ self.B0.T
            cauchy_stress = E0 * (strains @ self.C0.T)
        else:
            strains = np.einsum("esd,ed->es", self.B0_batch, ue)
            cauchy_stress = E0 * (strains @ self.C0.T)

        vm_sq = np.sum((cauchy_stress @ self.V_vm) * cauchy_stress, axis=1)
        von_mises_stress = np.sqrt(np.maximum(0.0, vm_sq))

        return cauchy_stress, von_mises_stress

    def compute_element_von_mises_batch(
        self,
        u_steps: np.ndarray,
        E0: float = 1.0,
    ) -> np.ndarray:
        """Vectorized computation of unpenalized von Mises stress across multiple time steps/load cases.

        Evaluates element centroid von Mises stress without Python loops using np.einsum:
        - sigma_c(t_k) = (D0 @ B0) @ u_e(t_k)
        - sigma_vm^2(t_k) = sigma_c^T @ V_vm @ sigma_c

        Parameters
        ----------
        u_steps : np.ndarray
            Array of displacement vectors of shape (n_cases, ndof) or (ndof,).
        E0 : float
            Base reference Young's modulus (default 1.0 for unit-modulus unpenalized stress).

        Returns
        -------
        von_mises : np.ndarray
            Element centroid von Mises stresses of shape (n_cases, n_elem) or (n_elem,).
        """
        u_arr = np.asarray(u_steps, dtype=np.float64)
        is_1d = (u_arr.ndim == 1)
        if is_1d:
            if u_arr.shape[0] != self.ndof:
                raise ValueError(f"Displacement vector length {u_arr.shape[0]} != ndof {self.ndof}")
            u_cases = u_arr[None, :]
        else:
            if u_arr.shape[1] != self.ndof:
                raise ValueError(f"Displacement array shape {u_arr.shape} does not match ndof {self.ndof}")
            u_cases = u_arr

        # Extract element displacement batch: shape (n_cases, n_elem, n_elem_dof)
        ue_cases = u_cases[:, self.mesh.edofMat]

        if self.is_uniform:
            DB0_scaled = self._DB0 if abs(E0 - 1.0) < 1e-15 else (float(E0) * self._DB0)
            # Cauchy stress batch: shape (n_cases, n_elem, n_stress_components)
            sigma_cases = np.einsum("sd,ced->ces", DB0_scaled, ue_cases)
            # Quadratic form: vm_sq = sigma @ V_vm @ sigma
            vm_sq = np.einsum("ces,st,cet->ce", sigma_cases, self.V_vm, sigma_cases)
        else:
            DB_batch = np.einsum("st,etd->esd", self.C0, self.B0_batch) * float(E0)
            sigma_cases = np.einsum("esd,ced->ces", DB_batch, ue_cases)
            vm_sq = np.einsum("ces,st,cet->ce", sigma_cases, self.V_vm, sigma_cases)

        von_mises = np.sqrt(np.maximum(0.0, vm_sq))
        return von_mises[0] if is_1d else von_mises

