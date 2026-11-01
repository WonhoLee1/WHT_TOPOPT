"""Asymptotic numerical homogenization and exact self-adjoint sensitivity analysis.

Implements 2D and 3D computational homogenization on periodic unit cells with
master-slave periodic boundary condition (PBC) reduction matrix P_pbc,
macroscopic unit test strains, mutual energy homogenized elasticity tensor E^H,
and exact self-adjoint element sensitivities d(E^H)/d(rho_e) without adjoint solves.
Based on Xia & Breitkopf (2015) and Andreassen & Andreasen (2014).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D


class BaseHomogenizer(ABC):
    """Abstract base class for asymptotic finite element homogenizers."""

    def __init__(
        self,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
        penal: float = 3.0,
        material_model: str = "simp_modified",
    ) -> None:
        """Initialize base homogenizer material parameters.

        Parameters
        ----------
        e0 : float
            Solid base Young's modulus E0 > 0.
        emin : float
            Void base Young's modulus Emin > 0 (avoids singularity).
        nu : float
            Poisson's ratio (-1.0 < nu < 0.5).
        penal : float
            SIMP/RAMP interpolation penalty exponent p >= 1.0.
        material_model : str
            Material interpolation model ('simp_modified', 'simp', or 'ramp').
        """
        if e0 <= 0.0 or emin <= 0.0:
            raise ValueError(f"e0 and emin must be positive, got ({e0}, {emin})")
        if emin >= e0:
            raise ValueError(f"emin must be strictly less than e0, got emin={emin}, e0={e0}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")
        if penal < 1.0:
            raise ValueError(f"penal must be >= 1.0, got {penal}")

        self.e0: float = float(e0)
        self.emin: float = float(emin)
        self.nu: float = float(nu)
        self.penal: float = float(penal)
        self.material_model: str = material_model.lower()

        self.last_EH: Optional[np.ndarray] = None
        self.last_dEH: Optional[np.ndarray] = None
        self.last_chi: Optional[np.ndarray] = None

    def interpolate_material(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute interpolated Young's modulus E_e and derivative dE_e/drho_e.

        Parameters
        ----------
        rho : np.ndarray
            Element density array of shape (N_elem,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (E_e, dE_e_drho_e) arrays of shape (N_elem,).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        p = self.penal
        e0 = self.e0
        emin = self.emin

        if self.material_model == "simp_modified":
            # E(rho) = emin + rho^p * (e0 - emin)
            delta = e0 - emin
            E_e = emin + (rho_arr ** p) * delta
            dE_e = p * (rho_arr ** (p - 1.0)) * delta
        elif self.material_model == "simp":
            # E(rho) = rho^p * e0
            E_e = (rho_arr ** p) * e0
            dE_e = p * (rho_arr ** (p - 1.0)) * e0
        elif self.material_model == "ramp":
            # RAMP: E(rho) = emin + rho / (1 + q*(1 - rho)) * (e0 - emin)
            q = p
            delta = e0 - emin
            denom = 1.0 + q * (1.0 - rho_arr)
            E_e = emin + (rho_arr / denom) * delta
            dE_e = ((1.0 + q) / (denom ** 2)) * delta
        else:
            raise ValueError(f"Unknown material model: {self.material_model}")

        return E_e, dE_e

    @abstractmethod
    def homogenize(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute homogenized elasticity tensor and its exact sensitivities."""
        pass


class Homogenizer2D(BaseHomogenizer):
    """2D Periodic Unit Cell asymptotic homogenizer with master-slave PBC reduction.

    Solves the 3 unit cell problems for macroscopic test strains:
        m = 0: [1, 0, 0]^T  (eps_xx = 1)
        m = 1: [0, 1, 0]^T  (eps_yy = 1)
        m = 2: [0, 0, 1]^T  (gamma_xy = 1)
    """

    def __init__(
        self,
        mesh: StructuredMesh2D,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
        penal: float = 3.0,
        plane: str = "plane_stress",
        material_model: str = "simp_modified",
    ) -> None:
        """Initialize 2D periodic homogenizer.

        Parameters
        ----------
        mesh : StructuredMesh2D
            Periodic unit cell rectangular mesh.
        e0 : float
            Solid base Young's modulus.
        emin : float
            Void Young's modulus.
        nu : float
            Poisson's ratio.
        penal : float
            SIMP penalty exponent.
        plane : str
            'plane_stress' or 'plane_strain'.
        material_model : str
            Material interpolation model.
        """
        super().__init__(
            e0=e0,
            emin=emin,
            nu=nu,
            penal=penal,
            material_model=material_model,
        )
        if not isinstance(mesh, StructuredMesh2D):
            raise TypeError(f"Homogenizer2D requires StructuredMesh2D, got {type(mesh)}")

        self.mesh: StructuredMesh2D = mesh
        self.plane: str = plane.lower()
        self.fem: FEMEngine = FEMEngine(
            self.mesh,
            nu=self.nu,
            thickness=1.0,
            plane=self.plane,
            formulation="FI",
        )
        self.Ke0: np.ndarray = self.fem.Ke0  # (8, 8)
        self.C0: np.ndarray = self.fem.C0    # (3, 3)
        self.cell_volume: float = float(self.mesh.lx * self.mesh.ly)
        self.n_cases: int = 3

        # Construct Master-Slave Periodic Boundary Condition matrix P_pbc
        self.P_pbc: sp.csc_matrix = self._build_pbc_projection()

        # Construct macroscopic unit strain displacement matrix U0 of shape (ndof, 3)
        self.U0: np.ndarray = self._build_macroscopic_displacements()

    def _build_pbc_projection(self) -> sp.csc_matrix:
        """Construct sparse boolean Master-Slave PBC reduction matrix P_pbc.

        Maps periodic boundary node pairs:
            - Left boundary nodes (ix = 0) are Master, Right (ix = nx) are Slave.
            - Bottom boundary nodes (iy = 0) are Master, Top (iy = ny) are Slave.
            - All 4 corners (0,0), (nx,0), (0,ny), (nx,ny) are constrained to 0
              to eliminate rigid body translations.
            - Interior nodes are Master.
        """
        nx = self.mesh.nelx
        ny = self.mesh.nely

        master_map = {}
        master_nodes = []
        node_to_master = np.full(self.mesh.n_nodes, -1, dtype=np.int64)

        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nid = iy * (nx + 1) + ix
                ix_c = 0 if ix == nx else ix
                iy_c = 0 if iy == ny else iy

                if (ix_c, iy_c) == (0, 0):
                    # Corner node: constrained to 0
                    node_to_master[nid] = -1
                else:
                    can = (ix_c, iy_c)
                    if can not in master_map:
                        master_map[can] = len(master_nodes)
                        master_nodes.append(can)
                    node_to_master[nid] = master_map[can]

        n_masters = len(master_nodes)
        rows = []
        cols = []
        data = []

        for nid in range(self.mesh.n_nodes):
            mid = node_to_master[nid]
            if mid >= 0:
                for d in range(2):
                    rows.append(nid * 2 + d)
                    cols.append(mid * 2 + d)
                    data.append(1.0)

        P_pbc = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.mesh.ndof, n_masters * 2),
            dtype=np.float64,
        ).tocsc()
        return P_pbc

    def _build_macroscopic_displacements(self) -> np.ndarray:
        """Build exact nodal displacements for unit macroscopic strains.

        Returns
        -------
        np.ndarray
            (N_dof, 3) matrix containing:
            Column 0: eps_xx = 1 -> u(x, y) = x, v(x, y) = 0
            Column 1: eps_yy = 1 -> u(x, y) = 0, v(x, y) = y
            Column 2: gamma_xy = 1 -> u(x, y) = 0.5 * y, v(x, y) = 0.5 * x
        """
        coords = self.mesh.node_coords
        U0 = np.zeros((self.mesh.ndof, 3), dtype=np.float64)

        # m=0: eps_xx = 1
        U0[0::2, 0] = coords[:, 0]
        # m=1: eps_yy = 1
        U0[1::2, 1] = coords[:, 1]
        # m=2: gamma_xy = 1
        U0[0::2, 2] = 0.5 * coords[:, 1]
        U0[1::2, 2] = 0.5 * coords[:, 0]

        return U0

    def homogenize(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D homogenized elasticity tensor E^H and exact self-adjoint sensitivities.

        Parameters
        ----------
        rho : np.ndarray
            Physical density field array of shape (N_elem,).

        Returns
        -------
        EH : np.ndarray
            Homogenized 3x3 symmetric elasticity matrix E^H.
        dEH_drho : np.ndarray
            Exact analytical sensitivity tensor of shape (3, 3, N_elem).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        if rho_arr.shape[0] != self.mesh.n_elem:
            raise ValueError(f"rho shape {rho_arr.shape} does not match N_elem {self.mesh.n_elem}")

        E_e, dE_drho = self.interpolate_material(rho_arr)
        K = self.fem.assemble_K(E_e)

        # Reduced periodic stiffness: K_red = P_pbc^T @ K @ P_pbc
        K_red = self.P_pbc.T @ K @ self.P_pbc
        K_red_csc = K_red.tocsc()

        # Macroscopic load: F_macro = K @ U0, F_red = P_pbc^T @ F_macro
        F_macro = K @ self.U0
        F_red = self.P_pbc.T @ F_macro

        # Factorize K_red once and solve for all 3 macroscopic load cases
        solve_op = spla.factorized(K_red_csc)
        chi_red = np.column_stack([solve_op(F_red[:, m]) for m in range(self.n_cases)])
        chi = self.P_pbc @ chi_red  # (ndof, 3)

        # Compute mutual energies and element sensitivities
        edof = self.mesh.edofMat  # (N_elem, 8)
        Ke0 = self.Ke0            # (8, 8)
        Y_vol = self.cell_volume

        EH = np.zeros((3, 3), dtype=np.float64)
        dEH_drho = np.zeros((3, 3, self.mesh.n_elem), dtype=np.float64)

        # Vectorized mutual energy calculation
        # For each case m, extract element nodal displacement: u_elem_m of shape (N_elem, 8)
        u_elem = [self.U0[edof, m] - chi[edof, m] for m in range(self.n_cases)]

        for m in range(self.n_cases):
            u_m = u_elem[m]  # (N_elem, 8)
            u_m_Ke0 = u_m @ Ke0  # (N_elem, 8)
            for n in range(m, self.n_cases):
                u_n = u_elem[n]  # (N_elem, 8)
                # Element mutual strain energy: q_e^(mn) = sum_j (u_m @ Ke0)_j * u_n,j
                qe = np.sum(u_m_Ke0 * u_n, axis=1)  # (N_elem,)

                # Homogenized tensor component
                val = float(np.sum(E_e * qe) / Y_vol)
                EH[m, n] = val
                EH[n, m] = val

                # Exact self-adjoint sensitivity: dEH_mn / drho_e = 1/|Y| * dE_e/drho_e * qe
                grad = (dE_drho * qe) / Y_vol
                dEH_drho[m, n, :] = grad
                dEH_drho[n, m, :] = grad

        self.last_EH = EH
        self.last_dEH = dEH_drho
        self.last_chi = chi

        return EH, dEH_drho

    def effective_tensor(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate and return only the homogenized 3x3 elasticity matrix E^H."""
        EH, _ = self.homogenize(rho)
        return EH

    def bulk_modulus(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective 2D bulk modulus K^H and exact gradient dK^H/drho.

        K^H = 0.25 * (E_11^H + E_22^H + 2 * E_12^H).

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (K_H, dKH_drho) where K_H is scalar and dKH_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        KH = 0.25 * (EH[0, 0] + EH[1, 1] + 2.0 * EH[0, 1])
        dKH = 0.25 * (dEH[0, 0, :] + dEH[1, 1, :] + 2.0 * dEH[0, 1, :])
        return float(KH), dKH

    def shear_modulus(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective 2D shear modulus G^H and exact gradient dG^H/drho.

        G^H = E_33^H.

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (G_H, dGH_drho) where G_H is scalar and dGH_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        GH = float(EH[2, 2])
        dGH = dEH[2, 2, :]
        return GH, dGH

    def poissons_ratio(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective Poisson's ratio nu_12^H and exact gradient dnu^H/drho.

        nu_12^H = E_12^H / E_22^H.

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (nu_H, dnu_drho) where nu_H is scalar and dnu_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        e12 = EH[0, 1]
        e22 = EH[1, 1]

        if abs(e22) < 1e-14:
            raise ZeroDivisionError("E_22^H is too close to zero to evaluate Poisson's ratio")

        nu_H = float(e12 / e22)
        # Quotient rule: d(u/v) = (u' * v - u * v') / v^2
        dnu_drho = (dEH[0, 1, :] * e22 - e12 * dEH[1, 1, :]) / (e22 ** 2)
        return nu_H, dnu_drho


class Homogenizer3D(BaseHomogenizer):
    """3D Periodic Unit Cell asymptotic homogenizer with master-slave PBC reduction.

    Solves the 6 unit cell problems for macroscopic test strains in 3D Voigt order:
        m = 0: [1, 0, 0, 0, 0, 0]^T  (eps_xx = 1)
        m = 1: [0, 1, 0, 0, 0, 0]^T  (eps_yy = 1)
        m = 2: [0, 0, 1, 0, 0, 0]^T  (eps_zz = 1)
        m = 3: [0, 0, 0, 1, 0, 0]^T  (gamma_xy = 1)
        m = 4: [0, 0, 0, 0, 1, 0]^T  (gamma_yz = 1)
        m = 5: [0, 0, 0, 0, 0, 1]^T  (gamma_zx = 1)
    """

    def __init__(
        self,
        mesh: StructuredMesh3D,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
        penal: float = 3.0,
        material_model: str = "simp_modified",
    ) -> None:
        """Initialize 3D periodic homogenizer.

        Parameters
        ----------
        mesh : StructuredMesh3D
            Periodic unit cell 3D structured mesh.
        e0 : float
            Solid base Young's modulus.
        emin : float
            Void Young's modulus.
        nu : float
            Poisson's ratio.
        penal : float
            SIMP penalty exponent.
        material_model : str
            Material interpolation model.
        """
        super().__init__(
            e0=e0,
            emin=emin,
            nu=nu,
            penal=penal,
            material_model=material_model,
        )
        if not isinstance(mesh, StructuredMesh3D):
            raise TypeError(f"Homogenizer3D requires StructuredMesh3D, got {type(mesh)}")

        self.mesh: StructuredMesh3D = mesh
        self.fem: FEMEngine = FEMEngine(
            self.mesh,
            nu=self.nu,
            thickness=1.0,
            formulation="FI",
        )
        self.Ke0: np.ndarray = self.fem.Ke0  # (24, 24)
        self.C0: np.ndarray = self.fem.C0    # (6, 6)
        self.cell_volume: float = float(self.mesh.lx * self.mesh.ly * self.mesh.lz)
        self.n_cases: int = 6

        # Construct Master-Slave Periodic Boundary Condition matrix P_pbc
        self.P_pbc: sp.csc_matrix = self._build_pbc_projection()

        # Construct macroscopic unit strain displacement matrix U0 of shape (ndof, 6)
        self.U0: np.ndarray = self._build_macroscopic_displacements()

    def _build_pbc_projection(self) -> sp.csc_matrix:
        """Construct sparse boolean Master-Slave PBC reduction matrix P_pbc in 3D.

        Maps periodic boundary node pairs:
            - Face x = 0 is Master, x = nx is Slave.
            - Face y = 0 is Master, y = ny is Slave.
            - Face z = 0 is Master, z = nz is Slave.
            - All 8 corners mapping to (0,0,0) are constrained to 0
              to eliminate rigid body translations.
            - Interior nodes are Master.
        """
        nx = self.mesh.nelx
        ny = self.mesh.nely
        nz = self.mesh.nelz

        master_map = {}
        master_nodes = []
        node_to_master = np.full(self.mesh.n_nodes, -1, dtype=np.int64)

        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nid = (iz * (ny + 1) + iy) * (nx + 1) + ix
                    ix_c = 0 if ix == nx else ix
                    iy_c = 0 if iy == ny else iy
                    iz_c = 0 if iz == nz else iz

                    if (ix_c, iy_c, iz_c) == (0, 0, 0):
                        # Corner node: constrained to 0
                        node_to_master[nid] = -1
                    else:
                        can = (ix_c, iy_c, iz_c)
                        if can not in master_map:
                            master_map[can] = len(master_nodes)
                            master_nodes.append(can)
                        node_to_master[nid] = master_map[can]

        n_masters = len(master_nodes)
        rows = []
        cols = []
        data = []

        for nid in range(self.mesh.n_nodes):
            mid = node_to_master[nid]
            if mid >= 0:
                for d in range(3):
                    rows.append(nid * 3 + d)
                    cols.append(mid * 3 + d)
                    data.append(1.0)

        P_pbc = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.mesh.ndof, n_masters * 3),
            dtype=np.float64,
        ).tocsc()
        return P_pbc

    def _build_macroscopic_displacements(self) -> np.ndarray:
        """Build exact nodal displacements for 6 unit macroscopic strains in 3D.

        Returns
        -------
        np.ndarray
            (N_dof, 6) matrix containing linear displacement fields for:
            0: eps_xx = 1 -> u = x
            1: eps_yy = 1 -> v = y
            2: eps_zz = 1 -> w = z
            3: gamma_xy = 1 -> u = 0.5 * y, v = 0.5 * x
            4: gamma_yz = 1 -> v = 0.5 * z, w = 0.5 * y
            5: gamma_zx = 1 -> u = 0.5 * z, w = 0.5 * x
        """
        coords = self.mesh.node_coords
        U0 = np.zeros((self.mesh.ndof, 6), dtype=np.float64)

        # 0: eps_xx
        U0[0::3, 0] = coords[:, 0]
        # 1: eps_yy
        U0[1::3, 1] = coords[:, 1]
        # 2: eps_zz
        U0[2::3, 2] = coords[:, 2]
        # 3: gamma_xy
        U0[0::3, 3] = 0.5 * coords[:, 1]
        U0[1::3, 3] = 0.5 * coords[:, 0]
        # 4: gamma_yz
        U0[1::3, 4] = 0.5 * coords[:, 2]
        U0[2::3, 4] = 0.5 * coords[:, 1]
        # 5: gamma_zx
        U0[0::3, 5] = 0.5 * coords[:, 2]
        U0[2::3, 5] = 0.5 * coords[:, 0]

        return U0

    def homogenize(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3D homogenized elasticity tensor E^H and exact self-adjoint sensitivities.

        Parameters
        ----------
        rho : np.ndarray
            Physical density field array of shape (N_elem,).

        Returns
        -------
        EH : np.ndarray
            Homogenized 6x6 symmetric elasticity matrix E^H.
        dEH_drho : np.ndarray
            Exact analytical sensitivity tensor of shape (6, 6, N_elem).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        if rho_arr.shape[0] != self.mesh.n_elem:
            raise ValueError(f"rho shape {rho_arr.shape} does not match N_elem {self.mesh.n_elem}")

        E_e, dE_drho = self.interpolate_material(rho_arr)
        K = self.fem.assemble_K(E_e)

        # Reduced periodic stiffness: K_red = P_pbc^T @ K @ P_pbc
        K_red = self.P_pbc.T @ K @ self.P_pbc
        K_red_csc = K_red.tocsc()

        # Macroscopic load: F_macro = K @ U0, F_red = P_pbc^T @ F_macro
        F_macro = K @ self.U0
        F_red = self.P_pbc.T @ F_macro

        # Factorize K_red once and solve for all 6 macroscopic load cases
        try:
            import pypardiso
            chi_red = pypardiso.spsolve(K_red.tocsr(), F_red)
        except Exception:
            solve_op = spla.factorized(K_red_csc)
            chi_red = np.column_stack([solve_op(F_red[:, m]) for m in range(self.n_cases)])
        chi = self.P_pbc @ chi_red  # (ndof, 6)

        # Compute mutual energies and element sensitivities
        edof = self.mesh.edofMat  # (N_elem, 24)
        Ke0 = self.Ke0            # (24, 24)
        Y_vol = self.cell_volume

        EH = np.zeros((6, 6), dtype=np.float64)
        dEH_drho = np.zeros((6, 6, self.mesh.n_elem), dtype=np.float64)

        # Vectorized mutual energy calculation
        u_elem = [self.U0[edof, m] - chi[edof, m] for m in range(self.n_cases)]

        for m in range(self.n_cases):
            u_m = u_elem[m]  # (N_elem, 24)
            u_m_Ke0 = u_m @ Ke0  # (N_elem, 24)
            for n in range(m, self.n_cases):
                u_n = u_elem[n]  # (N_elem, 24)
                qe = np.sum(u_m_Ke0 * u_n, axis=1)  # (N_elem,)

                val = float(np.sum(E_e * qe) / Y_vol)
                EH[m, n] = val
                EH[n, m] = val

                grad = (dE_drho * qe) / Y_vol
                dEH_drho[m, n, :] = grad
                dEH_drho[n, m, :] = grad

        self.last_EH = EH
        self.last_dEH = dEH_drho
        self.last_chi = chi

        return EH, dEH_drho

    def effective_tensor(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate and return only the homogenized 6x6 elasticity matrix E^H."""
        EH, _ = self.homogenize(rho)
        return EH

    def bulk_modulus(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective 3D bulk modulus K^H and exact gradient dK^H/drho.

        K^H = 1/9 * sum_{i=1}^3 sum_{j=1}^3 E_{ij}^H.

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (K_H, dKH_drho) where K_H is scalar and dKH_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        KH = float(np.sum(EH[:3, :3]) / 9.0)
        dKH = np.sum(dEH[:3, :3, :], axis=(0, 1)) / 9.0
        return KH, dKH

    def shear_modulus(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective 3D shear modulus G^H and exact gradient dG^H/drho.

        G^H = 1/3 * (E_44^H + E_55^H + E_66^H).

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (G_H, dGH_drho) where G_H is scalar and dGH_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        GH = float((EH[3, 3] + EH[4, 4] + EH[5, 5]) / 3.0)
        dGH = (dEH[3, 3, :] + dEH[4, 4, :] + dEH[5, 5, :]) / 3.0
        return GH, dGH

    def poissons_ratio(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute effective 3D Poisson's ratio nu_12^H and exact gradient dnu^H/drho.

        nu_12^H = E_12^H / E_22^H.

        Parameters
        ----------
        rho : np.ndarray
            Element density array (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (nu_H, dnu_drho) where nu_H is scalar and dnu_drho is shape (N_elem,).
        """
        EH, dEH = self.homogenize(rho)
        e12 = EH[0, 1]
        e22 = EH[1, 1]

        if abs(e22) < 1e-14:
            raise ZeroDivisionError("E_22^H is too close to zero to evaluate Poisson's ratio")

        nu_H = float(e12 / e22)
        dnu_drho = (dEH[0, 1, :] * e22 - e12 * dEH[1, 1, :]) / (e22 ** 2)
        return nu_H, dnu_drho
