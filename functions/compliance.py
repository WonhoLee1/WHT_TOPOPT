"""Compliance objective function for single and multi-load case topology optimization."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import ObjectiveFunction
from wht_topopt.algorithms.base import GradientSource


class ComplianceFunction(ObjectiveFunction):
    """Linear static compliance objective function:
    
        c(rho) = sum_l w_l * f_l^T * u_l = sum_l w_l * sum_e E_e(rho_e) * u_{l,e}^T * K_e0 * u_{l,e}
        
    With self-adjoint analytical sensitivities:
        dc / drho_e = - sum_l w_l * (dE_e / drho_e) * (u_{l,e}^T * K_e0 * u_{l,e})
    """

    name: str = "compliance"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        material_model: str = "simp_modified",
        elem_dofs: Optional[np.ndarray] = None,
        K_e0: Optional[np.ndarray] = None,
        fem: Optional[Any] = None,
        solver: Optional[Any] = None,
        bcm: Optional[Any] = None,
        ramp_q: float = 3.0,
    ) -> None:
        """Initialize ComplianceFunction.
        
        Args:
            weights: Weights w_l for multi-load case summation (default equal weights).
            penal: SIMP / RAMP penalty exponent p.
            e0: Base solid Young's modulus E0.
            emin: Void Young's modulus Emin (avoids singularity).
            material_model: 'simp_modified', 'simp', or 'ramp'.
            elem_dofs: Optional array of degree-of-freedom indices per element (N_elem, d_e).
            K_e0: Unit element stiffness matrix (d_e, d_e) or batch (N_elem, d_e, d_e).
            fem: Optional FEMEngine for self-contained assembly and solve.
            solver: Optional FEASolver for self-contained displacement solve.
            bcm: Optional BoundaryConditionManager for self-contained displacement solve.
            ramp_q: RAMP penalty factor if material_model == 'ramp'.
        """
        self.weights = weights
        self.penal = float(penal)
        self.e0 = float(e0)
        self.emin = float(emin)
        self.material_model = material_model.lower()
        self.fem = fem
        self.solver = solver
        self.bcm = bcm
        self.ramp_q = float(ramp_q)

        if elem_dofs is not None:
            self.elem_dofs: Optional[np.ndarray] = np.asarray(elem_dofs, dtype=np.int64)
        elif fem is not None:
            self.elem_dofs = np.asarray(fem.mesh.edofMat, dtype=np.int64)
        else:
            self.elem_dofs = None

        if K_e0 is not None:
            self.K_e0: Optional[np.ndarray] = np.asarray(K_e0, dtype=float)
        elif fem is not None:
            self.K_e0 = np.asarray(fem.Ke0 if fem.is_uniform else fem.Ke0_batch, dtype=float)
        else:
            self.K_e0 = None

        # Cached element strain energies for sensitivity reuse
        self._last_strain_energies: Optional[np.ndarray] = None  # (N_cases, N_elem)
        self._last_rho: Optional[np.ndarray] = None
        self._last_u: Optional[np.ndarray] = None

    def interpolate_material(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute interpolated Young's modulus E_e and its derivative dE_e/drho_e.
        
        Returns:
            (E_e, dE_e_drho_e)
        """
        rho_arr = np.asarray(rho, dtype=float)
        p = self.penal
        e0 = self.e0
        emin = self.emin

        if self.material_model == "simp_modified":
            # E(rho) = emin + rho^p * (e0 - emin)
            delta = e0 - emin
            E = emin + (rho_arr ** p) * delta
            dE = p * (rho_arr ** (p - 1.0)) * delta
            return E, dE
        elif self.material_model == "simp":
            # E(rho) = e0 * rho^p
            E = e0 * (rho_arr ** p)
            dE = p * e0 * (rho_arr ** (p - 1.0))
            return E, dE
        elif self.material_model == "ramp":
            # E(rho) = emin + (rho / (1 + p*(1 - rho))) * (e0 - emin)
            delta = e0 - emin
            den = 1.0 + p * (1.0 - rho_arr)
            E = emin + (rho_arr / den) * delta
            dE = ((p + 1.0) * delta) / (den ** 2)
            return E, dE
        else:
            raise ValueError(f"Unknown material model: {self.material_model}")

    def _solve_if_needed(self, rho_arr: np.ndarray) -> np.ndarray:
        """Solve K(rho) u = F using attached fem, solver, and bcm."""
        if self.fem is None or self.solver is None or self.bcm is None:
            raise ValueError(
                "Either (element_strain_energies), (u, elem_dofs, K_e0), (u, loads), "
                "or (fem, solver, bcm) must be provided"
            )
        E_arr, _ = self.interpolate_material(rho_arr)
        K = self.fem.assemble_K(E_arr)
        res = self.solver.solve(
            K,
            self.bcm.F,
            free_dofs=self.bcm.free_dofs,
            fixed_dofs=self.bcm.fixed_dofs,
        )
        u_sol = res.u
        if u_sol.ndim == 2 and u_sol.shape[1] == 1:
            u_sol = u_sol.ravel()
        elif u_sol.ndim == 2:
            u_sol = u_sol.T
        self._last_u = u_sol.copy()
        return u_sol

    def compute_element_strain_energies(
        self,
        u: np.ndarray,
        elem_dofs: Optional[np.ndarray] = None,
        K_e0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute base unit strain energy s_{l,e} = u_{l,e}^T * K_e0 * u_{l,e}.
        
        Args:
            u: Nodal displacement array of shape (N_dofs,) or (N_cases, N_dofs).
            elem_dofs: Degree-of-freedom indices per element (N_elem, d_e).
            K_e0: Base unit stiffness matrix (d_e, d_e) or (N_elem, d_e, d_e).
            
        Returns:
            Array of shape (N_cases, N_elem) containing u_{l,e}^T K_e0 u_{l,e}.
        """
        edofs = elem_dofs if elem_dofs is not None else self.elem_dofs
        ke0 = K_e0 if K_e0 is not None else self.K_e0

        if edofs is None or ke0 is None:
            raise ValueError("elem_dofs and K_e0 are required to compute element strain energies")

        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(1, -1)
        elif u_arr.ndim == 2 and u_arr.shape[1] <= int(np.max(edofs)) and u_arr.shape[0] > int(np.max(edofs)):
            u_arr = u_arr.T

        n_cases = u_arr.shape[0]
        n_elem = edofs.shape[0]

        # Extract element displacements: (n_cases, n_elem, d_e)
        u_elem = u_arr[:, edofs]  # fancy indexing: shape (n_cases, n_elem, d_e)

        if ke0.ndim == 2:
            # Single base matrix shared across all elements
            # Einstein summation: c = case, e = element, i/j = element dof
            energies = np.einsum("cei,ij,cej->ce", u_elem, ke0, u_elem)
        elif ke0.ndim == 3:
            # Per-element unique stiffness matrices
            energies = np.einsum("cei,eij,cej->ce", u_elem, ke0, u_elem)
        else:
            raise ValueError(f"Invalid K_e0 shape {ke0.shape}")

        return energies

    def evaluate(
        self,
        rho: np.ndarray,
        u: Optional[np.ndarray] = None,
        loads: Optional[np.ndarray] = None,
        elem_dofs: Optional[np.ndarray] = None,
        K_e0: Optional[np.ndarray] = None,
        element_strain_energies: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate scalar compliance c = sum_l w_l * f_l^T * u_l.
        
        Can evaluate either directly via f^T * u or from element strain energies.
        """
        rho_arr = np.asarray(rho, dtype=float)
        self._last_rho = rho_arr.copy()

        # 1. If element strain energies or FE matrices are available
        edofs = elem_dofs if elem_dofs is not None else self.elem_dofs
        ke0 = K_e0 if K_e0 is not None else self.K_e0

        if u is None and element_strain_energies is None and self.fem is not None:
            u = self._solve_if_needed(rho_arr)

        if element_strain_energies is not None:
            se = np.asarray(element_strain_energies, dtype=float)
            if se.ndim == 1:
                se = se.reshape(1, -1)
            self._last_strain_energies = se
        elif u is not None and edofs is not None and ke0 is not None:
            self._last_u = np.asarray(u, dtype=float).copy()
            se = self.compute_element_strain_energies(u, edofs, ke0)
            self._last_strain_energies = se
        elif u is not None and loads is not None:
            # Global force-displacement dot product
            u_arr = np.asarray(u, dtype=float)
            f_arr = np.asarray(loads, dtype=float)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape(1, -1)
            if f_arr.ndim == 1:
                f_arr = f_arr.reshape(1, -1)

            n_cases = u_arr.shape[0]
            w = self.weights if self.weights is not None else [1.0 / n_cases] * n_cases

            total_c = 0.0
            for l in range(n_cases):
                total_c += float(w[l]) * float(np.dot(f_arr[l], u_arr[l]))
            return total_c
        else:
            raise ValueError(
                "Either (element_strain_energies), (u, elem_dofs, K_e0), or (u, loads) must be provided"
            )

        # Summation via material interpolation
        E, _ = self.interpolate_material(rho_arr)
        n_cases = self._last_strain_energies.shape[0]
        w = self.weights if self.weights is not None else [1.0 / n_cases] * n_cases

        c_total = 0.0
        for l in range(n_cases):
            c_l = float(np.sum(E * self._last_strain_energies[l]))
            c_total += float(w[l]) * c_l

        return c_total

    def gradient(
        self,
        rho: np.ndarray,
        u: Optional[np.ndarray] = None,
        elem_dofs: Optional[np.ndarray] = None,
        K_e0: Optional[np.ndarray] = None,
        element_strain_energies: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute exact self-adjoint sensitivities:
        
            dc / drho_e = - sum_l w_l * (dE_e / drho_e) * (u_{l,e}^T * K_e0 * u_{l,e})
        """
        rho_arr = np.asarray(rho, dtype=float)
        _, dE = self.interpolate_material(rho_arr)

        # Retrieve or compute strain energies
        if element_strain_energies is not None:
            se = np.asarray(element_strain_energies, dtype=float)
            if se.ndim == 1:
                se = se.reshape(1, -1)
        elif self._last_strain_energies is not None and (
            self._last_rho is not None and np.allclose(rho_arr, self._last_rho)
        ):
            se = self._last_strain_energies
        elif u is not None:
            edofs = elem_dofs if elem_dofs is not None else self.elem_dofs
            ke0 = K_e0 if K_e0 is not None else self.K_e0
            se = self.compute_element_strain_energies(u, edofs, ke0)
        elif self.fem is not None:
            u_sol = self._solve_if_needed(rho_arr)
            se = self.compute_element_strain_energies(u_sol, self.elem_dofs, self.K_e0)
        else:
            raise ValueError("Displacements u or element_strain_energies are required for compliance gradient")

        n_cases = se.shape[0]
        w = self.weights if self.weights is not None else [1.0 / n_cases] * n_cases

        # Weighted combination across load cases
        weighted_se = np.zeros(len(rho_arr), dtype=float)
        for l in range(n_cases):
            weighted_se += float(w[l]) * se[l]

        # Self-adjoint gradient is negative: - dE * u_e^T K_e0 u_e
        grad = -dE * weighted_se
        return grad
