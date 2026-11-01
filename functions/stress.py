"""P-Norm von Mises stress constraint with qp-relaxation and exact adjoint sensitivity.

References:
    - Le et al. (2010), "Stress-based topology optimization for continua", SMO 41:605.
    - Holmberg et al. (2013), "Stress constrained topology optimization", SMO 48:33.
    - da Silva et al. (2019), "Stress-constrained topology optimization", SMO 60:885.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from .base import ConstraintFunction
from wht_topopt.algorithms.base import GradientSource
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.core.fem import FEMEngine
from wht_topopt.solvers.base import FEASolver


class PNormStressConstraint(ConstraintFunction):
    """Volume-weighted P-norm von Mises stress constraint with qp-relaxation.

    Implements:
        g_sigma(rho_hat) = (c_k * sigma_PN(rho_hat)) / sigma_allow - 1 <= 0

    where:
        sigma_e_vM0 = sqrt((sigma_e0)^T V sigma_e0 + sigma_reg^2)
        tilde_sigma_e_vM = rho_hat_e^q * sigma_e_vM0   (qp-relaxation, default q=0.5)
        sigma_PN = (sum_{e in cluster} v_e * (tilde_sigma_e_vM)^P)^(1/P)

    Exact Adjoint Sensitivity:
        dg_sigma / drho_hat_e = (c_k / sigma_allow) * (
            dsigma_PN / drho_hat_e|_u - (dE_e / drho_hat_e) * lambda_e^T Ke0 u_e
        )
        where K lambda = r_adj
        and r_adj_e = sigma_PN^(1-P) * v_e * (tilde_sigma_e_vM)^(P-1) * rho_hat_e^q * (1 / sigma_e_vM0) * B_e^T C_0^T V sigma_e0
    """

    name: str = "p_norm_stress"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        fem: FEMEngine,
        solver: FEASolver,
        bcm: BoundaryConditionManager,
        p: float = 8.0,
        sigma_allow: float = 1.0,
        q: float = 0.5,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-9,
        material_model: str = "simp_modified",
        sigma_reg: float = 1e-12,
        use_adaptive_scaling: bool = False,
        c_k: float = 1.0,
        clusters: Optional[List[np.ndarray]] = None,
        load_case: int = 0,
    ) -> None:
        """Initialize PNormStressConstraint.

        Parameters
        ----------
        fem : FEMEngine
            FEM engine containing reference Ke0, B0, C0, V_vm matrices.
        solver : FEASolver
            FEA linear solver.
        bcm : BoundaryConditionManager
            Boundary condition manager with fixed/free DOFs.
        p : float
            P-norm aggregation exponent (default 8.0).
        sigma_allow : float
            Allowable von Mises stress threshold.
        q : float
            Stress relaxation exponent in eta_S(rho) = rho^q (default 0.5).
        penal : float
            SIMP material penalty exponent for stiffness (default 3.0).
        e0 : float
            Base solid Young's modulus (default 1.0).
        emin : float
            Void Young's modulus (default 1e-9).
        material_model : str
            Stiffness interpolation model ('simp_modified' or 'simp').
        sigma_reg : float
            Regularization parameter to avoid 0/0 in unstressed elements (default 1e-12).
        use_adaptive_scaling : bool
            Whether to adaptively scale c_k = max(sigma_vM) / sigma_PN.
        c_k : float
            Initial scaling factor (default 1.0).
        clusters : list of np.ndarray, optional
            Subsets of element indices for multi-cluster active set stress aggregation.
        load_case : int
            Load case index to evaluate stress for (default 0).
        """
        self.fem = fem
        self.solver = solver
        self.bcm = bcm
        self.p = float(p)
        self.sigma_allow = float(sigma_allow)
        self.q = float(q)
        self.penal = float(penal)
        self.e0 = float(e0)
        self.emin = float(emin)
        self.material_model = material_model.lower()
        self.sigma_reg = float(sigma_reg)
        self.use_adaptive_scaling = bool(use_adaptive_scaling)
        self.c_k = float(c_k)
        self.clusters = clusters
        self.load_case = int(load_case)

        self.mesh = fem.mesh
        self.n_elem = fem.n_elem
        self.elem_volumes = np.asarray(fem.mesh.elem_volumes, dtype=np.float64)

        # Cache last computed stress fields for analysis and fast gradient reuse
        self._last_u: Optional[np.ndarray] = None
        self._last_rho: Optional[np.ndarray] = None
        self._last_sigma_vM0: Optional[np.ndarray] = None
        self._last_sigma_vM_relaxed: Optional[np.ndarray] = None
        self._last_sigma_PN: float = 0.0

    def interpolate_stiffness(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Young's modulus E(rho) and derivative dE/drho."""
        r = np.asarray(rho, dtype=np.float64)
        p = self.penal
        e0 = self.e0
        emin = self.emin

        if self.material_model == "simp_modified":
            delta = e0 - emin
            E = emin + (r ** p) * delta
            dE = p * (r ** (p - 1.0)) * delta
            return E, dE
        elif self.material_model == "simp":
            E = e0 * (r ** p)
            dE = p * e0 * (r ** (p - 1.0))
            return E, dE
        else:
            raise ValueError(f"Unsupported material model: {self.material_model}")

    def interpolate_stress_relaxation(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute stress relaxation factor eta_S(rho) = rho^q and derivative deta_S/drho = q * rho^(q-1)."""
        r = np.maximum(np.asarray(rho, dtype=np.float64), 1e-15)
        q = self.q
        eta = r ** q
        deta = q * (r ** (q - 1.0))
        return eta, deta

    def compute_element_micro_stresses(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unpenalized microscopic Cauchy stress sigma0 and von Mises stress sigma_vM0.

        Parameters
        ----------
        u : np.ndarray
            Full displacement vector (ndof,).

        Returns
        -------
        sigma0 : np.ndarray
            (n_elem, n_comp) microscopic Cauchy stresses in Voigt notation.
        sigma_vM0 : np.ndarray
            (n_elem,) regularized von Mises stress sqrt(sigma0^T V sigma0 + sigma_reg^2).
        """
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.ndim != 1 or u_arr.shape[0] != self.fem.ndof:
            raise ValueError(f"u shape must be ({self.fem.ndof},), got {u_arr.shape}")

        ue = u_arr[self.fem.mesh.edofMat]  # (n_elem, d)

        if self.fem.is_uniform:
            # Strains: eps = ue @ B0^T -> (n_elem, n_comp)
            strains = ue @ self.fem.B0.T
            sigma0 = self.e0 * (strains @ self.fem.C0.T)
        else:
            strains = np.einsum("esd,ed->es", self.fem.B0_batch, ue)
            sigma0 = self.e0 * (strains @ self.fem.C0.T)

        # Von Mises: sqrt(sigma0 @ V_vm @ sigma0 + sigma_reg^2)
        # V_vm is (n_comp, n_comp)
        sigma_V = sigma0 @ self.fem.V_vm  # (n_elem, n_comp)
        vm_sq = np.sum(sigma_V * sigma0, axis=1) + (self.sigma_reg ** 2)
        sigma_vM0 = np.sqrt(np.maximum(0.0, vm_sq))

        return sigma0, sigma_vM0

    def compute_relaxed_von_mises(
        self, rho: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute relaxed von Mises stress tilde_sigma_vM = rho^q * sigma_vM0.

        Returns
        -------
        sigma0 : (n_elem, n_comp)
        sigma_vM0 : (n_elem,)
        tilde_sigma_vM : (n_elem,)
        """
        sigma0, sigma_vM0 = self.compute_element_micro_stresses(u)
        eta, _ = self.interpolate_stress_relaxation(rho)
        tilde_sigma_vM = eta * sigma_vM0
        return sigma0, sigma_vM0, tilde_sigma_vM

    def compute_p_norm_stress(
        self, tilde_sigma_vM: np.ndarray, cluster: Optional[np.ndarray] = None
    ) -> float:
        """Compute volume-weighted P-norm aggregated stress.

        sigma_PN = (sum_{e in cluster} v_e * (tilde_sigma_e_vM)^P)^(1/P)
        """
        P = self.p
        if cluster is None:
            v = self.elem_volumes
            s = tilde_sigma_vM
        else:
            v = self.elem_volumes[cluster]
            s = tilde_sigma_vM[cluster]

        sum_term = np.sum(v * (s ** P))
        # Protect against non-positive sum
        if sum_term <= 0.0:
            return 0.0
        return float(sum_term ** (1.0 / P))

    def update_scaling(self, rho: np.ndarray, u: np.ndarray) -> float:
        """Update adaptive scaling parameter c_k = max(tilde_sigma_vM) / sigma_PN."""
        _, _, tilde_sigma_vM = self.compute_relaxed_von_mises(rho, u)
        sigma_pn = self.compute_p_norm_stress(tilde_sigma_vM)
        max_stress = float(np.max(tilde_sigma_vM))
        if sigma_pn > 1e-15:
            self.c_k = max_stress / sigma_pn
        else:
            self.c_k = 1.0
        return self.c_k

    def evaluate(self, rho: np.ndarray, u: Optional[np.ndarray] = None, **kwargs: Any) -> float:
        """Evaluate scalar constraint violation g_sigma(rho) <= 0.

        Parameters
        ----------
        rho : np.ndarray
            Physical density field (N_elem,).
        u : np.ndarray, optional
            Displacement field (ndof,). If None, solved via self.solver.
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        if u is None:
            E_arr, _ = self.interpolate_stiffness(rho_arr)
            K = self.fem.assemble_K(E_arr)
            f = self.bcm.F[:, self.load_case]
            res = self.solver.solve(K, f, free_dofs=self.bcm.free_dofs, fixed_dofs=self.bcm.fixed_dofs)
            u_arr = res.u.ravel()
        else:
            u_arr = np.asarray(u, dtype=np.float64).ravel()

        sigma0, sigma_vM0, tilde_sigma_vM = self.compute_relaxed_von_mises(rho_arr, u_arr)
        sigma_pn = self.compute_p_norm_stress(tilde_sigma_vM)

        # Cache results for gradient calculation
        self._last_u = u_arr.copy()
        self._last_rho = rho_arr.copy()
        self._last_sigma_vM0 = sigma_vM0
        self._last_sigma_vM_relaxed = tilde_sigma_vM
        self._last_sigma_PN = sigma_pn

        scale = self.c_k if self.use_adaptive_scaling else 1.0
        g_val = (scale * sigma_pn) / self.sigma_allow - 1.0
        return float(g_val)

    def gradient(self, rho: np.ndarray, u: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
        """Compute exact adjoint sensitivity dg_sigma / drho.

        Parameters
        ----------
        rho : np.ndarray
            Physical density field (N_elem,).
        u : np.ndarray, optional
            Displacement field (ndof,).

        Returns
        -------
        dg_drho : np.ndarray
            Sensitivity vector of shape (N_elem,).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)

        if u is None:
            E_arr, dE_drho = self.interpolate_stiffness(rho_arr)
            K = self.fem.assemble_K(E_arr)
            f = self.bcm.F[:, self.load_case]
            res = self.solver.solve(K, f, free_dofs=self.bcm.free_dofs, fixed_dofs=self.bcm.fixed_dofs)
            u_arr = res.u.ravel()
        else:
            u_arr = np.asarray(u, dtype=np.float64).ravel()
            _, dE_drho = self.interpolate_stiffness(rho_arr)
            K = kwargs.get("K", None)
            if K is None:
                E_arr, _ = self.interpolate_stiffness(rho_arr)
                K = self.fem.assemble_K(E_arr)

        sigma0, sigma_vM0 = self.compute_element_micro_stresses(u_arr)
        eta, deta_drho = self.interpolate_stress_relaxation(rho_arr)
        tilde_sigma_vM = eta * sigma_vM0
        sigma_pn = self.compute_p_norm_stress(tilde_sigma_vM)

        if sigma_pn <= 1e-15:
            return np.zeros(self.n_elem, dtype=np.float64)

        P = self.p
        v = self.elem_volumes
        scale = self.c_k if self.use_adaptive_scaling else 1.0

        # 1. Direct explicit density derivative term:
        # dsigma_PN / drho_e|_u = sigma_PN^(1-P) * v_e * (tilde_sigma_e)^(P-1) * deta_drho * sigma_vM0
        term_common = (sigma_pn ** (1.0 - P)) * v * (tilde_sigma_vM ** (P - 1.0))
        dsigma_PN_drho_u = term_common * deta_drho * sigma_vM0  # (n_elem,)

        # 2. Adjoint RHS vector assembly:
        # r_adj_e = term_common * eta * (1 / sigma_vM0) * (B_e^T C_0^T V sigma0_e)
        # alpha_e = term_common * eta / sigma_vM0
        alpha_e = term_common * eta / np.maximum(sigma_vM0, 1e-30)  # (n_elem,)

        # V sigma0_e: (n_elem, n_comp)
        V_sigma0 = sigma0 @ self.fem.V_vm  # (n_elem, n_comp)
        # C_0^T (V sigma0_e) = (V sigma0_e) @ C_0
        C_V_sigma0 = self.e0 * (V_sigma0 @ self.fem.C0)  # (n_elem, n_comp)

        # alpha_e * C_V_sigma0
        scaled_vec = alpha_e[:, None] * C_V_sigma0  # (n_elem, n_comp)

        # B_e^T @ scaled_vec_e: (n_elem, d)
        if self.fem.is_uniform:
            # B0 is (n_comp, d)
            # r_adj_e = scaled_vec @ B0: (n_elem, d)
            r_adj_e = scaled_vec @ self.fem.B0
        else:
            # B0_batch is (n_elem, n_comp, d)
            r_adj_e = np.einsum("es,esd->ed", scaled_vec, self.fem.B0_batch)

        # Assemble global adjoint load vector r_adj of shape (ndof,)
        r_adj = np.zeros(self.fem.ndof, dtype=np.float64)
        np.add.at(r_adj, self.fem.mesh.edofMat, r_adj_e)

        # 3. Solve adjoint equation K lambda = r_adj
        if hasattr(self.solver, "solve_cached"):
            try:
                lam = self.solver.solve_cached(r_adj)
            except Exception:
                res_adj = self.solver.solve(
                    K, r_adj, free_dofs=self.bcm.free_dofs, fixed_dofs=self.bcm.fixed_dofs
                )
                lam = res_adj.u.ravel()
        else:
            res_adj = self.solver.solve(
                K, r_adj, free_dofs=self.bcm.free_dofs, fixed_dofs=self.bcm.fixed_dofs
            )
            lam = res_adj.u.ravel()

        lam_arr = np.asarray(lam, dtype=np.float64).ravel()

        # 4. Adjoint inner product per element: lambda_e^T Ke0 u_e
        ue = u_arr[self.fem.mesh.edofMat]     # (n_elem, d)
        lame = lam_arr[self.fem.mesh.edofMat] # (n_elem, d)

        if self.fem.is_uniform:
            Ke_u = ue @ self.fem.Ke0  # (n_elem, d)
            lam_Ke_u = np.sum(lame * Ke_u, axis=1)  # (n_elem,)
        else:
            Ke_u = np.einsum("eij,ej->ei", self.fem.Ke0_batch, ue)
            lam_Ke_u = np.sum(lame * Ke_u, axis=1)

        # 5. Total sensitivity:
        # dg_drho = (scale / sigma_allow) * (dsigma_PN_drho_u - dE_drho * lam_Ke_u)
        dsigma_PN_total = dsigma_PN_drho_u - (dE_drho * lam_Ke_u)
        dg_drho = (scale / self.sigma_allow) * dsigma_PN_total

        return dg_drho
