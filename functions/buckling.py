"""Linearized buckling constraint function with exact non-self-adjoint sensitivities.

Implements the Ferrari, Sigmund & Guest (2021) inverse-eigenvalue formulation:
    K(rho) phi_j = - mu_j * KG(rho, u(rho)) phi_j
where mu_j = 1 / lambda_j is the inverse buckling load factor (BLF).
The fundamental buckling mode corresponds to the largest positive eigenvalue mu_1.
Multiple modes are aggregated via Kreisselmeier-Steinhauser (KS) aggregation:
    mu_KS = mu_1 + (1 / rho_KS) * ln(sum_j exp(rho_KS * (mu_j - mu_1)))
Constraint:
    g_buck(rho) = mu_KS * lambda_crit_target - 1.0 <= 0
    (equivalent to lambda_crit >= lambda_crit_target).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

from .base import ConstraintFunction
from wht_topopt.algorithms.base import GradientSource
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.solvers.base import FEASolver
from wht_topopt.solvers.scipy_solver import ScipySolver


class BucklingConstraint(ConstraintFunction):
    """Linearized buckling constraint function with Kreisselmeier-Steinhauser aggregation.

    Constraint formulation:
        g(rho) = mu_KS * lambda_crit_target - 1.0 <= 0

    Sensitivities are evaluated using the exact non-self-adjoint formulation
    including the stress-dependent adjoint displacement term w_KS:
        dmu_KS / drho_e = sum_j z_j * [ -phi_{j,e}^T (dEG_e KG_e^0 + mu_j dEK_e K_e0) phi_{j,e} ]
                          + dEK_e * (w_{KS,e}^T K_e0 u_e)
    where K w_KS = sum_j z_j psi_j is solved in a single adjoint solve.
    """

    name: str = "buckling_constraint"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        fem: FEMEngine,
        bcm: BoundaryConditionManager,
        solver: Optional[FEASolver] = None,
        lambda_crit_target: float = 1.0,
        n_modes: int = 5,
        rho_ks: float = 50.0,
        penal_k: float = 3.0,
        penal_g: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        load_case: int = 0,
    ) -> None:
        """Initialize BucklingConstraint.

        Parameters
        ----------
        fem : FEMEngine
            Vectorized finite element engine.
        bcm : BoundaryConditionManager
            Boundary condition manager with fixed DOFs and compressive load.
        solver : FEASolver, optional
            Sparse linear system solver (default: ScipySolver direct).
        lambda_crit_target : float
            Target minimum critical buckling load factor (default 1.0).
        n_modes : int
            Number of lowest buckling modes (largest mu) in KS aggregation (default 5).
        rho_ks : float
            Kreisselmeier-Steinhauser smoothness parameter (default 50.0).
        penal_k : float
            SIMP penalty exponent for elastic stiffness K (default 3.0).
        penal_g : float
            SIMP penalty exponent for geometric stiffness KG (default 3.0).
        e0 : float
            Base solid material Young's modulus (default 1.0).
        emin : float
            Void artificial Young's modulus for stiffness (default 1e-6).
        load_case : int
            Load case index for static prebuckling stress analysis (default 0).
        """
        self.fem = fem
        self.bcm = bcm
        self.solver = solver or ScipySolver(method="direct")
        self.lambda_crit_target = float(lambda_crit_target)
        self.n_modes = int(n_modes)
        self.rho_ks = float(rho_ks)
        self.penal_k = float(penal_k)
        self.penal_g = float(penal_g)
        self.e0 = float(e0)
        self.emin = float(emin)
        self.load_case = int(load_case)

        self.mesh = self.fem.mesh
        self.n_elem = self.mesh.n_elem
        self.ndof = self.fem.ndof
        self.edof = self.mesh.edofMat

        # Cache for value and gradient evaluation
        self._last_rho: Optional[np.ndarray] = None
        self._last_val: Optional[float] = None
        self._last_grad: Optional[np.ndarray] = None
        self._last_mu_ks: Optional[float] = None
        self._last_vals: Optional[np.ndarray] = None
        self._last_phis: Optional[np.ndarray] = None
        self._last_u: Optional[np.ndarray] = None

    def interpolate_material(
        self, rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute material stiffness and geometric stiffness interpolations and derivatives.

        Returns
        -------
        Ek : (n_elem,) elastic stiffness factor
        dEk : (n_elem,) dEk / drho
        Eg : (n_elem,) geometric stiffness factor
        dEg : (n_elem,) dEg / drho
        """
        r = np.asarray(rho, dtype=np.float64)
        pk = self.penal_k
        pg = self.penal_g
        e0 = self.e0
        emin = self.emin

        # Elastic stiffness: E_K = emin + rho^pk * (e0 - emin)
        delta_k = e0 - emin
        Ek = emin + (r ** pk) * delta_k
        dEk = pk * (r ** (pk - 1.0)) * delta_k

        # Geometric stiffness: E_G = rho^pg * e0 (strictly 0 for void)
        Eg = (r ** pg) * e0
        dEg = pg * (r ** (pg - 1.0)) * e0

        return Ek, dEk, Eg, dEg

    def solve_prebuckling(self, Ek: np.ndarray, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, sp.csc_matrix]:
        """Solve static equilibrium K(rho) u = F for prebuckling displacement."""
        K = self.fem.assemble_K(Ek)
        if u is not None:
            u_full = np.asarray(u, dtype=np.float64).ravel()
            return u_full, K

        f_vec = self.bcm.F[:, self.load_case]
        res = self.solver.solve(
            K,
            f_vec,
            free_dofs=self.bcm.free_dofs,
            fixed_dofs=self.bcm.fixed_dofs,
        )
        return res.u.ravel(), K

    def solve_buckling_modes(
        self,
        rho: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, sp.csc_matrix, sp.csc_matrix]:
        """Solve generalized buckling eigenvalue problem (-KG_ff) phi = mu K_ff phi.

        Returns
        -------
        vals : (n_modes,) largest positive eigenvalues mu_1 >= mu_2 >= ...
        phis : (n_modes, ndof) K-orthonormal global buckling mode shapes
        u : (ndof,) static displacement vector
        K : global stiffness matrix
        KG : global geometric stiffness matrix
        """
        Ek, _, Eg, _ = self.interpolate_material(rho)
        u_full, K = self.solve_prebuckling(Ek, u=u)

        # Assemble geometric stiffness KG
        KG = self.fem.assemble_KG(Eg, u_full)

        free = self.bcm.free_dofs
        n_free = len(free)
        n_req = min(self.n_modes, n_free)

        K_ff = K[free, :][:, free]
        KG_ff = KG[free, :][:, free]

        # Invert-free or dense solve depending on dimension
        if n_free <= 40 or n_req >= n_free - 1:
            # Dense generalized eigenvalue problem: (-KG_ff) phi = mu K_ff phi
            K_dense = K_ff.toarray()
            A_dense = -KG_ff.toarray()
            # Symmetric generalized eigenvalue solver
            all_vals, all_vecs = la.eigh(A_dense, K_dense)
            # Sort descending: largest algebraic eigenvalues (compressive buckling)
            sort_idx = np.argsort(all_vals)[::-1]
            vals = all_vals[sort_idx[:n_req]]
            vecs = all_vecs[:, sort_idx[:n_req]]
        else:
            # Sparse generalized eigenvalue problem using eigsh with shift-invert / Minv
            use_pardiso = (getattr(self.solver, "method", None) == "pardiso")
            if use_pardiso:
                try:
                    import pypardiso
                    ps = getattr(self.solver, "_last_pardiso", None) or pypardiso.PyPardisoSolver()
                    K_ff_csr = K_ff.tocsr()
                    A_ff_csr = (-KG_ff).tocsr()
                    ps.factorize(K_ff_csr)
                    Minv = spla.LinearOperator((n_free, n_free), matvec=lambda b: ps.solve(K_ff_csr, b))
                    M_mat = K_ff_csr
                    A_mat = A_ff_csr
                except Exception:
                    use_pardiso = False

            if not use_pardiso:
                K_ff_csc = K_ff.tocsc()
                A_ff_csc = (-KG_ff).tocsc()
                K_lu = spla.splu(K_ff_csc)
                Minv = spla.LinearOperator((n_free, n_free), matvec=K_lu.solve)
                M_mat = K_ff_csc
                A_mat = A_ff_csc

            try:
                raw_vals, raw_vecs = spla.eigsh(
                    A=A_mat,
                    k=n_req,
                    M=M_mat,
                    which="LA",
                    Minv=Minv,
                    tol=1e-6,
                    maxiter=2000,
                )
                sort_idx = np.argsort(raw_vals)[::-1]
                vals = raw_vals[sort_idx]
                vecs = raw_vecs[:, sort_idx]
            except Exception as e:
                if n_free <= 1000:
                    # Robust fallback to dense solver only for small dimension
                    all_vals, all_vecs = la.eigh(A_mat.toarray(), M_mat.toarray())
                    sort_idx = np.argsort(all_vals)[::-1]
                    vals = all_vals[sort_idx[:n_req]]
                    vecs = all_vecs[:, sort_idx[:n_req]]
                else:
                    raise RuntimeError(f"ARPACK eigensolver failed to converge for n_free={n_free}: {e}") from e

        # Reconstruct global mode shapes phi of shape (n_req, ndof)
        phis = np.zeros((n_req, self.ndof), dtype=np.float64)
        for j in range(n_req):
            phis[j, free] = vecs[:, j]

        return vals, phis, u_full, K, KG

    def _compute_state(
        self, rho: np.ndarray, **kwargs: Any
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, sp.csc_matrix, sp.csc_matrix]:
        """Compute all buckling eigenvalues, modes, displacements, and KS aggregation."""
        u_in = kwargs.get("u", None)
        vals, phis, u, K, KG = self.solve_buckling_modes(rho, u=u_in)

        # KS aggregation of eigenvalues mu_1, ..., mu_m
        mu_max = vals[0]
        exp_terms = np.exp(np.clip(self.rho_ks * (vals - mu_max), -700.0, 700.0))
        sum_exp = float(np.sum(exp_terms))
        mu_ks = float(mu_max + (1.0 / self.rho_ks) * np.log(sum_exp))
        z_weights = exp_terms / sum_exp

        val = mu_ks * self.lambda_crit_target - 1.0
        return val, mu_ks, vals, z_weights, phis, u, K, KG

    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the scalar buckling constraint g(rho) <= 0.

        Parameters
        ----------
        rho : np.ndarray
            Element physical density field (n_elem,).

        Returns
        -------
        float
            Constraint violation g(rho) = mu_KS * lambda_crit_target - 1.0.
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if self._last_rho is not None and np.array_equal(r, self._last_rho):
            if self._last_val is not None:
                return self._last_val

        val, mu_ks, vals, z, phis, u, K, KG = self._compute_state(r, **kwargs)

        self._last_rho = r.copy()
        self._last_val = val
        self._last_mu_ks = mu_ks
        self._last_vals = vals
        self._last_phis = phis
        self._last_u = u
        self._last_grad = None

        return val

    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate the analytical sensitivity dg / drho using exact adjoint solve.

        Parameters
        ----------
        rho : np.ndarray
            Element physical density field (n_elem,).

        Returns
        -------
        np.ndarray
            Sensitivity array dg / drho of shape (n_elem,).
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if (
            self._last_rho is not None
            and np.array_equal(r, self._last_rho)
            and self._last_grad is not None
        ):
            return self._last_grad

        # Ensure state is computed
        if (
            self._last_rho is None
            or not np.array_equal(r, self._last_rho)
            or self._last_vals is None
        ):
            self.evaluate(r, **kwargs)

        assert self._last_vals is not None
        assert self._last_phis is not None
        assert self._last_u is not None

        vals = self._last_vals
        phis = self._last_phis
        u = self._last_u
        n_modes = len(vals)

        # KS weights
        mu_max = vals[0]
        exp_terms = np.exp(np.clip(self.rho_ks * (vals - mu_max), -700.0, 700.0))
        z = exp_terms / np.sum(exp_terms)

        Ek, dEk, Eg, dEg = self.interpolate_material(r)
        ue = u[self.edof]
        KG0_e = self.fem.compute_element_kg0(u)  # (n_elem, d, d)

        # Direct eigenvalue sensitivities
        # dmu_j / drho_e (direct) = - phi_{j,e}^T (dEg KG0_e + mu_j dEk Ke0) phi_{j,e}
        grad_direct = np.zeros(self.n_elem, dtype=np.float64)
        for j in range(n_modes):
            phi_e = phis[j][self.edof]  # (n_elem, d)
            term_KG = np.einsum("ei,eij,ej->e", phi_e, KG0_e, phi_e)
            if self.fem.is_uniform:
                term_K = np.einsum("ei,ij,ej->e", phi_e, self.fem.Ke0, phi_e)
            else:
                term_K = np.einsum("ei,eij,ej->e", phi_e, self.fem.Ke0_batch, phi_e)

            dmu_j_direct = - (dEg * term_KG + vals[j] * dEk * term_K)
            grad_direct += z[j] * dmu_j_direct

        # Stress-dependent adjoint force psi_KS = sum_j z_j * psi_j
        psi_all = self.fem.compute_buckling_adjoint_forces(phis, Eg)  # (n_modes, ndof)
        psi_ks = np.tensordot(z, psi_all, axes=(0, 0))              # (ndof,)

        # Single adjoint solve K w_KS = psi_KS
        K = self.fem.assemble_K(Ek)
        w_res = self.solver.solve(
            K,
            psi_ks,
            free_dofs=self.bcm.free_dofs,
            fixed_dofs=self.bcm.fixed_dofs,
        )
        w_ks = w_res.u.ravel()
        w_e = w_ks[self.edof]

        # Adjoint term: dEk * w_e^T Ke0 u_e
        if self.fem.is_uniform:
            adj_term = dEk * np.einsum("ei,ij,ej->e", w_e, self.fem.Ke0, ue)
        else:
            adj_term = dEk * np.einsum("ei,eij,ej->e", w_e, self.fem.Ke0_batch, ue)

        # Total derivative of mu_KS
        dmu_ks = grad_direct + adj_term

        # Chain rule with target factor: dg / drho = lambda_crit_target * dmu_ks
        grad = self.lambda_crit_target * dmu_ks

        self._last_grad = grad
        return grad

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> Tuple[float, np.ndarray]:
        """Evaluate both constraint value and analytical sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad

    def get_critical_load_factor(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Return the first fundamental critical buckling load factor lambda_1 = 1 / mu_1."""
        self.evaluate(rho, **kwargs)
        assert self._last_vals is not None
        mu_1 = float(self._last_vals[0])
        if mu_1 <= 0.0:
            return float("inf")
        return 1.0 / mu_1

    def get_mode_load_factors(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return the critical buckling load factors lambda_j = 1 / mu_j for all computed modes."""
        self.evaluate(rho, **kwargs)
        assert self._last_vals is not None
        vals = self._last_vals
        blfs = np.zeros_like(vals)
        for i, mu in enumerate(vals):
            blfs[i] = (1.0 / mu) if mu > 0.0 else float("inf")
        return blfs
