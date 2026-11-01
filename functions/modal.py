"""Vibration eigenfrequency and compliant mechanism objective and constraint functions.

Provides:
1. EigenFrequencyFunction: Maximizes fundamental or aggregated vibration frequencies:
       min f(rho) = - omega_1^2
   using exact self-adjoint eigenvalue sensitivities:
       d(omega_j^2) / drho_e = phi_{j,e}^T (dEK_e K_e0 - omega_j^2 dM_e M_e0) phi_{j,e}
2. EigenFrequencyConstraint: Inequality constraint on natural frequency:
       g(rho) = 1.0 - (omega_1^2 / omega_target^2) <= 0
3. CompliantMechanismFunction: Maximizes output displacement for compliant mechanisms:
       min f(rho) = - u_out = - L^T u
   with input actuator spring k_in and workpiece output spring k_out:
       df / drho_e = - dEK_e * (lambda_e^T K_e0 u_e)
   where K_mod lambda = - L is solved via adjoint solve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

from .base import ObjectiveFunction, ConstraintFunction
from wht_topopt.algorithms.base import GradientSource
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.solvers.base import FEASolver
from wht_topopt.solvers.scipy_solver import ScipySolver


# ===========================================================================
# 1. Eigenfrequency Function & Constraint
# ===========================================================================

class EigenFrequencyFunction(ObjectiveFunction):
    """Objective function to maximize vibration natural frequencies (minimizes -omega^2).

    Solves generalized eigenvalue problem on free DOFs:
        K_ff(rho) phi_j = omega_j^2 M_ff(rho) phi_j
    with M-orthonormality phi_j^T M_ff phi_j = 1.0.

    Spurious void vibration modes are suppressed via SIMP stiffness interpolation
    and modified mass interpolation eta_M(rho) = rho_min_m + rho^q_m * (1 - rho_min_m).
    """

    name: str = "eigenfrequency_objective"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        fem: FEMEngine,
        bcm: BoundaryConditionManager,
        solver: Optional[FEASolver] = None,
        mode_idx: int = 0,
        n_modes: int = 3,
        penal_k: float = 3.0,
        penal_m: float = 1.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        rho_min_m: float = 1e-8,
        rho_base: float = 1.0,
    ) -> None:
        """Initialize EigenFrequencyFunction.

        Parameters
        ----------
        fem : FEMEngine
            Vectorized finite element engine.
        bcm : BoundaryConditionManager
            Boundary condition manager with fixed DOFs.
        solver : FEASolver, optional
            Linear solver adapter (default: ScipySolver direct).
        mode_idx : int
            Target vibration mode index (default 0 for fundamental frequency).
        n_modes : int
            Number of lowest modes to compute (default 3).
        penal_k : float
            SIMP penalty exponent for stiffness K (default 3.0).
        penal_m : float
            Penalty exponent q_M for mass M (default 1.0).
        e0 : float
            Base solid material Young's modulus (default 1.0).
        emin : float
            Void artificial Young's modulus for stiffness (default 1e-6).
        rho_min_m : float
            Minimum mass density factor to prevent zero-mass singular M (default 1e-8).
        rho_base : float
            Base solid material mass density (default 1.0).
        """
        self.fem = fem
        self.bcm = bcm
        self.solver = solver or ScipySolver(method="direct")
        self.mode_idx = int(mode_idx)
        self.n_modes = max(int(n_modes), self.mode_idx + 1)
        self.penal_k = float(penal_k)
        self.penal_m = float(penal_m)
        self.e0 = float(e0)
        self.emin = float(emin)
        self.rho_min_m = float(rho_min_m)
        self.rho_base = float(rho_base)

        self.mesh = self.fem.mesh
        self.n_elem = self.mesh.n_elem
        self.ndof = self.fem.ndof
        self.edof = self.mesh.edofMat

        # Cache for evaluation
        self._last_rho: Optional[np.ndarray] = None
        self._last_val: Optional[float] = None
        self._last_grad: Optional[np.ndarray] = None
        self._last_eigenvals: Optional[np.ndarray] = None
        self._last_phis: Optional[np.ndarray] = None

    def interpolate_material(
        self, rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute stiffness and mass density factors and derivatives."""
        r = np.asarray(rho, dtype=np.float64)
        pk = self.penal_k
        pm = self.penal_m
        e0 = self.e0
        emin = self.emin
        rmin_m = self.rho_min_m
        r_base = self.rho_base

        # Elastic stiffness: E_K = emin + rho^pk * (e0 - emin)
        delta_k = e0 - emin
        Ek = emin + (r ** pk) * delta_k
        dEk = pk * (r ** (pk - 1.0)) * delta_k

        # Mass density: M = rho_base * (rmin_m + rho^pm * (1 - rmin_m))
        delta_m = 1.0 - rmin_m
        Em = r_base * (rmin_m + (r ** pm) * delta_m)
        dEm = r_base * pm * (r ** (pm - 1.0)) * delta_m

        return Ek, dEk, Em, dEm

    def solve_eigenfrequencies(
        self, rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, sp.csc_matrix, sp.csc_matrix]:
        """Solve generalized eigenvalue problem K_ff phi = omega^2 M_ff phi for lowest frequencies.

        Returns
        -------
        eigenvals : (n_modes,) squared natural frequencies omega_1^2 <= omega_2^2 <= ...
        phis : (n_modes, ndof) M-orthonormal global mode shapes
        K : global stiffness matrix
        M : global mass matrix
        """
        Ek, _, Em, _ = self.interpolate_material(rho)
        K = self.fem.assemble_K(Ek)
        M = self.fem.assemble_M(Em)

        free = self.bcm.free_dofs
        n_free = len(free)
        n_req = min(self.n_modes, n_free)

        K_ff = K[free, :][:, free]
        M_ff = M[free, :][:, free]

        if n_free <= 40 or n_req >= n_free - 1:
            # Dense generalized eigenvalue problem: K_ff phi = omega^2 M_ff phi
            K_dense = K_ff.toarray()
            M_dense = M_ff.toarray()
            # Regularize slightly if ill-conditioned
            diag_k = np.mean(np.diag(K_dense))
            if np.any(np.isnan(K_dense)):
                raise ValueError("NaN detected in K_dense matrix")

            all_vals, all_vecs = la.eigh(K_dense, M_dense)
            # Sort ascending: lowest vibration frequencies
            sort_idx = np.argsort(all_vals)
            vals = all_vals[sort_idx[:n_req]]
            vecs = all_vecs[:, sort_idx[:n_req]]
        else:
            # Sparse generalized eigenvalue problem using eigsh with shift-invert at sigma = 0
            K_ff_csc = K_ff.tocsc()
            M_ff_csc = M_ff.tocsc()

            try:
                raw_vals, raw_vecs = spla.eigsh(
                    A=K_ff_csc,
                    k=n_req,
                    M=M_ff_csc,
                    sigma=0.0,
                    which="LM",
                    tol=1e-8,
                    maxiter=2000,
                )
                sort_idx = np.argsort(raw_vals)
                vals = raw_vals[sort_idx]
                vecs = raw_vecs[:, sort_idx]
            except Exception:
                # Dense fallback
                all_vals, all_vecs = la.eigh(K_ff_csc.toarray(), M_ff_csc.toarray())
                sort_idx = np.argsort(all_vals)
                vals = all_vals[sort_idx[:n_req]]
                vecs = all_vecs[:, sort_idx[:n_req]]

        # Ensure mode shapes are normalized: phi^T M_ff phi = 1.0
        for j in range(n_req):
            v = vecs[:, j]
            m_norm = float(v.T @ (M_ff @ v))
            if m_norm > 1e-14:
                vecs[:, j] = v / np.sqrt(m_norm)

        # Reconstruct global mode shapes phi of shape (n_req, ndof)
        phis = np.zeros((n_req, self.ndof), dtype=np.float64)
        for j in range(n_req):
            phis[j, free] = vecs[:, j]

        return vals, phis, K, M

    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate scalar objective f(rho) = - omega_{mode_idx}^2.

        Parameters
        ----------
        rho : np.ndarray
            Element physical density field (n_elem,).

        Returns
        -------
        float
            Objective value - omega_{mode_idx}^2.
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if self._last_rho is not None and np.array_equal(r, self._last_rho):
            if self._last_val is not None:
                return self._last_val

        vals, phis, _, _ = self.solve_eigenfrequencies(r)
        val = - float(vals[self.mode_idx])

        self._last_rho = r.copy()
        self._last_val = val
        self._last_eigenvals = vals
        self._last_phis = phis
        self._last_grad = None

        return val

    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate exact self-adjoint objective sensitivity df / drho.

        d(-omega_j^2) / drho_e = - phi_{j,e}^T (dEK_e K_e0 - omega_j^2 dM_e M_e0) phi_{j,e}
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if (
            self._last_rho is not None
            and np.array_equal(r, self._last_rho)
            and self._last_grad is not None
        ):
            return self._last_grad

        if (
            self._last_rho is None
            or not np.array_equal(r, self._last_rho)
            or self._last_eigenvals is None
        ):
            self.evaluate(r, **kwargs)

        assert self._last_eigenvals is not None
        assert self._last_phis is not None

        omega2 = self._last_eigenvals[self.mode_idx]
        phi = self._last_phis[self.mode_idx]
        phi_e = phi[self.edof]  # (n_elem, d)

        Ek, dEk, Em, dEm = self.interpolate_material(r)

        # d(omega^2) / drho_e = phi_e^T (dEk Ke0 - omega^2 dEm Me0) phi_e
        if self.fem.is_uniform:
            Ke_phi = phi_e @ self.fem.Ke0
            Me_phi = phi_e @ self.fem.Me0
        else:
            Ke_phi = np.einsum("eij,ej->ei", self.fem.Ke0_batch, phi_e)
            Me_phi = np.einsum("eij,ej->ei", self.fem.Me0_batch, phi_e)

        term_K = np.sum(phi_e * Ke_phi, axis=1)
        term_M = np.sum(phi_e * Me_phi, axis=1)

        d_omega2 = dEk * term_K - omega2 * dEm * term_M

        # Objective is -omega^2, so df / drho = - d(omega^2) / drho
        grad = - d_omega2
        self._last_grad = grad
        return grad

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> Tuple[float, np.ndarray]:
        """Evaluate both objective value and sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad

    def get_natural_frequencies_hz(self, rho: np.ndarray) -> np.ndarray:
        """Return natural frequencies f_j = omega_j / (2*pi) in Hz."""
        self.evaluate(rho)
        assert self._last_eigenvals is not None
        omegas = np.sqrt(np.maximum(0.0, self._last_eigenvals))
        return omegas / (2.0 * np.pi)


class EigenFrequencyConstraint(ConstraintFunction):
    """Inequality constraint on vibration frequency: g(rho) = 1.0 - (omega_1^2 / omega_target^2) <= 0."""

    name: str = "eigenfrequency_constraint"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        fem: FEMEngine,
        bcm: BoundaryConditionManager,
        omega_target: float,
        solver: Optional[FEASolver] = None,
        mode_idx: int = 0,
        penal_k: float = 3.0,
        penal_m: float = 1.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        rho_min_m: float = 1e-8,
        rho_base: float = 1.0,
    ) -> None:
        """Initialize EigenFrequencyConstraint."""
        self.omega_target = float(omega_target)
        if self.omega_target <= 0.0:
            raise ValueError(f"omega_target must be positive, got {self.omega_target}")

        self.fn = EigenFrequencyFunction(
            fem=fem,
            bcm=bcm,
            solver=solver,
            mode_idx=mode_idx,
            penal_k=penal_k,
            penal_m=penal_m,
            e0=e0,
            emin=emin,
            rho_min_m=rho_min_m,
            rho_base=rho_base,
        )

    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate constraint g(rho) = 1.0 - omega^2 / omega_target^2 <= 0."""
        # fn.evaluate returns -omega^2
        neg_omega2 = self.fn.evaluate(rho, **kwargs)
        omega2 = - neg_omega2
        return 1.0 - (omega2 / (self.omega_target ** 2))

    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate constraint sensitivity dg / drho."""
        # fn.gradient returns d(-omega^2) / drho = - d(omega^2) / drho
        d_neg_omega2 = self.fn.gradient(rho, **kwargs)
        # dg / drho = - 1 / omega_target^2 * d(omega^2) / drho = 1 / omega_target^2 * d(-omega^2) / drho
        return (1.0 / (self.omega_target ** 2)) * d_neg_omega2

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> Tuple[float, np.ndarray]:
        """Evaluate both constraint value and sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad


# ===========================================================================
# 2. Compliant Mechanism Function
# ===========================================================================

class CompliantMechanismFunction(ObjectiveFunction):
    """Compliant mechanism objective function to maximize output displacement (minimize -u_out).

    Formulation:
        min f(rho) = - u_out = - L^T u
    subject to:
        K_mod(rho) u = F_in
    where K_mod(rho) = K(rho) + K_springs includes actuator spring stiffness k_in
    at the input DOF and workpiece stiffness k_out at the output DOF.

    Sensitivity is evaluated via exact adjoint analysis:
        df / drho_e = - dEK_e * (lambda_e^T K_e0 u_e)
    where K_mod lambda = - L.
    """

    name: str = "compliant_mechanism_objective"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        fem: FEMEngine,
        bcm: BoundaryConditionManager,
        in_dof: int,
        out_dof: int,
        solver: Optional[FEASolver] = None,
        f_in: float = 1.0,
        L_vector: Optional[np.ndarray] = None,
        k_in: float = 1.0,
        k_out: float = 0.001,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
    ) -> None:
        """Initialize CompliantMechanismFunction.

        Parameters
        ----------
        fem : FEMEngine
            Vectorized finite element engine.
        bcm : BoundaryConditionManager
            Boundary condition manager with fixed DOFs.
        in_dof : int
            Global degree of freedom where input actuator load is applied.
        out_dof : int
            Global degree of freedom where output mechanism displacement is measured.
        solver : FEASolver, optional
            Linear solver adapter (default: ScipySolver direct).
        f_in : float
            Magnitude of input actuator load (default 1.0).
        L_vector : np.ndarray, optional
            Displacement output direction vector (default unit vector at out_dof).
        k_in : float
            Actuator spring stiffness attached at in_dof (default 1.0).
        k_out : float
            Workpiece resistance spring stiffness attached at out_dof (default 0.001).
        penal : float
            SIMP penalty exponent for stiffness (default 3.0).
        e0 : float
            Base solid material Young's modulus (default 1.0).
        emin : float
            Void artificial Young's modulus (default 1e-6).
        """
        self.fem = fem
        self.bcm = bcm
        self.in_dof = int(in_dof)
        self.out_dof = int(out_dof)
        self.solver = solver or ScipySolver(method="direct")
        self.f_in_mag = float(f_in)
        self.k_in = float(k_in)
        self.k_out = float(k_out)
        self.penal = float(penal)
        self.e0 = float(e0)
        self.emin = float(emin)

        self.mesh = self.fem.mesh
        self.n_elem = self.mesh.n_elem
        self.ndof = self.fem.ndof
        self.edof = self.mesh.edofMat

        # Construct input force vector F_in
        self.F_in = np.zeros(self.ndof, dtype=np.float64)
        self.F_in[self.in_dof] = self.f_in_mag

        # Construct output measurement vector L
        if L_vector is not None:
            self.L = np.asarray(L_vector, dtype=np.float64).ravel()
        else:
            self.L = np.zeros(self.ndof, dtype=np.float64)
            self.L[self.out_dof] = 1.0

        # Cache for states
        self._last_rho: Optional[np.ndarray] = None
        self._last_val: Optional[float] = None
        self._last_grad: Optional[np.ndarray] = None
        self._last_u: Optional[np.ndarray] = None
        self._last_lam: Optional[np.ndarray] = None

    def interpolate_material(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SIMP Young's modulus E(rho) and derivative dE / drho."""
        r = np.asarray(rho, dtype=np.float64)
        p = self.penal
        e0 = self.e0
        emin = self.emin
        delta = e0 - emin

        E = emin + (r ** p) * delta
        dE = p * (r ** (p - 1.0)) * delta
        return E, dE

    def assemble_modified_K(self, E: np.ndarray) -> sp.csc_matrix:
        """Assemble global stiffness matrix K_mod = K(rho) + K_springs."""
        K = self.fem.assemble_K(E)
        if self.k_in > 0.0 or self.k_out > 0.0:
            rows = []
            cols = []
            vals = []
            if self.k_in > 0.0:
                rows.append(self.in_dof)
                cols.append(self.in_dof)
                vals.append(self.k_in)
            if self.k_out > 0.0:
                rows.append(self.out_dof)
                cols.append(self.out_dof)
                vals.append(self.k_out)
            K_springs = sp.csc_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof), dtype=np.float64)
            return (K + K_springs).tocsc()
        return K

    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate scalar objective f(rho) = - u_out = - L^T u.

        Parameters
        ----------
        rho : np.ndarray
            Element physical density field (n_elem,).

        Returns
        -------
        float
            Negative output displacement -u_out.
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if self._last_rho is not None and np.array_equal(r, self._last_rho):
            if self._last_val is not None:
                return self._last_val

        E, _ = self.interpolate_material(r)
        K_mod = self.assemble_modified_K(E)

        # Solve forward equilibrium K_mod u = F_in (factorizes and caches LU)
        u_res = self.solver.solve(
            K_mod,
            self.F_in,
            free_dofs=self.bcm.free_dofs,
            fixed_dofs=self.bcm.fixed_dofs,
        )
        u = u_res.u.ravel()
        val = - float(np.dot(self.L, u))

        # Solve adjoint equation K_mod lambda = - L in O(N) using cached LU
        if hasattr(self.solver, "solve_cached"):
            lam = self.solver.solve_cached(-self.L).ravel()
        else:
            lam_res = self.solver.solve(
                K_mod,
                -self.L,
                free_dofs=self.bcm.free_dofs,
                fixed_dofs=self.bcm.fixed_dofs,
            )
            lam = lam_res.u.ravel()

        self._last_rho = r.copy()
        self._last_val = val
        self._last_u = u
        self._last_lam = lam
        self._last_grad = None

        return val

    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate exact adjoint sensitivity df / drho = - dEK * (lambda_e^T K_e0 u_e).

        Parameters
        ----------
        rho : np.ndarray
            Element physical density field (n_elem,).

        Returns
        -------
        np.ndarray
            Sensitivity vector of shape (n_elem,).
        """
        r = np.asarray(rho, dtype=np.float64).ravel()
        if (
            self._last_rho is not None
            and np.array_equal(r, self._last_rho)
            and self._last_grad is not None
        ):
            return self._last_grad

        if (
            self._last_rho is None
            or not np.array_equal(r, self._last_rho)
            or self._last_u is None
        ):
            self.evaluate(r, **kwargs)

        assert self._last_u is not None
        assert self._last_lam is not None

        u = self._last_u
        lam = self._last_lam
        ue = u[self.edof]        # (n_elem, d)
        lame = lam[self.edof]    # (n_elem, d)

        _, dE = self.interpolate_material(r)

        if self.fem.is_uniform:
            Ke_u = ue @ self.fem.Ke0
        else:
            Ke_u = np.einsum("eij,ej->ei", self.fem.Ke0_batch, ue)

        mutual_energy = np.sum(lame * Ke_u, axis=1)

        # df / drho = - dE * lambda_e^T K_e0 u_e
        grad = - dE * mutual_energy
        self._last_grad = grad
        return grad

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> Tuple[float, np.ndarray]:
        """Evaluate both objective value and sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad

    def get_output_displacement(self, rho: np.ndarray) -> float:
        """Return the actual mechanism output displacement u_out = L^T u."""
        val = self.evaluate(rho)
        return - val

    def get_mechanical_advantage(self, rho: np.ndarray) -> float:
        """Return mechanical advantage MA = F_out / F_in = (k_out * u_out) / F_in."""
        u_out = self.get_output_displacement(rho)
        if abs(self.f_in_mag) < 1e-14:
            return 0.0
        return float(self.k_out * u_out / self.f_in_mag)
