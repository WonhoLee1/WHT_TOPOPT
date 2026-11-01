"""SciPy sparse linear solver adapter supporting direct (SuperLU) and iterative (CG/MINRES) methods."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import FEASolver, SolverResult


def _run_solver(solver_fn: Any, A: Any, b: np.ndarray, tol: float, maxiter: int, M: Any = None, callback: Any = None) -> Tuple[np.ndarray, int]:
    """Execute iterative solver with compatibility across SciPy versions (rtol vs tol)."""
    try:
        return solver_fn(A, b, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    except TypeError:
        return solver_fn(A, b, tol=tol, maxiter=maxiter, M=M, callback=callback)


class ScipySolver(FEASolver):
    """Direct and iterative sparse FEA solver based on SciPy.
    
    Supports:
        - method="direct": splu LU factorization with reuse for multi-load cases and adjoint solves.
        - method="cg": Conjugate Gradient with Jacobi diagonal preconditioner.
        - method="minres": Minimal Residual solver with Jacobi diagonal preconditioner.
        - Regularization fallback (1e-12 * diag_mean * I) if matrix is near-singular.
    """

    solver_name: str = "scipy"

    def __init__(
        self,
        method: str = "direct",
        tol: float = 1e-10,
        maxiter: int = 5000,
        regularize_near_singular: bool = True,
        reg_factor: float = 1e-12,
    ) -> None:
        """Initialize ScipySolver.
        
        Args:
            method: 'direct', 'cg', or 'minres'.
            tol: Convergence tolerance for iterative solvers.
            maxiter: Maximum number of iterations for iterative solvers.
            regularize_near_singular: Whether to add diagonal perturbation if singular.
            reg_factor: Relative diagonal regularization scale (default 1e-12).
        """
        valid_methods = ("direct", "pardiso", "cg", "minres")
        if method.lower() not in valid_methods:
            raise ValueError(f"Unknown solver method {method!r}. Must be one of {valid_methods}")

        self.method = method.lower()
        self.tol = tol
        self.maxiter = maxiter
        self.regularize_near_singular = regularize_near_singular
        self.reg_factor = reg_factor

        # Cached structures for adjoint/subsequent solves
        self._last_lu: Optional[spla.SuperLU] = None
        self._last_pardiso: Any = None
        self._last_K_ff: Optional[sp.csc_matrix] = None
        self._last_free_dofs: Optional[np.ndarray] = None
        self._last_n_dof: int = 0
        self._last_precond: Optional[spla.LinearOperator] = None

    def _build_jacobi_preconditioner(self, A: sp.spmatrix) -> spla.LinearOperator:
        """Build Jacobi diagonal preconditioner M = diag(A)."""
        A_csr = A.tocsr()
        diag = A_csr.diagonal().copy()
        # Avoid division by zero or negative diagonal entries in near-singular/void regions
        diag_mean = float(np.mean(np.abs(diag)))
        threshold = max(1e-14 * diag_mean, 1e-30)
        diag[np.abs(diag) < threshold] = 1.0

        inv_diag = 1.0 / diag

        def matvec(v: np.ndarray) -> np.ndarray:
            if v.ndim == 1:
                return v * inv_diag
            return v * inv_diag[:, None]

        n = A.shape[0]
        return spla.LinearOperator((n, n), matvec=matvec, dtype=float)

    def _regularize_matrix(self, A: sp.csc_matrix) -> sp.csc_matrix:
        """Add diagonal regularization reg_factor * diag_mean * I."""
        diag_mean = float(np.mean(np.abs(A.diagonal())))
        shift = self.reg_factor * max(diag_mean, 1.0)
        n = A.shape[0]
        reg_eye = sp.eye(n, format="csc", dtype=float) * shift
        return A + reg_eye

    def solve(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        free_dofs: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
        M: Optional[sp.spmatrix] = None,
        node_coords: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve K_ff u_f = F_f and reconstruct full displacement field."""
        t0 = time.perf_counter()
        n_dof = K.shape[0]

        # Check inertia relief if node_coords given and no constraints
        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if len(free_dofs) == n_dof and node_coords is not None:
            dim = 2 if node_coords.shape[1] == 2 else 3
            F = self.apply_inertia_relief(F, M, node_coords, free_dofs=free_dofs, dim=dim)

        K_ff, F_f, free_dofs, fixed_dofs = self.partition_system(K, F, free_dofs, fixed_dofs, u_fixed)
        K_ff_csc = K_ff.tocsc()
        n_free = K_ff_csc.shape[0]

        self._last_free_dofs = free_dofs
        self._last_n_dof = n_dof
        self._last_K_ff = K_ff_csc

        total_iterations = 0
        u_free: np.ndarray

        if self.method == "direct":
            # Factorize with splu
            try:
                lu = spla.splu(K_ff_csc)
            except RuntimeError as exc:
                if self.regularize_near_singular:
                    K_ff_csc = self._regularize_matrix(K_ff_csc)
                    self._last_K_ff = K_ff_csc
                    lu = spla.splu(K_ff_csc)
                else:
                    raise exc

            self._last_lu = lu
            total_iterations = 1

            if F_f.ndim == 1:
                u_free = lu.solve(F_f)
            else:
                # Multi-load cases
                u_free = lu.solve(F_f)  # splu handles (n_free, n_cases) directly

        elif self.method == "pardiso":
            import pypardiso
            if self._last_pardiso is None:
                self._last_pardiso = pypardiso.PyPardisoSolver()
            K_ff_csr = K_ff_csc.tocsr()
            self._last_K_ff = K_ff_csr
            try:
                u_free = self._last_pardiso.solve(K_ff_csr, np.asarray(F_f, dtype=np.float64))
            except Exception as exc:
                if self.regularize_near_singular:
                    K_ff_csc = self._regularize_matrix(K_ff_csc)
                    K_ff_csr = K_ff_csc.tocsr()
                    self._last_K_ff = K_ff_csr
                    u_free = self._last_pardiso.solve(K_ff_csr, np.asarray(F_f, dtype=np.float64))
                else:
                    raise exc
            total_iterations = 1

        elif self.method in ("cg", "minres"):
            precond = self._build_jacobi_preconditioner(K_ff_csc)
            self._last_precond = precond
            solver_fn = spla.cg if self.method == "cg" else spla.minres

            if F_f.ndim == 1:
                iter_count = [0]
                def cb(_xk: np.ndarray) -> None:
                    iter_count[0] += 1

                u_free, info = _run_solver(
                    solver_fn,
                    K_ff_csc,
                    F_f,
                    tol=self.tol,
                    maxiter=self.maxiter,
                    M=precond,
                    callback=cb,
                )
                if info != 0 and self.regularize_near_singular:
                    K_ff_reg = self._regularize_matrix(K_ff_csc)
                    precond_reg = self._build_jacobi_preconditioner(K_ff_reg)
                    u_free, info = _run_solver(
                        solver_fn,
                        K_ff_reg,
                        F_f,
                        tol=self.tol,
                        maxiter=self.maxiter,
                        M=precond_reg,
                    )
                total_iterations = iter_count[0]
            else:
                n_cases = F_f.shape[1]
                u_free = np.zeros_like(F_f)
                total_iters = 0
                for c in range(n_cases):
                    b = F_f[:, c]
                    iter_count = [0]
                    def cb(_xk: np.ndarray) -> None:
                        iter_count[0] += 1
                    u_c, info = _run_solver(
                        solver_fn,
                        K_ff_csc,
                        b,
                        tol=self.tol,
                        maxiter=self.maxiter,
                        M=precond,
                        callback=cb,
                    )
                    u_free[:, c] = u_c
                    total_iters += iter_count[0]
                total_iterations = total_iters // max(1, n_cases)
        else:
            raise ValueError(f"Unknown method {self.method}")

        u_full = self.reconstruct_displacement(u_free, n_dof, free_dofs, fixed_dofs, u_fixed)
        solve_time = time.perf_counter() - t0

        # Evaluate residual norm ||K_ff u_free - F_f|| / max(1.0, ||F_f||)
        if F_f.ndim == 1:
            res = K_ff_csc @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))
        else:
            res = K_ff_csc @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))

        extra: Dict[str, Any] = {
            "method": self.method,
            "n_free": int(n_free),
            "n_dof": int(n_dof),
            "regularized": self.regularize_near_singular,
        }

        return SolverResult(
            u=u_full,
            residual_norm=res_norm,
            iterations=total_iterations,
            solve_time_sec=solve_time,
            solver_name=self.solver_name,
            extra=extra,
        )

    def solve_cached(self, rhs: np.ndarray) -> np.ndarray:
        """Solve adjoint equation K_ff u_f_adj = rhs_f in O(N) using cached LU factorization."""
        if self._last_free_dofs is None or self._last_n_dof == 0:
            raise RuntimeError("solve_cached called before any system was factorized with solve()")

        rhs_arr = np.asarray(rhs, dtype=float)
        free_dofs = self._last_free_dofs
        n_free = len(free_dofs)
        n_dof = self._last_n_dof

        # Determine if rhs is provided on full DOFs or free DOFs only
        if rhs_arr.shape[0] == n_dof:
            rhs_f = rhs_arr[free_dofs]
        elif rhs_arr.shape[0] == n_free:
            rhs_f = rhs_arr
        elif rhs_arr.ndim == 2 and rhs_arr.shape[1] == n_dof:
            rhs_f = rhs_arr[:, free_dofs].T
        elif rhs_arr.ndim == 2 and rhs_arr.shape[0] == n_dof:
            rhs_f = rhs_arr[free_dofs, :]
        else:
            raise ValueError(f"rhs shape {rhs_arr.shape} is incompatible with n_dof={n_dof}, n_free={n_free}")

        if self.method == "direct" and self._last_lu is not None:
            u_free_adj = self._last_lu.solve(rhs_f)
        elif self.method == "pardiso" and self._last_pardiso is not None and self._last_K_ff is not None:
            u_free_adj = self._last_pardiso.solve(self._last_K_ff, np.asarray(rhs_f, dtype=np.float64))
        else:
            # Fallback to iterative solver with preconditioning
            if self._last_K_ff is None:
                raise RuntimeError("No cached K_ff matrix available")
            precond = self._last_precond or self._build_jacobi_preconditioner(self._last_K_ff)
            solver_fn = spla.cg if self.method == "cg" else spla.minres
            if rhs_f.ndim == 1:
                u_free_adj, _ = _run_solver(solver_fn, self._last_K_ff, rhs_f, tol=self.tol, maxiter=self.maxiter, M=precond)
            else:
                n_cases = rhs_f.shape[1]
                u_free_adj = np.zeros_like(rhs_f)
                for c in range(n_cases):
                    u_free_adj[:, c], _ = _run_solver(solver_fn, self._last_K_ff, rhs_f[:, c], tol=self.tol, maxiter=self.maxiter, M=precond)

        return self.reconstruct_displacement(u_free_adj, n_dof, free_dofs)
