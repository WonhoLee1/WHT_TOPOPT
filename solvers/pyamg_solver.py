"""PyAMG smoothed aggregation algebraic multigrid solver adapter."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import FEASolver, SolverResult, build_rigid_body_modes


def _run_cg(A: Any, b: np.ndarray, tol: float, maxiter: int, M: Any = None, callback: Any = None) -> Tuple[np.ndarray, int]:
    """Execute CG with compatibility across SciPy versions (rtol vs tol)."""
    try:
        return spla.cg(A, b, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    except TypeError:
        return spla.cg(A, b, tol=tol, maxiter=maxiter, M=M, callback=callback)


class PyAMGSolver(FEASolver):
    """Algebraic Multigrid (AMG) solver using PyAMG smoothed aggregation preconditioned CG.
    
    Constructs near-nullspace candidate vectors (rigid-body translation and rotation modes)
    restricted to free degrees of freedom, building an optimal multiscale hierarchy for
    linear elasticity problems.
    """

    solver_name: str = "pyamg"

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 1000,
        cycle: str = "V",
        symmetry: str = "hermitian",
        strength: Union[str, tuple] = "symmetric",
        aggregate: str = "standard",
        smooth: Union[str, tuple] = ("jacobi", {"omega": 4.0 / 3.0}),
    ) -> None:
        """Initialize PyAMGSolver.
        
        Args:
            tol: Relative residual convergence tolerance.
            maxiter: Maximum CG iterations.
            cycle: Multigrid cycle type ('V', 'W', 'F').
            symmetry: 'hermitian' or 'symmetric'.
            strength: Strength of connection method (default 'symmetric').
            aggregate: Aggregation method (default 'standard').
            smooth: Tentative prolongator smoothing method.
        """
        self.tol = tol
        self.maxiter = maxiter
        self.cycle = cycle
        self.symmetry = symmetry
        self.strength = strength
        self.aggregate = aggregate
        self.smooth = smooth

        # Cached hierarchy for fast adjoint solves
        self._last_ml: Any = None
        self._last_precond: Optional[spla.LinearOperator] = None
        self._last_K_ff: Optional[sp.csr_matrix] = None
        self._last_free_dofs: Optional[np.ndarray] = None
        self._last_n_dof: int = 0

    def _build_near_nullspace(
        self,
        free_dofs: np.ndarray,
        node_coords: Optional[np.ndarray],
        n_dof: int,
    ) -> Optional[np.ndarray]:
        """Build rigid-body candidate vectors B restricted to free DOFs."""
        if node_coords is None:
            return None

        dim = 2 if node_coords.shape[1] == 2 else 3
        phi_full = build_rigid_body_modes(node_coords, dim=dim)
        if phi_full.shape[0] != n_dof:
            return None

        # Restrict candidate modes to free DOFs
        B = phi_full[free_dofs, :].copy()
        # Filter out columns that are all zero (e.g. if a rotation mode is completely fixed)
        col_norms = np.linalg.norm(B, axis=0)
        valid = col_norms > 1e-12
        if np.any(valid):
            return B[:, valid]
        return None

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
        """Solve K_ff u_f = F_f using PyAMG-preconditioned Conjugate Gradient."""
        try:
            import pyamg
        except ImportError as exc:
            raise ImportError(
                "pyamg is required for PyAMGSolver. Install with `pip install pyamg`."
            ) from exc

        t0 = time.perf_counter()
        n_dof = K.shape[0]

        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if len(free_dofs) == n_dof and node_coords is not None:
            dim = 2 if node_coords.shape[1] == 2 else 3
            F = self.apply_inertia_relief(F, M, node_coords, free_dofs=free_dofs, dim=dim)

        K_ff, F_f, free_dofs, fixed_dofs = self.partition_system(K, F, free_dofs, fixed_dofs, u_fixed)
        K_ff_csr = K_ff.tocsr().astype(float)
        n_free = K_ff_csr.shape[0]

        # Construct candidate nullspace B from physical node coordinates
        B = self._build_near_nullspace(free_dofs, node_coords, n_dof)

        # Build smoothed aggregation hierarchy
        ml = pyamg.smoothed_aggregation_solver(
            K_ff_csr,
            B=B,
            symmetry=self.symmetry,
            strength=self.strength,
            aggregate=self.aggregate,
            smooth=self.smooth,
        )
        precond = ml.aspreconditioner(cycle=self.cycle)

        self._last_ml = ml
        self._last_precond = precond
        self._last_K_ff = K_ff_csr
        self._last_free_dofs = free_dofs
        self._last_n_dof = n_dof

        residuals = []
        def cb(xk: np.ndarray) -> None:
            rk = F_f - K_ff_csr @ xk if F_f.ndim == 1 else None
            if rk is not None:
                residuals.append(float(np.linalg.norm(rk)))

        u_free: np.ndarray
        total_iterations = 0

        if F_f.ndim == 1:
            u_free, info = _run_cg(
                K_ff_csr,
                F_f,
                tol=self.tol,
                maxiter=self.maxiter,
                M=precond,
                callback=cb,
            )
            total_iterations = len(residuals)
            if info != 0:
                # If CG failed to reach tolerance, try running AMG direct cycle or fallback
                u_free = ml.solve(F_f, tol=self.tol, maxiter=self.maxiter, cycle=self.cycle)
        else:
            n_cases = F_f.shape[1]
            u_free = np.zeros_like(F_f)
            total_iters = 0
            for c in range(n_cases):
                b = F_f[:, c]
                case_res = []
                def cb_case(xk: np.ndarray) -> None:
                    case_res.append(float(np.linalg.norm(b - K_ff_csr @ xk)))
                u_c, info = _run_cg(
                    K_ff_csr,
                    b,
                    tol=self.tol,
                    maxiter=self.maxiter,
                    M=precond,
                    callback=cb_case,
                )
                if info != 0:
                    u_c = ml.solve(b, tol=self.tol, maxiter=self.maxiter, cycle=self.cycle)
                u_free[:, c] = u_c
                total_iters += len(case_res)
            total_iterations = total_iters // max(1, n_cases)

        u_full = self.reconstruct_displacement(u_free, n_dof, free_dofs, fixed_dofs, u_fixed)
        solve_time = time.perf_counter() - t0

        # Compute relative residual norm
        if F_f.ndim == 1:
            res = K_ff_csr @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))
        else:
            res = K_ff_csr @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))

        extra: Dict[str, Any] = {
            "amg_levels": len(ml.levels),
            "operator_complexity": float(ml.operator_complexity()),
            "grid_complexity": float(ml.grid_complexity()),
            "near_nullspace_modes": int(B.shape[1]) if B is not None else 1,
            "n_free": int(n_free),
            "n_dof": int(n_dof),
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
        """Solve adjoint equation K_ff u_f_adj = rhs_f reusing the pre-constructed AMG hierarchy."""
        if self._last_free_dofs is None or self._last_ml is None or self._last_K_ff is None:
            raise RuntimeError("solve_cached called before any system was factorized with solve()")

        rhs_arr = np.asarray(rhs, dtype=float)
        free_dofs = self._last_free_dofs
        n_free = len(free_dofs)
        n_dof = self._last_n_dof

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

        precond = self._last_precond or self._last_ml.aspreconditioner(cycle=self.cycle)

        if rhs_f.ndim == 1:
            u_free_adj, info = _run_cg(
                self._last_K_ff,
                rhs_f,
                tol=self.tol,
                maxiter=self.maxiter,
                M=precond,
            )
            if info != 0:
                u_free_adj = self._last_ml.solve(rhs_f, tol=self.tol, maxiter=self.maxiter, cycle=self.cycle)
        else:
            n_cases = rhs_f.shape[1]
            u_free_adj = np.zeros_like(rhs_f)
            for c in range(n_cases):
                u_c, info = _run_cg(
                    self._last_K_ff,
                    rhs_f[:, c],
                    tol=self.tol,
                    maxiter=self.maxiter,
                    M=precond,
                )
                if info != 0:
                    u_c = self._last_ml.solve(rhs_f[:, c], tol=self.tol, maxiter=self.maxiter, cycle=self.cycle)
                u_free_adj[:, c] = u_c

        return self.reconstruct_displacement(u_free_adj, n_dof, free_dofs)
