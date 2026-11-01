"""Method of Moving Asymptotes (MMA) optimizer for topology optimization.

References:
    - Svanberg, K. (1987), "The method of moving asymptotes—a new method for structural optimization",
      International Journal for Numerical Methods in Engineering, 24(2), 359-373.
    - Svanberg, K. (2002), "A class of globally convergent optimization methods based on conservative
      convex approximations", SIAM Journal on Optimization, 12(2), 555-573.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

try:
    import mmapy
    from mmapy import mmasub, gcmmasub, kktcheck
    HAS_MMAPY = True
except ImportError:
    HAS_MMAPY = False

from .base import GradientSource, Optimizer, TrustRegionController


class MMAOptimizer(Optimizer):
    """Method of Moving Asymptotes (MMA) mathematical programming solver.

    Solves general nonlinearly constrained topology optimization problems:
        min   f_0(rho)
        s.t.  g_i(rho) <= 0,   i = 1, ..., m
              rho_min <= rho_e <= rho_max

    Features:
        - Svanberg (1987/2002) moving asymptotes update with oscillation-damping logic.
        - Conservative separable convex approximations p_{ie}, q_{ie} with curvature regularization.
        - Elastic slack variables y_i >= 0 ensuring strictly feasible dual subproblems.
        - High-precision dual Newton-Raphson solver with L-BFGS-B fallback (tol <= 1e-9).
        - Support for passive solid, passive void, and adaptive trust-region move limits.
    """

    def __init__(
        self,
        move: float = 0.2,
        s_init: float = 0.5,
        s_incr: float = 1.2,
        s_decr: float = 0.7,
        rho_min: float = 1e-3,
        rho_max: float = 1.0,
        a0: float = 1.0,
        a_i: float = 0.0,
        c_i: float = 1000.0,
        d_i: float = 1.0,
        tol: float = 1e-9,
        max_sub_iter: int = 150,
        trust_region: Optional[TrustRegionController] = None,
        gradient_source: GradientSource = GradientSource.EXACT_ADJOINT,
        backend: str = "mmapy",
    ) -> None:
        """Initialize MMAOptimizer.

        Parameters
        ----------
        move : float
            Maximum design variable move limit per iteration (default 0.2).
        s_init : float
            Initial moving asymptote distance factor (default 0.5).
        s_incr : float
            Asymptote expansion factor when consecutive steps are monotonic (default 1.2).
        s_decr : float
            Asymptote contraction factor when consecutive steps oscillate (default 0.7).
        rho_min : float
            Lower bound for design variables (default 1e-3).
        rho_max : float
            Upper bound for design variables (default 1.0).
        a0 : float
            Objective artificial variable penalty (default 1.0).
        a_i : float
            Constraint artificial variable weight (default 0.0).
        c_i : float
            Elastic slack variable linear penalty (default 1000.0).
        d_i : float
            Elastic slack variable quadratic penalty (default 1.0).
        tol : float
            Dual subproblem convergence tolerance (default 1e-9).
        max_sub_iter : int
            Maximum subproblem dual iterations.
        trust_region : Optional[TrustRegionController]
            Adaptive trust-region controller.
        gradient_source : GradientSource
            Source of gradient information.
        """
        if move <= 0.0:
            raise ValueError(f"move must be positive, got {move}")
        if not (0.0 <= rho_min < rho_max):
            raise ValueError(f"Invalid bounds: rho_min={rho_min}, rho_max={rho_max}")

        self.move = float(move)
        self.s_init = float(s_init)
        self.s_incr = float(s_incr)
        self.s_decr = float(s_decr)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.a0 = float(a0)
        self.a_i = float(a_i)
        self.c_i = float(c_i)
        self.d_i = float(d_i)
        self.tol = float(tol)
        self.max_sub_iter = int(max_sub_iter)
        self.trust_region = trust_region
        self.gradient_source = gradient_source
        self.backend = backend.lower()

        # Iteration history state
        self.iter: int = 0
        self.rho_old1: Optional[np.ndarray] = None
        self.rho_old2: Optional[np.ndarray] = None
        self.low: Optional[np.ndarray] = None
        self.upp: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal optimization history."""
        self.iter = 0
        self.rho_old1 = None
        self.rho_old2 = None
        self.low = None
        self.upp = None

    def update(
        self,
        rho: np.ndarray,
        df_drho: np.ndarray,
        g_vals: np.ndarray,
        dg_drho: np.ndarray,
        elem_volumes: Optional[np.ndarray] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute one MMA iteration to compute updated design variables rho^(k+1).

        Parameters
        ----------
        rho : np.ndarray
            Current design variables array (N_elem,).
        df_drho : np.ndarray
            Objective sensitivity vector (N_elem,).
        g_vals : np.ndarray
            Array of constraint function values g_i(rho) <= 0.
        dg_drho : np.ndarray
            Constraint sensitivity matrix of shape (m, N_elem) or (N_elem,).
        elem_volumes : Optional[np.ndarray]
            Element volume array (N_elem,).
        passive_solid : Optional[np.ndarray]
            Boolean mask or indices of elements fixed to solid (rho=rho_max).
        passive_void : Optional[np.ndarray]
            Boolean mask or indices of elements fixed to void (rho=rho_min).

        Returns
        -------
        rho_new : np.ndarray
            Updated design variables array of shape (N_elem,).
        """
        rho_curr = np.asarray(rho, dtype=np.float64).copy()
        n_elem = len(rho_curr)

        # Process constraints: m constraints
        g_arr = np.atleast_1d(np.asarray(g_vals, dtype=np.float64))
        m = len(g_arr)

        if m > 0:
            dg_mat = np.asarray(dg_drho, dtype=np.float64)
            if dg_mat.ndim == 1:
                dg_mat = dg_mat.reshape(1, n_elem)
            if dg_mat.shape != (m, n_elem):
                raise ValueError(
                    f"dg_drho shape {dg_mat.shape} does not match (m={m}, n_elem={n_elem})"
                )
        else:
            dg_mat = np.zeros((0, n_elem), dtype=np.float64)

        current_move = self.move
        if self.trust_region is not None:
            current_move = self.trust_region.move

        # Passive region masks
        solid_mask = np.zeros(n_elem, dtype=bool)
        if passive_solid is not None:
            if np.asarray(passive_solid).dtype == bool:
                solid_mask = np.asarray(passive_solid, dtype=bool)
            else:
                solid_mask[np.asarray(passive_solid, dtype=np.int64)] = True

        void_mask = np.zeros(n_elem, dtype=bool)
        if passive_void is not None:
            if np.asarray(passive_void).dtype == bool:
                void_mask = np.asarray(passive_void, dtype=bool)
            else:
                void_mask[np.asarray(passive_void, dtype=np.int64)] = True

        # -------------------------------------------------------------------
        # 0. Fast-path: Svanberg Official GCMMA/MMA Python Engine (mmapy)
        # -------------------------------------------------------------------
        if self.backend == "mmapy" and HAS_MMAPY:
            self.iter += 1
            xval = rho_curr.reshape(-1, 1)
            xmin = np.full((n_elem, 1), self.rho_min, dtype=np.float64)
            xmax = np.full((n_elem, 1), self.rho_max, dtype=np.float64)

            xold1 = self.rho_old1.reshape(-1, 1) if self.rho_old1 is not None else xval.copy()
            xold2 = self.rho_old2.reshape(-1, 1) if self.rho_old2 is not None else xold1.copy()

            low_in = self.low.reshape(-1, 1) if self.low is not None else xmin.copy()
            upp_in = self.upp.reshape(-1, 1) if self.upp is not None else xmax.copy()

            f0val = 0.0
            df0dx = np.asarray(df_drho, dtype=np.float64).reshape(-1, 1).copy()
            fval = g_arr.reshape(-1, 1) if m > 0 else np.zeros((0, 1), dtype=np.float64)
            dfdx = dg_mat.copy()

            # Zero out gradients for passive elements so MMA does not mutate them
            passive_any = solid_mask | void_mask
            if np.any(passive_any):
                df0dx[passive_any] = 0.0
                if m > 0:
                    dfdx[:, passive_any] = 0.0

            a = np.full((m, 1), self.a_i, dtype=np.float64)
            c = np.full((m, 1), self.c_i, dtype=np.float64)
            d = np.full((m, 1), self.d_i, dtype=np.float64)

            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low_out, upp_out = mmasub(
                m,
                n_elem,
                self.iter,
                xval,
                xmin,
                xmax,
                xold1,
                xold2,
                f0val,
                df0dx,
                fval,
                dfdx,
                low_in,
                upp_in,
                self.a0,
                a,
                c,
                d,
                move=current_move,
                asyinit=self.s_init,
                asydecr=self.s_decr,
                asyincr=self.s_incr,
            )

            # Update historical states
            self.low = low_out.ravel()
            self.upp = upp_out.ravel()
            self.rho_old2 = None if self.rho_old1 is None else self.rho_old1.copy()
            self.rho_old1 = rho_curr.copy()

            rho_new = xmma.ravel()
            if np.any(solid_mask):
                rho_new[solid_mask] = self.rho_max
            if np.any(void_mask):
                rho_new[void_mask] = self.rho_min

            return rho_new

        # -------------------------------------------------------------------
        # 1. Update Moving Asymptotes L_e and U_e
        # -------------------------------------------------------------------
        self.iter += 1
        delta_rho = self.rho_max - self.rho_min

        if self.iter <= 2 or self.rho_old1 is None or self.rho_old2 is None or self.low is None or self.upp is None:
            low = rho_curr - self.s_init * delta_rho
            upp = rho_curr + self.s_init * delta_rho
        else:
            # Oscillation detection: gamma = (rho - rho_old1) * (rho_old1 - rho_old2)
            gamma = (rho_curr - self.rho_old1) * (self.rho_old1 - self.rho_old2)
            factor = np.ones(n_elem, dtype=np.float64)
            factor[gamma > 0.0] = self.s_incr  # Monotonic -> expand asymptotes
            factor[gamma < 0.0] = self.s_decr  # Oscillating -> contract asymptotes

            low = rho_curr - factor * (self.rho_old1 - self.low)
            upp = rho_curr + factor * (self.upp - self.rho_old1)

            # Safety bounds: [1e-5, 10.0] * delta_rho
            low = np.clip(low, rho_curr - 10.0 * delta_rho, rho_curr - 1e-5 * delta_rho)
            upp = np.clip(upp, rho_curr + 1e-5 * delta_rho, rho_curr + 10.0 * delta_rho)

        self.low = low.copy()
        self.upp = upp.copy()
        self.rho_old2 = None if self.rho_old1 is None else self.rho_old1.copy()
        self.rho_old1 = rho_curr.copy()

        # -------------------------------------------------------------------
        # 2. Dynamic Bounds alpha_e, beta_e
        # -------------------------------------------------------------------
        alpha = np.maximum(
            self.rho_min,
            np.maximum(low + 0.1 * (rho_curr - low), rho_curr - current_move),
        )
        beta = np.minimum(
            self.rho_max,
            np.minimum(upp - 0.1 * (upp - rho_curr), rho_curr + current_move),
        )

        # Handle passive regions
        if np.any(solid_mask):
            alpha[solid_mask] = self.rho_max
            beta[solid_mask] = self.rho_max
        if np.any(void_mask):
            alpha[void_mask] = self.rho_min
            beta[void_mask] = self.rho_min

        # Ensure alpha <= beta strictly
        beta = np.maximum(beta, alpha)

        # -------------------------------------------------------------------
        # 3. Construct Convex Separable Approximations p_{ie}, q_{ie}
        # -------------------------------------------------------------------
        # Stack objective (i=0) and constraints (i=1..m): shape (m+1, n_elem)
        df0 = np.asarray(df_drho, dtype=np.float64).ravel()
        grad_all = np.vstack([df0.reshape(1, n_elem), dg_mat])

        p_mat = np.zeros((m + 1, n_elem), dtype=np.float64)
        q_mat = np.zeros((m + 1, n_elem), dtype=np.float64)

        u_diff = upp - rho_curr
        l_diff = rho_curr - low
        reg_const = 1e-5 * delta_rho

        for i in range(m + 1):
            df_i = grad_all[i]
            pos = df_i > 0.0
            neg = df_i < 0.0
            zero = ~pos & ~neg

            # Positive gradient elements
            p_mat[i, pos] = (u_diff[pos] ** 2) * (1.001 * df_i[pos]) + reg_const
            q_mat[i, pos] = 0.001 * (l_diff[pos] ** 2) * df_i[pos] + reg_const

            # Negative gradient elements
            p_mat[i, neg] = 0.001 * (u_diff[neg] ** 2) * (-df_i[neg]) + reg_const
            q_mat[i, neg] = (l_diff[neg] ** 2) * (1.001 * (-df_i[neg])) + reg_const

            # Zero gradient elements
            p_mat[i, zero] = reg_const
            q_mat[i, zero] = reg_const

        p0 = p_mat[0]  # (n_elem,)
        q0 = q_mat[0]  # (n_elem,)
        pi = p_mat[1:] if m > 0 else np.zeros((0, n_elem)) # (m, n_elem)
        qi = q_mat[1:] if m > 0 else np.zeros((0, n_elem)) # (m, n_elem)

        # Constant terms r_i
        # r_0: objective offset
        r0 = -float(np.sum(p0 / u_diff + q0 / l_diff))
        # r_i for constraints i=1..m
        if m > 0:
            curr_approx = np.sum(pi / u_diff[None, :] + qi / l_diff[None, :], axis=1) # (m,)
            r_vec = g_arr - curr_approx  # (m,)
        else:
            r_vec = np.zeros(0, dtype=np.float64)

        # -------------------------------------------------------------------
        # 4. Dual Subproblem Optimization
        # -------------------------------------------------------------------
        if m == 0:
            # Unconstrained case: lambda is empty
            rho_star = (np.sqrt(p0) * low + np.sqrt(q0) * upp) / (np.sqrt(p0) + np.sqrt(q0))
            rho_new = np.clip(rho_star, alpha, beta)
            if np.any(solid_mask):
                rho_new[solid_mask] = self.rho_max
            if np.any(void_mask):
                rho_new[void_mask] = self.rho_min
            return rho_new

        # Primal mapping rho(lambda)
        def primal_solution(lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            # P_e = p0_e + sum_i lam_i * pi_ie
            # Q_e = q0_e + sum_i lam_i * qi_ie
            P = p0 + lam @ pi  # (n_elem,)
            Q = q0 + lam @ qi  # (n_elem,)
            sqrt_P = np.sqrt(P)
            sqrt_Q = np.sqrt(Q)
            rho_star = (sqrt_P * low + sqrt_Q * upp) / (sqrt_P + sqrt_Q)
            rho_val = np.clip(rho_star, alpha, beta)
            unclipped = (rho_star > alpha + 1e-12) & (rho_star < beta - 1e-12)
            return rho_val, P, Q, unclipped

        # Dual function objective and gradient (minimizing -W(lambda))
        def dual_objective_and_grad(lam: np.ndarray) -> Tuple[float, np.ndarray]:
            lam_safe = np.maximum(lam, 0.0)
            rho_val, P, Q, _ = primal_solution(lam_safe)

            u_d = upp - rho_val
            l_d = rho_val - low

            # Slack variable y_i*(lambda_i) = max(0, (lambda_i - c_i) / d_i)
            y_star = np.maximum(0.0, (lam_safe - self.c_i) / self.d_i)

            # Constraint approximation: tilde_g_i(lambda)
            # tilde_g_i = r_i + sum_e (pi_ie / u_d_e + qi_ie / l_d_e)
            tilde_g = r_vec + np.sum(pi / u_d[None, :] + qi / l_d[None, :], axis=1)

            # W(lambda) = r0 + sum_i lam_i * r_i + sum_e (P / u_d + Q / l_d) - sum_i (c_i y + 0.5 d_i y^2 - lam y)
            # Slack contribution: c_i y + 0.5 d_i y^2 - lam y = -0.5 * d_i * (y^*)^2
            W = (
                r0
                + float(np.dot(lam_safe, r_vec))
                + float(np.sum(P / u_d + Q / l_d))
                - float(0.5 * self.d_i * np.sum(y_star ** 2))
            )

            # Gradient: d(-W)/dlambda_i = -(tilde_g_i - y_star_i)
            grad_dual = -(tilde_g - y_star)
            return -W, grad_dual

        def dual_hessian(lam: np.ndarray) -> np.ndarray:
            lam_safe = np.maximum(lam, 0.0)
            rho_val, P, Q, unclipped = primal_solution(lam_safe)

            u_d = upp - rho_val
            l_d = rho_val - low

            # H_ee = 2*P / u_d^3 + 2*Q / l_d^3
            H_ee = (2.0 * P) / (u_d ** 3) + (2.0 * Q) / (l_d ** 3)

            # d_tilde_g / d_rho_e: shape (m, n_elem)
            d_g_d_rho = pi / (u_d[None, :] ** 2) - qi / (l_d[None, :] ** 2)

            # Hessian of -W:
            # sum_{e unclipped} (1 / H_ee) * d_g_d_rho[:, e] @ d_g_d_rho[:, e]^T + diag(I(lam > c) / d)
            H = np.zeros((m, m), dtype=np.float64)
            if np.any(unclipped):
                inv_H = 1.0 / H_ee[unclipped]
                weighted_dg = d_g_d_rho[:, unclipped] * np.sqrt(inv_H)[None, :]
                H = weighted_dg @ weighted_dg.T

            slack_active = (lam_safe > self.c_i)
            if np.any(slack_active):
                H += np.diag(slack_active.astype(np.float64) / self.d_i)

            return H

        # -------------------------------------------------------------------
        # 5. Dual Solve: Newton-Raphson with Line Search & Fallback
        # -------------------------------------------------------------------
        lam = np.zeros(m, dtype=np.float64)
        converged = False

        for _ in range(self.max_sub_iter):
            val, grad = dual_objective_and_grad(lam)

            # Check projected KKT gradient tolerance:
            # If lam_i == 0 and grad_i >= 0, it satisfies KKT stationary condition.
            # If grad_i < 0, lam_i needs to increase.
            proj_grad = np.where((lam <= 1e-12) & (grad >= 0.0), 0.0, grad)
            if np.max(np.abs(proj_grad)) < self.tol:
                converged = True
                break

            H = dual_hessian(lam)
            # Regularize Hessian for stability
            H_reg = H + 1e-9 * np.eye(m)
            try:
                d_lam = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                d_lam = -grad

            # Projected backtracking line search
            step = 1.0
            cur_val = val
            success_step = False
            for _ls in range(25):
                lam_trial = np.maximum(0.0, lam + step * d_lam)
                val_trial, _ = dual_objective_and_grad(lam_trial)
                # Armijo sufficient decrease
                if val_trial <= cur_val + 1e-4 * step * float(np.dot(grad, lam_trial - lam)):
                    lam = lam_trial
                    success_step = True
                    break
                step *= 0.5

            if not success_step:
                # Step failed to decrease, fallback to gradient step
                lam = np.maximum(0.0, lam - 1e-3 * grad)

            if np.max(np.abs(step * d_lam)) < 1e-12:
                converged = True
                break

        if not converged:
            # Fallback to Scipy L-BFGS-B
            bnds = [(0.0, None) for _ in range(m)]
            res = minimize(
                dual_objective_and_grad,
                lam,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds,
                tol=self.tol,
                options={"maxiter": self.max_sub_iter, "ftol": 1e-14, "gtol": self.tol},
            )
            lam = np.maximum(0.0, res.x)

        # -------------------------------------------------------------------
        # 6. Extract Primal Solution and Apply Passive Domains
        # -------------------------------------------------------------------
        rho_new, _, _, _ = primal_solution(lam)

        if np.any(solid_mask):
            rho_new[solid_mask] = self.rho_max
        if np.any(void_mask):
            rho_new[void_mask] = self.rho_min

        return rho_new
