"""Optimality Criteria (OC) mathematical optimization algorithm for minimum compliance."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .base import GradientSource, Optimizer, TrustRegionController


class OptimalityCriteria(Optimizer):
    """Volume-weighted Optimality Criteria (OC) updating algorithm.
    
    Implements the classical Bendsøe & Sigmund OC update:
        B_e = (-df/drho_e) / max(1e-30, lambda * dv/drho_e)
        rho_e_new = clip(rho_e * B_e^eta, max(rho_min, rho_e - move), min(1.0, rho_e + move))
        
    Enforces passive solid (rho=1) and passive void (rho=rho_min) design domains
    within a robust logarithmic bisection loop finding multiplier lambda.
    """

    def __init__(
        self,
        volfrac: float = 0.5,
        move: float = 0.2,
        eta: float = 0.5,
        rho_min: float = 1e-3,
        tol: float = 1e-9,
        max_bisection_iter: int = 300,
        trust_region: Optional[TrustRegionController] = None,
        gradient_source: GradientSource = GradientSource.EXACT_ADJOINT,
    ) -> None:
        """Initialize OptimalityCriteria optimizer.
        
        Args:
            volfrac: Prescribed upper limit volume fraction V* / V0.
            move: Maximum design change per iteration ||rho^(k+1) - rho^(k)||_inf.
            eta: Numerical damping exponent (typically 0.5 for compliance).
            rho_min: Minimum physical density lower bound to avoid stiffness singularity.
            tol: Bisection convergence tolerance for relative volume constraint violation.
            max_bisection_iter: Maximum bisection iterations.
            trust_region: Optional trust-region controller adapting move limit.
            gradient_source: Source of gradients (affects trust region behavior).
        """
        if not (0.0 < volfrac <= 1.0):
            raise ValueError(f"volfrac must be in (0, 1], got {volfrac}")
        if move <= 0.0:
            raise ValueError(f"move must be positive, got {move}")

        self.volfrac = float(volfrac)
        self.move = float(move)
        self.eta = float(eta)
        self.rho_min = float(rho_min)
        self.tol = float(tol)
        self.max_bisection_iter = int(max_bisection_iter)
        self.trust_region = trust_region
        self.gradient_source = gradient_source

    def update(
        self,
        rho: np.ndarray,
        df_drho: np.ndarray,
        g_vals: Optional[np.ndarray] = None,
        dg_drho: Optional[np.ndarray] = None,
        elem_volumes: Optional[np.ndarray] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute volume-constrained OC bisection update."""
        rho_curr = np.asarray(rho, dtype=float).copy()
        n_elem = len(rho_curr)

        if elem_volumes is None:
            volumes = np.ones(n_elem, dtype=float)
        else:
            volumes = np.asarray(elem_volumes, dtype=float)

        total_volume = float(np.sum(volumes))
        target_vol = self.volfrac * total_volume

        # Determine effective move limit (using trust region if available)
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

        # Sensitivity numerator: for compliance minimization, df_drho is typically negative,
        # so -df_drho is positive strain energy density.
        num = np.maximum(-df_drho, 1e-30)

        # Volume derivative per element: dv/drho_e
        if dg_drho is not None and dg_drho.size == n_elem:
            den = np.asarray(dg_drho, dtype=float).ravel()
            den = np.where(den > 0.0, den, volumes)
        else:
            den = volumes.copy()

        # Logarithmic bisection bounds for Lagrange multiplier lambda
        l1, l2 = 1e-15, 1e15

        # Check bounds: ensure l1 gives volume > target_vol and l2 gives volume < target_vol
        def evaluate_candidate(lmbda: float) -> Tuple[np.ndarray, float]:
            be = (num / np.maximum(lmbda * den, 1e-30)) ** self.eta
            cand = rho_curr * be
            # Dynamic move limits
            lb = np.maximum(self.rho_min, rho_curr - current_move)
            ub = np.minimum(1.0, rho_curr + current_move)
            rho_new = np.clip(cand, lb, ub)

            # Enforce passive domains
            if np.any(solid_mask):
                rho_new[solid_mask] = 1.0
            if np.any(void_mask):
                rho_new[void_mask] = self.rho_min

            vol = float(np.sum(rho_new * volumes))
            return rho_new, vol

        # Expand bounds if necessary
        _, v1 = evaluate_candidate(l1)
        while v1 < target_vol and l1 > 1e-40:
            l1 *= 0.01
            _, v1 = evaluate_candidate(l1)

        _, v2 = evaluate_candidate(l2)
        while v2 > target_vol and l2 < 1e40:
            l2 *= 100.0
            _, v2 = evaluate_candidate(l2)

        rho_final = rho_curr
        for _ in range(self.max_bisection_iter):
            mid = np.sqrt(l1 * l2)
            rho_cand, vol_cand = evaluate_candidate(mid)
            rho_final = rho_cand

            rel_err = abs(vol_cand - target_vol) / total_volume
            if rel_err < self.tol or (l2 / l1 - 1.0) < 1e-12:
                break

            if vol_cand > target_vol:
                l1 = mid
            else:
                l2 = mid

        return rho_final


OCOptimizer = OptimalityCriteria
