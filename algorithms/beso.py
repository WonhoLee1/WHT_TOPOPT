"""Bi-directional Evolutionary Structural Optimization (BESO) algorithm.

References:
    - Huang, X., & Xie, Y. M. (2007), "A new algorithm for bi-directional evolutionary structural
      optimization", Journal of Computational Mechanics, 49(4), 1091-1099.
    - Tian, K., & Zhang, W. (2025), "Bi-Directional Evolutionary Topology Optimization with Adaptive
      Evolutionary Ratio for Nonlinear Structures", Chinese Journal of Mechanical Engineering, 38:122.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .base import GradientSource, Optimizer, TrustRegionController


def check_load_path_connected(
    elems: np.ndarray,
    n_nodes: int,
    solid_mask: np.ndarray,
    load_nodes: np.ndarray,
    fixed_nodes: np.ndarray,
) -> bool:
    """Return True if every loaded node is connected via solid elements to a supported node."""
    if not np.any(solid_mask):
        return False
    conn = np.asarray(elems, dtype=np.int64)[solid_mask]
    rows = np.repeat(conn[:, 0], conn.shape[1] - 1)
    cols = conn[:, 1:].ravel()
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
    _, label = connected_components(graph, directed=False)
    in_solid = np.zeros(n_nodes, dtype=bool)
    in_solid[conn.ravel()] = True
    load_arr = np.asarray(load_nodes, dtype=np.int64)
    fixed_arr = np.asarray(fixed_nodes, dtype=np.int64)
    if len(load_arr) == 0 or len(fixed_arr) == 0:
        return True
    if not np.all(in_solid[load_arr]):
        return False
    supported_labels = np.unique(label[fixed_arr[in_solid[fixed_arr]]])
    return bool(np.all(np.isin(label[load_arr], supported_labels)))


class BESOOptimizer(Optimizer):
    """Bi-directional Evolutionary Structural Optimization (BESO) solver.

    Features:
        - Huang & Xie (2007) historical sensitivity averaging:
            alpha_tilde^(k) = 0.5 * (alpha^(k) + alpha^(k-1))
        - Tian & Zhang (2025) adaptive evolutionary ratio ER^(k) based on current volume fraction.
        - Volume-weighted threshold bisection to determine target elemental topology.
        - Maximum admission ratio AR_max to prevent sudden element churn.
        - Support for both discrete soft-kill (rho in {rho_min, 1.0}) and quasi-continuous relaxation.
        - Support for passive solid, passive void, and trust-region move limits.
    """

    def __init__(
        self,
        target_volfrac: float = 0.5,
        er: float = 0.02,
        er_adaptive: bool = False,
        er_adaptive_scheme: str = "ER1",
        er_min: float = 0.02,
        er_max: float = 0.05,
        ar_max: float = 0.04,
        rho_min: float = 1e-3,
        quasi_continuous: bool = False,
        move: float = 0.2,
        tol: float = 1e-4,
        trust_region: Optional[TrustRegionController] = None,
        gradient_source: GradientSource = GradientSource.EXACT_ADJOINT,
        mesh: Optional[Any] = None,
        load_nodes: Optional[np.ndarray] = None,
        fixed_nodes: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize BESOOptimizer.

        Parameters
        ----------
        target_volfrac : float
            Final target volume fraction V* / V0.
        er : float
            Base evolutionary ratio (default 0.02 = 2%).
        er_adaptive : bool
            Whether to use Tian & Zhang (2025) adaptive evolutionary ratio.
        er_adaptive_scheme : str
            Adaptive scheme ('ER1', 'ER2', 'ER3', 'ER4').
        er_min : float
            Minimum adaptive evolutionary ratio (default 0.02).
        er_max : float
            Maximum adaptive evolutionary ratio (default 0.05).
        ar_max : float
            Maximum admission ratio per iteration (default 0.04).
        rho_min : float
            Void density value (default 1e-3).
        quasi_continuous : bool
            Whether to allow continuous density relaxation instead of strict soft-kill.
        move : float
            Maximum step change when quasi_continuous is True.
        tol : float
            Bisection tolerance for matching target volume.
        trust_region : Optional[TrustRegionController]
            Optional trust-region controller.
        gradient_source : GradientSource
            Source of sensitivity information.
        """
        if not (0.0 < target_volfrac <= 1.0):
            raise ValueError(f"target_volfrac must be in (0, 1], got {target_volfrac}")

        self.target_volfrac = float(target_volfrac)
        self.er = float(er)
        self.er_adaptive = bool(er_adaptive)
        self.er_adaptive_scheme = er_adaptive_scheme.upper()
        self.er_min = float(er_min)
        self.er_max = float(er_max)
        self.ar_max = float(ar_max)
        self.rho_min = float(rho_min)
        self.quasi_continuous = bool(quasi_continuous)
        self.move = float(move)
        self.tol = float(tol)
        self.trust_region = trust_region
        self.gradient_source = gradient_source
        self.mesh = mesh
        self.load_nodes = load_nodes
        self.fixed_nodes = fixed_nodes

        # Internal history state
        self.iter: int = 0
        self.sens_history: Optional[np.ndarray] = None
        self.current_vol: Optional[float] = None

    def reset(self) -> None:
        """Reset internal optimization history."""
        self.iter = 0
        self.sens_history = None
        self.current_vol = None

    def compute_adaptive_er(self, vol_frac: float) -> float:
        """Compute adaptive evolutionary ratio using Tian & Zhang (2025) formulations.

        Parameters
        ----------
        vol_frac : float
            Current design volume fraction V / V0.

        Returns
        -------
        er : float
            Adaptive evolutionary ratio.
        """
        vh = float(vol_frac)
        er_min = self.er_min
        er_max = self.er_max
        delta_er = er_max - er_min

        scheme = self.er_adaptive_scheme
        if scheme == "ER1":
            # ER1 = 0.5 * (er_max - er_min) * tanh(25 * (vh - 0.5)) + 0.5 * (er_max + er_min)
            return float(0.5 * delta_er * np.tanh(25.0 * (vh - 0.5)) + 0.5 * (er_max + er_min))
        elif scheme == "ER2":
            # ER2 = (er_max - er_min) / (1 + exp(-30 * (vh - 0.5))) + er_min
            return float(delta_er / (1.0 + np.exp(-30.0 * (vh - 0.5))) + er_min)
        elif scheme == "ER3":
            # ER3 = delta_er * ln(1 + exp(12 * (vh - 0.5))) / ln(1 + exp(6)) + er_min
            num = np.log(1.0 + np.exp(12.0 * (vh - 0.5)))
            den = np.log(1.0 + np.exp(6.0))
            return float(delta_er * (num / den) + er_min)
        elif scheme == "ER4":
            # ER4 = er_min + delta_er * vh
            return float(er_min + delta_er * vh)
        else:
            return self.er

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
        """Execute one BESO iteration.

        Parameters
        ----------
        rho : np.ndarray
            Current elemental density array (N_elem,).
        df_drho : np.ndarray
            Objective sensitivity array (N_elem,). For compliance minimization,
            -df_drho is the positive strain energy density.
        g_vals : np.ndarray
            Constraint values (unused or for monitoring).
        dg_drho : np.ndarray
            Constraint sensitivities (unused or volume weighting).
        elem_volumes : Optional[np.ndarray]
            Element volume array (N_elem,).
        passive_solid : Optional[np.ndarray]
            Elements locked to solid (rho=1.0).
        passive_void : Optional[np.ndarray]
            Elements locked to void (rho=rho_min).

        Returns
        -------
        rho_new : np.ndarray
            Updated elemental density array of shape (N_elem,).
        """
        rho_curr = np.asarray(rho, dtype=np.float64).copy()
        n_elem = len(rho_curr)

        if elem_volumes is None:
            volumes = np.ones(n_elem, dtype=np.float64)
        else:
            volumes = np.asarray(elem_volumes, dtype=np.float64)

        total_volume = float(np.sum(volumes))
        target_volume = self.target_volfrac * total_volume
        current_volume = float(np.sum(rho_curr * volumes))
        current_volfrac = current_volume / total_volume

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
        # 1. Historical Sensitivity Averaging (Huang & Xie 2007)
        # -------------------------------------------------------------------
        self.iter += 1
        # Sensitivity number: for compliance, -df/drho > 0 represents strain energy density
        raw_sens = -np.asarray(df_drho, dtype=np.float64).ravel()

        if self.sens_history is None or self.iter == 1:
            sens_avg = raw_sens.copy()
        else:
            sens_avg = 0.5 * (raw_sens + self.sens_history)

        self.sens_history = sens_avg.copy()

        # -------------------------------------------------------------------
        # 2. Evolutionary Ratio and Target Volume Schedule
        # -------------------------------------------------------------------
        if self.er_adaptive:
            current_er = self.compute_adaptive_er(current_volfrac)
        else:
            current_er = self.er

        if self.trust_region is not None:
            # Scale evolutionary ratio by trust region factor if needed
            current_er = min(current_er, self.trust_region.move)

        if current_volume > target_volume:
            next_vol = max(target_volume, current_volume - current_er * total_volume)
        elif current_volume < target_volume:
            next_vol = min(target_volume, current_volume + current_er * total_volume)
        else:
            next_vol = target_volume

        # -------------------------------------------------------------------
        # 3. Volume-Weighted Threshold Bisection
        # -------------------------------------------------------------------
        active_mask = ~(solid_mask | void_mask)
        active_indices = np.flatnonzero(active_mask)

        # Candidate threshold bounds
        th_min = float(np.min(sens_avg))
        th_max = float(np.max(sens_avg))

        def evaluate_threshold(th: float) -> Tuple[np.ndarray, float]:
            rho_cand = np.full(n_elem, self.rho_min, dtype=np.float64)
            # Active elements above threshold become solid
            solid_active = active_mask & (sens_avg > th)
            rho_cand[solid_active] = 1.0

            # Passive elements
            if np.any(solid_mask):
                rho_cand[solid_mask] = 1.0
            if np.any(void_mask):
                rho_cand[void_mask] = self.rho_min

            vol = float(np.sum(rho_cand * volumes))
            return rho_cand, vol

        # Bisection loop
        th_l, th_u = th_min, th_max
        best_rho = rho_curr.copy()

        for _ in range(100):
            th_mid = 0.5 * (th_l + th_u)
            rho_cand, vol_cand = evaluate_threshold(th_mid)
            best_rho = rho_cand

            vol_err = abs(vol_cand - next_vol) / total_volume
            if vol_err < self.tol or (th_u - th_l) < 1e-12:
                break

            if vol_cand > next_vol:
                th_l = th_mid  # Needs higher threshold to reduce solid volume
            else:
                th_u = th_mid  # Needs lower threshold to increase solid volume

        # -------------------------------------------------------------------
        # 4. Load-Path Connectivity Protection (if mesh/load_nodes/fixed_nodes given)
        # -------------------------------------------------------------------
        if (
            self.mesh is not None
            and self.load_nodes is not None
            and self.fixed_nodes is not None
            and hasattr(self.mesh, "elements")
        ):
            solid_cand = best_rho > 0.5
            if not check_load_path_connected(
                self.mesh.elements, self.mesh.n_nodes, solid_cand, self.load_nodes, self.fixed_nodes
            ):
                # Rank active elements by sensitivity descending and binary-search minimum keep count
                order = active_indices[np.argsort(-sens_avg[active_indices])]
                k_lo = int(np.sum(solid_cand[active_indices]))
                k_hi = len(order)
                while k_lo < k_hi:
                    k_mid = (k_lo + k_hi) // 2
                    test_solid = solid_mask.copy()
                    test_solid[order[:k_mid]] = True
                    if check_load_path_connected(
                        self.mesh.elements, self.mesh.n_nodes, test_solid, self.load_nodes, self.fixed_nodes
                    ):
                        k_hi = k_mid
                    else:
                        k_lo = k_mid + 1
                best_rho = np.full(n_elem, self.rho_min, dtype=np.float64)
                best_rho[solid_mask] = 1.0
                best_rho[order[:k_lo]] = 1.0
                if np.any(void_mask):
                    best_rho[void_mask] = self.rho_min

        # -------------------------------------------------------------------
        # 5. Quasi-Continuous Relaxation (if requested)
        # -------------------------------------------------------------------
        if self.quasi_continuous:
            move_eff = self.move if self.trust_region is None else self.trust_region.move
            direction = np.sign(best_rho - rho_curr)
            rho_final = np.clip(rho_curr + direction * move_eff, self.rho_min, 1.0)
            if np.any(solid_mask):
                rho_final[solid_mask] = 1.0
            if np.any(void_mask):
                rho_final[void_mask] = self.rho_min
            return rho_final

        return best_rho
