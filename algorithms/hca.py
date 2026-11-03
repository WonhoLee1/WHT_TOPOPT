"""Stress-Energy Adaptive Hybrid Cellular Automata (SE-AHCA) Optimizer Module.

Implements the advanced Stress-Energy Adaptive Hybrid Cellular Automata algorithm
specified in `dev_log/advanced_hca_implementation_and_plan.md`.

Key Features:
1. Mixed Strain Energy Density (SED) & Fully Stressed Design (FSD) state variables.
2. qp-Relaxation (Le et al. 2010) on von Mises stress: hat_sigma = (rho^q) * sigma_0.
3. Multi-case & transient dynamic temporal envelope via P-mean / max pooling.
4. Bounded saturation error function: err = tanh(beta * (S_bar - S*) / S*).
5. Exact volume-preserving bisection setpoint (S*) search.
6. Element-wise Adaptive PID gain scheduling (2025/2026) detecting error oscillations.
7. Drop-in compatibility with wht_topopt Optimizer abstract base class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from .base import Optimizer


@dataclass
class SEAHCAConfig:
    """Configuration dataclass for Stress-Energy Adaptive Hybrid Cellular Automata (SE-AHCA)."""
    volfrac_target: float = 0.30
    xmin: float = 1e-3
    xmax: float = 1.0
    # Stiffness (SED U_e) vs Stress (von Mises sigma_vm) blend weight eta_sigma in [0.0, 1.0]
    # eta_sigma = 0.0: pure minimum compliance / SED leveling
    # eta_sigma = 0.45: balanced compliance + peak stress leveling
    # eta_sigma = 1.0: pure fully stressed design (FSD)
    stress_weight_eta: float = 0.45
    sigma_allow: float = 0.0               # Allowable von Mises stress in the problem's stress unit; <= 0: 90th percentile of the field
    qp_relax_exp: float = 0.50             # Low-density stress singularity relaxation exponent q
    overstress_exponent: float = 3.0       # Asymmetric penalty exponent when sigma > sigma_allow
    p_envelope: float = 12.0               # Multi-load / transient time-step P-mean envelope exponent
    # Adaptive PID Gains (Adaptive HCA 2025/2026)
    kp_init: float = 0.16
    kp_min: float = 0.02
    kp_max: float = 0.38
    ki: float = 0.04
    kd: float = 0.02
    gain_inc_factor: float = 1.15          # Gain acceleration factor under monotonic convergence
    gain_dec_factor: float = 0.70          # Gain damping factor under sign reversal (oscillation)
    move_limit: float = 0.15               # Max design change per iteration
    error_sharpness_beta: float = 1.0      # tanh saturation error slope
    history_window: int = 2                # Moving average history window


class StressEnergyAdaptiveHCA:
    """Stress-Energy Adaptive Hybrid Cellular Automata (SE-AHCA) Core Engine.
    
    A gradient-free structural optimization engine leveling strain energy density and
    mitigating local von Mises stress concentration without requiring adjoint PDE solves.
    """

    def __init__(
        self,
        elem_volumes: np.ndarray,
        config: Optional[SEAHCAConfig] = None,
        passive_solid_mask: Optional[np.ndarray] = None,
        passive_void_mask: Optional[np.ndarray] = None,
    ):
        self.volumes = np.asarray(elem_volumes, dtype=float)
        self.n_elem = len(self.volumes)
        self.cfg = config or SEAHCAConfig()
        self.total_vol = float(np.sum(self.volumes))
        self.target_vol = self.cfg.volfrac_target * self.total_vol

        self.passive_solid = (
            np.zeros(self.n_elem, dtype=bool)
            if passive_solid_mask is None
            else np.asarray(passive_solid_mask, dtype=bool)
        )
        self.passive_void = (
            np.zeros(self.n_elem, dtype=bool)
            if passive_void_mask is None
            else np.asarray(passive_void_mask, dtype=bool)
        )

        # Independent element-wise adaptive proportional gains K_P and PID state
        self.kp_elem = np.full(self.n_elem, self.cfg.kp_init, dtype=float)
        self.prev_err = np.zeros(self.n_elem, dtype=float)
        self.int_err = np.zeros(self.n_elem, dtype=float)
        self.signal_history: List[np.ndarray] = []
        self.iter = 0

    def aggregate_multicase_fields(
        self,
        rho_phys: np.ndarray,
        strain_energy_cases: np.ndarray,                           # shape: (n_cases, n_elem) or (n_elem,)
        von_mises_unpenalized_cases: Optional[np.ndarray] = None,  # shape: (n_cases, n_elem) or (n_elem,)
        case_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate strain energy density and qp-relaxed von Mises stress across multiple cases/time steps."""
        u_arr = np.asarray(strain_energy_cases, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr[None, :]
        n_cases = u_arr.shape[0]

        if case_weights is None:
            w = np.full(n_cases, 1.0 / n_cases, dtype=float)
        else:
            w = np.asarray(case_weights, dtype=float)
            w = w / np.maximum(np.sum(w), 1e-16)

        # 1. Weighted Average Strain Energy Density U_e^{eff} = sum(w_k * U_e_k) / v_e
        u_eff = (w @ u_arr) / np.maximum(self.volumes, 1e-18)

        # 2. qp-Relaxed von Mises Stress P-mean envelope
        if von_mises_unpenalized_cases is None:
            sigma_peak = np.zeros(self.n_elem, dtype=float)
        else:
            s0_arr = np.asarray(von_mises_unpenalized_cases, dtype=float)
            if s0_arr.ndim == 1:
                s0_arr = s0_arr[None, :]
            # qp-Relaxation: hat_sigma = (rho_phys ^ q) * sigma_0
            relax_factor = np.maximum(rho_phys, self.cfg.xmin) ** self.cfg.qp_relax_exp
            s_relaxed = s0_arr * relax_factor[None, :]

            if n_cases == 1:
                sigma_peak = s_relaxed[0]
            else:
                P = self.cfg.p_envelope
                s_max_per_elem = np.maximum(np.max(s_relaxed, axis=0), 1e-12)
                norm_s = s_relaxed / s_max_per_elem[None, :]
                sigma_peak = s_max_per_elem * ((np.mean(norm_s ** P, axis=0)) ** (1.0 / P))

        return u_eff, sigma_peak

    def compute_cell_state_signal(
        self,
        rho_phys: np.ndarray,
        strain_energy_cases: np.ndarray,
        von_mises_unpenalized_cases: Optional[np.ndarray] = None,
        case_weights: Optional[np.ndarray] = None,
        neighborhood_filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Combine SED and relaxed stress, apply spatial neighborhood filter and temporal smoothing."""
        u_eff, sigma_peak = self.aggregate_multicase_fields(
            rho_phys, strain_energy_cases, von_mises_unpenalized_cases, case_weights
        )

        active_mask = rho_phys > (self.cfg.xmin * 2.0)
        if np.any(active_mask):
            u_ref = max(float(np.percentile(u_eff[active_mask], 85)), 1e-12)
        else:
            u_ref = max(float(np.percentile(u_eff, 85)), 1e-12)
        u_norm = u_eff / u_ref

        eta = self.cfg.stress_weight_eta
        if eta > 0.0 and np.max(sigma_peak) > 0.0:
            s_allow = self.cfg.sigma_allow
            if s_allow <= 0.0:
                s_allow = max(float(np.percentile(sigma_peak[active_mask], 90)), 1e-12)
            r_stress = sigma_peak / max(s_allow, 1e-12)
            m_exp = self.cfg.overstress_exponent
            phi_stress = np.where(r_stress > 1.0, r_stress ** m_exp, r_stress ** 2)
            raw_signal = (1.0 - eta) * u_norm + eta * phi_stress
        else:
            raw_signal = u_norm

        # Spatial neighborhood averaging
        if neighborhood_filter_fn is not None:
            spatial_signal = neighborhood_filter_fn(raw_signal)
        else:
            spatial_signal = raw_signal

        # Temporal moving average smoothing
        self.signal_history.append(spatial_signal.copy())
        if len(self.signal_history) > self.cfg.history_window:
            self.signal_history.pop(0)
        smoothed_signal = np.mean(self.signal_history, axis=0)

        # Diagnostic stress uniformity index
        solid_mask = rho_phys >= 0.5
        if np.sum(solid_mask) > 10:
            stress_cov = float(
                np.std(sigma_peak[solid_mask]) / max(float(np.mean(sigma_peak[solid_mask])), 1e-12)
            )
            max_solid_stress = float(np.max(sigma_peak[solid_mask]))
        else:
            stress_cov = 0.0
            max_solid_stress = float(np.max(sigma_peak))

        metrics = {
            "max_relaxed_stress": float(np.max(sigma_peak)),
            "max_solid_stress": max_solid_stress,
            "solid_stress_cov": stress_cov,
            "mean_sed": float(np.mean(u_eff)),
        }
        return smoothed_signal, metrics

    def step(
        self,
        xi_current: np.ndarray,
        cell_signal: np.ndarray,
        forward_projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform one SE-AHCA iteration step with volume-preserving bisection.
        
        Args:
            xi_current: (N_elem,) Current mathematical design variable array.
            cell_signal: (N_elem,) Smoothed cell state signal S_bar_e.
            forward_projection_fn: Optional projection mapping xi -> rho_phys.
            
        Returns:
            (xi_next, rho_phys_next, s_star_optimal)
        """
        self.iter += 1
        proj_fn = forward_projection_fn if forward_projection_fn is not None else (lambda x: x)

        s_field = np.maximum(cell_signal, 1e-16)
        low_s = float(np.min(s_field)) * 1e-4
        high_s = float(np.max(s_field)) * 1e4
        beta = self.cfg.error_sharpness_beta

        def evaluate_trial(s_star: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            err = np.tanh(beta * (s_field - s_star) / max(s_star, 1e-15))
            d_xi = (
                self.kp_elem * err
                + self.cfg.ki * (self.int_err + err)
                + self.cfg.kd * (err - self.prev_err)
            )
            d_xi = np.clip(d_xi, -self.cfg.move_limit, self.cfg.move_limit)
            xi_try = np.clip(xi_current + d_xi, self.cfg.xmin, self.cfg.xmax)
            if np.any(self.passive_solid):
                xi_try[self.passive_solid] = self.cfg.xmax
            if np.any(self.passive_void):
                xi_try[self.passive_void] = self.cfg.xmin
            rho_try = proj_fn(xi_try)
            return xi_try, rho_try, err

        # Exact volume-preserving bisection for S*
        for _ in range(50):
            mid_s = 0.5 * (low_s + high_s)
            _, rho_try, _ = evaluate_trial(mid_s)
            if float(np.dot(self.volumes, rho_try)) > self.target_vol:
                low_s = mid_s
            else:
                high_s = mid_s

        s_star_opt = 0.5 * (low_s + high_s)
        xi_next, rho_next, curr_err = evaluate_trial(s_star_opt)

        # Element-wise adaptive proportional gain scheduling (Adaptive HCA 2025/2026)
        if self.iter >= 2:
            osc_indicator = curr_err * self.prev_err
            self.kp_elem = np.where(
                osc_indicator < 0.0,
                self.kp_elem * self.cfg.gain_dec_factor,
                np.where(
                    osc_indicator > 0.0,
                    self.kp_elem * self.cfg.gain_inc_factor,
                    self.kp_elem,
                ),
            )
            self.kp_elem = np.clip(self.kp_elem, self.cfg.kp_min, self.cfg.kp_max)

        # Anti-windup and PID state update
        self.int_err = np.clip(self.int_err + curr_err, -2.5, 2.5)
        self.prev_err = curr_err.copy()
        return xi_next, rho_next, s_star_opt


class HCAOptimizer(Optimizer):
    """Adapter wrapping StressEnergyAdaptiveHCA into the standard wht_topopt Optimizer interface.
    
    Provides 100% plug-and-play drop-in compatibility with MMAOptimizer and SEMDOTOptimizer.
    """

    def __init__(
        self,
        volfrac_target: float = 0.30,
        move: float = 0.15,
        stress_weight: float = 0.45,
        sigma_allow: float = 0.0,
        rho_min: float = 1e-3,
        rho_max: float = 1.0,
        config: Optional[SEAHCAConfig] = None,
    ):
        super().__init__()
        self.cfg = config or SEAHCAConfig(
            volfrac_target=volfrac_target,
            move_limit=move,
            stress_weight_eta=stress_weight,
            sigma_allow=sigma_allow,
            xmin=rho_min,
            xmax=rho_max,
        )
        self._engine: Optional[StressEnergyAdaptiveHCA] = None
        self.last_metrics: Dict[str, float] = {}

    def update(
        self,
        rho: np.ndarray,
        df_drho: np.ndarray,
        g_vals: np.ndarray,
        dg_drho: np.ndarray,
        elem_volumes: Optional[np.ndarray] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
        elem_strain_energy: Optional[np.ndarray] = None,
        elem_von_mises: Optional[np.ndarray] = None,
        forward_projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        neighborhood_filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """Standard update interface conforming to Optimizer ABC."""
        n_elem = len(rho)
        vols = elem_volumes if elem_volumes is not None else np.ones(n_elem, dtype=float)

        if self._engine is None:
            self._engine = StressEnergyAdaptiveHCA(
                elem_volumes=vols,
                config=self.cfg,
                passive_solid_mask=passive_solid,
                passive_void_mask=passive_void,
            )

        # 1. Determine cell strain energy
        if elem_strain_energy is not None:
            u_e = elem_strain_energy
        else:
            # Fallback for standard SIMP gradient loops: df_drho is proportional to -c_e
            # where c_e = p * rho^(p-1) * u_e^T * K_0 * u_e.
            # Using |df_drho| as surrogate strain energy proxy
            u_e = np.abs(df_drho) * np.maximum(rho, 1e-3)

        # 2. Cell state signal
        signal, metrics = self._engine.compute_cell_state_signal(
            rho_phys=rho,
            strain_energy_cases=u_e,
            von_mises_unpenalized_cases=elem_von_mises,
            neighborhood_filter_fn=neighborhood_filter_fn,
        )
        self.last_metrics = metrics

        # 3. Step forward
        xi_next, _, _ = self._engine.step(
            xi_current=rho,
            cell_signal=signal,
            forward_projection_fn=forward_projection_fn,
        )
        return xi_next  # the design variable: returning the projected field would filter it again every iteration
