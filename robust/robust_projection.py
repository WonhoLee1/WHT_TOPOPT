"""Robust topology optimization formulation based on 3-field projection.

Implements Wang, Lazarov & Sigmund (2011) 3-field robust topology optimization wrapper:
- Evaluates underlying ObjectiveFunction across eroded, intermediate, and dilated fields.
- Supports worst-case min-max (max_m f_m), smooth Kreisselmeier-Steinhauser (KS)
  min-max aggregation, and weighted sum modes.
- Exact chain rule sensitivity backpropagation to rho_tilde and unfiltered rho.
- Dynamic volume fraction scaling on the dilated field V_d* = (V_d / V_i) * V_i*
  to ensure the intermediate blueprint design converges to target volume fraction V_i*.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from wht_topopt.functions.base import ObjectiveFunction
from wht_topopt.manufacturing.minimum_length import ThreeFieldProjection


class RobustFormulation(ObjectiveFunction):
    """3-Field Robust Topology Optimization formulation (Wang, Lazarov & Sigmund 2011)."""

    name: str = "robust_formulation"

    def __init__(
        self,
        objective: ObjectiveFunction,
        projection: Optional[ThreeFieldProjection] = None,
        mode: str = "worst_case",
        weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ks_param: float = 16.0,
        filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        filter_gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_volfrac_intermediate: float = 0.5,
        elem_volumes: Optional[np.ndarray] = None,
        fe_solver: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> None:
        """Initialize RobustFormulation wrapper.

        Parameters
        ----------
        objective : ObjectiveFunction
            Base objective function (e.g., ComplianceFunction).
        projection : Optional[ThreeFieldProjection]
            ThreeFieldProjection instance. Defaults to beta=8.0, eta_i=0.5, delta_eta=0.25.
        mode : str
            Aggregation mode:
            - 'worst_case': max(f_e, f_i, f_d) with subgradient / argmax sensitivity.
            - 'ks': Smooth Kreisselmeier-Steinhauser aggregation.
            - 'weighted_sum': w_e * f_e + w_i * f_i + w_d * f_d.
        weights : Tuple[float, float, float]
            Weights (w_e, w_i, w_d) used for weighted_sum mode.
        ks_param : float
            Smoothness parameter rho_ks for KS aggregation (> 0).
        filter_fn : Optional[Callable]
            Forward filter mapping design variable rho to filtered field rho_tilde.
        filter_gradient_fn : Optional[Callable]
            Adjoint filter mapping sensitivity dJ/d(rho_tilde) back to dJ/d(rho).
        target_volfrac_intermediate : float
            Target volume fraction V_i* for intermediate blueprint design.
        elem_volumes : Optional[np.ndarray]
            Element volumes v_e (N_elem,). Defaults to uniform unit volumes.
        fe_solver : Optional[Callable[[np.ndarray], Any]]
            Optional FE solution callback fe_solver(rho_hat) returning displacement u.
        """
        self.objective: ObjectiveFunction = objective
        self.projection: ThreeFieldProjection = (
            projection if projection is not None else ThreeFieldProjection()
        )
        self.mode: str = mode.lower()
        if self.mode not in ("worst_case", "ks", "weighted_sum"):
            raise ValueError(f"Unknown mode '{mode}'. Must be 'worst_case', 'ks', or 'weighted_sum'")

        self.weights: Tuple[float, float, float] = tuple(float(w) for w in weights)  # type: ignore
        if self.mode == "weighted_sum" and any(w < 0.0 for w in self.weights):
            raise ValueError(f"Weights must be non-negative, got {self.weights}")

        self.ks_param: float = float(ks_param)
        if self.ks_param <= 0.0:
            raise ValueError(f"ks_param must be positive, got {self.ks_param}")

        self.filter_fn = filter_fn
        self.filter_gradient_fn = filter_gradient_fn
        self.fe_solver = fe_solver

        self.target_volfrac_intermediate: float = float(target_volfrac_intermediate)
        self.dilated_volfrac_target: float = self.target_volfrac_intermediate
        self.elem_volumes: Optional[np.ndarray] = (
            np.asarray(elem_volumes, dtype=np.float64) if elem_volumes is not None else None
        )

        # Cached state from last forward evaluation
        self._last_rho_tilde: Optional[np.ndarray] = None
        self._last_rho_e: Optional[np.ndarray] = None
        self._last_rho_i: Optional[np.ndarray] = None
        self._last_rho_d: Optional[np.ndarray] = None
        self._last_f_e: Optional[float] = None
        self._last_f_i: Optional[float] = None
        self._last_f_d: Optional[float] = None
        self._last_weights: Optional[np.ndarray] = None
        self._last_extra_e: Dict[str, Any] = {}
        self._last_extra_i: Dict[str, Any] = {}
        self._last_extra_d: Dict[str, Any] = {}

    def get_three_fields(
        self, rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute filtered field rho_tilde and projected fields (rho_e, rho_i, rho_d).

        Parameters
        ----------
        rho : np.ndarray
            Design variable array of shape (N_elem,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (rho_tilde, rho_e, rho_i, rho_d).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        if self.filter_fn is not None:
            rho_tilde = self.filter_fn(rho_arr)
        else:
            rho_tilde = rho_arr

        rho_e, rho_i, rho_d = self.projection.project(rho_tilde)
        return rho_tilde, rho_e, rho_i, rho_d

    def _resolve_kwargs_per_field(
        self,
        rho_e: np.ndarray,
        rho_i: np.ndarray,
        rho_d: np.ndarray,
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Resolve FE displacements or other field-specific kwargs."""
        kw_e = dict(kwargs)
        kw_i = dict(kwargs)
        kw_d = dict(kwargs)

        if self.fe_solver is not None:
            # If FE solver is attached, solve for each field
            kw_e["u"] = self.fe_solver(rho_e)
            kw_i["u"] = self.fe_solver(rho_i)
            kw_d["u"] = self.fe_solver(rho_d)
        else:
            # Check if kwargs explicitly provided per-field inputs
            if "u_e" in kwargs:
                kw_e["u"] = kwargs["u_e"]
            if "u_i" in kwargs:
                kw_i["u"] = kwargs["u_i"]
            if "u_d" in kwargs:
                kw_d["u"] = kwargs["u_d"]

        return kw_e, kw_i, kw_d

    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate robust objective function.

        Parameters
        ----------
        rho : np.ndarray
            Design variable array of shape (N_elem,).
        **kwargs : Any
            Additional solver / evaluation arguments.

        Returns
        -------
        float
            Aggregated objective value.
        """
        rho_tilde, rho_e, rho_i, rho_d = self.get_three_fields(rho)
        kw_e, kw_i, kw_d = self._resolve_kwargs_per_field(rho_e, rho_i, rho_d, kwargs)

        f_e = float(self.objective.evaluate(rho_e, **kw_e))
        f_i = float(self.objective.evaluate(rho_i, **kw_i))
        f_d = float(self.objective.evaluate(rho_d, **kw_d))

        self._last_rho_tilde = rho_tilde
        self._last_rho_e = rho_e
        self._last_rho_i = rho_i
        self._last_rho_d = rho_d
        self._last_f_e = f_e
        self._last_f_i = f_i
        self._last_f_d = f_d
        self._last_extra_e = kw_e
        self._last_extra_i = kw_i
        self._last_extra_d = kw_d

        f_vals = np.array([f_e, f_i, f_d], dtype=np.float64)

        if self.mode == "worst_case":
            idx_max = np.argmax(f_vals)
            f_val = float(f_vals[idx_max])
            w = np.zeros(3, dtype=np.float64)
            w[idx_max] = 1.0
            self._last_weights = w
            return f_val

        elif self.mode == "ks":
            f_max = float(np.max(f_vals))
            shift = self.ks_param * (f_vals - f_max)
            # Clip shift to prevent overflow
            shift = np.clip(shift, -60.0, 0.0)
            exp_shift = np.exp(shift)
            sum_exp = float(np.sum(exp_shift))

            f_ks = f_max + (1.0 / self.ks_param) * np.log(sum_exp)
            self._last_weights = exp_shift / sum_exp
            return float(f_ks)

        elif self.mode == "weighted_sum":
            w = np.array(self.weights, dtype=np.float64)
            self._last_weights = w
            f_val = float(np.dot(w, f_vals))
            return f_val

        raise RuntimeError(f"Unhandled mode: {self.mode}")

    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate exact analytical sensitivity d(f_robust)/d(rho).

        Parameters
        ----------
        rho : np.ndarray
            Design variable array of shape (N_elem,).
        **kwargs : Any
            Additional solver / evaluation arguments.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (N_elem,).
        """
        # Ensure forward evaluation is up to date
        rho_arr = np.asarray(rho, dtype=np.float64)
        if (
            self._last_rho_tilde is None
            or self._last_weights is None
            or not np.allclose(self._last_rho_tilde, self.get_three_fields(rho_arr)[0], atol=1e-14)
        ):
            self.evaluate(rho_arr, **kwargs)

        assert self._last_weights is not None
        assert self._last_rho_e is not None
        assert self._last_rho_i is not None
        assert self._last_rho_d is not None
        assert self._last_rho_tilde is not None

        w_e, w_i, w_d = self._last_weights

        # Compute sensitivities for active projection fields
        df_drho_e = (
            self.objective.gradient(self._last_rho_e, **self._last_extra_e)
            if w_e > 1e-12
            else None
        )
        df_drho_i = (
            self.objective.gradient(self._last_rho_i, **self._last_extra_i)
            if w_i > 1e-12
            else None
        )
        df_drho_d = (
            self.objective.gradient(self._last_rho_d, **self._last_extra_d)
            if w_d > 1e-12
            else None
        )

        grad_e = w_e * df_drho_e if df_drho_e is not None else None
        grad_i = w_i * df_drho_i if df_drho_i is not None else None
        grad_d = w_d * df_drho_d if df_drho_d is not None else None

        # Adjoint vector-Jacobian product through ThreeFieldProjection
        dJ_drho_tilde = self.projection.backward(
            df_drho_e=grad_e,
            df_drho_i=grad_i,
            df_drho_d=grad_d,
            rho_tilde=self._last_rho_tilde,
        )

        # Chain rule back to unfiltered design variables rho
        if self.filter_gradient_fn is not None:
            dJ_drho = self.filter_gradient_fn(dJ_drho_tilde)
        else:
            dJ_drho = dJ_drho_tilde

        return dJ_drho

    def update_dilated_volume_target(
        self,
        rho: Optional[np.ndarray] = None,
        alpha: float = 1.0,
    ) -> float:
        """Update dynamic dilated volume fraction target V_d* = (V_d / V_i) * V_i*.

        Parameters
        ----------
        rho : Optional[np.ndarray]
            Current design variables. If None, uses cached projection fields.
        alpha : float
            Relaxation factor in (0, 1] for updating V_d* (1.0 = full update).

        Returns
        -------
        float
            Updated dilated volume target V_d*.
        """
        if rho is not None:
            _, rho_e, rho_i, rho_d = self.get_three_fields(rho)
        else:
            if self._last_rho_i is None or self._last_rho_d is None:
                raise RuntimeError("Must call evaluate() or pass rho before update_dilated_volume_target")
            rho_i = self._last_rho_i
            rho_d = self._last_rho_d

        v = (
            self.elem_volumes
            if self.elem_volumes is not None
            else np.ones_like(rho_i, dtype=np.float64)
        )
        total_vol = float(np.sum(v))

        V_i = float(np.sum(v * rho_i) / total_vol)
        V_d = float(np.sum(v * rho_d) / total_vol)

        ratio = V_d / max(V_i, 1e-12)
        new_target = float(ratio * self.target_volfrac_intermediate)

        # Relaxation update
        self.dilated_volfrac_target = float(
            (1.0 - alpha) * self.dilated_volfrac_target + alpha * new_target
        )
        return self.dilated_volfrac_target

    def dilated_volume_constraint(
        self, rho: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Evaluate dilated volume fraction constraint g_d(rho) = V(rho_d) / V_0 - V_d* <= 0.

        Parameters
        ----------
        rho : np.ndarray
            Design variable array of shape (N_elem,).

        Returns
        -------
        Tuple[float, np.ndarray]
            (constraint_value, constraint_gradient).
        """
        rho_tilde, _, _, rho_d = self.get_three_fields(rho)
        v = (
            self.elem_volumes
            if self.elem_volumes is not None
            else np.ones_like(rho_d, dtype=np.float64)
        )
        total_vol = float(np.sum(v))

        V_d = float(np.sum(v * rho_d) / total_vol)
        g_val = float(V_d - self.dilated_volfrac_target)

        # Gradient of V_d with respect to rho_d is v / total_vol
        dg_drho_d = v / total_vol

        # Backpropagate through dilated projection
        dg_drho_tilde = self.projection.backward(
            df_drho_e=None,
            df_drho_i=None,
            df_drho_d=dg_drho_d,
            rho_tilde=rho_tilde,
        )

        if self.filter_gradient_fn is not None:
            dg_drho = self.filter_gradient_fn(dg_drho_tilde)
        else:
            dg_drho = dg_drho_tilde

        return g_val, dg_drho
