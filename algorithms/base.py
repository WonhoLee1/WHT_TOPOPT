"""Base interfaces and data structures for optimization algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class GradientSource(str, Enum):
    """Source of objective and constraint gradient information."""
    EXACT_ADJOINT = "exact_adjoint"
    AUTODIFF = "autodiff"
    SURROGATE = "surrogate"
    LEARNED = "learned"
    FINITE_DIFFERENCE = "finite_difference"
    NONE = "none"


@dataclass
class OptState:
    """Snapshot of the optimization state at a given design iteration.
    
    Attributes:
        iteration: Optimization iteration counter.
        rho: Current mathematical design variable array (N_elem,).
        rho_tilde: Filtered design density array (N_elem,).
        rho_hat: Projected physical density array after Heaviside projection (N_elem,).
        obj: Objective function value.
        constraints: Array of evaluated constraint function values g(x) <= 0.
        change: Maximum change in design variables ||rho - rho_prev||_inf.
        grayness: Non-discreteness / grayness index in percent [0, 100].
        extra: Additional diagnostic values (e.g. step sizes, multiplier history).
    """
    iteration: int
    rho: np.ndarray
    rho_tilde: Optional[np.ndarray] = None
    rho_hat: Optional[np.ndarray] = None
    rho_phys: Optional[np.ndarray] = None
    obj: float = 0.0
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    g: Optional[np.ndarray] = None
    vol: float = 0.0
    change: float = 1.0
    penal: float = 3.0
    beta: float = 1.0
    grayness: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=float)
        if self.rho_phys is not None and self.rho_hat is None:
            self.rho_hat = np.asarray(self.rho_phys, dtype=float)
        if self.rho_hat is not None:
            self.rho_hat = np.asarray(self.rho_hat, dtype=float)
            if self.rho_phys is None:
                self.rho_phys = self.rho_hat
            if self.grayness == 0.0 and len(self.rho_hat) > 0:
                # Sigmund non-discreteness measure: 4 * sum(x * (1 - x)) / N * 100%
                self.grayness = float(4.0 * np.mean(self.rho_hat * (1.0 - self.rho_hat)) * 100.0)
        if self.g is not None and len(self.constraints) == 0:
            self.constraints = np.asarray(self.g, dtype=float)
        elif self.g is None:
            self.g = np.asarray(self.constraints, dtype=float)


class TrustRegionController:
    """Trust-region controller dynamically adapting move limits for surrogate or learned gradients.
    
    rho_TR = (J_prev - J_actual) / (J_prev - J_predicted)
    """

    def __init__(
        self,
        initial_move: float = 0.2,
        min_move: float = 0.01,
        max_move: float = 0.35,
        shrink_factor: float = 0.5,
        expand_factor: float = 1.2,
        accept_threshold: float = 0.1,
        reject_non_improving: bool = True,
    ) -> None:
        self.move = initial_move
        self.min_move = min_move
        self.max_move = max_move
        self.shrink_factor = shrink_factor
        self.expand_factor = expand_factor
        self.accept_threshold = accept_threshold
        self.reject_non_improving = reject_non_improving

    def evaluate_step(
        self,
        actual_reduction: float,
        predicted_reduction: float,
        gradient_source: GradientSource = GradientSource.SURROGATE,
    ) -> Tuple[bool, float]:
        """Evaluate step quality and adjust move limit.
        
        Args:
            actual_reduction: J_prev - J_current (positive indicates true improvement).
            predicted_reduction: Predicted objective reduction from surrogate/model.
            gradient_source: Origin of gradient information.
            
        Returns:
            (step_accepted, new_move_limit)
        """
        if gradient_source in (GradientSource.EXACT_ADJOINT, GradientSource.AUTODIFF):
            # Exact gradients do not require trust-region shrinking unless explicitly desired
            return True, self.move

        if abs(predicted_reduction) <= 1e-15:
            rho_tr = 1.0 if actual_reduction >= 0.0 else -1.0
        else:
            rho_tr = actual_reduction / predicted_reduction

        accepted = True
        if self.reject_non_improving and actual_reduction < 0.0:
            accepted = False
            self.move = max(self.min_move, self.move * self.shrink_factor)
        elif rho_tr < 0.25:
            self.move = max(self.min_move, self.move * self.shrink_factor)
            accepted = (rho_tr >= self.accept_threshold)
        elif rho_tr > 0.75:
            self.move = min(self.max_move, self.move * self.expand_factor)
            accepted = True

        return accepted, self.move


class Optimizer(ABC):
    """Abstract base class for topology mathematical optimization solvers."""

    @abstractmethod
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
        """Compute updated design variables rho^(k+1).
        
        Args:
            rho: Current design variables array (N_elem,).
            df_drho: Objective sensitivity vector (N_elem,).
            g_vals: Array of constraint values (N_constraints,).
            dg_drho: Constraint sensitivity matrix (N_constraints, N_elem) or (N_elem,).
            elem_volumes: Element volume array v_e of shape (N_elem,).
            passive_solid: Boolean mask or indices of elements locked to solid (rho=1).
            passive_void: Boolean mask or indices of elements locked to void (rho=rho_min).
            
        Returns:
            Updated design variables array rho^(k+1) of shape (N_elem,).
        """
        pass
