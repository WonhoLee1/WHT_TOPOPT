"""Base interfaces for objective and constraint functions in topology optimization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from wht_topopt.algorithms.base import GradientSource


class ObjectiveFunction(ABC):
    """Abstract base class for all topology optimization objective functions."""

    name: str = "objective"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    @abstractmethod
    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the scalar objective function value f(rho).
        
        Args:
            rho: Physical density field array (N_elem,).
            **kwargs: Additional evaluation inputs (e.g. displacements, FE matrices).
            
        Returns:
            Scalar objective value.
        """
        pass

    @abstractmethod
    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate the objective sensitivity vector df/drho of shape (N_elem,).
        
        Args:
            rho: Physical density field array (N_elem,).
            **kwargs: Additional evaluation inputs (e.g. displacements, FE matrices).
            
        Returns:
            Sensitivity array of shape (N_elem,).
        """
        pass

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> tuple[float, np.ndarray]:
        """Evaluate both scalar value and sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad


class ConstraintFunction(ABC):
    """Abstract base class for all inequality constraint functions g(rho) <= 0."""

    name: str = "constraint"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    @abstractmethod
    def evaluate(self, rho: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the scalar constraint violation g(rho) <= 0.
        
        Args:
            rho: Physical density field array (N_elem,).
            **kwargs: Additional evaluation inputs.
            
        Returns:
            Scalar constraint value g(rho). Value <= 0 means feasible.
        """
        pass

    @abstractmethod
    def gradient(self, rho: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate the constraint sensitivity vector dg/drho of shape (N_elem,).
        
        Args:
            rho: Physical density field array (N_elem,).
            **kwargs: Additional evaluation inputs.
            
        Returns:
            Sensitivity array of shape (N_elem,).
        """
        pass

    def is_satisfied(self, rho: np.ndarray, tol: float = 1e-4, **kwargs: Any) -> bool:
        """Check whether the constraint is satisfied within tolerance g(rho) <= tol."""
        val = self.evaluate(rho, **kwargs)
        return val <= tol

    def value_and_gradient(self, rho: np.ndarray, **kwargs: Any) -> tuple[float, np.ndarray]:
        """Evaluate both scalar constraint value and sensitivity vector."""
        val = self.evaluate(rho, **kwargs)
        grad = self.gradient(rho, **kwargs)
        return val, grad
