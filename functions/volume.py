"""Volume and mass objective and constraint functions."""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from .base import ConstraintFunction, ObjectiveFunction
from wht_topopt.algorithms.base import GradientSource


class VolumeFractionFunction(ConstraintFunction):
    """Volume fraction inequality constraint function g(rho) <= 0.
    
    Forms:
    1. Normalized ratio form (normalize=True):
        g(rho) = (sum_e v_e * rho_e) / (V_0 * V_frac*) - 1 <= 0
        dg / drho_e = v_e / (V_0 * V_frac*)
        
    2. Direct difference form (normalize=False):
        g(rho) = (sum_e v_e * rho_e) / V_0 - V_frac* <= 0
        dg / drho_e = v_e / V_0
    """

    name: str = "volume_fraction"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        target_volfrac: float = 0.5,
        elem_volumes: Optional[np.ndarray] = None,
        normalize: bool = False,
        mesh: Optional[Any] = None,
        volfrac: Optional[float] = None,
    ) -> None:
        """Initialize VolumeFractionFunction.
        
        Args:
            target_volfrac: Target volume fraction V_frac* in (0, 1].
            elem_volumes: Optional array of element volumes v_e (N_elem,).
            normalize: If True, uses (V / V_target - 1 <= 0); if False, uses (V / V0 - V_target <= 0).
            mesh: Optional mesh from which elem_volumes are extracted if elem_volumes is None.
            volfrac: Optional alias for target_volfrac.
        """
        vf = float(volfrac) if volfrac is not None else float(target_volfrac)
        if not (0.0 < vf <= 1.0):
            raise ValueError(f"target_volfrac must be in (0, 1], got {vf}")

        self.target_volfrac = vf
        if elem_volumes is not None:
            self.elem_volumes: Optional[np.ndarray] = np.asarray(elem_volumes, dtype=float)
        elif mesh is not None and hasattr(mesh, "elem_volumes"):
            self.elem_volumes = np.asarray(mesh.elem_volumes, dtype=float)
        else:
            self.elem_volumes = None
        self.normalize = bool(normalize)

    def _resolve_volumes(self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None) -> np.ndarray:
        if elem_volumes is not None:
            return np.asarray(elem_volumes, dtype=float)
        elif self.elem_volumes is not None:
            return self.elem_volumes
        else:
            return np.ones(len(rho), dtype=float)

    def evaluate(self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None, **kwargs: Any) -> float:
        """Evaluate constraint g(rho) <= 0."""
        rho_arr = np.asarray(rho, dtype=float)
        v_e = self._resolve_volumes(rho_arr, elem_volumes)
        v_total = float(np.sum(v_e))
        current_vol = float(np.sum(rho_arr * v_e))

        current_frac = current_vol / v_total

        if self.normalize:
            # (V / (V0 * V_target)) - 1 <= 0
            return (current_vol / (v_total * self.target_volfrac)) - 1.0
        else:
            # V / V0 - V_target <= 0
            return current_frac - self.target_volfrac

    def gradient(self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
        """Evaluate exact gradient dg/drho."""
        rho_arr = np.asarray(rho, dtype=float)
        v_e = self._resolve_volumes(rho_arr, elem_volumes)
        v_total = float(np.sum(v_e))

        if self.normalize:
            # dg / drho_e = v_e / (V0 * V_target)
            return v_e / (v_total * self.target_volfrac)
        else:
            # dg / drho_e = v_e / V0
            return v_e / v_total


VolumeConstraint = VolumeFractionFunction


class MassFunction(ObjectiveFunction):
    """Total structural mass objective function:
    
        M(rho) = sum_e rho_mat,e * v_e * rho_e
        dM / drho_e = rho_mat,e * v_e
    """

    name: str = "mass"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        elem_volumes: Optional[np.ndarray] = None,
        base_density: Union[float, np.ndarray] = 1.0,
    ) -> None:
        """Initialize MassFunction.
        
        Args:
            elem_volumes: Element volume array v_e of shape (N_elem,).
            base_density: Solid material mass density rho_mat (scalar or N_elem array).
        """
        self.elem_volumes = np.asarray(elem_volumes, dtype=float) if elem_volumes is not None else None
        self.base_density = base_density

    def _resolve_properties(
        self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rho_arr = np.asarray(rho, dtype=float)
        n = len(rho_arr)

        if elem_volumes is not None:
            v_e = np.asarray(elem_volumes, dtype=float)
        elif self.elem_volumes is not None:
            v_e = self.elem_volumes
        else:
            v_e = np.ones(n, dtype=float)

        if isinstance(self.base_density, (int, float)):
            rho_mat = np.full(n, float(self.base_density))
        else:
            rho_mat = np.asarray(self.base_density, dtype=float)

        return v_e, rho_mat

    def evaluate(self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None, **kwargs: Any) -> float:
        """Evaluate total mass M = sum_e rho_mat,e * v_e * rho_e."""
        rho_arr = np.asarray(rho, dtype=float)
        v_e, rho_mat = self._resolve_properties(rho_arr, elem_volumes)
        return float(np.sum(rho_mat * v_e * rho_arr))

    def gradient(self, rho: np.ndarray, elem_volumes: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
        """Evaluate exact mass gradient dM/drho_e = rho_mat,e * v_e."""
        rho_arr = np.asarray(rho, dtype=float)
        v_e, rho_mat = self._resolve_properties(rho_arr, elem_volumes)
        return rho_mat * v_e
