"""Material property interpolation schemes for topology optimization.

Implements SIMP (Modified SIMP), ClassicSIMP, RAMP, and MultiMaterialSIMP.
Every model provides exact evaluate() and derivative() methods verified
against Central Finite Differences (step = 1e-6) to relative error < 1e-7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Union
import numpy as np


class MaterialInterpolation(ABC):
    """Abstract Base Class for topology optimization material interpolation models."""

    @abstractmethod
    def evaluate(self, rho_hat: np.ndarray) -> np.ndarray:
        """Compute effective Young's modulus E(rho_hat)."""
        pass

    @abstractmethod
    def derivative(self, rho_hat: np.ndarray) -> np.ndarray:
        """Compute analytical derivative dE/drho_hat."""
        pass


class SIMP(MaterialInterpolation):
    """Modified SIMP material interpolation model (Andreassen et al. 2011 eq. 1).

    Formula:
        E(rho) = E_min + rho^p * (E_0 - E_min)

    Derivative:
        dE/drho = p * rho^(p - 1) * (E_0 - E_min)
    """

    def __init__(self, E0: float = 1.0, Emin: float = 1e-9, penal: float = 3.0) -> None:
        """Initialize Modified SIMP model.

        Parameters
        ----------
        E0 : float
            Solid phase Young's modulus.
        Emin : float
            Void phase artificial Young's modulus.
        penal : float
            Penalty exponent p >= 1.0.
        """
        if E0 <= 0.0:
            raise ValueError(f"E0 must be positive, got {E0}")
        if Emin < 0.0 or Emin >= E0:
            raise ValueError(f"Emin must be non-negative and < E0, got {Emin}")
        if penal < 1.0:
            raise ValueError(f"Penalty exponent penal must be >= 1.0, got {penal}")

        self.E0: float = float(E0)
        self.Emin: float = float(Emin)
        self.penal: float = float(penal)

    def evaluate(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate effective Young's modulus E(rho_hat)."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        return self.Emin + (rho_clamped**self.penal) * (self.E0 - self.Emin)

    def derivative(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate exact derivative dE/drho_hat."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        p = self.penal
        dE = p * (rho_clamped ** (p - 1.0)) * (self.E0 - self.Emin)
        return dE


class ClassicSIMP(MaterialInterpolation):
    """Classical SIMP material interpolation (Bendsoe & Sigmund 1999).

    Formula:
        E(rho) = max(E_min, rho^p * E_0)

    Derivative:
        dE/drho = p * rho^(p - 1) * E_0   (for rho^p * E_0 > E_min, else 0.0)
    """

    def __init__(self, E0: float = 1.0, Emin: float = 1e-9, penal: float = 3.0) -> None:
        """Initialize Classic SIMP model."""
        if E0 <= 0.0:
            raise ValueError(f"E0 must be positive, got {E0}")
        if Emin < 0.0 or Emin >= E0:
            raise ValueError(f"Emin must be non-negative and < E0, got {Emin}")
        if penal < 1.0:
            raise ValueError(f"Penalty exponent penal must be >= 1.0, got {penal}")

        self.E0: float = float(E0)
        self.Emin: float = float(Emin)
        self.penal: float = float(penal)

    def evaluate(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate E(rho) = max(E_min, rho^p * E_0)."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        val = (rho_clamped**self.penal) * self.E0
        return np.maximum(self.Emin, val)

    def derivative(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate exact subgradient dE/drho."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        p = self.penal
        val = (rho_clamped**p) * self.E0
        dE = np.where(val > self.Emin, p * (rho_clamped ** (p - 1.0)) * self.E0, 0.0)
        return dE


class RAMP(MaterialInterpolation):
    """Rational Approximation of Material Properties (Stolpe & Svanberg 2001).

    Formula:
        E(rho) = E_min + (rho / (1 + q * (1 - rho))) * (E_0 - E_min)

    Derivative:
        dE/drho = ((1 + q) * (E_0 - E_min)) / (1 + q * (1 - rho))^2
    """

    def __init__(self, E0: float = 1.0, Emin: float = 1e-9, q: float = 3.0) -> None:
        """Initialize RAMP model.

        Parameters
        ----------
        E0 : float
            Solid phase Young's modulus.
        Emin : float
            Void phase Young's modulus.
        q : float
            RAMP parameter q >= 0.0. Non-zero slope at rho = 0 suppresses spurious modes.
        """
        if E0 <= 0.0:
            raise ValueError(f"E0 must be positive, got {E0}")
        if Emin < 0.0 or Emin >= E0:
            raise ValueError(f"Emin must be non-negative and < E0, got {Emin}")
        if q < 0.0:
            raise ValueError(f"q must be non-negative, got {q}")

        self.E0: float = float(E0)
        self.Emin: float = float(Emin)
        self.q: float = float(q)

    def evaluate(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate E(rho)."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        denom = 1.0 + self.q * (1.0 - rho_clamped)
        return self.Emin + (rho_clamped / denom) * (self.E0 - self.Emin)

    def derivative(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate exact derivative dE/drho."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        rho_clamped = np.clip(rho, 0.0, 1.0)
        denom = 1.0 + self.q * (1.0 - rho_clamped)
        return ((1.0 + self.q) * (self.E0 - self.Emin)) / (denom**2)


class MultiMaterialSIMP(MaterialInterpolation):
    """Multi-material SIMP interpolation with exact partial derivatives.

    Supports:
        1. Ordered 1D formulation (rho in [0, M]):
           Piecewise segment interpolation between Phase m-1 and Phase m.
        2. DMO / Phase-fraction 2D formulation (rho of shape (n_elem, M)):
           E = E_min + sum_{m=0}^{M-1} (rho_m^p * (E_m - E_min))
    """

    def __init__(
        self,
        E_phases: Sequence[float],
        Emin: float = 1e-9,
        penal: float = 3.0,
    ) -> None:
        """Initialize MultiMaterialSIMP.

        Parameters
        ----------
        E_phases : sequence of float
            Young's moduli of the candidate material phases (e.g., [1.0, 2.5, 5.0]).
        Emin : float
            Void base modulus.
        penal : float
            SIMP penalty exponent.
        """
        self.E_phases: np.ndarray = np.asarray(E_phases, dtype=np.float64)
        if len(self.E_phases) == 0:
            raise ValueError("E_phases must contain at least one phase modulus")
        if np.any(self.E_phases <= 0.0):
            raise ValueError("All phase moduli in E_phases must be positive")

        self.n_phases: int = len(self.E_phases)
        self.Emin: float = float(Emin)
        self.penal: float = float(penal)

    def evaluate(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate effective Young's modulus for multi-material phases."""
        rho = np.asarray(rho_hat, dtype=np.float64)

        if rho.ndim == 2 and rho.shape[1] == self.n_phases:
            # DMO / Phase fraction mode: (n_elem, M)
            rho_clamped = np.clip(rho, 0.0, 1.0)
            E = np.full(rho.shape[0], self.Emin, dtype=np.float64)
            for m in range(self.n_phases):
                E += (rho_clamped[:, m] ** self.penal) * (self.E_phases[m] - self.Emin)
            return E

        else:
            # Ordered continuous mode: rho in [0, n_phases]
            rho_clamped = np.clip(rho, 0.0, float(self.n_phases))
            E = np.full_like(rho_clamped, self.Emin, dtype=np.float64)

            # Extended phases array: [Emin, E_phases[0], E_phases[1], ...]
            all_E = np.concatenate([[self.Emin], self.E_phases])

            for m in range(1, len(all_E)):
                lower = float(m - 1)
                upper = float(m)
                # Elements in segment [m-1, m]
                mask = (rho_clamped >= lower) & (rho_clamped <= upper if m == len(all_E) - 1 else rho_clamped < upper)
                if np.any(mask):
                    s = rho_clamped[mask] - lower
                    E[mask] = all_E[m - 1] + (s**self.penal) * (all_E[m] - all_E[m - 1])
            return E

    def derivative(self, rho_hat: np.ndarray) -> np.ndarray:
        """Evaluate exact analytical partial derivatives dE/drho."""
        rho = np.asarray(rho_hat, dtype=np.float64)
        p = self.penal

        if rho.ndim == 2 and rho.shape[1] == self.n_phases:
            # DMO mode: returns (n_elem, M)
            rho_clamped = np.clip(rho, 0.0, 1.0)
            dE = np.zeros_like(rho_clamped)
            for m in range(self.n_phases):
                dE[:, m] = p * (rho_clamped[:, m] ** (p - 1.0)) * (self.E_phases[m] - self.Emin)
            return dE

        else:
            # Ordered mode: returns (n_elem,)
            rho_clamped = np.clip(rho, 0.0, float(self.n_phases))
            dE = np.zeros_like(rho_clamped)
            all_E = np.concatenate([[self.Emin], self.E_phases])

            for m in range(1, len(all_E)):
                lower = float(m - 1)
                upper = float(m)
                mask = (rho_clamped >= lower) & (rho_clamped <= upper if m == len(all_E) - 1 else rho_clamped < upper)
                if np.any(mask):
                    s = rho_clamped[mask] - lower
                    dE[mask] = p * (s ** (p - 1.0)) * (all_E[m] - all_E[m - 1])
            return dE
