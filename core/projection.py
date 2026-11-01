"""Heaviside projection and volume-preserving threshold optimization.

Implements smooth Heaviside projection with exact reverse-mode derivatives
and safeguarded Newton-bisection root finder for volume preservation.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class HeavisideProjection:
    """Smooth Heaviside projection for crisp 0/1 topologies (Wang, Lazarov & Sigmund 2011).

    Forward:
        rho_hat = (tanh(beta * eta) + tanh(beta * (rho_tilde - eta))) /
                  (tanh(beta * eta) + tanh(beta * (1 - eta)))

    Derivative:
        d(rho_hat)/d(rho_tilde) = beta * (1 - tanh^2(beta * (rho_tilde - eta))) /
                                  (tanh(beta * eta) + tanh(beta * (1 - eta)))
    """

    def __init__(self, beta: float = 1.0, eta: float = 0.5) -> None:
        """Initialize Heaviside projection.

        Parameters
        ----------
        beta : float
            Sharpness parameter beta >= 0.0. When beta <= 1e-12, acts as identity.
        eta : float
            Threshold parameter 0.0 < eta < 1.0 (default 0.5).
        """
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if not (0.0 < eta < 1.0):
            raise ValueError(f"eta must be strictly in (0, 1), got {eta}")

        self.beta: float = float(beta)
        self.eta: float = float(eta)

    def forward(self, rho_tilde: np.ndarray, eta: Optional[float] = None) -> np.ndarray:
        """Project filtered field: rho_tilde -> rho_hat."""
        rho = np.asarray(rho_tilde, dtype=np.float64)
        if self.beta <= 1e-12:
            return rho.copy()

        e = self.eta if eta is None else float(eta)
        b = self.beta

        # Numerator: tanh(b*e) + tanh(b*(rho - e))
        # Denominator: tanh(b*e) + tanh(b*(1 - e))
        t_be = np.tanh(b * e)
        t_b1e = np.tanh(b * (1.0 - e))
        denom = t_be + t_b1e
        if denom == 0.0:
            return rho.copy()

        t_diff = np.tanh(b * (rho - e))
        rho_hat = (t_be + t_diff) / denom
        return np.clip(rho_hat, 0.0, 1.0)

    def derivative(self, rho_tilde: np.ndarray, eta: Optional[float] = None) -> np.ndarray:
        """Exact derivative d(rho_hat)/d(rho_tilde)."""
        rho = np.asarray(rho_tilde, dtype=np.float64)
        if self.beta <= 1e-12:
            return np.ones_like(rho)

        e = self.eta if eta is None else float(eta)
        b = self.beta

        t_be = np.tanh(b * e)
        t_b1e = np.tanh(b * (1.0 - e))
        denom = t_be + t_b1e
        if denom == 0.0:
            return np.ones_like(rho)

        t_diff = np.tanh(b * (rho - e))
        d_hat = b * (1.0 - t_diff**2) / denom
        return d_hat

    def backward(
        self,
        df_drho_hat: np.ndarray,
        rho_tilde: np.ndarray,
        eta: Optional[float] = None,
    ) -> np.ndarray:
        """Reverse-mode vector-Jacobian product: df/drho_tilde = df/drho_hat * d(rho_hat)/d(rho_tilde)."""
        return np.asarray(df_drho_hat, dtype=np.float64) * self.derivative(rho_tilde, eta=eta)

    def derivative_eta(self, rho_tilde: np.ndarray, eta: float) -> np.ndarray:
        """Exact derivative d(rho_hat)/d(eta) for Newton search."""
        rho = np.asarray(rho_tilde, dtype=np.float64)
        if self.beta <= 1e-12:
            return np.zeros_like(rho)

        b = self.beta
        e = float(eta)

        t_be = np.tanh(b * e)
        t_b1e = np.tanh(b * (1.0 - e))
        w = t_be + t_b1e
        if abs(w) < 1e-14:
            return np.zeros_like(rho)

        t_diff = np.tanh(b * (rho - e))
        u = t_be + t_diff
        rho_hat = u / w

        # sech^2(x) = 1 - tanh^2(x)
        sech2_be = 1.0 - t_be**2
        sech2_b1e = 1.0 - t_b1e**2
        sech2_diff = 1.0 - t_diff**2

        # u'(eta) = b * (sech^2(b*e) - sech^2(b*(rho - e)))
        # w'(eta) = b * (sech^2(b*e) - sech^2(b*(1 - e)))
        u_prime = b * (sech2_be - sech2_diff)
        w_prime = b * (sech2_be - sech2_b1e)

        d_eta = (u_prime - rho_hat * w_prime) / w
        return d_eta

    def find_volume_preserving_eta(
        self,
        rho_tilde: np.ndarray,
        target_vol: float,
        elem_volumes: Optional[np.ndarray] = None,
        tol: float = 1e-10,
        max_iter: int = 50,
    ) -> float:
        """Find threshold eta* such that V(rho_hat(eta*)) == target_vol.

        Uses safeguarded Newton-Raphson bisection root finding (Ferrari & Sigmund 2020).

        Parameters
        ----------
        rho_tilde : np.ndarray
            Intermediate filtered density array.
        target_vol : float
            Desired target volume fraction in (0, 1).
        elem_volumes : np.ndarray, optional
            Element volume array (if None, uniform weighting is used).
        tol : float
            Convergence tolerance for volume residual |V - target_vol|.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        eta_star : float
            Optimized threshold parameter in (0, 1).
        """
        if self.beta <= 1e-12:
            return self.eta

        rho = np.asarray(rho_tilde, dtype=np.float64)
        if elem_volumes is None:
            v = np.ones_like(rho)
        else:
            v = np.asarray(elem_volumes, dtype=np.float64)
        total_vol = float(np.sum(v))

        # Root search for g(eta) = (sum(v * rho_hat(eta)) / total_vol) - target_vol = 0
        eta_low = 1e-6
        eta_high = 1.0 - 1e-6
        eta = float(np.clip(self.eta, 0.05, 0.95))

        for _ in range(max_iter):
            rho_hat = self.forward(rho, eta=eta)
            current_vol = float(np.sum(v * rho_hat)) / total_vol
            res = current_vol - target_vol

            if abs(res) < tol:
                break

            # Update bracketing interval
            # Note: Increasing eta decreases rho_hat (decreases current_vol)
            if res > 0:
                # Volume too high -> need higher eta
                eta_low = max(eta_low, eta)
            else:
                # Volume too low -> need lower eta
                eta_high = min(eta_high, eta)

            # Compute derivative dg/deta
            d_hat_d_eta = self.derivative_eta(rho, eta=eta)
            dg_d_eta = float(np.sum(v * d_hat_d_eta)) / total_vol

            # Safeguarded Newton step
            step_taken = False
            if dg_d_eta < -1e-12:
                eta_newton = eta - res / dg_d_eta
                if eta_low < eta_newton < eta_high:
                    eta = eta_newton
                    step_taken = True

            if not step_taken:
                # Bisection fallback
                eta = 0.5 * (eta_low + eta_high)

            if abs(eta_high - eta_low) < tol:
                break

        self.eta = eta
        return eta
