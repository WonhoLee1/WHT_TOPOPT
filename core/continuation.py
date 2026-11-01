"""Continuation parameter scheduler and non-discreteness metric.

Manages progressive updates for SIMP penalization (penal), Heaviside sharpness (beta),
RAMP nonlinearity (q), and OC move limit (move) across optimization iterations.
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np


class ContinuationScheduler:
    """Manages parameter continuation schedules across optimization loops."""

    def __init__(
        self,
        penal_init: float = 3.0,
        penal_max: float = 3.0,
        penal_interval: int = 10,
        penal_step: float = 0.5,
        beta_init: float = 1.0,
        beta_max: float = 64.0,
        beta_interval: int = 10,
        beta_factor: float = 2.0,
        q_init: float = 0.0,
        q_max: float = 0.0,
        q_interval: int = 10,
        q_step: float = 1.0,
        move_init: float = 0.2,
        move_min: float = 0.05,
        move_damping: float = 1.0,
    ) -> None:
        """Initialize ContinuationScheduler.

        Parameters
        ----------
        penal_init, penal_max, penal_interval, penal_step :
            SIMP penalty exponent schedule: increases by penal_step every penal_interval iterations up to penal_max.
        beta_init, beta_max, beta_interval, beta_factor :
            Heaviside beta schedule: multiplied by beta_factor every beta_interval iterations up to beta_max.
        q_init, q_max, q_interval, q_step :
            RAMP q parameter schedule.
        move_init, move_min, move_damping :
            OC/MMA design variable move limit schedule.
        """
        self.penal: float = float(penal_init)
        self.penal_max: float = float(penal_max)
        self.penal_interval: int = int(penal_interval)
        self.penal_step: float = float(penal_step)

        self.beta: float = float(beta_init)
        self.beta_max: float = float(beta_max)
        self.beta_interval: int = int(beta_interval)
        self.beta_factor: float = float(beta_factor)

        self.q: float = float(q_init)
        self.q_max: float = float(q_max)
        self.q_interval: int = int(q_interval)
        self.q_step: float = float(q_step)

        self.move: float = float(move_init)
        self.move_min: float = float(move_min)
        self.move_damping: float = float(move_damping)

        self.current_iter: int = 0
        self.history: list[Dict[str, float]] = []

    def step(self, iteration: Optional[int] = None, change: Optional[float] = None) -> Dict[str, float]:
        """Advance continuation schedule by one iteration or specified iteration count.

        Parameters
        ----------
        iteration : int, optional
            Current iteration number. If omitted, increments internal counter.
        change : float, optional
            Maximum change in design variables (used for adaptive continuation).

        Returns
        -------
        dict
            Updated values of {'penal', 'beta', 'q', 'move'}.
        """
        if iteration is not None:
            self.current_iter = int(iteration)
        else:
            self.current_iter += 1

        it = self.current_iter

        # 1. Update penal
        if self.penal < self.penal_max and self.penal_interval > 0:
            if it > 0 and it % self.penal_interval == 0:
                self.penal = min(self.penal_max, self.penal + self.penal_step)

        # 2. Update beta
        if self.beta < self.beta_max and self.beta_interval > 0:
            if it > 0 and it % self.beta_interval == 0:
                self.beta = min(self.beta_max, self.beta * self.beta_factor)

        # 3. Update q
        if self.q < self.q_max and self.q_interval > 0:
            if it > 0 and it % self.q_interval == 0:
                self.q = min(self.q_max, self.q + self.q_step)

        # 4. Update move limit
        if self.move_damping < 1.0 and self.move > self.move_min:
            self.move = max(self.move_min, self.move * self.move_damping)

        state = {
            "iter": it,
            "penal": self.penal,
            "beta": self.beta,
            "q": self.q,
            "move": self.move,
        }
        self.history.append(state)
        return state

    @staticmethod
    def compute_grayness(rho_hat: np.ndarray, percent: bool = False) -> float:
        """Compute measure of non-discreteness (M_nd grayness indicator).

        Formula (Ferrari & Sigmund 2020 eq. 3.3):
            M_nd = (4 / N_e) * sum(rho_hat * (1 - rho_hat))

        Values:
            0.0: Perfectly discrete 0/1 design.
            1.0 (or 100%): Uniformly grey 0.5 design.

        Parameters
        ----------
        rho_hat : np.ndarray
            Physical density field.
        percent : bool
            If True, returns percentage in [0, 100]. Default False [0, 1].

        Returns
        -------
        float
            Grayness metric.
        """
        rho = np.asarray(rho_hat, dtype=np.float64)
        n = rho.size
        if n == 0:
            return 0.0

        m_nd = float(np.sum(rho * (1.0 - rho)) * 4.0 / n)
        return m_nd * 100.0 if percent else m_nd
