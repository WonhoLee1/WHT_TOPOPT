"""Generalized-alpha time integration (Chung & Hulbert 1993) for linear structural dynamics.

Equilibrium is enforced at the intermediate point

    M a_{n+1-am} + C v_{n+1-af} + K u_{n+1-af} = f_{n+1-af},   x_{n+1-a} = (1-a) x_{n+1} + a x_n,

with the Newmark update for v_{n+1}, a_{n+1}. Written in u_{n+1} this is ``K_hat u_{n+1} = r_hat``. The examples share this
single right-hand side so the formula cannot drift apart between copies again.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


def generalized_alpha_parameters(rho_inf: float) -> Tuple[float, float, float, float]:
    """Return (alpha_m, alpha_f, beta, gamma), second-order accurate and unconditionally stable for rho_inf in [0, 1]."""
    if not 0.0 <= rho_inf <= 1.0:
        raise ValueError(f"rho_inf must be in [0, 1], got {rho_inf}")
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    gamma = 0.5 - alpha_m + alpha_f
    return alpha_m, alpha_f, beta, gamma


def generalized_alpha_lhs(M, C, K, dt: float, alpha_m: float, alpha_f: float, beta: float, gamma: float):
    """Effective matrix ``K_hat`` multiplying u_{n+1}."""
    return ((1.0 - alpha_m) / (beta * dt**2)) * M + ((1.0 - alpha_f) * gamma / (beta * dt)) * C + (1.0 - alpha_f) * K


def generalized_alpha_rhs(M, C, K, u: np.ndarray, v: np.ndarray, a: np.ndarray, f_mid: np.ndarray, dt: float,
                          alpha_m: float, alpha_f: float, beta: float, gamma: float) -> np.ndarray:
    """Effective right-hand side ``r_hat`` from the state at t_n; ``f_mid = (1-af) f_{n+1} + af f_n``."""
    return (
        f_mid
        + M @ ((1.0 - alpha_m) / (beta * dt**2) * u
               + (1.0 - alpha_m) / (beta * dt) * v
               + ((1.0 - alpha_m) / (2.0 * beta) - 1.0) * a)
        + C @ ((1.0 - alpha_f) * gamma / (beta * dt) * u
               + ((1.0 - alpha_f) * gamma / beta - 1.0) * v
               + (1.0 - alpha_f) * dt * (gamma / (2.0 * beta) - 1.0) * a)
        - alpha_f * (K @ u)
    )


def newmark_update(u: np.ndarray, v: np.ndarray, a: np.ndarray, u_next: np.ndarray, dt: float, beta: float,
                   gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """(v_{n+1}, a_{n+1}) from the Newmark relations."""
    a_next = (u_next - u) / (beta * dt**2) - v / (beta * dt) - (1.0 - 2.0 * beta) / (2.0 * beta) * a
    v_next = v + dt * ((1.0 - gamma) * a + gamma * a_next)
    return v_next, a_next


#: density below which the Du & Olhoff (2007) mass penalty applies
MASS_CUTOFF = 0.10


def du_olhoff_mass_density(rho: np.ndarray) -> np.ndarray:
    """Mass interpolation m(rho): rho above the cutoff, c rho^6 below it (continuous at the cutoff).

    With E ~ rho^3 and a linear mass, low-density elements get a low stiffness-to-mass ratio and spurious local modes."""
    rho = np.asarray(rho, dtype=float)
    return np.where(rho < MASS_CUTOFF, MASS_CUTOFF ** (-5.0) * rho**6, rho)


def du_olhoff_mass_derivative(rho: np.ndarray) -> np.ndarray:
    """dm/drho of :func:`du_olhoff_mass_density`."""
    rho = np.asarray(rho, dtype=float)
    return np.where(rho < MASS_CUTOFF, 6.0 * MASS_CUTOFF ** (-5.0) * rho**5, 1.0)


def check_time_step(K_ff, M_ff, dt: float, limit: float = 0.5) -> float:
    """omega_1 dt of the free system; warns when the first mode is not resolved (omega_1 dt > ``limit``)."""
    from scipy.sparse.linalg import eigsh

    omega2 = float(eigsh(K_ff.tocsc(), k=1, M=M_ff.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)[0])
    omega_dt = float(np.sqrt(max(omega2, 0.0)) * dt)
    if omega_dt > limit:
        warnings.warn(f"time step does not resolve the first mode: omega_1 dt = {omega_dt:.2f} > {limit:g}; "
                      f"use dt <= {limit / np.sqrt(omega2):.3e}", stacklevel=2)
    return omega_dt
