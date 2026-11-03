"""What the three dynamic 2D basics share: the load history, the time scale, the damping and the study set-up.

The material stays non-dimensional (E = 1, F = 1). The density is set so the uniform starting design has f1 = 1 Hz: with real steel a 180 x 60 mm
cantilever has f1 ~ 1.5 kHz, a 20 s window would be quasi-static and take ~380 000 steps per iteration. Time is the only dimensional quantity.

The transient starts from rest, so a ramp comes before the 20 s window; the load limits, the peaks and the audit hold on the window only.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from wht_topopt.examples.src_v1.common.pillar1_algorithms import LoadSet
from wht_topopt.examples.src_v1.common.pillar2_static_dynamic import LoadHistory, esl_peak_selector
from wht_topopt.examples.src_v1.common.pillar4_manufacturing import RHO_MIN, DensityMap
from wht_topopt.examples.src_v1.common.pillar5_audit import StiffnessTest, TestRig
from wht_topopt.examples.src_v1.common.structure import Structure
from wht_topopt.examples.src_v1.common.study import Study

FAMILY = "ex00-dyn"
NOT_APPLICABLE = {
    3: "체적 완화 목표를 두지 않은 2D 기본 동적 시험",
    4: "2D 평면 문제: 압출, 스킨-리브 필터 대상 아님",
}
DEFAULT_ALGOS = ("mma",)  # Pillar 2 compares against the static MMA design

F1_TARGET_HZ = 1.0
RAMP_S, WINDOW_S = 2.0, 20.0
DT_MAX_S = 0.02  # ~16 points per period of the fastest component
N_PEAKS, PEAK_SEPARATION_S = 5, 0.5
MAG_RANGE = (0.5, 1.5)  # times the static load
ANGLE_RANGE_DEG = (-45.0, 45.0)  # from -y
# (Hz, amplitude, phase in rad). Above the starting f1 on purpose: 0/1 designs measured up to 2.8 x the f1 of the uniform design.
MAG_HARMONICS = ((0.13, 1.0, 0.3), (0.37, 0.8, 1.9), (0.71, 0.7, 4.1), (1.07, 1.0, 2.6), (1.53, 0.8, 5.2), (2.3, 0.5, 0.9), (3.1, 0.4, 3.4))
ANGLE_HARMONICS = ((0.17, 1.0, 1.1), (0.43, 0.8, 3.7), (0.89, 0.8, 0.5), (1.31, 0.7, 5.5), (2.1, 0.5, 2.2), (2.9, 0.4, 4.6))
ZETA, ZETA_F1_HZ, ZETA_F2_HZ = 0.02, 1.0, 5.0


def rayleigh_for(zeta: float, f1_hz: float, f2_hz: float) -> Tuple[float, float]:
    """(alpha, beta) with damping ratio ``zeta`` at both frequencies."""
    w1, w2 = 2.0 * np.pi * f1_hz, 2.0 * np.pi * f2_hz
    return 2.0 * zeta * w1 * w2 / (w1 + w2), 2.0 * zeta / (w1 + w2)


def _harmonics(table: Sequence[Tuple[float, float, float]], tau):
    return sum(amp * np.sin(2.0 * np.pi * f * tau + phase) for f, amp, phase in table)


def _ranged(table: Sequence[Tuple[float, float, float]], limits: Tuple[float, float]) -> Callable[[float], float]:
    """The combination mapped so its extremes over the window are exactly the limits."""
    h = _harmonics(table, np.linspace(0.0, WINDOW_S, 40001))
    lo, hi = float(h.min()), float(h.max())
    return lambda tau: limits[0] + (limits[1] - limits[0]) * (_harmonics(table, tau) - lo) / (hi - lo)


magnitude = _ranged(MAG_HARMONICS, MAG_RANGE)
angle_deg = _ranged(ANGLE_HARMONICS, ANGLE_RANGE_DEG)


def load_formula() -> str:
    fmt = lambda table: " + ".join(f"{a:g} sin(2 pi {f:g} tau + {p:g})" for f, a, p in table)
    return (
        f"F(t) = F0 s(t) a(tau) [sin phi(tau), -cos phi(tau)], tau = t - {RAMP_S:g} s, s(t) = sin^2(pi t / {2.0 * RAMP_S:g}) for t < {RAMP_S:g} s then 1; "
        f"a(tau) is h_a scaled to [{MAG_RANGE[0]:g}, {MAG_RANGE[1]:g}] over the {WINDOW_S:g} s window, h_a = {fmt(MAG_HARMONICS)}; "
        f"phi(tau) is h_phi scaled to [{ANGLE_RANGE_DEG[0]:g}, {ANGLE_RANGE_DEG[1]:g}] deg, h_phi = {fmt(ANGLE_HARMONICS)}. "
        f"Non-dimensional E and F: the mm, N and MPa labels of the tables carry no unit here; density set for f1 = {F1_TARGET_HZ:g} Hz of the uniform start, zeta = {ZETA:g} at {ZETA_F1_HZ:g} and {ZETA_F2_HZ:g} Hz."
    )


def calibrate_density(s: Structure, rmin: float, volfrac: float, f1_hz: float = F1_TARGET_HZ) -> float:
    """Set the density so the uniform starting design has first frequency ``f1_hz``; f ~ 1 / sqrt(density)."""
    dmap = DensityMap(s, rmin)
    rho = np.maximum(dmap.forward(dmap.fix_passive(np.full(s.n_elem, volfrac))), RHO_MIN)
    s.density = 1.0
    f_unit = s.natural_frequencies(s.stiffness(rho), s.mass(rho), k=1)[0]
    s.density = (f_unit / f1_hz) ** 2
    return s.density


def load_history(s: Structure, force: np.ndarray) -> LoadHistory:
    """The static load's nodal pattern driven through the history. The static load must point along -y."""
    pattern = -force[1::2]
    if np.any(force[0::2] != 0.0) or np.any(pattern < 0.0) or not np.any(pattern > 0.0):
        raise ValueError("the static load must act along -y only")
    nodes = np.flatnonzero(pattern)
    total = float(pattern.sum())

    def force_at(t: float) -> np.ndarray:
        tau = t - RAMP_S
        ramp = np.sin(np.pi * t / (2.0 * RAMP_S)) ** 2 if t < RAMP_S else 1.0
        amp = ramp * magnitude(tau)
        phi = np.radians(angle_deg(tau))
        f = np.zeros(force.size)
        f[0::2] = amp * np.sin(phi) * pattern
        f[1::2] = -amp * np.cos(phi) * pattern
        return f

    return LoadHistory(
        name=f"{WINDOW_S:g} s harmonic load, |F| {MAG_RANGE[0]:g}-{MAG_RANGE[1]:g} x static, direction -y +/- 45 deg",
        formula=load_formula(),
        duration=RAMP_S + WINDOW_S,
        force_at=force_at,
        peak_force=MAG_RANGE[1] * total,
        tip_response=lambda u: np.array([float(np.mean(u[2 * nodes])), float(np.mean(u[2 * nodes + 1]))]),
        tip_labels=("Ux", "Uy"),
        load_end=RAMP_S + WINDOW_S,
        window_start=RAMP_S,
        dt_max=DT_MAX_S,
        peak_selector=esl_peak_selector(N_PEAKS, PEAK_SEPARATION_S),
        store_all=True,
        has_residual_phase=False,
    )


def dynamic_study_2d(name: str, mesh, fixed: np.ndarray, force: np.ndarray, volfrac: float, rmin: float, passive_solid: Sequence[int] = (), passive_void: Sequence[int] = (), iters: Optional[int] = None, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    """The static ex00 problem plus a mass and the load history. Model 1 is optimised for the static load, model 2 for the ESL of the history."""
    s = Structure(mesh, fixed, e0=1.0, nu=0.3, density=1.0, yield_stress=None, passive_solid=passive_solid, passive_void=passive_void, emin_ratio=1e-9, rayleigh=rayleigh_for(ZETA, ZETA_F1_HZ, ZETA_F2_HZ))
    calibrate_density(s, rmin, volfrac)
    pattern = -force[1::2]
    total = float(pattern.sum())
    nodes = np.flatnonzero(pattern)
    force_x = np.zeros(force.size)
    force_x[0::2] = pattern
    tests = [
        StiffnessTest("K_load", force, total, lambda u: float(force @ u) / total, "F/u"),
        StiffnessTest("K_x", force_x, total, lambda u: float(force_x @ u) / total, "F/u"),
    ]
    loads = LoadSet(force, np.ones(1), ["load"])
    return Study(
        name=name, family=FAMILY, structure=s, loads=loads, rig=TestRig(tests, loads, load_history(s, force), drop_detached=True), volfrac=volfrac, rmin=rmin,
        make_map=lambda: DensityMap(s, rmin), not_applicable=NOT_APPLICABLE, smoke=smoke, iters=iters, max_steps=max_steps,
    )
