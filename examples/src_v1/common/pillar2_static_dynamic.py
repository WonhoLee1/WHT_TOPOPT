"""Pillar 2: static multi-load optimisation against dynamic optimisation at the same volume.

The dynamic model uses zeroth-order equivalent static loads: at the critical instants t_k of the transient response of the current design,
f_k = K u(t_k), so the static sub-problem has the response u(t_k) and needs no solve. The gradient is that of the ESL sub-problem; there is no
transient adjoint and no KKT guarantee (docs/references/investigations/code_review_wht_topopt_20261103.md, R7). Weights follow the strain
energy at the instants, w_k ~ SE(t_k).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from wht_topopt.algorithms.mma_wrapper import MMAOptimizer

from .pillar1_algorithms import DesignRun, LoadSet, Timers, _keep_frame, run_mma
from .pillar4_manufacturing import RHO_MIN
from .structure import Structure, TransientResult, steps_for_first_mode


@dataclass
class LoadHistory:
    """Dynamic load: the formula is part of the deliverable (spec, Pillar 5.1)."""

    name: str
    formula: str  # readable definition for the report
    duration: float  # s, simulated window
    force_at: Callable[[float], np.ndarray]  # full-space load at time t, zero at t = 0
    peak_force: float  # N
    tip_response: Callable[[np.ndarray], np.ndarray]  # full-space displacement -> measured (e.g. Uy, Uz) in mm
    tip_labels: Tuple[str, ...] = ("Uy", "Uz")
    load_end: Optional[float] = None  # s, when the load has stopped; residual vibration is measured after it
    window_start: float = 0.0  # s, peaks and audit values are taken from here on (what comes before is a start-up ramp)
    dt_max: Optional[float] = None  # s, resolves the fastest load component; the first-mode rule alone does not see it
    peak_selector: Optional[Callable[..., np.ndarray]] = None  # (s, tr, load) -> stored positions; None: critical_instants
    store_all: bool = False  # keep every step, so no peak falls between stored states
    has_residual_phase: bool = True  # False: the load lasts to the end of the window, so there is no residual vibration to measure


def time_grid(s: Structure, rho: np.ndarray, load: LoadHistory, max_steps: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Uniform times with omega_1 dt <= 0.5 of this design; a cap below that is reported."""
    K, M = s.stiffness(np.maximum(rho, RHO_MIN)), s.mass(np.maximum(rho, RHO_MIN))
    f1 = s.natural_frequencies(K, M, k=1)[0]
    n, notes = steps_for_first_mode(f1, load.duration, max_steps=max_steps)
    if load.dt_max is not None:
        needed = int(np.ceil(load.duration / load.dt_max - 1e-9)) + 1
        if max_steps is not None and needed > max_steps:
            notes.append(f"time step above the load's limit: {max_steps} steps used, dt <= {load.dt_max:g} s needs {needed}")
            n = max_steps
        else:
            n = max(n, needed)
    return np.linspace(0.0, load.duration, max(n, 2)), notes


def critical_instants(s: Structure, tr: TransientResult, load: LoadHistory, n: int = 4) -> np.ndarray:
    """Positions in ``tr.stored``: peak strain energy, peak of each measured response, then the next largest energies, all distinct."""
    energy = tr.energy[tr.stored]
    chosen = [int(np.argmax(energy))]
    tips = np.stack([load.tip_response(s.expand(tr.u[i])) for i in range(tr.stored.size)])
    for col in range(tips.shape[1]):
        i = int(np.argmax(np.abs(tips[:, col])))
        if i not in chosen:
            chosen.append(i)
    for i in np.argsort(-energy):
        if len(chosen) >= n:
            break
        if int(i) not in chosen and energy[i] > 0.0:
            chosen.append(int(i))
    return np.array(sorted(chosen[:n]))


def esl_peak_instants(times: np.ndarray, energy: np.ndarray, tips: np.ndarray, n: int = 5, min_separation: float = 0.5, start: float = 0.0) -> np.ndarray:
    """``n`` positions at separate response peaks, in time order: the energy peak, the peak of each measured response, then the largest energy maxima.

    Neighbouring steps of one peak carry the same load, so positions closer than ``min_separation`` do not count twice. Fewer than ``n``
    separate peaks raise: padding with non-peaks would hide that the window does not hold the requested peaks."""
    times, energy, tips = np.asarray(times), np.asarray(energy), np.atleast_2d(tips)
    window = np.flatnonzero(times >= start)
    if window.size == 0:
        raise ValueError(f"no stored state at or after t = {start}")
    chosen: List[int] = []

    def take(i: int) -> None:
        if len(chosen) < n and all(abs(times[i] - times[j]) >= min_separation for j in chosen):
            chosen.append(int(i))

    take(window[np.argmax(energy[window])])
    for col in range(tips.shape[1]):
        take(window[np.argmax(np.abs(tips[window, col]))])
    inner = window[(window > 0) & (window < energy.size - 1)]
    maxima = inner[(energy[inner] > 0.0) & (energy[inner] >= energy[inner - 1]) & (energy[inner] >= energy[inner + 1])]
    for i in maxima[np.argsort(-energy[maxima], kind="stable")]:
        take(i)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} separate response peaks at least {min_separation:g} s apart in the window, {n} requested")
    return np.array(sorted(chosen))


def _peak_record(s: Structure, tr: TransientResult, load: LoadHistory, idx: np.ndarray, energies: np.ndarray, w: np.ndarray) -> List[dict]:
    """Time, u^T K u, weight and load resultant at each chosen instant."""
    t = tr.times[tr.stored]
    return [{"t_s": float(t[i]), "energy": float(e), "weight": float(wk), "force": [float(v) for v in load.force_at(float(t[i])).reshape(-1, s.dim).sum(axis=0)]} for i, e, wk in zip(idx, energies, w)]


def esl_peak_selector(n: int = 5, min_separation: float = 0.5) -> Callable[[Structure, TransientResult, LoadHistory], np.ndarray]:
    """``LoadHistory.peak_selector`` from :func:`esl_peak_instants`."""

    def select(s: Structure, tr: TransientResult, load: LoadHistory) -> np.ndarray:
        tips = np.stack([load.tip_response(s.expand(tr.u[i])) for i in range(tr.stored.size)])
        return esl_peak_instants(tr.times[tr.stored], tr.energy[tr.stored], tips, n, min_separation, load.window_start)

    return select


def optimize_dynamic(
    s: Structure,
    dmap,
    load: LoadHistory,
    volfrac: float,
    max_iters: int = 40,
    max_steps: Optional[int] = None,
    store_limit: int = 200,
    tol: float = 0.008,
    label: str = "dynamic",
) -> DesignRun:
    """MMA on the ESL sub-problem. The time grid is set from the design's first mode at the first iteration and the last."""
    t0 = time.perf_counter()
    t = Timers()
    opt = MMAOptimizer(rho_min=RHO_MIN, rho_max=1.0, move=0.2)
    x = dmap.fix_passive(np.full(s.n_elem, volfrac))
    vol = s.volumes / s.total_active_volume
    hist_c, hist_v, frames = [], [], []
    notes: List[str] = ["zeroth-order ESL: no transient adjoint, no KKT guarantee"]
    c_ref = None
    times = None
    converged = False
    it = 0
    for it in range(1, max_iters + 1):
        dmap.set_iteration(it)
        rho = dmap.forward(x)
        rho_a = np.maximum(rho, RHO_MIN)
        t1 = time.perf_counter()
        if times is None:
            times, grid_notes = time_grid(s, rho_a, load, max_steps)
            notes += grid_notes
        K, M = s.stiffness(rho_a), s.mass(rho_a)
        tr = s.transient(K, M, load.force_at, times, store_every=1 if load.store_all else max(1, (len(times) - 1) // store_limit))
        idx = (load.peak_selector or critical_instants)(s, tr, load)
        t.fem += time.perf_counter() - t1
        t1 = time.perf_counter()
        u_full = [s.expand(tr.u[i]) for i in idx]
        energies = np.array([float(tr.u[i] @ (K @ tr.u[i])) for i in idx])
        total = float(energies.sum())
        w = energies / total if total > 0.0 else np.full(idx.size, 1.0 / idx.size)
        peaks = _peak_record(s, tr, load, idx, energies, w)
        c = float(w @ energies)
        ce = sum(wk * s.unit_energy(uk) for wk, uk in zip(w, u_full))
        c_ref = c_ref or max(c, 1e-30)
        dc = dmap.backward(-s.dyoung(rho) * ce)
        v = s.volume_fraction(rho)
        dv = dmap.backward(vol) / volfrac
        t.sens += time.perf_counter() - t1
        g = np.array([v / volfrac - 1.0])
        t1 = time.perf_counter()
        x_new = dmap.fix_passive(opt.update(x, dc / c_ref, g, dv.reshape(1, -1)))
        t.opt += time.perf_counter() - t1
        change = float(np.max(np.abs(x_new - x)))
        x = x_new
        hist_c.append(c)
        hist_v.append(v)
        _keep_frame(it, frames, rho)
        print(f"    {label} it {it}: c = {c:.5g}, V = {v:.3f}, {len(times)} steps", flush=True)
        if change < tol and abs(g[0]) <= 0.005 and it >= 20:
            converged = True
            break
    return DesignRun(label, dmap.forward(x), it, converged, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames, notes, {"esl_peaks_last_iteration": peaks})


def run_static_vs_dynamic(s: Structure, make_map: Callable[[], object], loads: LoadSet, load: LoadHistory, volfrac: float, iters: Optional[int] = None, max_steps: Optional[int] = None) -> Tuple[DesignRun, DesignRun]:
    """Model 1 (static multi-load) and Model 2 (dynamic ESL) at the same volume, same design map."""
    model1 = run_mma(s, make_map(), loads, volfrac, iters, label="model1_static")
    model2 = optimize_dynamic(s, make_map(), load, volfrac, iters or 40, max_steps, label="model2_dynamic")
    return model1, model2
