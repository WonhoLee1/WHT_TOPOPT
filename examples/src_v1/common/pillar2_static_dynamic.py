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


def time_grid(s: Structure, rho: np.ndarray, load: LoadHistory, max_steps: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Uniform times with omega_1 dt <= 0.5 of this design; a cap below that is reported."""
    K, M = s.stiffness(np.maximum(rho, RHO_MIN)), s.mass(np.maximum(rho, RHO_MIN))
    f1 = s.natural_frequencies(K, M, k=1)[0]
    n, notes = steps_for_first_mode(f1, load.duration, max_steps=max_steps)
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
        tr = s.transient(K, M, load.force_at, times, store_every=max(1, (len(times) - 1) // store_limit))
        idx = critical_instants(s, tr, load)
        t.fem += time.perf_counter() - t1
        t1 = time.perf_counter()
        u_full = [s.expand(tr.u[i]) for i in idx]
        energies = np.array([float(tr.u[i] @ (K @ tr.u[i])) for i in idx])
        total = float(energies.sum())
        w = energies / total if total > 0.0 else np.full(idx.size, 1.0 / idx.size)
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
    return DesignRun(label, dmap.forward(x), it, converged, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames, notes)


def run_static_vs_dynamic(s: Structure, make_map: Callable[[], object], loads: LoadSet, load: LoadHistory, volfrac: float, iters: Optional[int] = None, max_steps: Optional[int] = None) -> Tuple[DesignRun, DesignRun]:
    """Model 1 (static multi-load) and Model 2 (dynamic ESL) at the same volume, same design map."""
    model1 = run_mma(s, make_map(), loads, volfrac, iters, label="model1_static")
    model2 = optimize_dynamic(s, make_map(), load, volfrac, iters or 40, max_steps, label="model2_dynamic")
    return model1, model2
