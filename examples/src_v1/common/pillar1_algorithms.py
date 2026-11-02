"""Pillar 1 on a 2D or 3D structure with multiple load cases: MMA, GCMMA, SEMDOT, BESO, SE-AHCA.

Kratos variants need the Kratos engine; where it is missing they are recorded as skipped, never replaced by a built-in run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wht_topopt.algorithms.adapter import AlgorithmRegistry, OptimizationStepInput
from wht_topopt.algorithms.beso import BESOOptimizer
from wht_topopt.algorithms.hca import SEAHCAConfig, StressEnergyAdaptiveHCA
from wht_topopt.algorithms.mma_wrapper import MMAOptimizer
from wht_topopt.algorithms.oc import OCOptimizer
from wht_topopt.algorithms.semdot import SEMDOTOptimizer

from .pillar4_manufacturing import RHO_MIN, DensityMap
from .structure import Structure

DEFAULT_ITERS = {"mma": 60, "gcmma": 50, "semdot": 60, "beso": 60, "se_ahca": 100}
ALGORITHMS = tuple(DEFAULT_ITERS)
KRATOS_ALGORITHMS = ("kratos_mma", "kratos_gcmma", "kratos_semdot", "kratos_beso")
SEMDOT_MIN_ITERS = 50


@dataclass
class DesignRun:
    """One optimised design."""

    label: str
    rho: np.ndarray
    iterations: int
    converged: bool
    t_total: float
    t_fem: float = 0.0
    t_sens: float = 0.0
    t_opt: float = 0.0
    history_c: List[float] = field(default_factory=list)
    history_v: List[float] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class LoadSet:
    """Static load cases (ncase, ndof) and their weights."""

    forces: np.ndarray
    weights: np.ndarray
    names: List[str]

    def __post_init__(self) -> None:
        self.forces = np.atleast_2d(self.forces)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.forces.shape[0] != self.weights.size or self.weights.size != len(self.names):
            raise ValueError("one weight and one name per load case")


class Timers:
    def __init__(self) -> None:
        self.fem = self.sens = self.opt = 0.0


def static_response(s: Structure, rho: np.ndarray, loads: LoadSet, t: Timers) -> Tuple[float, np.ndarray, np.ndarray]:
    """(weighted compliance, unit energy per element weighted over the cases, displacements)."""
    t0 = time.perf_counter()
    K = s.stiffness(rho)
    u = s.solve_static(K, loads.forces)
    t.fem += time.perf_counter() - t0
    t0 = time.perf_counter()
    c = float(loads.weights @ np.einsum("cd,cd->c", loads.forces, u))
    ce = sum(w * s.unit_energy(u[k]) for k, w in enumerate(loads.weights))
    t.sens += time.perf_counter() - t0
    return c, ce, u


def _keep_frame(it: int, frames: List[np.ndarray], rho: np.ndarray, every: int = 5) -> None:
    if it == 1 or it % every == 0:
        frames.append(rho.astype(np.float32))


def run_mma(s: Structure, dmap, loads: LoadSet, volfrac: float, max_iters: Optional[int] = None, gcmma: bool = False, tol: float = 0.008, label: Optional[str] = None) -> DesignRun:
    t0 = time.perf_counter()
    name = "gcmma" if gcmma else "mma"
    t = Timers()
    opt = AlgorithmRegistry.get("gcmma", rho_min=RHO_MIN, rho_max=1.0, move=0.2) if gcmma else MMAOptimizer(rho_min=RHO_MIN, rho_max=1.0, move=0.2)
    x = dmap.fix_passive(np.full(s.n_elem, volfrac))
    vol = s.volumes / s.total_active_volume
    hist_c, hist_v, frames = [], [], []
    c_ref = None
    converged = False
    it = 0
    for it in range(1, (max_iters or DEFAULT_ITERS[name]) + 1):
        dmap.set_iteration(it)
        rho = dmap.forward(x)
        c, ce, _ = static_response(s, np.maximum(rho, RHO_MIN), loads, t)
        c_ref = c_ref or max(c, 1e-30)
        t1 = time.perf_counter()
        dc = dmap.backward(-s.dyoung(rho) * ce)
        v = s.volume_fraction(rho)
        dv = dmap.backward(vol) / volfrac
        t.sens += time.perf_counter() - t1
        g = np.array([v / volfrac - 1.0])
        t1 = time.perf_counter()
        if gcmma:
            def true_values(xt: np.ndarray, _ref: float = c_ref) -> Tuple[float, np.ndarray]:
                r = dmap.forward(xt)
                ct, _, _ = static_response(s, np.maximum(r, RHO_MIN), loads, t)
                return ct / _ref, np.array([s.volume_fraction(r) / volfrac - 1.0])

            x_new = opt.step(OptimizationStepInput(iteration=it, rho=x, df_drho=dc / c_ref, g_vals=g, dg_drho=dv.reshape(1, -1), f0_val=c / c_ref, eval_fn=true_values)).rho_next
            dmap.forward(x)  # the inner loop moved the map's state; the next forward call resets it anyway
        else:
            x_new = opt.update(x, dc / c_ref, g, dv.reshape(1, -1))
        x_new = dmap.fix_passive(x_new)
        t.opt += time.perf_counter() - t1
        change = float(np.max(np.abs(x_new - x)))
        x = x_new
        hist_c.append(c)
        hist_v.append(v)
        _keep_frame(it, frames, rho)
        if change < tol and abs(g[0]) <= 0.005 and it >= 20:
            converged = True
            break
    return DesignRun(label or name, dmap.forward(x), it, converged, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames)


def run_semdot(s: Structure, dmap: DensityMap, loads: LoadSet, volfrac: float, max_iters: Optional[int] = None, tol: float = 0.005, n_sub: int = 4) -> DesignRun:
    """Linear modulus in the analysis, sub-element cut of the filtered x. ``dmap`` supplies the density filter only."""
    t0 = time.perf_counter()
    if not isinstance(dmap, DensityMap) or dmap.beta0 > 0.0:
        raise ValueError("SEMDOT makes its own boundary: it takes a plain density map, without projection or a manufacturing filter")
    t = Timers()
    xmin = 1e-6
    saved = s.penal, s.emin
    s.penal, s.emin = 1.0, s.e0 * xmin
    try:
        oc = OCOptimizer(volfrac=volfrac * s.total_active_volume / s.volumes.sum(), move=0.2, rho_min=RHO_MIN)
        semdot = SEMDOTOptimizer(mesh=s.grid_mesh, volfrac=volfrac, n_sub=n_sub, beta_init=0.5, beta_max=8.0, rho_min=xmin, move=0.2)
        target = volfrac * s.total_active_volume / s.volumes.sum()
        project = lambda xt: np.maximum(semdot.project_filtered(dmap.H @ xt, target_volfrac=target, passive_solid=np.flatnonzero(s.solid_mask), passive_void=np.flatnonzero(s.void_mask)), xmin)
        x = dmap.fix_passive(np.full(s.n_elem, target))
        psi = project(x)
        dv = dmap.H.T @ s.volumes
        dc_prev = None
        hist_c, hist_v, frames = [], [], []
        converged = False
        it = 0
        for it in range(1, (max_iters or DEFAULT_ITERS["semdot"]) + 1):
            c, ce, _ = static_response(s, psi, loads, t)
            t1 = time.perf_counter()
            dc = -((1.0 - psi) * xmin + psi) * ce * s.design_mask
            if dc_prev is not None:
                dc = 0.35 * dc + 0.65 * dc_prev
            dc_prev = dc
            dc_x = dmap.H.T @ dc
            t.sens += time.perf_counter() - t1
            t1 = time.perf_counter()
            x_new = oc.update(x, dc_x, None, dv, elem_volumes=s.volumes, passive_solid=s.solid_mask, passive_void=s.void_mask)
            psi_new = project(x_new)
            t.opt += time.perf_counter() - t1
            change = float(np.mean(np.abs(x_new - x)))
            hist_c.append(c)
            hist_v.append(s.volume_fraction(psi))
            _keep_frame(it, frames, psi)
            x, psi = x_new, psi_new
            if change < tol and it >= SEMDOT_MIN_ITERS:
                converged = True
                break
    finally:
        s.penal, s.emin = saved
    run = DesignRun("semdot", psi, it, converged, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames, ["own compliance uses the linear modulus"])
    run.__dict__["semdot"] = semdot
    return run


def run_beso(s: Structure, dmap: DensityMap, loads: LoadSet, volfrac: float, max_iters: Optional[int] = None) -> DesignRun:
    """0/1 design from the full domain; load and support elements stay, the load path is kept connected."""
    t0 = time.perf_counter()
    t = Timers()
    saved = s.emin
    s.emin = s.e0 * 1e-6
    try:
        grid_nodes = int(s.conn.max()) + 1
        loaded = np.any(loads.forces != 0.0, axis=0)
        load_nodes = np.unique(np.flatnonzero(loaded) // s.dim)
        for rb in s.rigid_bodies:
            if np.isin([rb.ref_node, rb.rot_node], load_nodes).any():
                load_nodes = np.union1d(np.setdiff1d(load_nodes, [rb.ref_node, rb.rot_node]), rb.nodes)
        touches = np.zeros(grid_nodes, dtype=bool)
        touches[load_nodes[load_nodes < grid_nodes]] = True
        solid = s.solid_mask | (touches[s.conn].any(axis=1) & ~s.void_mask)
        target = volfrac * s.total_active_volume / s.volumes.sum()
        beso = BESOOptimizer(target_volfrac=target, er=0.03, er_adaptive=True, er_adaptive_scheme="ER1", er_min=0.015, er_max=0.05, rho_min=RHO_MIN)
        x = np.where(s.void_mask, RHO_MIN, 1.0)
        hist_c, hist_v, frames, stable = [], [], [], []
        converged = False
        it = 0
        for it in range(1, (max_iters or DEFAULT_ITERS["beso"]) + 1):
            c, ce, _ = static_response(s, np.maximum(x, RHO_MIN), loads, t)
            t1 = time.perf_counter()
            alpha = dmap.H @ (x ** (s.penal - 1.0) * ce)
            t.sens += time.perf_counter() - t1
            t1 = time.perf_counter()
            x_new = beso.update(x, -alpha, np.array([0.0]), s.volumes, elem_volumes=s.volumes, passive_solid=solid, passive_void=s.void_mask)
            t.opt += time.perf_counter() - t1
            hist_c.append(c)
            hist_v.append(s.volume_fraction(x))
            _keep_frame(it, frames, x)
            x = x_new
            if abs(s.volume_fraction(x) - volfrac) <= 0.005:
                stable.append(c)
                if len(stable) >= 8 and it >= 25 and abs(np.mean(stable[-4:]) - np.mean(stable[-8:-4])) / max(c, 1e-30) <= 0.003:
                    converged = True
                    break
    finally:
        s.emin = saved
    return DesignRun("beso", dmap.fix_passive(np.where(x > 0.5, 1.0, RHO_MIN)), it, converged, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames)


def run_se_ahca(s: Structure, rmin: float, loads: LoadSet, volfrac: float, max_iters: Optional[int] = None, stress_weight: float = 0.30) -> DesignRun:
    """Penalised signal rho^(p-1) u0, solid-modulus stress (the engine relaxes it), Heaviside projection."""
    t0 = time.perf_counter()
    t = Timers()
    dmap = DensityMap(s, rmin, beta0=1.0, beta_max=8.0, beta_every=20)
    target = volfrac * s.total_active_volume / s.volumes.sum()
    cfg = SEAHCAConfig(volfrac_target=target, stress_weight_eta=stress_weight, kp_init=0.18, ki=0.03, kd=0.02, move_limit=0.18, history_window=1, sigma_allow=s.yield_stress or 0.0)  # 0: relative to the field, for a non-dimensional problem

    hca = StressEnergyAdaptiveHCA(elem_volumes=s.volumes, config=cfg, passive_solid_mask=s.solid_mask, passive_void_mask=s.void_mask)
    x = dmap.fix_passive(np.full(s.n_elem, target))
    hist_c, hist_v, frames = [], [], []
    it = 0
    for it in range(1, (max_iters or DEFAULT_ITERS["se_ahca"]) + 1):
        dmap.set_iteration(it)
        rho = dmap.forward(x)
        c, _, u = static_response(s, np.maximum(rho, RHO_MIN), loads, t)
        t1 = time.perf_counter()
        floor = np.maximum(rho, RHO_MIN) ** (s.penal - 1.0)
        energy = np.stack([floor * s.unit_energy(u[k]) for k in range(u.shape[0])])
        vm = np.stack([s.von_mises(u[k]) for k in range(u.shape[0])])
        signal, _ = hca.compute_cell_state_signal(rho, strain_energy_cases=energy, von_mises_unpenalized_cases=vm, case_weights=loads.weights, neighborhood_filter_fn=dmap.smooth_signal)
        t.sens += time.perf_counter() - t1
        t1 = time.perf_counter()
        x, _, _ = hca.step(x, signal, forward_projection_fn=dmap.forward)
        t.opt += time.perf_counter() - t1
        hist_c.append(c)
        hist_v.append(s.volume_fraction(rho))
        _keep_frame(it, frames, rho)
    return DesignRun("se_ahca", dmap.forward(x), it, False, time.perf_counter() - t0, t.fem, t.sens, t.opt, hist_c, hist_v, frames)


def kratos_available() -> bool:
    try:
        import KratosMultiphysics  # noqa: F401

        return True
    except ImportError:
        return False


def run_algorithms(s: Structure, rmin: float, loads: LoadSet, volfrac: float, algorithms: Sequence[str] = ALGORITHMS, max_iters: Optional[int] = None) -> Tuple[List[DesignRun], Dict[str, str]]:
    """The built-in methods, and the reason each Kratos variant did not run."""
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown algorithm(s) {unknown}; available: {list(ALGORITHMS)}")
    dmap = DensityMap(s, rmin)
    runs: List[DesignRun] = []
    for name in algorithms:
        if name == "mma":
            r = run_mma(s, dmap, loads, volfrac, max_iters)
        elif name == "gcmma":
            r = run_mma(s, dmap, loads, volfrac, max_iters, gcmma=True)
        elif name == "semdot":
            r = run_semdot(s, dmap, loads, volfrac, max_iters)
        elif name == "beso":
            r = run_beso(s, dmap, loads, volfrac, max_iters)
        else:
            r = run_se_ahca(s, rmin, loads, volfrac, max_iters)
        print(f"  {name}: V = {s.volume_fraction(r.rho):.3f}, {r.iterations} it, {r.t_total:.1f} s", flush=True)
        runs.append(r)
    reason = "Kratos 환경에서 실행하지 않았다: 이 러너는 내장 해석기만 쓴다" if kratos_available() else "KratosMultiphysics가 이 환경에 없다"
    return runs, {k: reason for k in KRATOS_ALGORITHMS}
