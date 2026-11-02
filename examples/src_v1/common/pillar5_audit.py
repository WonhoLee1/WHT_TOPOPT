"""Pillar 5: every design on one virtual test rig, static and dynamic results kept apart.

Designs are cut to 0/1 at their target volume. All values come from the solves below; if a modal solve fails the error propagates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .pillar1_algorithms import LoadSet
from .pillar2_static_dynamic import LoadHistory, time_grid
from .structure import Structure


@dataclass
class StiffnessTest:
    """Load ``force`` (magnitude ``amount``, N or N*mm); ``measure`` reads the conjugate displacement or rotation from the full field."""

    label: str  # K_x, K_y, K_z, K_theta_x
    force: np.ndarray
    amount: float
    measure: Callable[[np.ndarray], float]
    unit: str  # N/mm or N*mm/rad


@dataclass
class TestRig:
    stiffness_tests: List[StiffnessTest]
    operating: LoadSet
    dynamic: Optional[LoadHistory] = None  # None: a static problem, no dynamic block


def binary_design(s: Structure, rho: np.ndarray, volfrac: float) -> np.ndarray:
    """The passive solid plus the densest design elements, to the target volume."""
    x = np.zeros(s.n_elem)
    x[s.solid_mask] = 1.0
    design = np.flatnonzero(s.design_mask)
    n_active = int(np.sum(~s.void_mask))
    n_keep = int(round(volfrac * n_active)) - int(s.solid_mask.sum())
    x[design[np.argsort(-rho[design], kind="stable")[: max(n_keep, 0)]]] = 1.0
    return x


def load_path_connected(s: Structure, xb: np.ndarray, forces: np.ndarray) -> bool:
    """Every loaded grid node on the solid, in a part held in every direction.

    Reaching some support is not enough: a half MBB beam touching only its symmetry plane is held in x alone and hangs on the void.
    Loads on a rigid-body reference count through its slave nodes."""
    n_grid = int(s.conn.max()) + 1
    conn = s.conn[xb > 0.5]
    if conn.size == 0:
        return False
    rows = np.repeat(conn[:, 0], conn.shape[1] - 1)
    _, label = connected_components(coo_matrix((np.ones(rows.size), (rows, conn[:, 1:].ravel())), shape=(n_grid, n_grid)), directed=False)
    on_solid = np.zeros(n_grid, dtype=bool)
    on_solid[conn.ravel()] = True
    loaded = np.unique(np.flatnonzero(np.any(np.atleast_2d(forces) != 0.0, axis=0)) // s.dim)
    for rb in s.rigid_bodies:
        if np.isin([rb.ref_node, rb.rot_node], loaded).any():
            loaded = np.union1d(np.setdiff1d(loaded, [rb.ref_node, rb.rot_node]), rb.nodes)
    if not on_solid[loaded].all():
        return False
    fixed_nodes, fixed_dir = s.fixed_dofs // s.dim, s.fixed_dofs % s.dim
    keep = (fixed_nodes < n_grid) & on_solid[np.minimum(fixed_nodes, n_grid - 1)]
    fixed_nodes, fixed_dir = fixed_nodes[keep], fixed_dir[keep]
    return all(np.unique(fixed_dir[label[fixed_nodes] == part]).size == s.dim for part in np.unique(label[loaded]))


def audit_design(s: Structure, rig: TestRig, rho: np.ndarray, volfrac: float, max_steps: Optional[int] = None, n_modes: int = 3) -> Dict:
    xb = binary_design(s, rho, volfrac)
    K = s.stiffness(xb)
    notes: List[str] = []
    connected = load_path_connected(s, xb, rig.operating.forces)
    if not connected:
        notes.append("load path cut in the 0/1 design: the results measure the void stiffness")

    # static block
    tests = np.stack([t.force for t in rig.stiffness_tests])
    u_tests = s.solve_static(K, tests)
    stiffness = {t.label: float(t.amount / abs(t.measure(u_tests[i]))) for i, t in enumerate(rig.stiffness_tests)}
    u_op = s.solve_static(K, rig.operating.forces)
    case_c = np.einsum("cd,cd->c", rig.operating.forces, u_op)
    solid = (xb > 0.5) & (~s.void_mask)
    vm_op = np.stack([s.von_mises(u_op[k])[solid] for k in range(u_op.shape[0])])
    sigma_static = float(vm_op.max())
    static = {
        "stiffness": stiffness,
        "stiffness_units": {t.label: t.unit for t in rig.stiffness_tests},
        "compliance_mJ": float(rig.operating.weights @ case_c),
        "compliance_per_case_mJ": {n: float(c) for n, c in zip(rig.operating.names, case_c)},
        "von_mises_max_MPa": sigma_static,
        "von_mises_p995_MPa": float(np.percentile(vm_op, 99.5)),
        "safety_factor": None if s.yield_stress is None else s.yield_stress / sigma_static,
    }
    if rig.dynamic is not None:
        static["tip_displacement_mm"] = {n: [float(v) for v in rig.dynamic.tip_response(u_op[k])] for k, n in enumerate(rig.operating.names)}
    base = {"volume_fraction": s.volume_fraction(xb), "mass_g": None if s.density is None else s.mass_g(xb), "load_path_connected": connected, "static": static, "dynamic": None, "notes": notes, "xb": xb}
    if s.yield_stress is None:
        notes.append("non-dimensional problem: no yield check")
    if rig.dynamic is None or s.density is None:
        return base
    M = s.mass(xb)

    # dynamic block
    freqs = s.natural_frequencies(K, M, k=n_modes)
    times, grid_notes = time_grid(s, xb, rig.dynamic, max_steps)
    notes += grid_notes
    tr = s.transient(K, M, rig.dynamic.force_at, times, store_every=max(1, (len(times) - 1) // 400))
    tips = np.stack([rig.dynamic.tip_response(s.expand(tr.u[i])) for i in range(tr.stored.size)])
    t_stored = times[tr.stored]
    peak = {lab: float(np.max(np.abs(tips[:, c]))) for c, lab in enumerate(rig.dynamic.tip_labels)}
    end = rig.dynamic.load_end if rig.dynamic.load_end is not None else 0.8 * rig.dynamic.duration
    tail = t_stored >= end
    rms = {lab: float(np.sqrt(np.mean(tips[tail, c] ** 2))) for c, lab in enumerate(rig.dynamic.tip_labels)} if tail.any() else {}
    settle: Dict[str, Optional[float]] = {}
    for c, lab in enumerate(rig.dynamic.tip_labels):
        band = 0.05 * max(peak[lab], 1e-30)
        after = np.flatnonzero((t_stored >= end) & (np.abs(tips[:, c]) > band))
        # None: still outside the 5 % band at the end of the window, so no settling time can be stated
        settle[lab] = None if after.size and after[-1] == t_stored.size - 1 else (float(t_stored[after[-1]] - end) * 1e3 if after.size else 0.0)
    peak_i = int(np.argmax(np.max(np.abs(tips), axis=1)))
    vm_dyn = np.max(np.stack([s.von_mises(s.expand(tr.u[i]))[solid] for i in np.unique([peak_i, int(np.argmax(tr.energy[tr.stored]))])]), axis=0)
    sigma_dyn = float(vm_dyn.max())
    dynamic = {
        "load_profile": {"name": rig.dynamic.name, "formula": rig.dynamic.formula, "peak_force_N": rig.dynamic.peak_force, "duration_s": rig.dynamic.duration, "load_end_s": end},
        "frequencies_hz": freqs,
        "time_steps": len(times),
        "peak": peak,
        "residual_rms": rms,
        "settling_time_ms": settle,
        "von_mises_max_MPa": sigma_dyn,
        "safety_factor": None if s.yield_stress is None else s.yield_stress / sigma_dyn,
        "response": {"t_ms": [float(v) for v in 1e3 * t_stored], **{lab: [float(v) for v in tips[:, c]] for c, lab in enumerate(rig.dynamic.tip_labels)}},
    }
    base["dynamic"] = dynamic
    return base
