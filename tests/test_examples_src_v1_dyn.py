"""examples/src_v1/ex00_dyn against independent values: the load limits, a known signal for the peak picker, modes of a known design, a smoke run."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh2D
from wht_topopt.examples.src_v1.common.pillar2_static_dynamic import esl_peak_instants, time_grid
from wht_topopt.examples.src_v1.common.pillar5_audit import drop_detached_solid
from wht_topopt.examples.src_v1.common.structure import Structure
from wht_topopt.examples.src_v1.common.study import main
from wht_topopt.examples.src_v1.ex00_dyn import _common as dyn
from wht_topopt.examples.src_v1.ex00_dyn import ex00_dyn_a_cantilever, ex00_dyn_b_lbracket, ex00_dyn_c_mbb

EXAMPLES = [ex00_dyn_a_cantilever, ex00_dyn_b_lbracket, ex00_dyn_c_mbb]


def _small_cantilever():
    return ex00_dyn_a_cantilever.build(nelx=30, nely=10)


def test_load_stays_in_the_limits_and_reaches_them():
    st = _small_cantilever()
    load = st.rig.dynamic
    static = st.loads.forces[0]
    total = float(np.abs(static).sum())
    t = np.linspace(load.window_start, load.duration, 20001)
    f = np.array([load.force_at(float(v)).reshape(-1, 2).sum(axis=0) for v in t])
    mag = np.hypot(f[:, 0], f[:, 1]) / total
    angle = np.degrees(np.arctan2(f[:, 0], -f[:, 1]))
    assert mag.min() == pytest.approx(0.5, abs=1e-3) and mag.max() == pytest.approx(1.5, abs=1e-3)
    assert angle.min() == pytest.approx(-45.0, abs=0.1) and angle.max() == pytest.approx(45.0, abs=0.1)
    assert load.duration - load.window_start == pytest.approx(20.0)


def test_load_starts_from_zero_on_the_static_nodes():
    st = _small_cantilever()
    load = st.rig.dynamic
    assert np.all(load.force_at(0.0) == 0.0)
    static_y = st.loads.forces[0][1::2]
    f = load.force_at(7.3)
    assert np.array_equal(np.flatnonzero(f[1::2]), np.flatnonzero(static_y))
    assert np.array_equal(np.flatnonzero(f[0::2]), np.flatnonzero(static_y))
    ratio = f[1::2][static_y != 0.0] / static_y[static_y != 0.0]
    assert np.allclose(ratio, ratio[0])  # the nodal pattern of the static load is kept


def test_a_static_load_off_the_y_axis_is_refused():
    st = _small_cantilever()
    bad = st.loads.forces[0].copy()
    bad[0] = 0.1
    with pytest.raises(ValueError, match="-y"):
        dyn.load_history(st.structure, bad)


def test_peak_picker_finds_the_known_peaks():
    """Six known energy humps; the peak of Ux lies on a small hump, so it must be picked ahead of bigger ones."""
    t = np.linspace(0.0, 22.0, 1101)
    centres = np.array([4.0, 7.0, 10.0, 13.0, 16.0, 19.0])
    heights = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 0.5])
    energy = sum(h * np.exp(-((t - c) / 0.2) ** 2) for h, c in zip(heights, centres))
    tips = np.column_stack([np.exp(-((t - 19.0) / 0.2) ** 2), -energy])
    got = t[esl_peak_instants(t, energy, tips, n=5, min_separation=0.5, start=2.0)]
    assert got == pytest.approx([7.0, 10.0, 13.0, 16.0, 19.0], abs=0.02)


def test_peak_picker_skips_the_ramp_and_keeps_the_separation():
    t = np.linspace(0.0, 22.0, 1101)
    energy = 100.0 * np.exp(-((t - 1.0) / 0.1) ** 2) + (1.0 + 0.5 * np.sin(2.0 * np.pi * t)) ** 2
    idx = esl_peak_instants(t, energy, np.zeros((t.size, 1)), n=5, min_separation=0.5, start=2.0)
    assert t[idx].min() >= 2.0 and idx.size == 5
    assert np.diff(t[idx]).min() >= 0.5


def test_peak_picker_refuses_to_pad():
    t = np.linspace(0.0, 22.0, 1101)
    energy = np.exp(-((t - 10.0) / 0.5) ** 2)  # one peak only
    with pytest.raises(ValueError, match="separate response peaks"):
        esl_peak_instants(t, energy, np.zeros((t.size, 1)), n=5, min_separation=0.5, start=2.0)


def test_density_sets_the_first_frequency():
    st = _small_cantilever()
    s = st.structure
    dmap = st.make_map()
    rho = np.maximum(dmap.forward(dmap.fix_passive(np.full(s.n_elem, st.volfrac))), 1e-3)
    assert s.natural_frequencies(s.stiffness(rho), s.mass(rho), k=1)[0] == pytest.approx(1.0, rel=1e-3)


def test_rayleigh_gives_the_damping_ratio_at_both_frequencies():
    alpha, beta = dyn.rayleigh_for(0.02, 1.0, 5.0)
    for f in (1.0, 5.0):
        w = 2.0 * np.pi * f
        assert alpha / (2.0 * w) + beta * w / 2.0 == pytest.approx(0.02, rel=1e-12)


def test_time_step_follows_the_load_not_only_the_first_mode():
    st = _small_cantilever()
    s, load = st.structure, st.rig.dynamic
    rho = np.full(s.n_elem, st.volfrac)
    times, notes = time_grid(s, rho, load)
    assert times[1] - times[0] <= load.dt_max + 1e-12 and notes == []
    capped, notes = time_grid(s, rho, load, max_steps=100)
    assert capped.size == 100 and any("load's limit" in n for n in notes)


def test_full_mbb_carries_the_x_load_where_the_half_model_cannot():
    st = ex00_dyn_c_mbb.build(nelx=48, nely=8)
    s, load = st.structure, st.rig.dynamic
    assert np.count_nonzero(s.reduce(load.force_at(7.3))) == 2  # both load components reach free dofs
    fx = np.zeros(s.ndof)
    fx[0::2] = -st.loads.forces[0][1::2]
    u = s.solve_static(s.stiffness(np.ones(s.n_elem)), fx)[0]
    assert abs(load.tip_response(u)[0]) > 0.0
    # the supports hold the rigid motions: pin (2) + roller (1)
    assert s.fixed_dofs.size == 3


def test_detached_solid_is_removed_and_its_spurious_mode_goes_with_it():
    """A horizontal bar on a cantilever root, plus a solid island: the island on void stiffness has a near-zero frequency."""
    mesh = StructuredMesh2D(nelx=40, nely=12, lx=40.0, ly=12.0)
    left = np.flatnonzero(mesh.nodes[:, 0] < 1e-9)
    s = Structure(mesh, np.concatenate([2 * left, 2 * left + 1]), 1.0, 0.3, 1.0, None, emin_ratio=1e-9)
    c = s.centroids
    bar = (c[:, 1] > 4.0) & (c[:, 1] < 8.0)
    island = (c[:, 0] > 20.0) & (c[:, 0] < 24.0) & (c[:, 1] < 3.0)
    xb = (bar | island).astype(float)
    tip = np.flatnonzero((mesh.nodes[:, 0] > 40.0 - 1e-9) & (np.abs(mesh.nodes[:, 1] - 6.0) < 1e-9))
    force = np.zeros(mesh.ndof)
    force[2 * tip + 1] = -1.0
    before = s.natural_frequencies(s.stiffness(xb), s.mass(xb), k=1)[0]
    out, removed = drop_detached_solid(s, xb, force)
    assert removed == int(island.sum())
    assert np.array_equal(out > 0.5, bar)
    after = s.natural_frequencies(s.stiffness(out), s.mass(out), k=1)[0]
    inertia, area = 4.0**3 / 12.0, 4.0
    theory = (1.875104**2 / (2.0 * np.pi)) * np.sqrt(inertia / (area * 40.0**4))  # Euler-Bernoulli bar, E = density = 1
    assert before < 0.1 * theory
    assert after == pytest.approx(theory, rel=0.1)


def test_detached_removal_keeps_a_connected_design_unchanged():
    st = _small_cantilever()
    s = st.structure
    xb = np.ones(s.n_elem)
    out, removed = drop_detached_solid(s, xb, st.loads.forces[0])
    assert removed == 0 and np.array_equal(out, xb)


@pytest.mark.parametrize("module", EXAMPLES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_smoke_run_writes_the_layout(module, tmp_path):
    m = main(module.build, module.FULL, module.SMOKE, ["--smoke", "-o", str(tmp_path)], default_algos=dyn.DEFAULT_ALGOS)
    assert m["family"] == "ex00-dyn" and m["example"].startswith("ex00-dyn-")
    assert m["algorithms"] == ["mma"] and m["smoke"] is True
    out = tmp_path / m["family"] / m["example"]
    name = m["example"]
    for rel in m["files"]:
        assert (out / rel).stat().st_size > 0, rel
    for rel in (f"pillar5_audit/{name}_transient_response.png", f"pillar5_audit/{name}_load_history_peaks.png", f"pillar2_static_dynamic/{name}_static_vs_dynamic.png",
                f"pillar6_deliverables/{name}_summary.md", f"pillar6_deliverables/{name}_comprehensive_master_report.html"):
        assert rel in m["files"]
    assert m["pillars"]["2"]["status"] == "done"
    assert m["pillars"]["3"]["status"] == "not_applicable" and m["pillars"]["4"]["status"] == "not_applicable"
    metrics = json.loads((out / f"pillar5_audit/{name}_cross_stiffness_metrics.json").read_text(encoding="utf-8"))
    peaks = metrics["timing"]["model2_dynamic"]["esl_peaks_last_iteration"]
    assert len(peaks) == 5 and all(q["t_s"] >= 2.0 for q in peaks)
    assert sum(q["weight"] for q in peaks) == pytest.approx(1.0)
    for model in ("model1_static", "model2_dynamic"):
        d = metrics["models"][model]["dynamic"]
        assert set(d["peak"]) == {"Ux", "Uy"} and d["residual_rms"] == {} and d["settling_time_ms"] == {}
        assert d["peak"]["Ux"] > 0.0
        assert len(d["frequencies_hz"]) == 3
