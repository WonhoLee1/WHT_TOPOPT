"""examples/src_v1 3D modules against independent values: beam theory, finite differences, a closed-form transient, the output layout."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.examples.src_v1.common.pillar1_algorithms import LoadSet, run_algorithms, static_response, Timers
from wht_topopt.examples.src_v1.common.pillar2_static_dynamic import LoadHistory, critical_instants
from wht_topopt.examples.src_v1.common.pillar3_volume_relaxed import dominance_table
from wht_topopt.examples.src_v1.common.pillar4_manufacturing import DensityMap, ExtrusionMap, RibMap, SkinMap, skin_elements
from wht_topopt.examples.src_v1.common.pillar5_audit import binary_design, load_path_connected
from wht_topopt.examples.src_v1.common.pillar6_deliverables import pillar_status
from wht_topopt.examples.src_v1.common.structure import Structure, steps_for_first_mode

E, NU, DENSITY = 1000.0, 0.3, 1e-6
L, B, H = 40.0, 4.0, 4.0


def _beam(nx=40, ny=4, nz=4, **kw) -> Structure:
    mesh = StructuredMesh3D(nelx=nx, nely=ny, nelz=nz, lx=L, ly=B, lz=H)
    left = np.flatnonzero(mesh.node_coords[:, 0] < 1e-9)
    return Structure(mesh, (3 * left[:, None] + np.arange(3)).ravel(), E, NU, DENSITY, 100.0, **kw)


def _tip_force(s: Structure, axis: int = 1, amount: float = 1.0) -> np.ndarray:
    tip = np.flatnonzero(s.nodes[:, 0] > L - 1e-9)
    f = np.zeros(s.ndof)
    f[3 * tip + axis] = amount / tip.size
    return f


def test_static_deflection_and_frequency_match_beam_theory():
    s = _beam()
    rho = np.ones(s.n_elem)
    K, M = s.stiffness(rho), s.mass(rho)
    u = s.solve_static(K, _tip_force(s))
    tip = np.flatnonzero(s.nodes[:, 0] > L - 1e-9)
    inertia = B * H**3 / 12.0
    assert np.mean(u[0, 3 * tip + 1]) == pytest.approx(L**3 / (3.0 * E * inertia), rel=0.08)
    f1 = s.natural_frequencies(K, M, k=2)[0]
    area = B * H
    theory = (1.875104**2 / (2.0 * np.pi)) * np.sqrt(E * inertia / (DENSITY * area * L**4))
    assert f1 == pytest.approx(theory, rel=0.08)


def test_stress_of_a_uniform_bar_is_the_applied_stress():
    mesh = StructuredMesh3D(nelx=10, nely=2, nelz=2, lx=10.0, ly=2.0, lz=2.0)
    n = mesh.node_coords
    left, right = np.flatnonzero(n[:, 0] < 1e-9), np.flatnonzero(n[:, 0] > 10.0 - 1e-9)
    fixed = np.concatenate([3 * left, 3 * np.flatnonzero((n[:, 1] < 1e-9) & (n[:, 2] < 1e-9)) + 1, 3 * np.flatnonzero(n[:, 2] < 1e-9) + 2, 3 * np.flatnonzero(n[:, 1] < 1e-9) + 1])
    s = Structure(mesh, np.unique(fixed), E, 0.0, DENSITY, 100.0)
    # consistent nodal forces of a uniform traction on the right face: uniform stress 5 needs the corner-weighted pattern
    f = np.zeros(s.ndof)
    y, z = n[right, 1], n[right, 2]
    weight = (1.0 - 0.5 * ((y < 1e-9) | (y > 2.0 - 1e-9))) * (1.0 - 0.5 * ((z < 1e-9) | (z > 2.0 - 1e-9)))
    f[3 * right] = 5.0 * 4.0 * weight / weight.sum()
    u = s.solve_static(s.stiffness(np.ones(s.n_elem)), f)[0]
    assert s.von_mises(u) == pytest.approx(np.full(s.n_elem, 5.0), rel=1e-6)


def test_transient_follows_the_quasi_static_limit():
    s = _beam(nx=20, ny=2, nz=2, rayleigh=(0.0, 0.0))
    rho = np.ones(s.n_elem)
    K, M = s.stiffness(rho), s.mass(rho)
    f = _tip_force(s)
    f1 = s.natural_frequencies(K, M, k=1)[0]
    period = 1.0 / f1
    # a load rising slowly over many periods: the response is the static one
    ramp = 40.0 * period
    times = np.linspace(0.0, ramp, int(ramp / (period / 25.0)) + 1)
    tr = s.transient(K, M, lambda t: f * (t / ramp), times, store_every=len(times) - 1)
    full_end = s.expand(tr.u[-1])
    full_static = s.solve_static(K, f)[0]
    tip = np.flatnonzero(s.nodes[:, 0] > L - 1e-9)
    assert np.mean(full_end[3 * tip + 1]) == pytest.approx(np.mean(full_static[3 * tip + 1]), rel=0.02)


def test_the_transient_refuses_a_load_at_t_zero():
    s = _beam(nx=10, ny=2, nz=2)
    K, M = s.stiffness(np.ones(s.n_elem)), s.mass(np.ones(s.n_elem))
    f = _tip_force(s)
    with pytest.raises(ValueError, match="from rest"):
        s.transient(K, M, lambda t: f, np.linspace(0.0, 1.0, 5))


def test_time_step_rule_reports_a_cap():
    n, notes = steps_for_first_mode(100.0, 0.05, limit=0.5)
    assert n == int(np.ceil(0.05 * 2.0 * np.pi * 100.0 / 0.5)) + 1 and not notes
    n, notes = steps_for_first_mode(100.0, 0.05, limit=0.5, max_steps=10)
    assert n == 10 and "above the limit" in notes[0]


@pytest.mark.parametrize("make", [lambda s: DensityMap(s, 2.0), lambda s: DensityMap(s, 2.0, beta0=2.0, beta_max=2.0), lambda s: ExtrusionMap(s, 2.0, "y"), lambda s: ExtrusionMap(s, 2.0, "z"),
                                  lambda s: SkinMap(s, 2.0, np.arange(8)), lambda s: RibMap(s, 2.0)], ids=["density", "projection", "extrude_y", "extrude_z", "skin", "rib"])
def test_design_maps_have_exact_gradients(make):
    s = _beam(nx=12, ny=6, nz=4)
    dmap = make(s)
    dmap.set_iteration(1)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.8, s.n_elem)
    d = rng.standard_normal(s.n_elem)
    loads = LoadSet(_tip_force(s), np.ones(1), ["tip"])
    h = 1e-6

    def compliance(xv):
        return static_response(s, np.maximum(dmap.forward(xv), 1e-3), loads, Timers())

    rho = dmap.forward(x)
    _, ce, _ = compliance(x)
    rho = dmap.forward(x)
    grad = dmap.backward(-s.dyoung(rho) * ce)
    fd = (compliance(x + h * d)[0] - compliance(x - h * d)[0]) / (2.0 * h)
    assert grad @ d == pytest.approx(fd, rel=1e-4)


def test_extrusion_gives_one_density_per_column():
    s = _beam(nx=12, ny=6, nz=4)
    dmap = ExtrusionMap(s, 0.5, "y")
    rho = dmap.forward(np.random.default_rng(1).uniform(0.0, 1.0, s.n_elem)).reshape(4, 6, 12)
    assert np.ptp(rho, axis=1).max() == pytest.approx(0.0, abs=1e-12)


def test_skin_elements_select_the_exposed_layers():
    inside = np.ones(5 * 4 * 3, dtype=bool)
    assert skin_elements(inside, (5, 4, 3), (1.0, 1.0, 1.0), 1.0, ["z-"]).tolist() == list(range(20))
    assert skin_elements(inside, (5, 4, 3), (1.0, 1.0, 1.0), 2.0, ["z+"]).tolist() == list(range(20, 60))
    with pytest.raises(ValueError):
        skin_elements(inside, (5, 4, 3), (1.0, 1.0, 1.0), 1.0, ["up"])


def test_rigid_pin_transfers_the_load():
    from wht_topopt.core.stl_domain import RigidBody

    mesh = StructuredMesh3D(nelx=10, nely=2, nelz=2, lx=10.0, ly=2.0, lz=2.0)

    class WithRef:  # a grid with two extra reference nodes, as the STL problems have
        elements = mesh.elements
        nodes = np.vstack([mesh.node_coords, [[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]]])
        grid = (10, 2, 2)

    n = WithRef.nodes
    ref, rot = n.shape[0] - 2, n.shape[0] - 1
    right = np.flatnonzero(mesh.node_coords[:, 0] > 10.0 - 1e-9)
    left = np.flatnonzero(mesh.node_coords[:, 0] < 1e-9)
    s = Structure(WithRef, (3 * left[:, None] + np.arange(3)).ravel(), E, NU, DENSITY, 100.0, rigid_bodies=[RigidBody(right, ref, rot)])
    f = np.zeros(s.ndof)
    f[3 * ref] = 8.0
    u = s.solve_static(s.stiffness(np.ones(s.n_elem)), f)[0]
    assert u[3 * ref] == pytest.approx(8.0 * 10.0 / (E * 4.0), rel=0.05)  # F L / (E A)
    assert u[3 * right] == pytest.approx(np.full(right.size, u[3 * ref]))  # the slaves follow


def test_audit_helpers():
    s = _beam(nx=12, ny=4, nz=4, passive_solid=np.arange(4))
    x = binary_design(s, np.random.default_rng(0).uniform(0.0, 1.0, s.n_elem), 0.3)
    assert s.volume_fraction(x) == pytest.approx(0.3, abs=1.0 / s.n_elem)
    assert np.all(x[:4] == 1.0)
    f = _tip_force(s)
    c = s.centroids
    assert load_path_connected(s, np.ones(s.n_elem), f)
    assert not load_path_connected(s, (c[:, 0] < 6.0 * L / 12.0).astype(float), f)  # the loaded end is not on the solid


def test_every_pillar_must_be_accounted_for():
    assert pillar_status([1, 5, 6], {2: "a", 3: "b", 4: "c"})["2"]["status"] == "not_applicable"
    assert pillar_status([1, 2, 3, 4, 5, 6], {}, {1: "kratos"})["1"]["status"] == "partial"
    with pytest.raises(ValueError, match="pillar 2"):
        pillar_status([1, 3, 4, 5, 6], {})


def test_dominance_table_judges_the_spec_targets():
    m = lambda k, u, f: {"static": {"stiffness": {"K_y": k}}, "dynamic": {"peak": {"Uz": u}, "frequencies_hz": [f]}}
    rows = dominance_table(m(100.0, 10.0, 300.0), m(90.0, 5.0, 320.0), m(125.0, 2.0, 330.0), "K_y", "Uz")
    assert [r["passed"] for r in rows] == [True, True, True]
    rows = dominance_table(m(100.0, 10.0, 300.0), m(90.0, 5.0, 320.0), m(110.0, 5.0, 310.0), "K_y", "Uz")
    assert [r["passed"] for r in rows] == [False, False, False]
    assert all(isinstance(r["passed"], bool) for r in rows)


def test_algorithms_refuse_unknown_and_record_kratos():
    s = _beam(nx=12, ny=4, nz=4)
    loads = LoadSet(_tip_force(s), np.ones(1), ["tip"])
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_algorithms(s, 2.0, loads, 0.4, ["simulated_annealing"])
    runs, skipped = run_algorithms(s, 2.0, loads, 0.4, ["mma", "semdot", "se_ahca"], 3)
    assert [r.label for r in runs] == ["mma", "semdot", "se_ahca"]
    assert set(skipped) == {"kratos_mma", "kratos_gcmma", "kratos_semdot", "kratos_beso"} and all(skipped.values())
    assert s.penal == 3.0 and s.emin == pytest.approx(E * 1e-6)  # SEMDOT's temporary model does not leak


@pytest.mark.parametrize("modname", ["ex09_robot_arm.ex09_robot_arm", "ex07_ge_bracket.ex07_ge_bracket"])
def test_full_study_writes_the_layout(modname, tmp_path):
    import importlib
    from pathlib import Path

    mod = importlib.import_module(f"wht_topopt.examples.src_v1.{modname}")
    if "ex07" in modname and not (Path(mod.ROOT) / "external" / "freeto_stls" / "GE_domain.STL").exists():
        pytest.skip("FreeTO STL files are not available")
    from wht_topopt.examples.src_v1.common.study import main

    m = main(mod.build, mod.FULL, mod.SMOKE, ["--smoke", "--algos", "mma", "se_ahca", "-o", str(tmp_path)])
    out = tmp_path / m["family"] / m["example"]
    assert m["smoke"] is True and {"model1_static", "model2_dynamic", "model3_relaxed"} <= set(m["models"])
    for rel in m["files"]:
        assert (out / rel).stat().st_size > 0, rel
    assert m["pillars"]["1"]["status"] == "partial" and "Kratos" in m["pillars"]["1"]["reason"]
    metrics = json.loads((out / f"pillar5_audit/{m['example']}_cross_stiffness_metrics.json").read_text(encoding="utf-8"))
    three = [metrics["models"][k] for k in ("model1_static", "model2_dynamic", "model3_relaxed")]
    assert three[2]["mass_g"] > three[0]["mass_g"]  # Model 3 has the relaxed volume
    for a in three:
        assert a["dynamic"]["frequencies_hz"][0] > 0.0 and a["static"]["safety_factor"] > 0.0
    html = (out / f"pillar6_deliverables/{m['example']}_comprehensive_master_report.html").read_text(encoding="utf-8")
    for needle in ("정적 강성 평가표", "동적 성능 평가표", "내부 단면 절단", "__models"):
        assert needle in html
