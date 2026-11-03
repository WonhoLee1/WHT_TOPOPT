"""examples/src_v1 2D: the problem definition, the output layout and a smoke run of every ex00 example."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh2D
from wht_topopt.examples.src_v1.common.pillar1_algorithms import ALGORITHMS, LoadSet, run_algorithms
from wht_topopt.examples.src_v1.common.pillar5_audit import TestRig, binary_design, load_path_connected
from wht_topopt.examples.src_v1.common.structure import Structure
from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.ex00_2d_basic import ex00a_cantilever, ex00b_lbracket, ex00c_mbb

EXAMPLES = [ex00a_cantilever, ex00b_lbracket, ex00c_mbb]


def test_the_material_has_no_default():
    params = inspect.signature(Structure.__init__).parameters
    for name in ("e0", "nu", "density", "yield_stress"):
        assert params[name].default is inspect.Parameter.empty, name


def test_bad_material_is_refused():
    mesh = StructuredMesh2D(nelx=4, nely=2, lx=4.0, ly=2.0)
    for e0, nu in ((0.0, 0.3), (1.0, 0.6)):
        with pytest.raises(ValueError):
            Structure(mesh, np.arange(4), e0, nu, None, None)


def test_a_pillar_neither_run_nor_declared_is_refused():
    st = ex00c_mbb.build(nelx=30, nely=10)
    with pytest.raises(ValueError, match="pillar 2"):
        Study(name="x", family="y", structure=st.structure, loads=st.loads, rig=st.rig, volfrac=0.4, rmin=2.4)


def test_passive_solid_above_the_target_is_refused():
    st = ex00c_mbb.build(nelx=30, nely=10)
    s = Structure(st.structure.mesh, st.structure.fixed_dofs, 1.0, 0.3, None, None, passive_solid=np.arange(200))
    with pytest.raises(ValueError, match="passive solid alone"):
        Study(name="x", family="y", structure=s, loads=st.loads, rig=st.rig, volfrac=0.4, rmin=2.4, not_applicable={2: "a", 3: "b", 4: "c"})


def test_an_unknown_algorithm_is_refused():
    st = ex00c_mbb.build(nelx=30, nely=10)
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_algorithms(st.structure, 2.4, st.loads, 0.4, ["simulated_annealing"])


def test_the_load_path_check_wants_a_part_held_in_every_direction():
    st = ex00c_mbb.build(nelx=30, nely=10)
    s, f = st.structure, st.loads.forces
    c = s.centroids
    assert load_path_connected(s, np.ones(s.n_elem), f)
    assert not load_path_connected(s, (c[:, 0] < 10.0).astype(float), f)  # touches only the symmetry plane: held in x alone
    assert not load_path_connected(s, (c[:, 0] > 10.0).astype(float), f)  # the load node is not on the solid


def test_binary_design_meets_the_volume_and_keeps_the_passive_regions():
    s = ex00b_lbracket.build(nel=30).structure
    xb = binary_design(s, np.random.default_rng(0).uniform(0.0, 1.0, s.n_elem), 0.4)
    assert s.volume_fraction(xb) == pytest.approx(0.4, abs=2.0 / np.sum(~s.void_mask))
    assert np.all(xb[s.solid_mask] == 1.0) and np.all(xb[s.void_mask] == 0.0)


@pytest.mark.parametrize("module", EXAMPLES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_example_runs_every_method_and_writes_the_layout(module, tmp_path):
    m = main(module.build, module.FULL, module.SMOKE, ["--smoke", "-o", str(tmp_path)])
    out = tmp_path / m["family"] / m["example"]
    assert m["algorithms"] == list(ALGORITHMS) and m["smoke"] is True
    name = m["example"]
    for rel in m["files"]:
        assert (out / rel).stat().st_size > 0, rel
    for rel in (f"pillar1_algorithms/{name}_progression.gif", f"pillar1_algorithms/{name}_algorithms.png", f"pillar5_audit/{name}_cross_stiffness_metrics.json",
                f"pillar6_deliverables/{name}_summary.md", f"pillar6_deliverables/{name}_comprehensive_master_report.html"):
        assert rel in m["files"]
    for p in ("2", "3", "4"):
        assert m["pillars"][p]["status"] == "not_applicable" and m["pillars"][p]["reason"]
    for p in ("1", "5", "6"):
        assert m["pillars"][p]["status"] == "done"
    metrics = json.loads((out / f"pillar5_audit/{name}_cross_stiffness_metrics.json").read_text(encoding="utf-8"))
    assert metrics["conditions"]["E0"] == 1.0 and len(metrics["algorithms"]) == len(ALGORITHMS)
    for a in metrics["algorithms"].values():
        assert np.isfinite(a["static"]["compliance_mJ"]) and a["static"]["von_mises_max_MPa"] > 0.0 and a["dynamic"] is None


def test_l_bracket_void_stays_void_in_every_method():
    st = ex00b_lbracket.build(nel=30)
    s = st.structure
    runs, _ = run_algorithms(s, st.rmin, st.loads, st.volfrac, ALGORITHMS, 6)
    for r in runs:
        assert np.all(r.rho[s.void_mask] <= 1e-3 + 1e-12), r.label
        assert np.all(r.rho[s.solid_mask] == pytest.approx(1.0)), r.label
        if r.label != "beso":  # BESO comes down from the full domain at a limited rate
            assert s.volume_fraction(r.rho) == pytest.approx(st.volfrac, abs=0.12), r.label
