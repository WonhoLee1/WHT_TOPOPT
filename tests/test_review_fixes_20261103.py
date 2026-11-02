"""Regression tests for the defects of docs/references/investigations/code_review_wht_topopt_20261103.md (R1-R10 and section 4)."""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp

from wht_topopt.core.stl_domain import BuiltinAdapter, LoadCase, TopOptEvaluator, TopOptProblem, run_topology_optimization, structured_hex
from wht_topopt.core.time_integration import (
    check_time_step,
    du_olhoff_mass_density,
    du_olhoff_mass_derivative,
    generalized_alpha_lhs,
    generalized_alpha_parameters,
    generalized_alpha_rhs,
    newmark_update,
)


def _sdof_final(n_steps: int, rho_inf: float = 0.85, c: float = 0.0) -> float:
    """u(5T) of m u'' + c u' + k u = 0, m = k = 1, u0 = 1, v0 = 0."""
    am, af, beta, gamma = generalized_alpha_parameters(rho_inf)
    M, C, K = np.array([[1.0]]), np.array([[c]]), np.array([[1.0]])
    dt = 5.0 * 2.0 * np.pi / n_steps
    u, v = np.array([1.0]), np.array([0.0])
    a = -(c * v + u)
    khat = generalized_alpha_lhs(M, C, K, dt, am, af, beta, gamma)
    for _ in range(n_steps):
        r = generalized_alpha_rhs(M, C, K, u, v, a, np.zeros(1), dt, am, af, beta, gamma)
        un = np.linalg.solve(khat, r)
        v, a = newmark_update(u, v, a, un, dt, beta, gamma)
        u = un
    return float(u[0])


def test_r1_generalized_alpha_converges_at_second_order():
    errors = [abs(_sdof_final(n) - 1.0) for n in (100, 400, 1600)]
    assert errors[-1] < 1e-5
    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(4.0)
    assert np.all(orders > 1.8), orders


def test_r1_generalized_alpha_damped_case_matches_the_exact_solution():
    c = 0.1
    wd = np.sqrt(1.0 - (c / 2.0) ** 2)
    t = 5.0 * 2.0 * np.pi
    exact = np.exp(-c / 2.0 * t) * (np.cos(wd * t) + c / (2.0 * wd) * np.sin(wd * t))
    assert _sdof_final(3200, c=c) == pytest.approx(exact, abs=1e-5)


def test_r1_examples_use_the_shared_right_hand_side():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples"
    for name in ("ex09_esl_dynamic_robot_arm.py", "ex09_compare_dynamic_vs_static.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "generalized_alpha_rhs(" in src and "p_term" not in src, name


def _cantilever(**kw) -> TopOptProblem:
    mesh = structured_hex(12, 4, 4)
    n = mesh.nodes
    fixed = (3 * np.flatnonzero(n[:, 0] < 1e-9)[:, None] + np.arange(3)).ravel()
    f = np.zeros(mesh.n_dofs)
    tip = np.flatnonzero((n[:, 0] > 12 - 1e-9) & (n[:, 1] < 1e-9))
    f[3 * tip + 1] = -1.0 / len(tip)
    base = dict(mesh=mesh, fixed_dofs=fixed, load_cases=[LoadCase("tip", f)], volfrac=0.4, rmin=1.5, max_iters=12, tol=0.0)
    base.update(kw)
    return TopOptProblem(**base)


def test_r10_compliance_gradient_matches_finite_differences():
    p = _cantilever()
    ev = TopOptEvaluator(p, BuiltinAdapter())
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.9, p.mesh.n_elems)
    d = rng.standard_normal(x.size)
    h = 1e-6
    fd = (ev.evaluate(x + h * d, gradients=False).compliance - ev.evaluate(x - h * d, gradients=False).compliance) / (2 * h)
    assert ev.evaluate(x).dc @ d == pytest.approx(fd, rel=1e-5)


def test_r2_unknown_optimizer_is_refused():
    with pytest.raises(ValueError, match="unknown optimizer"):
        run_topology_optimization(_cantilever(optimizer="simulated_annealing"), BuiltinAdapter())


def test_r2_each_optimizer_runs_its_own_update():
    # BESO evolves from the full design at a limited rate, so it needs more iterations to reach the target volume
    iters = {"mma": 12, "gcmma": 12, "oc": 12, "beso": 60, "semdot": 12}
    results = {n: run_topology_optimization(_cantilever(optimizer=n, max_iters=k), BuiltinAdapter()) for n, k in iters.items()}
    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(results[a].xphys, results[b].xphys), f"{a} and {b} gave the same design"
    first = run_topology_optimization(_cantilever(optimizer="mma", max_iters=1), BuiltinAdapter()).compliance
    for n, r in results.items():
        assert r.compliance < first, n
        assert abs(r.volume_fraction - 0.4) < 0.05, (n, r.volume_fraction)


def _extrusion_groups(mesh) -> np.ndarray:
    nx, ny, nz = mesh.grid
    iz, iy, ix = np.unravel_index(np.arange(mesh.n_elems), (nz, ny, nx))
    return iz * nx + ix  # extruded along y


@pytest.mark.parametrize("optimizer", ["mma", "gcmma"])
def test_r3_variable_groups_are_enforced(optimizer):
    p = _cantilever(optimizer=optimizer)
    p = replace(p, variable_groups=_extrusion_groups(p.mesh), filter_type="none")
    r = run_topology_optimization(p, BuiltinAdapter())
    nx, ny, nz = p.mesh.grid
    assert r.x.shape == (nx * nz,)
    assert np.ptp(r.xphys.reshape(nz, ny, nx), axis=1).max() == pytest.approx(0.0, abs=1e-12)
    assert r.xphys.std() > 0.05  # it did optimise


def test_r3_x0_per_element_is_reduced_to_the_groups_and_a_wrong_length_is_refused():
    p = _cantilever(optimizer="mma", max_iters=1)
    p = replace(p, variable_groups=_extrusion_groups(p.mesh))
    assert run_topology_optimization(p, BuiltinAdapter(), x0=np.full(p.mesh.n_elems, 0.4)).x.shape == (p.mesh.grid[0] * p.mesh.grid[2],)
    with pytest.raises(ValueError, match="x0 must have shape"):
        run_topology_optimization(p, BuiltinAdapter(), x0=np.full(7, 0.4))


def test_r3_groups_are_refused_by_optimizers_without_group_support():
    p = _cantilever(optimizer="beso")
    with pytest.raises(ValueError, match="variable_groups"):
        run_topology_optimization(replace(p, variable_groups=_extrusion_groups(p.mesh)), BuiltinAdapter())


# ---- R4: the callers hand the HCA a solid-modulus stress


def _source(name: str) -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "examples" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["ex07_ge_bracket_advanced_algorithms.py", "ex11_ge_bracket_hca.py"])
def test_r4_no_penalised_stress_goes_into_the_hca(name):
    assert not re.search(r"vm_all = calculate_von_mises\([^\n]*, E_field,", _source(name))


def test_r4_hca_applies_the_relaxation_once():
    from wht_topopt.algorithms.hca import SEAHCAConfig, StressEnergyAdaptiveHCA

    hca = StressEnergyAdaptiveHCA(np.ones(3), SEAHCAConfig(sigma_allow=1.0))
    rho = np.array([1.0, 0.5, 0.25])
    _, sigma = hca.aggregate_multicase_fields(rho, np.ones(3), np.full(3, 100.0))
    assert sigma == pytest.approx(100.0 * rho**0.5)


# ---- R5: weights follow the strain energy; R7: no static solve of a known answer


def test_r5_r7_ex09_uses_energy_weights_all_instants_and_no_static_solve():
    src = _source("ex09_esl_dynamic_robot_arm.py")
    assert "np.array([0.30, 0.25, 0.30, 0.15])" not in src
    assert "u_peak = u_static_cases[0]" not in src
    assert "self.pardiso_static.solve(K_ff, F_ff)" not in src


# ---- R6: signal smoothing keeps values above one


def test_r6_signal_smoothing_neither_projects_nor_clips():
    from wht_topopt.core.mesh import StructuredMesh3D
    from wht_topopt.manufacturing.extrusion import PlanarRibExtrusionFilter

    mesh = StructuredMesh3D(nelx=8, nely=6, nelz=4, lx=8.0, ly=6.0, lz=4.0)
    flt = PlanarRibExtrusionFilter(mesh, r_min_xy=1.5, bottom_layers=1, side_layers=1, beta=8.0)
    out = flt.smooth_signal(np.full(mesh.n_elem, 7.0))
    assert out == pytest.approx(np.full(mesh.n_elem, 7.0))
    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 50.0, mesh.n_elem)
    assert flt.smooth_signal(s).max() > 1.0


# ---- R8: mass interpolation and time step check


def test_r8_du_olhoff_mass_is_continuous_and_its_derivative_exact():
    rho = np.array([1e-3, 0.05, 0.0999, 0.1, 0.1001, 0.5, 1.0])
    m = du_olhoff_mass_density(rho)
    assert du_olhoff_mass_density(np.array([0.1 - 1e-12])) == pytest.approx(du_olhoff_mass_density(np.array([0.1 + 1e-12])), rel=1e-6)
    assert m[-2:] == pytest.approx(rho[-2:])
    assert m[0] < rho[0] * 1e-3  # far lighter than linear: no spurious low-density modes
    h = 1e-7
    probe = np.array([0.02, 0.08, 0.3, 0.9])
    fd = (du_olhoff_mass_density(probe + h) - du_olhoff_mass_density(probe - h)) / (2 * h)
    assert du_olhoff_mass_derivative(probe) == pytest.approx(fd, rel=1e-6)


def test_r8_time_step_check_warns_only_when_the_mode_is_unresolved():
    K = sp.diags([4.0, 9.0], format="csc")
    M = sp.identity(2, format="csc")  # omega_1 = 2
    assert check_time_step(K, M, dt=0.1) == pytest.approx(0.2)
    with pytest.warns(UserWarning, match="does not resolve the first mode"):
        assert check_time_step(K, M, dt=1.0) == pytest.approx(2.0)


# ---- R9: TET10


def _tet10_nodes() -> np.ndarray:
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]], float)


def test_r9_tet10_mass_matrix_has_full_rank_and_the_right_total_mass():
    from wht_topopt.core.fe_builders import tet10_ke0_me0

    Ke, Me, _ = tet10_ke0_me0(_tet10_nodes(), 0.3)
    assert np.linalg.matrix_rank(Me) == 30
    assert np.min(np.linalg.eigvalsh(Me)) > 0.0
    assert Me[::3, ::3].sum() == pytest.approx(1.0 / 6.0)


def test_r9_gmsh_tet10_order_is_converted():
    gmsh = pytest.importorskip("gmsh")
    from wht_topopt.core.fe_builders import tet10_ke0_me0
    from wht_topopt.core.tetra_mesher import mesh_stl_tetra

    trimesh = pytest.importorskip("trimesh")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        stl = Path(d) / "box.stl"
        trimesh.creation.box((2.0, 2.0, 2.0)).export(str(stl))
        mesh = mesh_stl_tetra(stl, target_elem_size=1.0, order=2)
    assert mesh.elems.shape[1] == 10
    for conn in mesh.elems[:20]:
        c = mesh.nodes[conn]
        assert c[8] == pytest.approx(0.5 * (c[1] + c[3]))  # edge 1-3
        assert c[9] == pytest.approx(0.5 * (c[2] + c[3]))  # edge 2-3
        tet10_ke0_me0(c, 0.3)  # raises on a negative Jacobian


def test_bc_tagging_finds_a_node_inside_a_large_triangle():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("gmsh")
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from wht_topopt.core.tetra_mesher import tag_nodes_near_stl

    with tempfile.TemporaryDirectory() as d:
        stl = Path(d) / "tri.stl"
        trimesh.Trimesh(vertices=[[0, 0, 0], [100, 0, 0], [0, 100, 0]], faces=[[0, 1, 2]], process=False).export(str(stl))
        nodes = np.array([[33.0, 11.0, 0.0], [33.0, 11.0, 5.0]])  # on the triangle, far from every old sample; off the surface
        found = tag_nodes_near_stl(SimpleNamespace(nodes=nodes), stl, tolerance=2.0)
    assert found.tolist() == [0]


# ---- other review items


def test_hca_adapter_returns_the_design_variable_not_the_projected_field():
    from wht_topopt.algorithms.hca import HCAOptimizer

    opt = HCAOptimizer(volfrac_target=0.5)
    args = (np.full(4, 0.5), -np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(1), np.ones((1, 4)))
    opt.update(*args)  # creates the engine
    xi, projected = np.full(4, 0.2), np.full(4, 0.9)
    opt._engine.step = lambda **_: (xi, projected, 1.0)
    assert opt.update(*args) is xi
    assert opt.cfg.sigma_allow == 0.0  # a default in Pa would silently switch the stress term off in an MPa problem


def test_measured_phase_times_come_from_the_run():
    from wht_topopt.core.stl_domain import measured_phase_times

    r = run_topology_optimization(_cantilever(optimizer="mma", max_iters=3), BuiltinAdapter())
    fem, sens, opt = measured_phase_times(r)
    assert fem > 0.0 and sens >= 0.0 and opt > 0.0
    assert _source("ex07_ge_bracket_advanced_algorithms.py").count("* 0.70") == 0


@pytest.mark.parametrize("name", ["ex07_ge_bracket_advanced_algorithms.py", "ex07_se_ahca_signal_study.py"])
def test_ge_studies_pass_the_material(name):
    assert re.search(r"build_stl_problem\(CASE_GE,[^\n]*E0=E_TI_MPA, nu=NU_TI\)", _source(name))
