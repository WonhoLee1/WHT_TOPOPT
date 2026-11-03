"""SEMDOT on structured Hex8 meshes: trilinear sub-grid, volume matching, passive regions, boundary extraction, driver loop."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from wht_topopt.algorithms.semdot import SEMDOTOptimizer
from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.core.stl_domain import BuiltinAdapter, LoadCase, TopOptEvaluator, TopOptProblem, run_topology_optimization, structured_hex


def _coords(mesh):
    return mesh.node_coords if hasattr(mesh, "node_coords") else mesh.nodes


def _conn(mesh):
    return mesh.elements if hasattr(mesh, "elements") else mesh.elems


@pytest.mark.parametrize("mesh", [StructuredMesh3D(nelx=4, nely=3, nelz=2, lx=8.0, ly=3.0, lz=1.0), structured_hex(4, 3, 2, size=(2.0, 1.0, 0.5))],
                         ids=["StructuredMesh3D", "stl_hex"])
def test_samples_are_the_trilinear_interpolant_of_the_nodal_field(mesh):
    opt = SEMDOTOptimizer(mesh=mesh, n_sub=3)
    conn, coords = _conn(mesh), _coords(mesh)
    n_elem = conn.shape[0]
    rho = np.random.default_rng(0).uniform(0.0, 1.0, n_elem)
    active = np.ones(n_elem, dtype=bool)
    pts = np.array([0.0, 0.5, 1.0])
    samples = opt._trilinear_samples(rho, pts, active, np.zeros(n_elem, dtype=bool))
    nodal = opt._rho_nodes
    # by-hand interpolation from the coordinates, independent of the node numbering
    size = opt._hex["size"]
    for e in (0, n_elem // 2, n_elem - 1):
        lo = coords[conn[e]].min(axis=0)
        k = 0
        for z in pts:
            for y in pts:
                for x in pts:
                    expected = 0.0
                    for node in conn[e]:
                        t = (coords[node] - lo) / size
                        w = (x if t[0] > 0.5 else 1 - x) * (y if t[1] > 0.5 else 1 - y) * (z if t[2] > 0.5 else 1 - z)
                        expected += w * nodal[node]
                    assert samples[e, k] == pytest.approx(expected)
                    k += 1


def test_boundary_is_resolved_inside_the_elements():
    mesh = structured_hex(6, 2, 2)
    opt = SEMDOTOptimizer(mesh=mesh, volfrac=0.5, n_sub=4, beta_init=8.0)
    rho = np.where(mesh.centroids()[:, 0] < 3.0, 1.0, 0.0)
    psi = opt.project_filtered(rho, target_volfrac=0.5)
    assert psi.mean() == pytest.approx(0.5, abs=1e-3)
    gray = psi[(psi > 0.05) & (psi < 0.95)]
    assert gray.size > 0  # the elements at the cut are partly solid: the cut lies inside them
    assert np.all(psi[mesh.centroids()[:, 0] < 1.0] > 0.95) and np.all(psi[mesh.centroids()[:, 0] > 5.0] < 0.05)


def test_passive_regions():
    mesh = structured_hex(6, 3, 3)
    c = mesh.centroids()
    void = np.flatnonzero(c[:, 0] > 4.0)
    solid = np.flatnonzero(c[:, 0] < 1.0)
    opt = SEMDOTOptimizer(mesh=mesh, volfrac=0.5, n_sub=3, beta_init=4.0)
    rho = np.full(mesh.n_elems, 0.5)
    psi = opt.project_filtered(rho, target_volfrac=0.5, passive_solid=solid, passive_void=void)
    assert psi[solid] == pytest.approx(1.0)
    assert psi[void] == pytest.approx(opt.rho_min)
    assert psi.mean() == pytest.approx(0.5, abs=1e-3)
    # the nodal density next to the void region is not pulled down by the void elements
    nodes_on_void_face = np.flatnonzero(np.abs(mesh.nodes[:, 0] - 4.0) < 1e-9)
    assert opt._rho_nodes[nodes_on_void_face] == pytest.approx(0.5)


def test_update_uses_the_trilinear_grid_and_keeps_the_volume():
    mesh = StructuredMesh3D(nelx=6, nely=3, nelz=3, lx=6.0, ly=3.0, lz=3.0)
    opt = SEMDOTOptimizer(mesh=mesh, volfrac=0.4, n_sub=3, beta_init=4.0)
    rng = np.random.default_rng(0)
    out = opt.update(np.full(mesh.n_elem, 0.4), -rng.uniform(0.1, 1.0, mesh.n_elem), np.zeros(1), np.ones(mesh.n_elem) / mesh.n_elem)
    assert out.mean() == pytest.approx(0.4, abs=1e-3)
    assert opt._rho_nodes is not None


def test_a_mesh_without_structure_warns():
    opt = SEMDOTOptimizer(mesh=None, volfrac=0.5, n_sub=2)
    with pytest.warns(UserWarning, match="no boundary is resolved"):
        opt.project_filtered(np.full(8, 0.5), target_volfrac=0.5)


def _cantilever(nx: int = 24, ny: int = 8, nz: int = 4, **kw) -> TopOptProblem:
    mesh = structured_hex(nx, ny, nz)
    n = mesh.nodes
    fixed = (3 * np.flatnonzero(n[:, 0] < 1e-9)[:, None] + np.arange(3)).ravel()
    f = np.zeros(mesh.n_dofs)
    tip = np.flatnonzero((n[:, 0] > nx - 1e-9) & (np.abs(n[:, 1] - ny / 2) < 1.0 + 1e-9))
    f[3 * tip + 1] = -1.0 / len(tip)
    c = mesh.centroids()
    interface = np.flatnonzero((c[:, 0] > nx - 2.0) & (np.abs(c[:, 1] - ny / 2) < 2.0))  # a thin SEMDOT member would leave load nodes on void
    base = dict(mesh=mesh, fixed_dofs=fixed, load_cases=[LoadCase("tip", f)], volfrac=0.3, rmin=2.0, optimizer="semdot", max_iters=40, tol=0.0,
                passive_solid=interface)
    base.update(kw)
    return TopOptProblem(**base)


def _linear_compliance(p: TopOptProblem, x: np.ndarray) -> float:
    """Compliance of a density field with the linear modulus. Not a 0/1 cut: on this coarse grid a cut's connectivity decides
    the value by orders of magnitude and flips with solver round-off."""
    q = replace(p, penalty=1.0, emin_ratio=1e-6, filter_type="none", rmin=0.0, projection_beta=0.0, optimizer="mma")
    return TopOptEvaluator(q, BuiltinAdapter()).evaluate(x, gradients=False).compliance


def test_driver_runs_real_semdot_and_extracts_a_closed_boundary():
    p = _cantilever(36, 12, 6)  # four elements through the thickness are too coarse: the compliance then rises with beta
    r = run_topology_optimization(p, BuiltinAdapter())
    assert abs(r.volume_fraction - 0.3) < 2e-3
    h = [row["compliance"] for row in r.history]
    assert h[-1] < h[0]
    assert float(np.mean(r.xphys * (1.0 - r.xphys))) < 0.1  # sharper than the uniform start (0.21)
    mma = run_topology_optimization(replace(p, optimizer="mma"), BuiltinAdapter())
    assert _linear_compliance(p, r.xphys) < 1.5 * _linear_compliance(p, mma.xphys)  # a real design, in the range of MMA's
    opt = r.extras["semdot"]
    nx, ny, nz = p.mesh.grid
    assert opt.fine_density_grid().shape == (nz * 4 + 1, ny * 4 + 1, nx * 4 + 1)
    assert abs(float(np.mean(opt.extract_grid())) - 0.3) < 0.05  # the boundary encloses about the target volume
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("skimage")
    verts, faces = opt.extract_isosurface()
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    assert tm.is_watertight
    assert abs(abs(tm.volume) - 0.3 * 36 * 12 * 6) / (0.3 * 36 * 12 * 6) < 0.15
    fine_cell = 1.0 / opt.n_sub
    assert verts.min() >= -fine_cell and np.all(verts.max(axis=0) <= np.array([36.0, 12.0, 6.0]) + fine_cell)


def test_driver_respects_passive_regions():
    p = _cantilever(max_iters=5)
    c = p.mesh.centroids()
    void = np.flatnonzero((c[:, 0] > 10.0) & (c[:, 0] < 14.0) & (c[:, 1] > 6.0))
    solid = np.flatnonzero(c[:, 0] < 1.0)
    r = run_topology_optimization(replace(p, passive_void=void, passive_solid=solid), BuiltinAdapter())
    assert r.xphys[solid] == pytest.approx(1.0)
    assert np.all(r.xphys[void] == 0.0)
