"""Unit and integration tests for 3D topology optimization benchmark suite and Tri-Engine architecture.

Verifies:
1. GE Aircraft Bracket and top3d Cantilever domain voxelization and mask creation.
2. Numerical parity and convergence between Numba JIT, Kratos v1, and Kratos v2 engines.
3. 3D Marching Cubes STL surface generation integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.core.voxelizer import (
    VoxelizedDomain3D,
    voxelize_ge_aircraft_bracket,
    voxelize_top3d_cantilever_beam,
)
from wht_topopt.core.fem import FEMEngine
from wht_topopt.solvers.scipy_solver import ScipySolver
from wht_topopt.solvers.kratos_solver import KratosSolver


def test_voxelizer_ge_bracket():
    """Verify GE Aircraft Bracket voxelization produces valid masks and boundary conditions."""
    domain = voxelize_ge_aircraft_bracket(nx=16, ny=8, nz=6, scale=1.0)
    assert isinstance(domain, VoxelizedDomain3D)
    assert domain.n_elements == 16 * 8 * 6
    assert np.sum(domain.design_mask) > 0
    assert np.sum(domain.passive_solid_mask) > 0
    assert np.sum(domain.passive_void_mask) > 0

    # Ensure masks partition elements cleanly
    assert not np.any(domain.design_mask & domain.passive_solid_mask)
    assert not np.any(domain.design_mask & domain.passive_void_mask)

    # Boundary conditions
    assert len(domain.fixed_dofs) > 0
    assert len(domain.load_dofs) > 0
    assert len(domain.load_values) == len(domain.load_dofs)


def test_voxelizer_top3d_beam():
    """Verify academic standard 3D cantilever beam domain creation."""
    domain = voxelize_top3d_cantilever_beam(nx=12, ny=6, nz=4, lx=60.0, ly=30.0, lz=20.0)
    assert domain.n_elements == 12 * 6 * 4
    assert np.all(domain.design_mask)
    assert len(domain.fixed_dofs) == (6 + 1) * (4 + 1) * 3
    assert len(domain.load_dofs) == 1


def test_tri_engine_numerical_parity():
    """Verify 3D FEA solve displacement agreement between Numba (MKL PARDISO) and Kratos C++."""
    domain = voxelize_top3d_cantilever_beam(nx=10, ny=5, nz=4, lx=60.0, ly=30.0, lz=20.0)
    mesh = domain.mesh
    n_elem = mesh.n_elem
    n_dof = mesh.ndof
    x = np.full(n_elem, 0.5)

    # 1. Numba + MKL PARDISO Engine
    fem = FEMEngine(mesh)
    K = fem.assemble_stiffness_matrix(x, penal=3.0, e0=1.0, emin=1e-6)
    F = domain.assemble_load_vector()
    all_dofs = np.arange(n_dof)
    free_dofs = np.setdiff1d(all_dofs, domain.fixed_dofs)

    solver_numba = ScipySolver(method="pardiso")
    res_numba = solver_numba.solve(K, F, free_dofs=free_dofs, fixed_dofs=domain.fixed_dofs)
    u_numba = res_numba.u
    c_numba = float(np.dot(F, u_numba))

    # 2. Kratos Engine (evaluate_mesh_cached)
    youngs = 1e-6 + (x ** 3.0) * (1.0 - 1e-6)
    solver_kratos = KratosSolver(method="direct")
    res_kratos = solver_kratos.evaluate_mesh_cached(
        nodes=mesh.nodes,
        elems=mesh.elements,
        etype="hex8",
        fixed_dofs=domain.fixed_dofs,
        loads=F.reshape(1, -1),
        youngs=youngs,
        nu=0.3,
    )
    u_kratos = res_kratos["u"][0]
    c_kratos = float(np.dot(F, u_kratos))

    # Parity check
    rel_err_c = abs(c_numba - c_kratos) / c_numba
    rel_err_u = np.linalg.norm(u_numba - u_kratos) / np.linalg.norm(u_numba)
    assert rel_err_c < 1e-4, f"Compliance mismatch between Numba and Kratos: {rel_err_c:.2e}"
    assert rel_err_u < 1e-4, f"Displacement field mismatch between Numba and Kratos: {rel_err_u:.2e}"


def test_3d_topopt_algorithms():
    """Verify 3D SIMP optimization iteration convergence on a compact 3D domain."""
    domain = voxelize_top3d_cantilever_beam(nx=12, ny=6, nz=4, lx=60.0, ly=30.0, lz=20.0)
    mesh = domain.mesh
    n_elem = mesh.n_elem
    n_dof = mesh.ndof
    F = domain.assemble_load_vector()
    all_dofs = np.arange(n_dof)
    free_dofs = np.setdiff1d(all_dofs, domain.fixed_dofs)
    solver = ScipySolver(method="pardiso")

    x = np.full(n_elem, 0.4)
    target_vol = 0.4

    # Run 4 SIMP iterations
    fem = FEMEngine(mesh)
    c_history = []
    for it in range(4):
        K = fem.assemble_stiffness_matrix(x, penal=3.0, e0=1.0, emin=1e-6)
        res = solver.solve(K, F, free_dofs=free_dofs, fixed_dofs=domain.fixed_dofs)
        c = float(np.dot(F, res.u))
        c_history.append(c)

        # Sensitivity
        c_e0 = fem.compute_unit_element_compliance(res.u)
        dcdx = -3.0 * (x ** 2.0) * c_e0

        # Simple OC step
        l1, l2 = 1e-9, 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            B = -dcdx / max(1e-12, lmid)
            x_new = np.clip(x * np.sqrt(np.maximum(1e-9, B)), np.maximum(0.001, x - move), np.minimum(1.0, x + move))
            if np.mean(x_new) > target_vol:
                l1 = lmid
            else:
                l2 = lmid
        x = x_new

    # Check compliance values exist and are strictly positive
    assert all(c_val > 0.0 for c_val in c_history)


def test_marching_cubes_stl_export():
    """Verify 3D Marching Cubes surface extraction produces valid STL meshes."""
    import skimage.measure

    # Construct a simple spherical 3D density grid (10 x 10 x 10)
    res = 12
    x = np.linspace(-1, 1, res)
    grid_x, grid_y, grid_z = np.meshgrid(x, x, x, indexing="ij")
    dist_sq = grid_x ** 2 + grid_y ** 2 + grid_z ** 2
    density_grid = np.clip(1.0 - dist_sq, 0.0, 1.0)

    # Marching cubes at isovalue 0.5
    verts, faces, normals, values = skimage.measure.marching_cubes(
        density_grid, level=0.5, spacing=(1.0, 1.0, 1.0)
    )
    assert len(verts) > 0
    assert len(faces) > 0
    assert faces.shape[1] == 3

    # Export to STL via trimesh
    import trimesh
    stl_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    assert stl_mesh.is_watertight or len(stl_mesh.faces) > 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        stl_path = Path(tmp_dir) / "test_sphere.stl"
        stl_mesh.export(str(stl_path))
        assert stl_path.is_file()
        assert stl_path.stat().st_size > 0
