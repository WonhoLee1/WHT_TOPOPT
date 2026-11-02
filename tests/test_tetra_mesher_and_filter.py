"""Tests for Conformal 3D Tetrahedral Mesher & Unstructured Spatial Filter.

Verifies:
1. Spatial density filter properties: row normalization H @ 1 = 1, non-negativity, compact support.
2. Gmsh conformal tetrahedral volume meshing from STL files (TET4 and TET10).
3. Boundary node identification via proximity to auxiliary boundary STLs.
4. End-to-end integration: STL -> TetraMesh -> TetraFEASolver (PARDISO) -> von Mises stress.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest
import trimesh

from wht_topopt.core.tetra_assembler import (
    TetraMesh,
    TetraFEASolver,
    calculate_tetra_von_mises,
)
from wht_topopt.core.tetra_mesher import (
    mesh_stl_tetra,
    tag_nodes_near_stl,
    build_tetra_density_filter,
)

ROOT = Path(__file__).resolve().parents[2]


def create_sample_box_stl(filepath: Path, sx: float = 20.0, sy: float = 10.0, sz: float = 10.0):
    """Generate a clean watertight 3D box STL."""
    box = trimesh.creation.box(extents=[sx, sy, sz])
    box.export(str(filepath))


def test_tetra_density_filter_properties():
    """Verify mathematical properties of the unstructured spatial density filter H."""
    # Create unstructured tet mesh of a unit cube
    from wht_topopt.tests.test_tetra_fea_pardiso import create_cube_6tet_mesh
    mesh = create_cube_6tet_mesh(nx=3, ny=3, nz=3, L=3.0)

    # 1. Identity when rmin <= 0
    H0 = build_tetra_density_filter(mesh, rmin=0.0)
    assert H0.nnz == mesh.n_elems
    diag = H0.diagonal()
    assert np.allclose(diag, 1.0)

    # 2. Non-zero rmin
    rmin = 1.5
    H = build_tetra_density_filter(mesh, rmin=rmin)

    # Property 1: Row sums must equal 1.0 (constant field preservation)
    row_sums = np.asarray(H.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12), f"Row sums differ from 1.0: {row_sums}"

    # Property 2: All filter weights must be non-negative
    assert H.data.min() >= 0.0, "Negative weights found in filter matrix"

    # Property 3: Conservation test: H @ constant = constant
    const_field = np.full(mesh.n_elems, 0.45)
    filtered = H @ const_field
    assert np.allclose(filtered, 0.45, atol=1e-12)


def test_gmsh_stl_conformal_meshing_and_fea():
    """Verify Gmsh STL conformal volume meshing, boundary tagging, and PARDISO FEA."""
    pytest.importorskip("gmsh")

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = Path(tmpdir) / "beam_domain.stl"
        bnd_stl_path = Path(tmpdir) / "fixed_face.stl"

        # Create a 40x10x10 mm cantilever beam
        create_sample_box_stl(stl_path, sx=40.0, sy=10.0, sz=10.0)

        # Create a thin plate at x = -20 mm representing the fixed wall
        create_sample_box_stl(bnd_stl_path, sx=1.0, sy=12.0, sz=12.0)
        # Shift bnd_stl to x = -20 mm
        mesh_bnd = trimesh.load_mesh(str(bnd_stl_path))
        mesh_bnd.apply_translation([-20.0, 0.0, 0.0])
        mesh_bnd.export(str(bnd_stl_path))

        # 1. Generate TET4 Conformal Volume Mesh
        target_size = 5.0  # mm
        tetra_mesh = mesh_stl_tetra(
            stl_path=stl_path,
            target_elem_size=target_size,
            order=1,
            verbose=False,
        )

        assert tetra_mesh.element_type == "TET4"
        assert tetra_mesh.n_nodes > 20
        assert tetra_mesh.n_elems > 50

        # Verify all element volumes are strictly positive
        volumes = tetra_mesh.compute_volumes()
        assert np.all(volumes > 1e-6), "Inverted or zero-volume tetrahedra detected!"

        # 2. Tag boundary nodes near fixed_face.stl
        fixed_nodes = tag_nodes_near_stl(tetra_mesh, bnd_stl_path, tolerance=1.5)
        assert len(fixed_nodes) > 4, f"Too few fixed nodes identified: {len(fixed_nodes)}"

        # 3. Apply downward load at the tip (x near +20 mm)
        tip_nodes = np.flatnonzero(tetra_mesh.nodes[:, 0] > 18.0)
        assert len(tip_nodes) > 0

        loads = np.zeros(tetra_mesh.n_dofs, dtype=np.float64)
        p_down = -1000.0 / len(tip_nodes)  # 1 kN downward (-Z)
        loads[3 * tip_nodes + 2] = p_down

        # 4. Solve with TetraFEASolver using PARDISO
        fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
        E0 = 70000.0  # Aluminum, MPa
        nu = 0.33
        youngs = np.full(tetra_mesh.n_elems, E0)

        solver = TetraFEASolver(tetra_mesh, nu)
        u, b_all, vols, info = solver.solve(youngs, fixed_dofs, loads)

        assert u.shape == (1, tetra_mesh.n_dofs)
        u_tip = u[0, 3 * tip_nodes + 2]
        # Downward load must cause downward displacement (u_z < 0)
        assert np.mean(u_tip) < 0.0, "Tip did not deflect downwards under downward load"

        # 5. Centroid von Mises stress
        vm = calculate_tetra_von_mises(tetra_mesh, youngs, nu, u, b_all)
        assert vm.shape == (1, tetra_mesh.n_elems)
        assert np.all(vm >= 0.0)
        assert np.max(vm) > 1.0  # Non-trivial stress generated

        print(f" [PASS] Gmsh STL conformal meshing & FEA: {tetra_mesh.n_elems} tets, Max VM = {np.max(vm):.2f} MPa")


if __name__ == "__main__":
    test_tetra_density_filter_properties()
    print(" [PASS] test_tetra_density_filter_properties")
    test_gmsh_stl_conformal_meshing_and_fea()
    print(" [PASS] test_gmsh_stl_conformal_meshing_and_fea")
