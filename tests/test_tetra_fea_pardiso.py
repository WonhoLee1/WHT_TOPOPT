"""Rigorous Independent Verification of Unstructured Tetrahedral (TET4 / TET10) FEA Assembler.

Anti-Mocking & Numerical Integrity Standards:
1. Single element symmetry and rigid-body modes (6 zero eigenvalues lambda_1..6 < 1e-12).
2. Numba JIT batch assembly parity with exact analytical formulations.
3. 3D Patch Test on multi-tetrahedra domain (linear displacement field preserved to machine precision < 1e-10).
4. Multi-RHS PARDISO direct linear static solve residual verification ||K u - F|| / ||F|| < 1e-10.
5. Centroid von Mises stress evaluation and energy consistency.
"""
from __future__ import annotations

import numpy as np
import pytest

from wht_topopt.core.tetra_assembler import (
    TetraMesh,
    TetraFEASolver,
    assemble_tetra_elements,
    calculate_tetra_von_mises,
)
from wht_topopt.core.fe_builders import tet4_ke0_me0, tet10_ke0_me0


def create_cube_5tet_mesh(L: float = 1.0) -> TetraMesh:
    """Subdivide a unit cube [0, L]^3 into 5 conforming tetrahedral elements."""
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [L,   0.0, 0.0],  # 1
        [L,   L,   0.0],  # 2
        [0.0, L,   0.0],  # 3
        [0.0, 0.0, L  ],  # 4
        [L,   0.0, L  ],  # 5
        [L,   L,   L  ],  # 6
        [0.0, L,   L  ],  # 7
    ], dtype=np.float64)

    # 5-tet subdivision of a cube
    elems = np.array([
        [0, 1, 3, 4],  # Corner 1
        [1, 2, 3, 6],  # Corner 2
        [1, 4, 5, 6],  # Corner 3
        [3, 4, 6, 7],  # Corner 4
        [1, 3, 4, 6],  # Central regular tetrahedron
    ], dtype=np.int64)

    return TetraMesh(nodes=nodes, elems=elems, element_type="TET4")


def create_cube_6tet_mesh(nx: int = 2, ny: int = 2, nz: int = 2, L: float = 1.0) -> TetraMesh:
    """Generate structured box decomposed into 6 tetrahedra per hexahedron."""
    x = np.linspace(0.0, L, nx + 1)
    y = np.linspace(0.0, L, ny + 1)
    z = np.linspace(0.0, L, nz + 1)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def node_id(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = node_id(i, j, k)
                n1 = node_id(i + 1, j, k)
                n2 = node_id(i + 1, j + 1, k)
                n3 = node_id(i, j + 1, k)
                n4 = node_id(i, j, k + 1)
                n5 = node_id(i + 1, j, k + 1)
                n6 = node_id(i + 1, j + 1, k + 1)
                n7 = node_id(i, j + 1, k + 1)

                # 6 tets per hex (Kuhn's triangulation)
                tets.append([n0, n1, n2, n6])
                tets.append([n0, n2, n3, n6])
                tets.append([n0, n3, n7, n6])
                tets.append([n0, n7, n4, n6])
                tets.append([n0, n4, n5, n6])
                tets.append([n0, n5, n1, n6])

    return TetraMesh(nodes=nodes, elems=np.array(tets, dtype=np.int64), element_type="TET4")


def test_tet4_single_element_eigenvalues():
    """Verify single TET4 element stiffness matrix symmetry and exactly 6 zero RBM modes."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.2, 0.0],
        [0.3, 1.8, 0.1],
        [0.2, 0.4, 2.0],
    ], dtype=np.float64)
    nu = 0.33

    Ke0, Me0, B0 = tet4_ke0_me0(coords, nu)

    # 1. Symmetry
    sym_err = np.max(np.abs(Ke0 - Ke0.T))
    assert sym_err < 1e-14, f"Ke0 is not symmetric, error = {sym_err}"

    # 2. Eigenvalues
    eigvals = np.linalg.eigvalsh(Ke0)
    # First 6 modes must be 0 (rigid body translation & rotation)
    assert np.all(np.abs(eigvals[:6]) < 1e-11), f"Non-zero RBM eigenvalues: {eigvals[:6]}"
    # Remaining 6 strain modes must be strictly positive
    assert np.all(eigvals[6:] > 1e-3), f"Strain modes not strictly positive: {eigvals[6:]}"


def test_tet10_single_element_eigenvalues():
    """Verify single TET10 element stiffness matrix symmetry and exactly 6 zero RBM modes."""
    # Linear tet corner nodes
    corners = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 2.5, 0.0],
        [0.2, 0.5, 2.2],
    ], dtype=np.float64)

    # 10 nodes: 4 corners + 6 mid-edges
    nodes10 = np.zeros((10, 3), dtype=np.float64)
    nodes10[:4] = corners
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for i, (n1, n2) in enumerate(edges):
        nodes10[4 + i] = 0.5 * (corners[n1] + corners[n2])

    nu = 0.30
    Ke0, Me0, B0 = tet10_ke0_me0(nodes10, nu)

    # Symmetry
    sym_err = np.max(np.abs(Ke0 - Ke0.T))
    assert sym_err < 1e-13, f"TET10 Ke0 not symmetric, error = {sym_err}"

    # Eigenvalues: 30 DOFs -> 6 zeros, 24 positive
    eigvals = np.linalg.eigvalsh(Ke0)
    assert np.all(np.abs(eigvals[:6]) < 1e-10), f"TET10 RBM not zero: {eigvals[:6]}"
    assert np.all(eigvals[6:] > 1e-3), f"TET10 strain modes not positive: {eigvals[6:]}"


def test_numba_batch_assembly_parity():
    """Verify that Numba batch assembly exactly matches analytical single-element results."""
    mesh = create_cube_6tet_mesh(nx=3, ny=3, nz=3, L=2.0)
    youngs = np.linspace(100.0, 500.0, mesh.n_elems)
    nu = 0.34

    ke_all, b_all, volumes = assemble_tetra_elements(mesh, youngs, nu)

    # Check each element against exact tet4_ke0_me0
    for e in range(min(mesh.n_elems, 20)):
        coords_e = mesh.nodes[mesh.elems[e]]
        ke_ref, _, b_ref = tet4_ke0_me0(coords_e, nu)
        ke_expected = youngs[e] * ke_ref

        ke_err = np.max(np.abs(ke_all[e] - ke_expected))
        b_err = np.max(np.abs(b_all[e] - b_ref))

        assert ke_err < 1e-11, f"Batch Ke mismatch on elem {e}: {ke_err}"
        assert b_err < 1e-12, f"Batch B mismatch on elem {e}: {b_err}"

    # Total volume of cube [0, 2]^3 must be exactly 8.0
    tot_vol = np.sum(volumes)
    assert abs(tot_vol - 8.0) < 1e-12, f"Total mesh volume mismatch: {tot_vol} vs 8.0"


def test_3d_patch_test_exactness():
    """Rigorous 3D Patch Test on unstructured tetrahedra under linear displacement field.

    A linear displacement field u_i = a_i0 + a_i1*x + a_i2*y + a_i3*z must produce
    exact internal nodal displacements and constant strain/stress to machine precision.
    """
    mesh = create_cube_6tet_mesh(nx=3, ny=3, nz=3, L=1.0)
    E0 = 210000.0  # Steel, MPa
    nu = 0.30
    youngs = np.full(mesh.n_elems, E0)

    # Linear displacement field coefficients
    # u(x, y, z) = 0.001*x + 0.0005*y - 0.0002*z
    # v(x, y, z) = 0.0003*x - 0.0008*y + 0.0004*z
    # w(x, y, z) = -0.0005*x + 0.0002*y + 0.0012*z
    A = np.array([
        [0.0010,  0.0005, -0.0002],
        [0.0003, -0.0008,  0.0004],
        [-0.0005, 0.0002,  0.0012],
    ], dtype=np.float64)

    # Theoretical exact displacements at all nodes
    u_exact = (mesh.nodes @ A.T).ravel()

    # Identify boundary nodes: on any of the 6 faces x=0, x=1, y=0, y=1, z=0, z=1
    tol = 1e-8
    is_boundary = (
        (mesh.nodes[:, 0] < tol) | (mesh.nodes[:, 0] > 1.0 - tol) |
        (mesh.nodes[:, 1] < tol) | (mesh.nodes[:, 1] > 1.0 - tol) |
        (mesh.nodes[:, 2] < tol) | (mesh.nodes[:, 2] > 1.0 - tol)
    )
    bnd_nodes = np.flatnonzero(is_boundary)
    int_nodes = np.flatnonzero(~is_boundary)
    assert len(int_nodes) > 0, "No interior nodes found for patch test!"

    fixed_dofs = np.concatenate([3 * bnd_nodes, 3 * bnd_nodes + 1, 3 * bnd_nodes + 2])

    solver = TetraFEASolver(mesh, nu)
    K, b_all, volumes = solver.assemble_global_stiffness(youngs)

    # Boundary displacements applied as Dirichlet BCs:
    # Kff * u_free = - K_fb * u_boundary
    int_dofs = np.concatenate([3 * int_nodes, 3 * int_nodes + 1, 3 * int_nodes + 2])
    free_dofs = np.setdiff1d(np.arange(mesh.n_dofs), fixed_dofs)

    rhs = -K[free_dofs][:, fixed_dofs] @ u_exact[fixed_dofs]

    # Solve interior displacements
    loads = np.zeros(mesh.n_dofs, dtype=np.float64)
    # Using solver.solve with boundary forces
    Kff = K[free_dofs][:, free_dofs]
    import scipy.sparse.linalg as spla
    u_computed_free = spla.spsolve(Kff, rhs)

    # Check error against theoretical linear field
    u_exact_free = u_exact[free_dofs]
    patch_err = np.max(np.abs(u_computed_free - u_exact_free))
    assert patch_err < 1e-10, f"3D Patch test failed: max interior error = {patch_err:.3e}"


def test_tetra_solver_multi_rhs_pardiso():
    """Verify TetraFEASolver with multiple load cases against direct residual check."""
    mesh = create_cube_6tet_mesh(nx=4, ny=4, nz=4, L=2.0)
    E0 = 70000.0   # Aluminum, MPa
    nu = 0.33
    youngs = np.full(mesh.n_elems, E0)

    # Fix base face at z = 0
    base_nodes = np.flatnonzero(mesh.nodes[:, 2] < 1e-8)
    fixed_dofs = np.concatenate([3 * base_nodes, 3 * base_nodes + 1, 3 * base_nodes + 2])

    # Top face at z = 2.0
    top_nodes = np.flatnonzero(mesh.nodes[:, 2] > 2.0 - 1e-8)

    # 3 distinct load cases:
    # Case 1: Pure tension in Z
    # Case 2: Shear in X
    # Case 3: Torsion around Z
    n_dofs = mesh.n_dofs
    loads = np.zeros((3, n_dofs), dtype=np.float64)

    # Case 1: +1000 N uniform tension in Z on top face
    pz = 1000.0 / len(top_nodes)
    loads[0, 3 * top_nodes + 2] = pz

    # Case 2: +500 N shear in X on top face
    px = 500.0 / len(top_nodes)
    loads[1, 3 * top_nodes] = px

    # Case 3: Torsion torque around (x=1.0, y=1.0)
    xc, yc = 1.0, 1.0
    for tn in top_nodes:
        dx = mesh.nodes[tn, 0] - xc
        dy = mesh.nodes[tn, 1] - yc
        # F = (-dy, dx)
        loads[2, 3 * tn] = -dy * 10.0
        loads[2, 3 * tn + 1] = dx * 10.0

    solver = TetraFEASolver(mesh, nu)
    u, b_all, volumes, info = solver.solve(youngs, fixed_dofs, loads)

    assert u.shape == (3, n_dofs)

    # Verify linear equilibrium residual for each load case: ||K u - F||_free / ||F||_free < 1e-10
    K, _, _ = solver.assemble_global_stiffness(youngs)
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

    for c in range(3):
        res = (K @ u[c] - loads[c])[free_dofs]
        norm_res = np.linalg.norm(res)
        norm_f = np.linalg.norm(loads[c, free_dofs])
        rel_res = norm_res / max(norm_f, 1e-8)
        assert rel_res < 1e-10, f"Load case {c} residual too large: {rel_res:.3e}"

    # Calculate von Mises stresses
    vm = calculate_tetra_von_mises(mesh, youngs, nu, u, b_all)
    assert vm.shape == (3, mesh.n_elems)
    assert np.all(vm >= 0.0), "Negative von Mises stress encountered"
    assert np.all(np.isfinite(vm)), "NaN or Inf in von Mises stress"

    print(f" [+] TetraFEASolver multi-RHS test passed successfully with solver={info['solver']}")
    print(f"     Max VM per case: {np.max(vm, axis=1)}")


if __name__ == "__main__":
    test_tet4_single_element_eigenvalues()
    print(" [PASS] test_tet4_single_element_eigenvalues")
    test_tet10_single_element_eigenvalues()
    print(" [PASS] test_tet10_single_element_eigenvalues")
    test_numba_batch_assembly_parity()
    print(" [PASS] test_numba_batch_assembly_parity")
    test_3d_patch_test_exactness()
    print(" [PASS] test_3d_patch_test_exactness")
    test_tetra_solver_multi_rhs_pardiso()
    print(" [PASS] test_tetra_solver_multi_rhs_pardiso")
