"""Rigorous numerical integrity test for Milestone 1 Part A (wht_topopt.core).

Performs:
1. Import verification
2. Rigid Body Mode (RBM) and symmetry eigenvalue analysis for all elements
3. Finite Difference gradient checks for analytical derivatives (tol < 1e-7)
4. Adjoint transpose consistency tests for filter operators
5. Volume-preserving root-finding verification
6. FEM global assembly, compliance, and stress evaluations
"""

from __future__ import annotations

import numpy as np

from wht_topopt.core import (
    StructuredMesh2D,
    StructuredMesh3D,
    UnstructuredMesh,
    isotropic_elasticity_2d,
    isotropic_elasticity_3d,
    tri3_ke0_me0,
    q4_ke0_me0,
    tet4_ke0_me0,
    tet10_ke0_me0,
    hex8_ke0_me0,
    hex20_ke0_me0,
    mitc3_ke0_me0,
    mitc4_ke0_me0,
    FEMEngine,
    DensityFilter,
    SensitivityFilter,
    HelmholtzPDEFilter,
    SIMP,
    ClassicSIMP,
    RAMP,
    MultiMaterialSIMP,
    HeavisideProjection,
    ContinuationScheduler,
    BoundaryConditionManager,
)


def test_meshes():
    m2d = StructuredMesh2D(nx=10, ny=5, lx=2.0, ly=1.0)
    assert m2d.n_elem == 50
    assert m2d.n_nodes == 66
    assert m2d.ndof == 132
    assert m2d.elements.shape == (50, 4)
    assert m2d.edofMat.shape == (50, 8)
    assert np.isclose(np.sum(m2d.elem_volumes), 2.0)

    left_nodes = m2d.get_nodes_at_x(0.0)
    assert len(left_nodes) == 6
    right_nodes = m2d.get_nodes_at_x(2.0)
    assert len(right_nodes) == 6
    bottom_nodes = m2d.get_nodes_at_y(0.0)
    assert len(bottom_nodes) == 11

    u2d = m2d.to_unstructured()
    assert u2d.n_elem == 50
    assert u2d.ndof == 132
    assert np.isclose(np.sum(u2d.elem_volumes), 2.0)

    m3d = StructuredMesh3D(nx=4, ny=3, nz=2, lx=4.0, ly=3.0, lz=2.0)
    assert m3d.n_elem == 24
    assert m3d.n_nodes == 60
    assert m3d.ndof == 180
    assert m3d.elements.shape == (24, 8)
    assert m3d.edofMat.shape == (24, 24)
    assert np.isclose(np.sum(m3d.elem_volumes), 24.0)

    u3d = m3d.to_unstructured()
    assert u3d.n_elem == 24
    assert np.isclose(np.sum(u3d.elem_volumes), 24.0)


def test_element_eigenvalues():
    c_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ke_tri, me_tri, _ = tri3_ke0_me0(c_tri, nu=0.3)
    assert np.max(np.abs(ke_tri - ke_tri.T)) < 1e-12
    assert np.max(np.abs(me_tri - me_tri.T)) < 1e-12
    w_tri = np.linalg.eigvalsh(ke_tri)
    assert np.sum(np.abs(w_tri) < 1e-10 * np.max(w_tri)) == 3

    c_q4 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for form in ["FI", "BBAR"]:
        ke_q4, me_q4, _ = q4_ke0_me0(c_q4, nu=0.3, formulation=form)
        assert np.max(np.abs(ke_q4 - ke_q4.T)) < 1e-12
        assert np.max(np.abs(me_q4 - me_q4.T)) < 1e-12
        w_q4 = np.linalg.eigvalsh(ke_q4)
        assert np.sum(np.abs(w_q4) < 1e-10 * np.max(w_q4)) == 3

    c_tet4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ke_tet4, me_tet4, _ = tet4_ke0_me0(c_tet4, nu=0.3)
    assert np.max(np.abs(ke_tet4 - ke_tet4.T)) < 1e-12
    assert np.max(np.abs(me_tet4 - me_tet4.T)) < 1e-12
    w_tet4 = np.linalg.eigvalsh(ke_tet4)
    assert np.sum(np.abs(w_tet4) < 1e-10 * np.max(w_tet4)) == 6

    c_tet10 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    ])
    ke_tet10, me_tet10, _ = tet10_ke0_me0(c_tet10, nu=0.3)
    assert np.max(np.abs(ke_tet10 - ke_tet10.T)) < 1e-12
    assert np.max(np.abs(me_tet10 - me_tet10.T)) < 1e-12
    w_tet10 = np.linalg.eigvalsh(ke_tet10)
    assert np.sum(np.abs(w_tet10) < 1e-10 * np.max(w_tet10)) == 6

    c_hex8 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    for form in ["FI", "BBAR"]:
        ke_hex8, me_hex8, _ = hex8_ke0_me0(c_hex8, nu=0.3, formulation=form)
        assert np.max(np.abs(ke_hex8 - ke_hex8.T)) < 1e-12
        assert np.max(np.abs(me_hex8 - me_hex8.T)) < 1e-12
        w_hex8 = np.linalg.eigvalsh(ke_hex8)
        assert np.sum(np.abs(w_hex8) < 1e-10 * np.max(w_hex8)) == 6

    c_hex20 = np.zeros((20, 3))
    from wht_topopt.core.fe_builders import _HEX20_CORNER_SIGNS, _HEX20_MIDSIDE_NODES
    for i in range(8):
        c_hex20[i] = 0.5 * (_HEX20_CORNER_SIGNS[i] + 1.0)
    for idx, xi, eta, zeta in _HEX20_MIDSIDE_NODES:
        c_hex20[idx] = 0.5 * (np.array([xi, eta, zeta]) + 1.0)
    ke_hex20, me_hex20, _ = hex20_ke0_me0(c_hex20, nu=0.3)
    assert np.max(np.abs(ke_hex20 - ke_hex20.T)) < 1e-12
    assert np.max(np.abs(me_hex20 - me_hex20.T)) < 1e-12
    w_hex20 = np.linalg.eigvalsh(ke_hex20)
    assert np.sum(np.abs(w_hex20) < 1e-10 * np.max(w_hex20)) == 6

    c_shell3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ke_m3, _ = mitc3_ke0_me0(c_shell3, nu=0.3, thickness=0.1)
    assert np.max(np.abs(ke_m3 - ke_m3.T)) < 1e-12
    w_m3 = np.linalg.eigvalsh(ke_m3)
    assert np.sum(np.abs(w_m3) < 1e-10 * np.max(w_m3)) == 6

    c_shell4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    ke_m4, _ = mitc4_ke0_me0(c_shell4, nu=0.3, thickness=0.1)
    assert np.max(np.abs(ke_m4 - ke_m4.T)) < 1e-12
    w_m4 = np.linalg.eigvalsh(ke_m4)
    assert np.sum(np.abs(w_m4) < 1e-10 * np.max(w_m4)) == 6


def test_derivatives_finite_diff():
    h = 1e-6
    test_points = np.linspace(0.05, 0.95, 19)

    simp = SIMP(E0=2.1, Emin=1e-8, penal=3.0)
    for rho in test_points:
        d_ana = simp.derivative(np.array([rho]))[0]
        d_fd = (simp.evaluate(np.array([rho + h]))[0] - simp.evaluate(np.array([rho - h]))[0]) / (2.0 * h)
        rel_err = abs(d_ana - d_fd) / (abs(d_ana) + 1e-12)
        assert rel_err < 1e-7

    csimp = ClassicSIMP(E0=2.1, Emin=1e-8, penal=3.0)
    for rho in test_points:
        d_ana = csimp.derivative(np.array([rho]))[0]
        d_fd = (csimp.evaluate(np.array([rho + h]))[0] - csimp.evaluate(np.array([rho - h]))[0]) / (2.0 * h)
        rel_err = abs(d_ana - d_fd) / (abs(d_ana) + 1e-12)
        assert rel_err < 1e-7

    ramp = RAMP(E0=2.1, Emin=1e-8, q=4.0)
    for rho in test_points:
        d_ana = ramp.derivative(np.array([rho]))[0]
        d_fd = (ramp.evaluate(np.array([rho + h]))[0] - ramp.evaluate(np.array([rho - h]))[0]) / (2.0 * h)
        rel_err = abs(d_ana - d_fd) / (abs(d_ana) + 1e-12)
        assert rel_err < 1e-7

    mmsimp = MultiMaterialSIMP(E_phases=[1.0, 3.0, 7.0], Emin=1e-8, penal=3.0)
    for rho in [0.3, 0.7, 1.2, 1.8, 2.3, 2.7]:
        d_ana = mmsimp.derivative(np.array([rho]))[0]
        d_fd = (mmsimp.evaluate(np.array([rho + h]))[0] - mmsimp.evaluate(np.array([rho - h]))[0]) / (2.0 * h)
        rel_err = abs(d_ana - d_fd) / (abs(d_ana) + 1e-12)
        assert rel_err < 1e-7

    rho_vec = np.array([[0.2, 0.5, 0.3]])
    d_ana_vec = mmsimp.derivative(rho_vec)[0]
    for m in range(3):
        e_plus = rho_vec.copy()
        e_plus[0, m] += h
        e_minus = rho_vec.copy()
        e_minus[0, m] -= h
        d_fd_m = (mmsimp.evaluate(e_plus)[0] - mmsimp.evaluate(e_minus)[0]) / (2.0 * h)
        rel_err = abs(d_ana_vec[m] - d_fd_m) / (abs(d_ana_vec[m]) + 1e-12)
        assert rel_err < 1e-7

    for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
        proj = HeavisideProjection(beta=beta, eta=0.45)
        for rho in test_points:
            d_ana = proj.derivative(np.array([rho]))[0]
            d_fd = (proj.forward(np.array([rho + h]))[0] - proj.forward(np.array([rho - h]))[0]) / (2.0 * h)
            err = abs(d_ana - d_fd) / abs(d_ana) if abs(d_ana) > 1e-4 else abs(d_ana - d_fd)
            assert err < 1e-7

    proj = HeavisideProjection(beta=16.0, eta=0.5)
    rng = np.random.default_rng(42)
    rho_test = rng.uniform(0.1, 0.9, size=100)
    target_v = 0.4
    eta_star = proj.find_volume_preserving_eta(rho_test, target_vol=target_v, tol=1e-10)
    achieved_v = np.mean(proj.forward(rho_test, eta=eta_star))
    assert abs(achieved_v - target_v) < 1e-9


def test_filters():
    m2d = StructuredMesh2D(nx=12, ny=6, lx=1.2, ly=0.6)
    n_elem = m2d.n_elem

    rng = np.random.default_rng(1234)
    x = rng.uniform(0.1, 0.9, size=n_elem)
    y = rng.uniform(0.1, 0.9, size=n_elem)

    d_filt = DensityFilter(m2d, r_min=0.25, volume_weighted=True)
    ones = np.ones(n_elem)
    assert np.max(np.abs(d_filt.forward(ones) - 1.0)) < 1e-12
    assert abs(np.dot(d_filt.forward(x), y) - np.dot(x, d_filt.backward(y))) < 1e-12

    pde_filt = HelmholtzPDEFilter(m2d, r_min=0.25)
    assert np.max(np.abs(pde_filt.forward(ones) - 1.0)) < 1e-4
    dot1_pde = np.dot(pde_filt.forward(x), y)
    dot2_pde = np.dot(x, pde_filt.backward(y))
    assert abs(dot1_pde - dot2_pde) / abs(dot1_pde) < 1e-12


def test_fem_engine_and_bcs():
    m2d = StructuredMesh2D(nx=20, ny=10, lx=2.0, ly=1.0)
    fem = FEMEngine(m2d, nu=0.3, thickness=1.0, formulation="BBAR")
    K = fem.assemble_K(np.ones(m2d.n_elem))
    assert K.shape == (m2d.ndof, m2d.ndof)

    bcm = BoundaryConditionManager.cantilever_2d(m2d, F_mag=1.0)
    free = bcm.free_dofs
    F = bcm.F[:, 0]

    import scipy.sparse.linalg as spla
    K_free = K[free, :][:, free]
    u = np.zeros(m2d.ndof)
    u[free] = spla.spsolve(K_free, F[free])

    c_elem = fem.compute_unit_element_compliance(u)
    assert np.isclose(np.sum(c_elem), np.dot(F, u), rtol=1e-10)

    cauchy, vm = fem.compute_element_stresses(u, E0=1.0)
    assert cauchy.shape == (m2d.n_elem, 3)
    assert vm.shape == (m2d.n_elem,)
    assert np.all(vm >= 0.0)
    assert np.max(vm) > 0.0

    sched = ContinuationScheduler(penal_init=1.0, penal_max=3.0, penal_interval=5, penal_step=0.5)
    for i in range(1, 16):
        sched.step(i)
    assert sched.penal == 2.5
    assert np.isclose(ContinuationScheduler.compute_grayness(np.full(100, 0.5)), 1.0)
    assert np.isclose(ContinuationScheduler.compute_grayness(np.array([0.0] * 50 + [1.0] * 50)), 0.0)
