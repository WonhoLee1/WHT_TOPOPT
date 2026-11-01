"""Unit and integration tests for linearized buckling and vibration / compliant mechanisms (Milestone 3).

Tests:
1. Central Finite Difference (FD, h=1e-6) verification of BucklingConstraint
   (including the non-self-adjoint stress-dependent adjoint displacement w_KS)
   on 2D and 3D compressed columns, asserting relative error < 1e-5.
2. Central Finite Difference verification of EigenFrequencyFunction and
   CompliantMechanismFunction (force_inverter_2d), asserting relative error < 1e-5.
3. Buckling-driven topology optimization of a compressed column showing that
   the critical buckling load factor lambda_1 = 1 / mu_1 increases significantly.
"""

from __future__ import annotations

import numpy as np
import pytest

from wht_topopt.core import (
    StructuredMesh2D,
    StructuredMesh3D,
    FEMEngine,
    BoundaryConditionManager,
    DensityFilter,
)
from wht_topopt.solvers import ScipySolver
from wht_topopt.algorithms import OptimalityCriteria
from wht_topopt.functions.buckling import BucklingConstraint
from wht_topopt.functions.modal import (
    EigenFrequencyFunction,
    EigenFrequencyConstraint,
    CompliantMechanismFunction,
)


def test_buckling_constraint_fd_2d_and_3d():
    """Test 1: Analytical vs FD sensitivities for BucklingConstraint in 2D and 3D."""
    solver = ScipySolver(method="direct")
    h = 1e-6

    # 1. 2D Compressed Column
    mesh_2d = StructuredMesh2D(nx=4, ny=3, lx=1.0, ly=1.0)
    fem_2d = FEMEngine(mesh_2d, nu=0.3, formulation="FI")
    bcm_2d = BoundaryConditionManager.compressed_column_2d(mesh_2d, F_mag=1.0)

    constr_2d = BucklingConstraint(
        fem=fem_2d,
        bcm=bcm_2d,
        solver=solver,
        lambda_crit_target=1.5,
        n_modes=3,
        rho_ks=20.0,
        penal_k=3.0,
        penal_g=3.0,
        e0=10.0,
        emin=1e-6,
    )

    rng = np.random.default_rng(42)
    rho_2d = rng.uniform(0.5, 0.9, size=mesh_2d.n_elem)

    val_2d, grad_2d = constr_2d.value_and_gradient(rho_2d)
    assert np.isfinite(val_2d)
    assert np.all(np.isfinite(grad_2d))

    blf_1 = constr_2d.get_critical_load_factor(rho_2d)
    assert blf_1 > 0.0

    # FD verification for 2D
    for idx in [0, 4, 7, 11]:
        rp = rho_2d.copy(); rp[idx] += h
        rm = rho_2d.copy(); rm[idx] -= h
        vp = constr_2d.evaluate(rp)
        vm = constr_2d.evaluate(rm)
        g_fd = (vp - vm) / (2.0 * h)
        rel_err = abs(grad_2d[idx] - g_fd) / abs(grad_2d[idx])
        assert rel_err < 1e-5, f"2D Buckling FD mismatch at {idx}: ana={grad_2d[idx]}, fd={g_fd}, rel_err={rel_err}"

    # 2. 3D Compressed Column
    mesh_3d = StructuredMesh3D(nx=3, ny=3, nz=3, lx=1.0, ly=1.0, lz=1.5)
    fem_3d = FEMEngine(mesh_3d, nu=0.3, formulation="FI")
    bcm_3d = BoundaryConditionManager.compressed_column_3d(mesh_3d, F_mag=1.0)

    constr_3d = BucklingConstraint(
        fem=fem_3d,
        bcm=bcm_3d,
        solver=solver,
        lambda_crit_target=1.2,
        n_modes=2,
        rho_ks=15.0,
        penal_k=3.0,
        penal_g=3.0,
        e0=10.0,
        emin=1e-6,
    )

    rho_3d = rng.uniform(0.5, 0.9, size=mesh_3d.n_elem)
    val_3d, grad_3d = constr_3d.value_and_gradient(rho_3d)
    assert np.isfinite(val_3d)
    assert np.all(np.isfinite(grad_3d))

    for idx in [0, 8, 15, 26]:
        rp = rho_3d.copy(); rp[idx] += h
        rm = rho_3d.copy(); rm[idx] -= h
        vp = constr_3d.evaluate(rp)
        vm = constr_3d.evaluate(rm)
        g_fd = (vp - vm) / (2.0 * h)
        rel_err = abs(grad_3d[idx] - g_fd) / abs(grad_3d[idx])
        assert rel_err < 1e-5, f"3D Buckling FD mismatch at {idx}: ana={grad_3d[idx]}, fd={g_fd}, rel_err={rel_err}"


def test_eigenfrequency_and_compliant_mechanism_fd():
    """Test 2: Analytical vs FD sensitivities for EigenFrequency and CompliantMechanism."""
    mesh = StructuredMesh2D(nx=4, ny=3, lx=1.5, ly=1.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    solver = ScipySolver(method="direct")
    h = 1e-6

    rng = np.random.default_rng(123)
    rho = rng.uniform(0.5, 0.9, size=mesh.n_elem)

    # 1. Eigenfrequency objective & constraint
    bcm_cant = BoundaryConditionManager.cantilever_2d(mesh)
    eig_fn = EigenFrequencyFunction(
        fem,
        bcm_cant,
        solver=solver,
        mode_idx=0,
        penal_k=3.0,
        penal_m=1.0,
        e0=10.0,
        emin=1e-6,
    )

    f0, grad_eig = eig_fn.value_and_gradient(rho)
    assert f0 < 0.0  # -omega^2 is negative
    freqs_hz = eig_fn.get_natural_frequencies_hz(rho)
    assert freqs_hz[0] > 0.0

    for idx in [0, 3, 6, 11]:
        h_step = 1e-5
        rp = rho.copy(); rp[idx] += h_step
        rm = rho.copy(); rm[idx] -= h_step
        fp = eig_fn.evaluate(rp)
        fm = eig_fn.evaluate(rm)
        g_fd = (fp - fm) / (2.0 * h_step)
        rel_err = abs(grad_eig[idx] - g_fd) / (abs(grad_eig[idx]) + 1e-8)
        assert rel_err < 1e-5, f"Eigenfrequency FD mismatch at {idx}: ana={grad_eig[idx]}, fd={g_fd}, rel_err={rel_err}"

    # EigenFrequencyConstraint
    eig_constr = EigenFrequencyConstraint(
        fem,
        bcm_cant,
        omega_target=1.0,
        solver=solver,
        e0=10.0,
    )
    c0, grad_c = eig_constr.value_and_gradient(rho)
    for idx in [0, 5, 10]:
        h_step = 1e-5
        rp = rho.copy(); rp[idx] += h_step
        rm = rho.copy(); rm[idx] -= h_step
        cp = eig_constr.evaluate(rp)
        cm = eig_constr.evaluate(rm)
        g_fd = (cp - cm) / (2.0 * h_step)
        rel_err = abs(grad_c[idx] - g_fd) / (abs(grad_c[idx]) + 1e-8)
        assert rel_err < 1e-5, f"Eigenfrequency constraint FD mismatch at {idx}: {rel_err}"

    # 2. Compliant Mechanism (Force Inverter 2D)
    bcm_inv = BoundaryConditionManager.force_inverter_2d(mesh, F_in=1.0, k_in=1.0, k_out=0.01)
    in_nodes = mesh.get_nodes_in_box((-1e-6, 1e-6), (-1e-6, 1e-6))
    out_nodes = mesh.get_nodes_in_box((mesh.lx - 1e-6, mesh.lx + 1e-6), (-1e-6, 1e-6))
    in_dof = int(2 * in_nodes[0])
    out_dof = int(2 * out_nodes[0])

    mech_fn = CompliantMechanismFunction(
        fem,
        bcm_inv,
        in_dof=in_dof,
        out_dof=out_dof,
        solver=solver,
        f_in=1.0,
        k_in=1.0,
        k_out=0.01,
        penal=3.0,
        e0=10.0,
    )

    m0, grad_mech = mech_fn.value_and_gradient(rho)
    u_out = mech_fn.get_output_displacement(rho)
    assert abs(u_out - (-m0)) < 1e-12
    ma = mech_fn.get_mechanical_advantage(rho)
    assert np.isfinite(ma)

    for idx in [0, 4, 7, 11]:
        rp = rho.copy(); rp[idx] += h
        rm = rho.copy(); rm[idx] -= h
        mp = mech_fn.evaluate(rp)
        mm = mech_fn.evaluate(rm)
        g_fd = (mp - mm) / (2.0 * h)
        rel_err = abs(grad_mech[idx] - g_fd) / (abs(grad_mech[idx]) + 1e-7)
        assert rel_err < 1e-5, f"Compliant mechanism FD mismatch at {idx}: ana={grad_mech[idx]}, fd={g_fd}, rel_err={rel_err}"


def test_buckling_column_optimization():
    """Test 3: Buckling-driven topology optimization of compressed column."""
    mesh = StructuredMesh2D(nx=10, ny=20, lx=1.0, ly=2.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.compressed_column_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)

    constr = BucklingConstraint(
        fem=fem,
        bcm=bcm,
        solver=solver,
        lambda_crit_target=1.0,
        n_modes=2,
        rho_ks=20.0,
        penal_k=3.0,
        penal_g=3.0,
        e0=1.0,
        emin=1e-6,
    )

    oc = OptimalityCriteria(volfrac=0.5, move=0.2, eta=0.5, rho_min=1e-3)
    rho = np.full(mesh.n_elem, 0.5)

    lambda_init = constr.get_critical_load_factor(filt.forward(rho))

    for it in range(10):
        rho_phys = filt.forward(rho)
        val, grad_phys = constr.value_and_gradient(rho_phys)
        grad_design = filt.backward(grad_phys)

        rho = oc.update(
            rho=rho,
            df_drho=grad_design,
            g_vals=np.array([np.mean(rho) - 0.5]),
            dg_drho=mesh.elem_volumes,
            elem_volumes=mesh.elem_volumes,
        )

    lambda_final = constr.get_critical_load_factor(filt.forward(rho))

    # Verify that the critical buckling load factor lambda_1 significantly increases
    assert lambda_final > 1.15 * lambda_init, (
        f"Expected BLF increase > 15%, got initial={lambda_init:.6f}, final={lambda_final:.6f} "
        f"(ratio={lambda_final/lambda_init:.3f})"
    )
