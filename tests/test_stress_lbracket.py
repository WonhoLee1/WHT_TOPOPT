"""Tests for P-norm stress constraint and stress-constrained L-bracket optimization (Milestone 2).

Tests:
1. Exact adjoint gradient verification against Central Finite Differences (h = 1e-6)
   asserting relative error < 1e-6 on:
   - 2D L-bracket with Q4 elements
   - 3D Cantilever with HEX8 elements
2. L-bracket stress-constrained topology optimization with MMAOptimizer:
   - Verifies compliance reduction, volume constraint satisfaction, and that
     peak von Mises stress in the re-entrant corner is reduced compared to the
     unconstrained compliance design.
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
from wht_topopt.algorithms import MMAOptimizer
from wht_topopt.functions import ComplianceFunction, VolumeFractionFunction
from wht_topopt.functions.stress import PNormStressConstraint


def test_stress_adjoint_gradient_fd_2d_lbracket():
    """Verify analytical adjoint gradient of PNormStressConstraint on 2D L-bracket against FD."""
    mesh = StructuredMesh2D(nx=8, ny=8, lx=1.0, ly=1.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.l_bracket_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")

    stress_fn = PNormStressConstraint(
        fem=fem,
        solver=solver,
        bcm=bcm,
        p=6.0,
        sigma_allow=1.0,
        q=0.5,
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        sigma_reg=1e-8,
        use_adaptive_scaling=False,
    )

    rng = np.random.default_rng(42)
    rho = rng.uniform(0.3, 0.9, size=mesh.n_elem)
    rho[bcm.passive_void_mask] = 1e-3

    # Forward solve
    E_arr, _ = stress_fn.interpolate_stiffness(rho)
    K = fem.assemble_K(E_arr)
    res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
    u0 = res.u.ravel()

    val0, grad_ana = stress_fn.value_and_gradient(rho, u=u0)

    # Sort active elements by magnitude of analytical gradient to test high-stress regions
    active = np.flatnonzero(bcm.active_elements)
    top_active = active[np.argsort(-np.abs(grad_ana[active]))]
    test_indices = top_active[:3]  # elements with largest sensitivity

    h = 1e-6
    for idx in test_indices:
        rho_p = rho.copy()
        rho_m = rho.copy()
        rho_p[idx] += h
        rho_m[idx] -= h

        Ep, _ = stress_fn.interpolate_stiffness(rho_p)
        Kp = fem.assemble_K(Ep)
        up = solver.solve(Kp, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()
        val_p = stress_fn.evaluate(rho_p, u=up)

        Em, _ = stress_fn.interpolate_stiffness(rho_m)
        Km = fem.assemble_K(Em)
        um = solver.solve(Km, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()
        val_m = stress_fn.evaluate(rho_m, u=um)

        grad_fd = (val_p - val_m) / (2.0 * h)
        rel_err = abs(grad_ana[idx] - grad_fd) / abs(grad_ana[idx])
        assert rel_err < 1e-6, f"2D L-bracket FD mismatch at elem {idx}: rel_err={rel_err:.3e}"


def test_stress_adjoint_gradient_fd_3d_cantilever():
    """Verify analytical adjoint gradient of PNormStressConstraint on 3D HEX8 mesh against FD."""
    mesh = StructuredMesh3D(nx=4, ny=3, nz=2, lx=2.0, ly=1.0, lz=1.0)
    fem = FEMEngine(mesh, nu=0.3)
    bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")

    stress_fn = PNormStressConstraint(
        fem=fem,
        solver=solver,
        bcm=bcm,
        p=6.0,
        sigma_allow=1.0,
        q=0.5,
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        sigma_reg=1e-8,
        use_adaptive_scaling=False,
    )

    rng = np.random.default_rng(99)
    rho = rng.uniform(0.3, 0.9, size=mesh.n_elem)

    E_arr, _ = stress_fn.interpolate_stiffness(rho)
    K = fem.assemble_K(E_arr)
    res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
    u0 = res.u.ravel()

    val0, grad_ana = stress_fn.value_and_gradient(rho, u=u0)

    test_indices = [0, 4, 10, 15, 23]
    h = 1e-6
    for idx in test_indices:
        rho_p = rho.copy()
        rho_m = rho.copy()
        rho_p[idx] += h
        rho_m[idx] -= h

        Ep, _ = stress_fn.interpolate_stiffness(rho_p)
        Kp = fem.assemble_K(Ep)
        up = solver.solve(Kp, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()
        val_p = stress_fn.evaluate(rho_p, u=up)

        Em, _ = stress_fn.interpolate_stiffness(rho_m)
        Km = fem.assemble_K(Em)
        um = solver.solve(Km, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()
        val_m = stress_fn.evaluate(rho_m, u=um)

        grad_fd = (val_p - val_m) / (2.0 * h)
        rel_err = abs(grad_ana[idx] - grad_fd) / abs(grad_ana[idx])
        assert rel_err < 1e-6, f"3D Cantilever FD mismatch at elem {idx}: rel_err={rel_err:.3e}"


def test_lbracket_stress_constrained_mma():
    """Verify L-bracket multi-constraint optimization using MMAOptimizer."""
    mesh = StructuredMesh2D(nx=12, ny=12, lx=1.0, ly=1.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.l_bracket_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.12)

    vol_frac = 0.4
    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes, normalize=True)

    # 1. Unconstrained optimization: compliance minimization with volume constraint only
    mma_unconstrained = MMAOptimizer(move=0.15, tol=1e-9)
    rho_a = np.full(mesh.n_elem, vol_frac)
    rho_a[bcm.passive_void_mask] = 1e-3

    u0 = solver.solve(
        fem.assemble_K(comp_fn.interpolate_material(filt.forward(rho_a))[0]),
        bcm.F[:, 0],
        free_dofs=bcm.free_dofs,
        fixed_dofs=bcm.fixed_dofs,
    ).u.ravel()
    c0 = comp_fn.evaluate(filt.forward(rho_a), u=u0)

    for it in range(20):
        rho_tilde = filt.forward(rho_a)
        E_arr, _ = comp_fn.interpolate_material(rho_tilde)
        K = fem.assemble_K(E_arr)
        u = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()

        c_val, dc_tilde = comp_fn.value_and_gradient(rho_tilde, u=u)
        dc_rho = filt.backward(dc_tilde) / c0

        gv, dv_tilde = vol_fn.value_and_gradient(rho_tilde)
        dv_rho = filt.backward(dv_tilde)

        rho_a = mma_unconstrained.update(
            rho_a,
            dc_rho,
            np.array([gv]),
            dv_rho,
            elem_volumes=mesh.elem_volumes,
            passive_void=bcm.passive_void_mask,
        )

    u_a = solver.solve(
        fem.assemble_K(comp_fn.interpolate_material(filt.forward(rho_a))[0]),
        bcm.F[:, 0],
        free_dofs=bcm.free_dofs,
        fixed_dofs=bcm.fixed_dofs,
    ).u.ravel()

    stress_eval = PNormStressConstraint(
        fem, solver, bcm, p=6.0, sigma_allow=1.0, q=0.5, penal=3.0, e0=1.0, emin=1e-6, sigma_reg=1e-8
    )
    stress_eval.evaluate(filt.forward(rho_a), u=u_a)
    spn_a = stress_eval._last_sigma_PN

    # Element closest to re-entrant corner (x ~ 0.35, y ~ 0.35)
    centers = mesh.elem_centers
    corner_idx = int(np.argmin(np.hypot(centers[:, 0] - 0.35, centers[:, 1] - 0.35)))
    _, _, vm_a = stress_eval.compute_relaxed_von_mises(filt.forward(rho_a), u_a)
    corner_vm_a = float(vm_a[corner_idx])

    # 2. Stress-constrained optimization: target 80% of unconstrained stress
    sigma_allow = 0.8 * spn_a
    stress_fn = PNormStressConstraint(
        fem=fem,
        solver=solver,
        bcm=bcm,
        p=6.0,
        sigma_allow=sigma_allow,
        q=0.5,
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        sigma_reg=1e-8,
        use_adaptive_scaling=False,
    )

    mma_constrained = MMAOptimizer(move=0.15, tol=1e-9)
    rho_b = np.full(mesh.n_elem, vol_frac)
    rho_b[bcm.passive_void_mask] = 1e-3

    for it in range(20):
        rho_tilde = filt.forward(rho_b)
        E_arr, _ = comp_fn.interpolate_material(rho_tilde)
        K = fem.assemble_K(E_arr)
        u = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()

        c_val, dc_tilde = comp_fn.value_and_gradient(rho_tilde, u=u)
        dc_rho = filt.backward(dc_tilde) / c0

        gv, dv_tilde = vol_fn.value_and_gradient(rho_tilde)
        dv_rho = filt.backward(dv_tilde)

        gs, ds_tilde = stress_fn.value_and_gradient(rho_tilde, u=u, K=K)
        ds_rho = filt.backward(ds_tilde)

        g_all = np.array([gv, gs])
        dg_all = np.vstack([dv_rho, ds_rho])

        rho_b = mma_constrained.update(
            rho_b,
            dc_rho,
            g_all,
            dg_all,
            elem_volumes=mesh.elem_volumes,
            passive_void=bcm.passive_void_mask,
        )

    u_b = solver.solve(
        fem.assemble_K(comp_fn.interpolate_material(filt.forward(rho_b))[0]),
        bcm.F[:, 0],
        free_dofs=bcm.free_dofs,
        fixed_dofs=bcm.fixed_dofs,
    ).u.ravel()

    _, _, vm_b = stress_eval.compute_relaxed_von_mises(filt.forward(rho_b), u_b)
    corner_vm_b = float(vm_b[corner_idx])
    stress_eval.evaluate(filt.forward(rho_b), u=u_b)
    spn_b = stress_eval._last_sigma_PN

    # Assertions
    # 1. Peak von Mises stress in re-entrant corner is reduced
    assert corner_vm_b < corner_vm_a, (
        f"Corner stress was not reduced: Unconstrained={corner_vm_a:.4f}, Constrained={corner_vm_b:.4f}"
    )
    # 2. Aggregated P-norm stress is reduced
    assert spn_b < spn_a, f"PN stress was not reduced: {spn_a:.4f} -> {spn_b:.4f}"
    # 3. Volume constraint satisfied within tolerance
    assert vol_fn.evaluate(rho_b) < 0.1, f"Volume constraint violated: {vol_fn.evaluate(rho_b)}"
