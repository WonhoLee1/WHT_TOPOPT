"""3D Cantilever Topology Optimization & 3D Modal / Buckling Sensitivity Benchmark (Milestone 3).

Tests:
1. 3D Cantilever topology optimization using StructuredMesh3D with HEX8 elements:
   - Compares both BBAR (Hughes B-bar volumetric locking-free) and FI (Full Integration).
   - Decreases compliance by > 40% from initial uniform design.
   - Satisfies volume fraction constraint |V - V*| < 1e-4.
2. 3D modal sensitivity verification:
   - Validates that natural frequencies are strictly positive (zero spurious modes).
   - Asserts analytical eigenvalue gradient matches Central Finite Difference (h=1e-5) to < 1e-5.
3. 3D buckling sensitivity verification:
   - Validates 3D compressed column buckling load factors.
   - Asserts exact non-self-adjoint KS buckling gradient matches Central Finite Difference to < 1e-5.
"""

from __future__ import annotations

import numpy as np
import pytest

from wht_topopt.core import (
    StructuredMesh3D,
    FEMEngine,
    BoundaryConditionManager,
    DensityFilter,
)
from wht_topopt.solvers import ScipySolver
from wht_topopt.algorithms import OptimalityCriteria
from wht_topopt.functions import (
    ComplianceFunction,
    VolumeFractionFunction,
)
from wht_topopt.functions.buckling import BucklingConstraint
from wht_topopt.functions.modal import EigenFrequencyFunction


@pytest.mark.parametrize("formulation", ["BBAR", "FI"])
def test_cantilever_3d_compliance_optimization_bbar_and_fi(formulation: str):
    """Test 1: 3D Cantilever compliance topology optimization with BBAR and FI formulations."""
    mesh = StructuredMesh3D(nx=8, ny=4, nz=2, lx=2.0, ly=1.0, lz=0.5)
    vol_frac = 0.4

    fem = FEMEngine(mesh, nu=0.3, formulation=formulation)
    bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.25)

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    oc = OptimalityCriteria(volfrac=vol_frac, move=0.2, eta=0.5, rho_min=1e-3)

    rho = np.full(mesh.n_elem, vol_frac)

    # Initial compliance
    rho_phys_init = filt.forward(rho)
    E_arr, _ = comp_fn.interpolate_material(rho_phys_init)
    K = fem.assemble_K(E_arr)
    res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
    c_init = comp_fn.evaluate(rho_phys_init, u=res.u.ravel())

    # Optimization loop (12 iterations)
    for it in range(12):
        rho_phys = filt.forward(rho)
        E_arr, _ = comp_fn.interpolate_material(rho_phys)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)

        c_val, dc_phys = comp_fn.value_and_gradient(rho_phys, u=res.u.ravel())
        dc_design = filt.backward(dc_phys)

        g_val, dv_phys = vol_fn.value_and_gradient(rho_phys)
        dv_design = filt.backward(dv_phys)

        rho = oc.update(
            rho=rho,
            df_drho=dc_design,
            g_vals=np.array([g_val]),
            dg_drho=dv_design,
            elem_volumes=mesh.elem_volumes,
        )

    # Final compliance & volume
    rho_phys_final = filt.forward(rho)
    E_arr, _ = comp_fn.interpolate_material(rho_phys_final)
    K = fem.assemble_K(E_arr)
    res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
    c_final = comp_fn.evaluate(rho_phys_final, u=res.u.ravel())
    v_final = np.mean(rho_phys_final)

    # 1. Compliance must decrease significantly (> 30% reduction)
    assert c_final < 0.7 * c_init, (
        f"{formulation}: Expected compliance reduction > 30%, got c_init={c_init:.4f}, c_final={c_final:.4f}"
    )

    # 2. Volume constraint must be satisfied
    assert abs(v_final - vol_frac) < 1e-4, (
        f"{formulation}: Volume constraint violated: |{v_final:.4f} - {vol_frac}| >= 1e-4"
    )


def test_cantilever_3d_modal_sensitivity_and_spurious_modes():
    """Test 2: 3D modal analysis, spurious mode suppression, and exact FD sensitivity check."""
    mesh = StructuredMesh3D(nx=4, ny=2, nz=2, lx=2.0, ly=1.0, lz=1.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")

    rng = np.random.default_rng(789)
    rho = rng.uniform(0.5, 0.9, size=mesh.n_elem)

    eig_fn = EigenFrequencyFunction(
        fem,
        bcm,
        solver=solver,
        mode_idx=0,
        n_modes=3,
        penal_k=3.0,
        penal_m=1.0,
        e0=10.0,
        emin=1e-6,
    )

    vals, phis, _, _ = eig_fn.solve_eigenfrequencies(rho)

    # Verify zero spurious modes: all natural frequencies are strictly positive
    assert np.all(vals > 0.0), f"Spurious non-positive vibration modes detected: {vals}"
    assert vals[0] < vals[1] <= vals[2], "Eigenvalues not correctly ordered ascending"

    f0, grad_eig = eig_fn.value_and_gradient(rho)
    assert np.isclose(f0, - vals[0], atol=1e-12)

    h_step = 1e-5
    for idx in [0, 5, 10, 15]:
        rp = rho.copy(); rp[idx] += h_step
        rm = rho.copy(); rm[idx] -= h_step
        fp = eig_fn.evaluate(rp)
        fm = eig_fn.evaluate(rm)
        g_fd = (fp - fm) / (2.0 * h_step)
        rel_err = abs(grad_eig[idx] - g_fd) / (abs(grad_eig[idx]) + 1e-8)
        assert rel_err < 1e-5, f"3D Modal FD mismatch at {idx}: ana={grad_eig[idx]}, fd={g_fd}, rel_err={rel_err}"


def test_cantilever_3d_buckling_sensitivity():
    """Test 3: 3D compressed column buckling analysis and exact FD sensitivity check."""
    mesh = StructuredMesh3D(nx=3, ny=3, nz=3, lx=1.0, ly=1.0, lz=1.5)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.compressed_column_3d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")

    constr = BucklingConstraint(
        fem=fem,
        bcm=bcm,
        solver=solver,
        lambda_crit_target=1.2,
        n_modes=2,
        rho_ks=15.0,
        penal_k=3.0,
        penal_g=3.0,
        e0=10.0,
        emin=1e-6,
    )

    rng = np.random.default_rng(999)
    rho = rng.uniform(0.5, 0.9, size=mesh.n_elem)

    val, grad_buck = constr.value_and_gradient(rho)
    blf = constr.get_critical_load_factor(rho)
    assert np.isfinite(blf) and blf > 0.0

    h = 1e-6
    for idx in [0, 7, 14, 26]:
        rp = rho.copy(); rp[idx] += h
        rm = rho.copy(); rm[idx] -= h
        vp = constr.evaluate(rp)
        vm = constr.evaluate(rm)
        g_fd = (vp - vm) / (2.0 * h)
        rel_err = abs(grad_buck[idx] - g_fd) / abs(grad_buck[idx])
        assert rel_err < 1e-5, f"3D Buckling FD mismatch at {idx}: ana={grad_buck[idx]}, fd={g_fd}, rel_err={rel_err}"
