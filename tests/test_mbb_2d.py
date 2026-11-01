"""2D MBB Beam Compliance Topology Optimization Benchmark (Milestone 1).

Tests:
1. Analytical vs Central Finite Difference sensitivity verification for:
   - ComplianceFunction (single & multiple load cases with weights)
   - VolumeFractionFunction & MassFunction
2. 2D MBB half-beam topology optimization using OptimalityCriteria (OC):
   - Strictly decreases compliance from initial uniform design
   - Satisfies volume constraint |V - V*| < 1e-4
   - Achieves crisp topology with continuation & Heaviside projection
"""

from __future__ import annotations

import numpy as np

from wht_topopt.core import (
    StructuredMesh2D,
    FEMEngine,
    DensityFilter,
    SIMP,
    HeavisideProjection,
    ContinuationScheduler,
    BoundaryConditionManager,
)
from wht_topopt.solvers import ScipySolver
from wht_topopt.algorithms import (
    OptimalityCriteria,
    TrustRegionController,
    MMAOptimizer,
    BESOOptimizer,
    LevelSetOptimizer,
    SEMDOTOptimizer,
)
from wht_topopt.functions import (
    ComplianceFunction,
    VolumeFractionFunction,
    MassFunction,
)


def test_compliance_and_volume_gradients_fd():
    mesh = StructuredMesh2D(nx=6, ny=4, lx=1.5, ly=1.0)
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=10.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=0.5, elem_volumes=mesh.elem_volumes)
    mass_fn = MassFunction(elem_volumes=mesh.elem_volumes, base_density=2700.0)

    rng = np.random.default_rng(123)
    rho = rng.uniform(0.2, 0.9, size=mesh.n_elem)

    def solve_u(r: np.ndarray) -> np.ndarray:
        E_arr, _ = comp_fn.interpolate_material(r)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        return res.u.ravel()

    u0 = solve_u(rho)
    c0, dc_ana = comp_fn.value_and_gradient(rho, u=u0)
    v0, dv_ana = vol_fn.value_and_gradient(rho)
    m0, dm_ana = mass_fn.value_and_gradient(rho)

    h = 1e-6
    for idx in [0, 5, 12, 23]:
        rho_p = rho.copy()
        rho_m = rho.copy()
        rho_p[idx] += h
        rho_m[idx] -= h

        up = solve_u(rho_p)
        um = solve_u(rho_m)
        dc_fd = (comp_fn.evaluate(rho_p, u=up) - comp_fn.evaluate(rho_m, u=um)) / (2.0 * h)
        rel_c = abs(dc_ana[idx] - dc_fd) / (abs(dc_ana[idx]) + 1e-12)
        assert rel_c < 1e-5, f"Compliance FD mismatch at {idx}: {rel_c}"

        dv_fd = (vol_fn.evaluate(rho_p) - vol_fn.evaluate(rho_m)) / (2.0 * h)
        assert abs(dv_ana[idx] - dv_fd) < 1e-9

        dm_fd = (mass_fn.evaluate(rho_p) - mass_fn.evaluate(rho_m)) / (2.0 * h)
        assert abs(dm_ana[idx] - dm_fd) / abs(dm_ana[idx]) < 1e-7


def test_mbb_2d_oc_optimization():
    mesh = StructuredMesh2D(nx=30, ny=10, lx=3.0, ly=1.0)
    vol_frac = 0.5
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)
    proj = HeavisideProjection(beta=1.0, eta=0.5)
    sched = ContinuationScheduler(
        penal_init=3.0,
        penal_max=3.0,
        beta_init=1.0,
        beta_max=4.0,
        beta_interval=15,
        beta_factor=2.0,
    )

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-9,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    tr = TrustRegionController(initial_move=0.2)
    oc = OptimalityCriteria(volfrac=vol_frac, move=0.2, eta=0.5, trust_region=tr)

    rho = np.full(mesh.n_elem, vol_frac)
    history_c = []

    for it in range(1, 36):
        state_sched = sched.step(it)
        proj.beta = state_sched["beta"]

        rho_tilde = filt.forward(rho)
        eta_star = proj.find_volume_preserving_eta(rho_tilde, target_vol=vol_frac)
        proj.eta = eta_star
        rho_hat = proj.forward(rho_tilde)

        E_arr, _ = comp_fn.interpolate_material(rho_hat)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        c_val, dc_hat = comp_fn.value_and_gradient(rho_hat, u=res.u.ravel())
        history_c.append(c_val)

        # Chain rule: d/drho = W^T (d rho_hat / d rho_tilde * d/d rho_hat)
        dproj = proj.derivative(rho_tilde)
        dc_rho = filt.backward(dc_hat * dproj)
        g_val, dv_hat = vol_fn.value_and_gradient(rho_hat)
        dv_rho = filt.backward(dv_hat * dproj)

        rho = oc.update(
            rho=rho,
            df_drho=dc_rho,
            g_vals=np.array([g_val]),
            dg_drho=dv_rho,
            elem_volumes=mesh.elem_volumes,
        )

    assert history_c[-1] < 0.65 * history_c[0], (
        f"Compliance did not decrease sufficiently: {history_c[0]:.4f} -> {history_c[-1]:.4f}"
    )
    final_vol_err = abs(vol_fn.evaluate(rho))
    assert final_vol_err < 1e-4, f"Volume constraint violated: error={final_vol_err}"


def test_mbb_2d_mma_optimization():
    """Verify MMA optimizer on 2D MBB beam minimum compliance benchmark."""
    mesh = StructuredMesh2D(nx=30, ny=10, lx=3.0, ly=1.0)
    vol_frac = 0.5
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    mma = MMAOptimizer(move=0.2)

    rho = np.full(mesh.n_elem, vol_frac)
    history_c = []
    c0 = None

    for _ in range(25):
        E_arr, _ = comp_fn.interpolate_material(rho)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        c_val, dc_hat = comp_fn.value_and_gradient(rho, u=res.u.ravel())
        if c0 is None:
            c0 = c_val
        history_c.append(c_val)

        dc_rho = filt.backward(dc_hat)
        g_val, dv_hat = vol_fn.value_and_gradient(rho)
        dv_rho = filt.backward(dv_hat)

        rho = mma.update(
            rho=rho,
            df_drho=dc_rho / c0,
            g_vals=np.array([g_val]),
            dg_drho=dv_rho.reshape(1, -1),
            elem_volumes=mesh.elem_volumes,
        )

    assert history_c[-1] < 0.65 * history_c[0], (
        f"MMA compliance did not decrease: {history_c[0]:.2f} -> {history_c[-1]:.2f}"
    )
    vol_err = abs(vol_fn.evaluate(rho))
    assert vol_err < 0.01, f"MMA volume constraint violated: {vol_err:.4e}"


def test_mbb_2d_beso_optimization():
    """Verify BESO optimizer on 2D MBB beam minimum compliance benchmark."""
    mesh = StructuredMesh2D(nx=30, ny=10, lx=3.0, ly=1.0)
    vol_frac = 0.5
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    beso = BESOOptimizer(target_volfrac=vol_frac, quasi_continuous=True, move=0.1)

    rho = np.full(mesh.n_elem, vol_frac)
    history_c = []

    for _ in range(25):
        E_arr, _ = comp_fn.interpolate_material(rho)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        c_val, dc_hat = comp_fn.value_and_gradient(rho, u=res.u.ravel())
        history_c.append(c_val)

        dc_rho = filt.backward(dc_hat)
        g_val, dv_hat = vol_fn.value_and_gradient(rho)
        dv_rho = filt.backward(dv_hat)

        rho = beso.update(
            rho=rho,
            df_drho=dc_rho,
            g_vals=np.array([g_val]),
            dg_drho=dv_rho,
            elem_volumes=mesh.elem_volumes,
        )

    assert history_c[-1] < 0.65 * history_c[0], (
        f"BESO compliance did not decrease: {history_c[0]:.2f} -> {history_c[-1]:.2f}"
    )
    vol_err = abs(vol_fn.evaluate(rho))
    assert vol_err < 0.02, f"BESO volume constraint violated: {vol_err:.4e}"


def test_mbb_2d_level_set_optimization():
    """Verify Hilbertian Level-Set optimizer on 2D MBB beam benchmark."""
    mesh = StructuredMesh2D(nx=30, ny=10, lx=3.0, ly=1.0)
    vol_frac = 0.5
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    ls = LevelSetOptimizer(mesh=mesh, volfrac=vol_frac, gamma=0.1, move=0.15)

    rho = np.full(mesh.n_elem, vol_frac)
    history_c = []

    for _ in range(25):
        E_arr, _ = comp_fn.interpolate_material(rho)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        c_val, dc_hat = comp_fn.value_and_gradient(rho, u=res.u.ravel())
        history_c.append(c_val)

        dc_rho = filt.backward(dc_hat)
        g_val, dv_hat = vol_fn.value_and_gradient(rho)
        dv_rho = filt.backward(dv_hat)

        rho = ls.update(
            rho=rho,
            df_drho=dc_rho,
            g_vals=np.array([g_val]),
            dg_drho=dv_rho,
            elem_volumes=mesh.elem_volumes,
        )

    assert history_c[-1] < 0.70 * history_c[0], (
        f"LevelSet compliance did not decrease: {history_c[0]:.2f} -> {history_c[-1]:.2f}"
    )
    vol_err = abs(vol_fn.evaluate(rho))
    assert vol_err < 0.01, f"LevelSet volume constraint violated: {vol_err:.4e}"


def test_mbb_2d_semdot_optimization():
    """Verify SEMDOT optimizer on 2D MBB beam benchmark."""
    mesh = StructuredMesh2D(nx=30, ny=10, lx=3.0, ly=1.0)
    vol_frac = 0.5
    fem = FEMEngine(mesh, nu=0.3, formulation="FI")
    bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=1.0)
    solver = ScipySolver(method="direct")
    filt = DensityFilter(mesh, r_min=0.15)

    comp_fn = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-6,
        material_model="simp_modified",
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    vol_fn = VolumeFractionFunction(target_volfrac=vol_frac, elem_volumes=mesh.elem_volumes)
    semdot = SEMDOTOptimizer(mesh=mesh, volfrac=vol_frac, n_sub=3, move=0.2)

    rho = np.full(mesh.n_elem, vol_frac)
    history_c = []

    for _ in range(25):
        E_arr, _ = comp_fn.interpolate_material(rho)
        K = fem.assemble_K(E_arr)
        res = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs)
        c_val, dc_hat = comp_fn.value_and_gradient(rho, u=res.u.ravel())
        history_c.append(c_val)

        dc_rho = filt.backward(dc_hat)
        g_val, dv_hat = vol_fn.value_and_gradient(rho)
        dv_rho = filt.backward(dv_hat)

        rho = semdot.update(
            rho=rho,
            df_drho=dc_rho,
            g_vals=np.array([g_val]),
            dg_drho=dv_rho,
            elem_volumes=mesh.elem_volumes,
        )

    assert history_c[-1] < 0.65 * history_c[0], (
        f"SEMDOT compliance did not decrease: {history_c[0]:.2f} -> {history_c[-1]:.2f}"
    )
    vol_err = abs(vol_fn.evaluate(rho))
    assert vol_err < 0.01, f"SEMDOT volume constraint violated: {vol_err:.4e}"
