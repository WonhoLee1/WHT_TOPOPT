"""Cross-verification test suite for all 4 FEA Solvers in wht_topopt.solvers.

Tests:
1. 3D HEX8 cantilever cross-verification across:
   - ScipySolver (direct & cg)
   - PyAMGSolver (Smoothed Aggregation AMG with rigid-body near-nullspace)
   - KratosSolver (In-Memory KM.Model & native TopologyOptimizationApplication)
   - FrontISTRSolver (local fistr1.exe SMP, FORM361=FI/BBAR)
   Verifying:
   - Relative displacement L2 error < 1e-5 across solvers
   - Compliance sensitivity cosine similarity > 0.9999
2. Inertia Relief projection (apply_inertia_relief) on free-floating body:
   - 6 rigid body residual reactions < 1e-12
"""

from __future__ import annotations

import numpy as np

from wht_topopt.core import (
    StructuredMesh3D,
    FEMEngine,
    SIMP,
    BoundaryConditionManager,
)
from wht_topopt.solvers import (
    ScipySolver,
    PyAMGSolver,
    KratosSolver,
    FrontISTRSolver,
    FEASolver,
    build_rigid_body_modes,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def test_3d_cantilever_cross_solvers():
    mesh = StructuredMesh3D(nx=4, ny=2, nz=2, lx=4.0, ly=2.0, lz=2.0)
    nu = 0.3
    E0 = 1000.0
    Emin = 1e-6

    fem = FEMEngine(mesh, nu=nu, formulation="FI")
    bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=10.0)
    fixed_dofs = bcm.fixed_dofs
    free_dofs = bcm.free_dofs
    F = bcm.F[:, 0]

    rng = np.random.default_rng(7)
    rho = rng.uniform(0.4, 1.0, size=mesh.n_elem)
    simp = SIMP(E0=E0, Emin=Emin, penal=3.0)
    E_elem = simp.evaluate(rho)
    dE_drho = simp.derivative(rho)

    K = fem.assemble_K(E_elem)

    # 1. ScipySolver (direct)
    scipy_direct = ScipySolver(method="direct")
    res_direct = scipy_direct.solve(K, F, free_dofs=free_dofs, fixed_dofs=fixed_dofs)
    u_ref = res_direct.u.ravel()
    ce_ref = fem.compute_unit_element_compliance(u_ref)
    dc_ref = -dE_drho * ce_ref

    # Test solve_cached for adjoint solve
    u_adj_direct = scipy_direct.solve_cached(F)
    assert np.linalg.norm(u_adj_direct - u_ref) / np.linalg.norm(u_ref) < 1e-12

    # 2. ScipySolver (cg)
    scipy_cg = ScipySolver(method="cg", tol=1e-10)
    res_cg = scipy_cg.solve(K, F, free_dofs=free_dofs, fixed_dofs=fixed_dofs)
    rel_err_cg = np.linalg.norm(res_cg.u.ravel() - u_ref) / np.linalg.norm(u_ref)
    assert rel_err_cg < 1e-7, f"Scipy CG relative error too high: {rel_err_cg}"

    # 3. PyAMGSolver (Smoothed Aggregation AMG with rigid-body near-nullspace)
    amg_solver = PyAMGSolver(tol=1e-10)
    res_amg = amg_solver.solve(
        K, F, free_dofs=free_dofs, fixed_dofs=fixed_dofs, node_coords=mesh.node_coords
    )
    rel_err_amg = np.linalg.norm(res_amg.u.ravel() - u_ref) / np.linalg.norm(u_ref)
    ce_amg = fem.compute_unit_element_compliance(res_amg.u.ravel())
    dc_amg = -dE_drho * ce_amg
    assert rel_err_amg < 1e-7, f"PyAMG relative error too high: {rel_err_amg}"
    assert _cosine_similarity(dc_ref, dc_amg) > 0.999999

    # 4. KratosSolver (In-Memory KratosMultiphysics StructuralMechanicsApplication)
    kratos_solver = KratosSolver()
    eval_kratos = kratos_solver.evaluate_mesh(
        nodes=mesh.node_coords,
        elems=mesh.elements,
        etype="hex8",
        fixed_dofs=fixed_dofs,
        loads=F,
        youngs=E_elem,
        nu=nu,
        x_phys=rho,
        penal=3.0,
        e0=E0,
        emin=Emin,
        use_simp_elements=True,
    )
    u_kratos = eval_kratos["u"].ravel()
    rel_err_kratos = np.linalg.norm(u_kratos - u_ref) / np.linalg.norm(u_ref)
    dc_kratos = eval_kratos["dcdx"].ravel()
    cos_kratos = _cosine_similarity(dc_ref, dc_kratos)
    assert rel_err_kratos < 1e-6, f"Kratos In-Memory displacement relative error: {rel_err_kratos}"
    assert cos_kratos > 0.9999, f"Kratos sensitivity cosine similarity: {cos_kratos}"

    # 5. FrontISTRSolver (fistr1.exe SMP with exact per-element material properties)
    fistr_solver = FrontISTRSolver(form361="FI", n_buckets=0, threads=2)
    res_fistr = fistr_solver.solve_mesh(
        nodes=mesh.node_coords,
        elems=mesh.elements,
        etype="hex8",
        fixed_dofs=fixed_dofs,
        loads=F,
        youngs=E_elem,
        nu=nu,
    )
    u_fistr = res_fistr.u.ravel()
    rel_err_fistr = np.linalg.norm(u_fistr - u_ref) / np.linalg.norm(u_ref)
    ce_fistr = fem.compute_unit_element_compliance(u_fistr)
    dc_fistr = -dE_drho * ce_fistr
    cos_fistr = _cosine_similarity(dc_ref, dc_fistr)
    assert rel_err_fistr < 1e-5, f"FrontISTR displacement relative error: {rel_err_fistr}"
    assert cos_fistr > 0.9999, f"FrontISTR sensitivity cosine similarity: {cos_fistr}"


def test_kratos_native_simp_application():
    mesh = StructuredMesh3D(nx=3, ny=2, nz=2, lx=3.0, ly=2.0, lz=2.0)
    bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=10.0)
    kratos_solver = KratosSolver()
    out = kratos_solver.run_native_kratos_simp(
        nodes=mesh.node_coords,
        elems=mesh.elements,
        fixed_dofs=bcm.fixed_dofs,
        loads=bcm.F[:, 0],
        target_volfrac=0.5,
        max_iterations=3,
        filter_radius=1.5,
        penal=3.0,
        e0=1000.0,
        emin=1e-6,
        nu=0.3,
    )
    assert "x_phys" in out and len(out["x_phys"]) == mesh.n_elem
    assert "history" in out and len(out["history"]) >= 1
    assert np.all(np.isfinite(out["x_phys"]))


def test_inertia_relief_free_floating_balance():
    mesh = StructuredMesh3D(nx=3, ny=2, nz=2, lx=3.0, ly=2.0, lz=2.0)
    fem = FEMEngine(mesh, nu=0.3)
    rho = np.ones(mesh.n_elem)
    M = fem.assemble_M(rho)

    # Arbitrary unbalanced external force vector
    rng = np.random.default_rng(99)
    F_ext = rng.standard_normal((mesh.ndof, 2))

    F_bal = FEASolver.apply_inertia_relief(F_ext, M, mesh.node_coords, dim=3)
    Phi = build_rigid_body_modes(mesh.node_coords, dim=3)
    # Check Phi^T * F_bal == 0 to machine precision
    residual_rbm = Phi.T @ F_bal
    assert np.max(np.abs(residual_rbm)) < 1e-12, f"RBM residual not zero: {residual_rbm}"
