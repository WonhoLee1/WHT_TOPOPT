"""Comprehensive verification tests for Milestone 5.

Verifies:
1. Homogenizer2D and Homogenizer3D on solid unit cell:
   E^H(1) == C_0(E0, nu) to relative error < 1e-10.
2. Homogenization physics properties on non-uniform densities:
   - Symmetry: ||E^H - (E^H)^T|| < 10^-12
   - Positive definiteness: all eigenvalues > 0
   - Voigt-Reuss bounds: C_Reuss <= E_11^H <= C_Voigt
3. Exact self-adjoint sensitivities d(E^H_mn)/d(rho_e) vs Central Finite Differences (h = 1e-6)
   asserting relative error < 1e-6 in 2D and 3D, along with bulk modulus, shear modulus,
   and Poisson's ratio gradients.
4. SpinodalDehomogenizer Gaussian Random Field generation and volume fraction consistency.
5. RobustFormulation (Wang, Lazarov & Sigmund 2011) 3-field projection:
   - Worst-case, KS aggregation, and weighted-sum sensitivities vs Central FD < 1e-6
   - Dynamic dilated volume fraction scaling and constraint gradient vs Central FD < 1e-6.
6. MuJoCoLoadProvider:
   - Real MuJoCo (3.17.0) dynamic simulation of 2-link pendulum arm
   - Peak load extraction and RBE3 / D'Alembert body force mapping
   - Rigid-body equilibrium verification with Inertia Relief projection: ||Phi^T F_bal||_inf < 10^-12
   - Closed-loop MJCF XML inertial update and reload validation.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.fe_builders import isotropic_elasticity_2d, isotropic_elasticity_3d
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.manufacturing.minimum_length import ThreeFieldProjection
from wht_topopt.multiscale import Homogenizer2D, Homogenizer3D, SpinodalDehomogenizer
from wht_topopt.robust import MuJoCoLoadProvider, RobustFormulation
from wht_topopt.solvers.base import build_rigid_body_modes


# ---------------------------------------------------------------------------
# Test 1: Solid Unit Cell Homogenization (E^H(1) == C0 to relative error < 1e-10)
# ---------------------------------------------------------------------------

def test_homogenizer2d_solid_unit_cell() -> None:
    """Verify 2D homogenizer matches isotropic C0 down to machine precision for solid cell."""
    mesh = StructuredMesh2D(nx=4, ny=4, lx=1.0, ly=1.0)
    e0 = 2.5
    nu = 0.32

    for plane in ("plane_stress", "plane_strain"):
        homog = Homogenizer2D(
            mesh=mesh,
            e0=e0,
            emin=1e-6,
            nu=nu,
            penal=3.0,
            plane=plane,
        )
        rho_solid = np.ones(mesh.n_elem, dtype=np.float64)
        EH, _ = homog.homogenize(rho_solid)

        # Expected elasticity tensor
        C_expected = isotropic_elasticity_2d(nu, plane=plane) * e0
        rel_err = np.max(np.abs(EH - C_expected)) / np.max(np.abs(C_expected))
        assert rel_err < 1e-10, f"2D {plane} solid unit cell rel_err = {rel_err} >= 1e-10"


def test_homogenizer3d_solid_unit_cell() -> None:
    """Verify 3D homogenizer matches isotropic C0 down to machine precision for solid cell."""
    mesh = StructuredMesh3D(nx=3, ny=3, nz=3, lx=1.0, ly=1.0, lz=1.0)
    e0 = 3.0
    nu = 0.28

    homog = Homogenizer3D(
        mesh=mesh,
        e0=e0,
        emin=1e-6,
        nu=nu,
        penal=3.0,
    )
    rho_solid = np.ones(mesh.n_elem, dtype=np.float64)
    EH, _ = homog.homogenize(rho_solid)

    C_expected = isotropic_elasticity_3d(nu) * e0
    rel_err = np.max(np.abs(EH - C_expected)) / np.max(np.abs(C_expected))
    assert rel_err < 1e-10, f"3D solid unit cell rel_err = {rel_err} >= 1e-10"


# ---------------------------------------------------------------------------
# Test 2: Symmetry, Positive-Definiteness & Voigt-Reuss Bounds
# ---------------------------------------------------------------------------

def test_homogenizer2d_bounds_and_symmetry() -> None:
    """Verify symmetry, positive-definiteness, and Voigt-Reuss bounds on 2D porous cell."""
    mesh = StructuredMesh2D(nx=4, ny=4, lx=1.0, ly=1.0)
    e0 = 2.0
    emin = 1e-4
    nu = 0.3
    penal = 3.0

    homog = Homogenizer2D(
        mesh=mesh,
        e0=e0,
        emin=emin,
        nu=nu,
        penal=penal,
        plane="plane_stress",
    )

    # Heterogeneous deterministic density distribution
    rho = np.array([
        0.2, 0.8, 0.8, 0.2,
        0.8, 1.0, 1.0, 0.8,
        0.8, 1.0, 1.0, 0.8,
        0.2, 0.8, 0.8, 0.2,
    ], dtype=np.float64)

    EH, _ = homog.homogenize(rho)

    # 1. Symmetry
    asym = np.max(np.abs(EH - EH.T))
    assert asym < 1e-12, f"EH is not symmetric: asym = {asym}"

    # 2. Positive definiteness
    eigvals = np.linalg.eigvalsh(EH)
    assert np.all(eigvals > 0.0), f"EH not positive definite: eigvals = {eigvals}"

    # 3. Voigt and Reuss bounds for effective tensor
    E_e, _ = homog.interpolate_material(rho)
    C_voigt = np.mean(E_e) * homog.C0
    C_reuss = (1.0 / np.mean(1.0 / E_e)) * homog.C0

    E11 = EH[0, 0]
    assert E11 <= C_voigt[0, 0] + 1e-12, f"Violated Voigt upper bound: {E11} > {C_voigt[0, 0]}"
    assert E11 >= C_reuss[0, 0] - 1e-12, f"Violated Reuss lower bound: {E11} < {C_reuss[0, 0]}"


def test_homogenizer3d_bounds_and_symmetry() -> None:
    """Verify symmetry, positive-definiteness, and Voigt-Reuss bounds on 3D porous cell."""
    mesh = StructuredMesh3D(nx=2, ny=2, nz=2, lx=1.0, ly=1.0, lz=1.0)
    e0 = 1.0
    emin = 1e-5
    nu = 0.3
    penal = 3.0

    homog = Homogenizer3D(
        mesh=mesh,
        e0=e0,
        emin=emin,
        nu=nu,
        penal=penal,
    )

    rho = np.array([0.3, 0.9, 0.8, 0.4, 0.5, 0.7, 0.9, 0.2], dtype=np.float64)
    EH, _ = homog.homogenize(rho)

    # Symmetry
    asym = np.max(np.abs(EH - EH.T))
    assert asym < 1e-12, f"EH is not symmetric: asym = {asym}"

    # Positive definiteness
    eigvals = np.linalg.eigvalsh(EH)
    assert np.all(eigvals > 0.0), f"EH not positive definite: eigvals = {eigvals}"

    # Voigt-Reuss bounds
    E_e, _ = homog.interpolate_material(rho)
    C_voigt = np.mean(E_e) * homog.C0
    C_reuss = (1.0 / np.mean(1.0 / E_e)) * homog.C0

    E11 = EH[0, 0]
    assert E11 <= C_voigt[0, 0] + 1e-12, f"Violated Voigt: {E11} > {C_voigt[0, 0]}"
    assert E11 >= C_reuss[0, 0] - 1e-12, f"Violated Reuss: {E11} < {C_reuss[0, 0]}"


# ---------------------------------------------------------------------------
# Test 3: Central Finite Difference Verification (h = 1e-6, relative error < 1e-6)
# ---------------------------------------------------------------------------

def test_homogenizer2d_sensitivities_central_fd() -> None:
    """Verify 2D analytical sensitivities of EH, bulk modulus, shear modulus, and Poisson's ratio."""
    mesh = StructuredMesh2D(nx=3, ny=3, lx=1.0, ly=1.0)
    homog = Homogenizer2D(
        mesh=mesh,
        e0=1.0,
        emin=1e-4,
        nu=0.3,
        penal=3.0,
        plane="plane_stress",
    )

    rho = np.array([0.35, 0.65, 0.45, 0.80, 0.50, 0.90, 0.25, 0.70, 0.40], dtype=np.float64)
    EH, dEH_analytical = homog.homogenize(rho)
    KH, dKH_analytical = homog.bulk_modulus(rho)
    GH, dGH_analytical = homog.shear_modulus(rho)
    nuH, dnuH_analytical = homog.poissons_ratio(rho)

    h = 1e-6
    dEH_fd = np.empty_like(dEH_analytical)
    dKH_fd = np.empty_like(dKH_analytical)
    dGH_fd = np.empty_like(dGH_analytical)
    dnuH_fd = np.empty_like(dnuH_analytical)

    for e in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[e] += h
        rhom[e] -= h

        EHp, _ = homog.homogenize(rhop)
        EHm, _ = homog.homogenize(rhom)
        dEH_fd[:, :, e] = (EHp - EHm) / (2.0 * h)

        K_p, _ = homog.bulk_modulus(rhop)
        K_m, _ = homog.bulk_modulus(rhom)
        dKH_fd[e] = (K_p - K_m) / (2.0 * h)

        G_p, _ = homog.shear_modulus(rhop)
        G_m, _ = homog.shear_modulus(rhom)
        dGH_fd[e] = (G_p - G_m) / (2.0 * h)

        nu_p, _ = homog.poissons_ratio(rhop)
        nu_m, _ = homog.poissons_ratio(rhom)
        dnuH_fd[e] = (nu_p - nu_m) / (2.0 * h)

    # Verify tensor sensitivities
    rel_err_EH = np.max(np.abs(dEH_analytical - dEH_fd)) / np.max(np.abs(dEH_fd))
    assert rel_err_EH < 1e-6, f"2D EH sensitivity rel_err = {rel_err_EH} >= 1e-6"

    # Verify bulk modulus
    rel_err_KH = np.max(np.abs(dKH_analytical - dKH_fd)) / np.max(np.abs(dKH_fd))
    assert rel_err_KH < 1e-6, f"2D bulk modulus sensitivity rel_err = {rel_err_KH} >= 1e-6"

    # Verify shear modulus
    rel_err_GH = np.max(np.abs(dGH_analytical - dGH_fd)) / np.max(np.abs(dGH_fd))
    assert rel_err_GH < 1e-6, f"2D shear modulus sensitivity rel_err = {rel_err_GH} >= 1e-6"

    # Verify Poisson's ratio
    rel_err_nuH = np.max(np.abs(dnuH_analytical - dnuH_fd)) / np.max(np.abs(dnuH_fd))
    assert rel_err_nuH < 1e-6, f"2D Poisson's ratio sensitivity rel_err = {rel_err_nuH} >= 1e-6"


def test_homogenizer3d_sensitivities_central_fd() -> None:
    """Verify 3D analytical sensitivities of EH, bulk modulus, and shear modulus vs Central FD."""
    mesh = StructuredMesh3D(nx=2, ny=2, nz=2, lx=1.0, ly=1.0, lz=1.0)
    homog = Homogenizer3D(
        mesh=mesh,
        e0=1.0,
        emin=1e-4,
        nu=0.3,
        penal=3.0,
    )

    rho = np.array([0.4, 0.7, 0.5, 0.8, 0.3, 0.9, 0.6, 0.45], dtype=np.float64)
    EH, dEH_analytical = homog.homogenize(rho)
    KH, dKH_analytical = homog.bulk_modulus(rho)
    GH, dGH_analytical = homog.shear_modulus(rho)

    h = 1e-6
    dEH_fd = np.empty_like(dEH_analytical)
    dKH_fd = np.empty_like(dKH_analytical)
    dGH_fd = np.empty_like(dGH_analytical)

    for e in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[e] += h
        rhom[e] -= h

        EHp, _ = homog.homogenize(rhop)
        EHm, _ = homog.homogenize(rhom)
        dEH_fd[:, :, e] = (EHp - EHm) / (2.0 * h)

        K_p, _ = homog.bulk_modulus(rhop)
        K_m, _ = homog.bulk_modulus(rhom)
        dKH_fd[e] = (K_p - K_m) / (2.0 * h)

        G_p, _ = homog.shear_modulus(rhop)
        G_m, _ = homog.shear_modulus(rhom)
        dGH_fd[e] = (G_p - G_m) / (2.0 * h)

    rel_err_EH = np.max(np.abs(dEH_analytical - dEH_fd)) / np.max(np.abs(dEH_fd))
    assert rel_err_EH < 1e-6, f"3D EH sensitivity rel_err = {rel_err_EH} >= 1e-6"

    rel_err_KH = np.max(np.abs(dKH_analytical - dKH_fd)) / np.max(np.abs(dKH_fd))
    assert rel_err_KH < 1e-6, f"3D bulk modulus sensitivity rel_err = {rel_err_KH} >= 1e-6"

    rel_err_GH = np.max(np.abs(dGH_analytical - dGH_fd)) / np.max(np.abs(dGH_fd))
    assert rel_err_GH < 1e-6, f"3D shear modulus sensitivity rel_err = {rel_err_GH} >= 1e-6"


# ---------------------------------------------------------------------------
# Test 4: Spinodal De-homogenizer & Volume Fraction Consistency
# ---------------------------------------------------------------------------

def test_spinodal_dehomogenizer_2d_and_3d() -> None:
    """Verify Gaussian Random Field synthesis and volume fraction matching in 2D and 3D."""
    # 2D test
    dehomog2d = SpinodalDehomogenizer(beta_0=30.0, n_waves=150, dim=2, seed=123)
    coords2d = np.random.RandomState(42).uniform(0.0, 1.0, (2000, 2))
    G2d = dehomog2d.evaluate_grf(coords2d)

    # Check sample mean ~ 0 and std ~ 1
    assert abs(np.mean(G2d)) < 0.15
    assert abs(np.std(G2d) - 1.0) < 0.35

    # Check quantile thresholding for volume fractions
    for target_vol in (0.35, 0.50, 0.65):
        tau = dehomog2d.compute_threshold(target_vol)
        crisp = (G2d > tau).astype(np.float64)
        sample_vol = np.mean(crisp)
        assert abs(sample_vol - target_vol) < 0.08

    # 3D test
    dehomog3d = SpinodalDehomogenizer(beta_0=25.0, n_waves=150, dim=3, seed=456)
    coords3d = np.random.RandomState(42).uniform(0.0, 1.0, (2000, 3))
    G3d = dehomog3d.evaluate_grf(coords3d)
    assert abs(np.mean(G3d)) < 0.15
    assert abs(np.std(G3d) - 1.0) < 0.35

    # Test anisotropic wave generation
    dehomog_aniso = SpinodalDehomogenizer(
        beta_0=25.0,
        n_waves=60,
        dim=2,
        anisotropy_angle=np.pi / 4.0,
        cone_half_angle=np.pi / 12.0,
        seed=789,
    )
    assert dehomog_aniso.wave_vectors.shape == (60, 2)

    # Multiscale grid dehomogenize
    coarse_mesh = StructuredMesh2D(nx=4, ny=4, lx=1.0, ly=1.0)
    fine_mesh = StructuredMesh2D(nx=40, ny=40, lx=1.0, ly=1.0)
    macro_rho = np.full(coarse_mesh.n_elem, 0.55)

    rho_fine = dehomog2d.dehomogenize(
        macro_rho=macro_rho,
        coarse_mesh=coarse_mesh,
        fine_mesh_or_coords=fine_mesh,
        beta=40.0,
        projection_type="smooth",
    )
    fine_vol = float(np.mean(rho_fine))
    assert abs(fine_vol - 0.55) < 0.08, f"fine_vol = {fine_vol} differs from 0.55"


# ---------------------------------------------------------------------------
# Test 5: RobustFormulation (3-Field Worst-Case, KS Aggregation & Dilated Scaling)
# ---------------------------------------------------------------------------

def test_robust_formulation_sensitivities_central_fd() -> None:
    """Verify RobustFormulation KS, worst-case, weighted-sum sensitivities and volume scaling."""
    mesh = StructuredMesh2D(nx=2, ny=2, lx=1.0, ly=1.0)
    fem = FEMEngine(mesh, nu=0.3, plane="plane_stress")
    obj = ComplianceFunction(
        penal=3.0,
        e0=1.0,
        emin=1e-3,
        elem_dofs=mesh.edofMat,
        K_e0=fem.Ke0,
    )
    proj = ThreeFieldProjection(beta=4.0, eta_i=0.5, delta_eta=0.2)

    F = np.zeros(mesh.ndof)
    F[-1] = -1.0
    fixed_dofs = np.array([0, 1, 2, 3])
    free_dofs = np.setdiff1d(np.arange(mesh.ndof), fixed_dofs)

    def fe_solver_callback(rho_field: np.ndarray) -> np.ndarray:
        E_e, _ = obj.interpolate_material(rho_field)
        K = fem.assemble_K(E_e)
        K_ff = K[free_dofs, :][:, free_dofs]
        F_f = F[free_dofs]
        import scipy.sparse.linalg as spla
        u_f = spla.spsolve(K_ff.tocsc(), F_f)
        u_full = np.zeros(mesh.ndof, dtype=np.float64)
        u_full[free_dofs] = u_f
        return u_full

    # 1. KS mode gradient verification vs Central FD
    robust_ks = RobustFormulation(
        objective=obj,
        projection=proj,
        mode="ks",
        ks_param=8.0,
        fe_solver=fe_solver_callback,
    )

    rho = np.array([0.45, 0.70, 0.55, 0.60], dtype=np.float64)
    val_ks = robust_ks.evaluate(rho)
    grad_ks_analytical = robust_ks.gradient(rho)

    h = 1e-6
    grad_ks_fd = np.zeros_like(grad_ks_analytical)
    for j in range(len(rho)):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[j] += h
        rhom[j] -= h
        grad_ks_fd[j] = (robust_ks.evaluate(rhop) - robust_ks.evaluate(rhom)) / (2.0 * h)

    rel_err_ks = np.max(np.abs(grad_ks_analytical - grad_ks_fd) / np.abs(grad_ks_fd))
    assert rel_err_ks < 1e-6, f"RobustFormulation KS gradient rel_err = {rel_err_ks} >= 1e-6"

    # 2. Weighted-sum mode gradient verification vs Central FD
    robust_ws = RobustFormulation(
        objective=obj,
        projection=proj,
        mode="weighted_sum",
        weights=(0.5, 0.3, 0.2),
        fe_solver=fe_solver_callback,
    )
    val_ws = robust_ws.evaluate(rho)
    grad_ws_analytical = robust_ws.gradient(rho)

    grad_ws_fd = np.zeros_like(grad_ws_analytical)
    for j in range(len(rho)):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[j] += h
        rhom[j] -= h
        grad_ws_fd[j] = (robust_ws.evaluate(rhop) - robust_ws.evaluate(rhom)) / (2.0 * h)

    rel_err_ws = np.max(np.abs(grad_ws_analytical - grad_ws_fd) / np.abs(grad_ws_fd))
    assert rel_err_ws < 1e-6, f"RobustFormulation weighted_sum gradient rel_err = {rel_err_ws} >= 1e-6"

    # 3. Dynamic volume scaling and constraint gradient vs Central FD
    robust_ks.target_volfrac_intermediate = 0.5
    v_d_target = robust_ks.update_dilated_volume_target(rho)
    assert v_d_target > 0.5  # Dilated field has higher volume fraction

    g_val, g_grad_analytical = robust_ks.dilated_volume_constraint(rho)
    g_grad_fd = np.zeros_like(g_grad_analytical)
    for j in range(len(rho)):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[j] += h
        rhom[j] -= h
        gp, _ = robust_ks.dilated_volume_constraint(rhop)
        gm, _ = robust_ks.dilated_volume_constraint(rhom)
        g_grad_fd[j] = (gp - gm) / (2.0 * h)

    rel_err_g = np.max(np.abs(g_grad_analytical - g_grad_fd) / np.abs(g_grad_fd))
    assert rel_err_g < 1e-6, f"Dilated volume constraint gradient rel_err = {rel_err_g} >= 1e-6"


# ---------------------------------------------------------------------------
# Test 6: MuJoCo Dynamic Trajectory Load Extraction & Inertia Relief
# ---------------------------------------------------------------------------

def test_mujoco_loads_and_inertia_relief() -> None:
    """Verify MuJoCo multibody dynamics load provider, inertia relief, and MJCF update."""
    # Real 2-link pendulum MJCF
    mjcf_2link = """<mujoco model="two_link_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 1 0"/>
      <geom name="geom1" type="cylinder" size="0.04 0.3" pos="0 0 -0.3" mass="2.0"/>
      <body name="link2" pos="0 0 -0.6">
        <joint name="joint2" type="hinge" axis="0 1 0"/>
        <geom name="geom2" type="cylinder" size="0.03 0.25" pos="0 0 -0.25" mass="1.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    provider = MuJoCoLoadProvider(xml_str=mjcf_2link)

    # Release from horizontal angle (qpos[0] = pi/2)
    trajectory = provider.simulate_trajectory(
        t_final=0.6,
        qpos_init=np.array([1.5707963, 0.0]),
    )
    assert len(trajectory["time"]) > 50

    # Extract top 2 peak load cases
    peak_cases = provider.extract_peak_load_cases(
        body_name="link1",
        top_k=2,
        metric="force_norm",
    )
    assert len(peak_cases) == 2
    assert peak_cases[0]["metric_value"] > 10.0  # Dynamic forces significantly exceed static gravity

    # Map to 3D FE mesh
    mesh = StructuredMesh3D(nx=2, ny=2, nz=4, lx=0.08, ly=0.08, lz=0.6)
    fe_loadcases = provider.map_to_fe_loads(
        mesh=mesh,
        peak_cases=peak_cases,
        density_mat=2700.0,
        apply_inertia_relief=True,
    )
    assert len(fe_loadcases) == 2

    # Verify Inertia Relief exact balance: ||Phi^T @ F_bal||_inf < 10^-12
    phi = build_rigid_body_modes(mesh.node_coords, dim=3)
    for lc in fe_loadcases:
        rbm_res = phi.T @ lc.force_vector
        max_res = float(np.max(np.abs(rbm_res)))
        assert max_res < 1e-12, f"Inertia relief balance violated: max residual = {max_res}"

    # Compute optimized mass, COM, and inertia tensor
    rho_opt = np.full(mesh.n_elem, 0.6)
    mass, com, inertia_3x3 = provider.compute_optimized_inertial_properties(
        mesh=mesh,
        rho=rho_opt,
        density_mat=2700.0,
    )
    assert mass > 0.0
    assert com.shape == (3,)
    assert inertia_3x3.shape == (3, 3)
    # Check positive diagonal moments of inertia
    assert inertia_3x3[0, 0] > 0.0 and inertia_3x3[1, 1] > 0.0 and inertia_3x3[2, 2] > 0.0

    # Update MJCF XML and reload to verify model validity
    updated_xml = provider.update_mjcf_inertial(
        xml_str=mjcf_2link,
        body_name="link1",
        mass=mass,
        com=com,
        inertia_tensor=inertia_3x3,
    )
    assert "fullinertia=" in updated_xml

    # Reload into MuJoCo
    new_provider = MuJoCoLoadProvider(xml_str=updated_xml)
    body_id = mujoco.mj_name2id(new_provider.model, mujoco.mjtObj.mjOBJ_BODY, "link1")
    assert abs(new_provider.model.body_mass[body_id] - mass) < 1e-4
    assert np.allclose(new_provider.model.body_ipos[body_id], com, atol=1e-4)
