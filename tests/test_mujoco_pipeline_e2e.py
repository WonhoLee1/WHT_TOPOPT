"""End-to-end tests for Milestone 6: Post-processing, Neural interfaces, 13-stage TopOptRunner, Report, and aieng bridge."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D
from wht_topopt.pipeline.config import (
    BCConfig,
    ContinuationConfig,
    FilterConfig,
    ManufacturingConfig,
    MaterialConfig,
    MeshConfig,
    MuJoCoConfig,
    OptimizerConfig,
    PostConfig,
    ProjectionConfig,
    RobustConfig,
    SolverConfig,
    TopOptConfig,
)
from wht_topopt.pipeline.report import (
    DEFAULT_FONT_SIZE,
    DEFAULT_PYVISTA_BG_COLOR,
    DEFAULT_PYVISTA_EDGE_COLOR,
    DEFAULT_PYVISTA_FONT_SIZE,
    TopOptReport,
)
from wht_topopt.pipeline.runner import TopOptRunner


PENDULUM_MJCF = """
<mujoco model="robot_link_e2e">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 1.0">
      <joint name="pin1" type="hinge" axis="0 1 0" damping="0.05"/>
      <inertial pos="0.25 0 0" mass="2.5" diaginertia="0.02 0.08 0.08"/>
      <geom type="box" pos="0.25 0 0" size="0.25 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="pin1" ctrlrange="-20 20" gear="1"/>
  </actuator>
</mujoco>
""".strip()


def test_topopt_config_json_roundtrip(tmp_path: Path) -> None:
    """Verify TopOptConfig Pydantic v2 validation and UTF-8 JSON round-trip."""
    cfg = TopOptConfig(
        name="roundtrip_test",
        mesh=MeshConfig(dimension=3, nelx=8, nely=4, nelz=2),
        optimizer=OptimizerConfig(algorithm="mma", volfrac=0.45, max_iter=15),
        manufacturing=ManufacturingConfig(overhang=True, build_dir="+z"),
    )
    json_path = tmp_path / "config.json"
    cfg.to_json(json_path)
    assert json_path.exists()

    loaded = TopOptConfig.from_json(json_path)
    assert loaded.name == "roundtrip_test"
    assert loaded.mesh.dimension == 3
    assert loaded.mesh.nelx == 8
    assert loaded.optimizer.algorithm == "mma"
    assert np.isclose(loaded.optimizer.volfrac, 0.45)
    assert loaded.manufacturing.overhang is True


def test_13_stage_pipeline_2d_mbb_and_report(tmp_path: Path) -> None:
    """Verify 13-stage TopOptRunner on 2D MBB beam and 2D Korean Matplotlib reporting."""
    report_dir = tmp_path / "report_2d"
    cfg = TopOptConfig(
        name="2D MBB 위상최적화 테스트",
        mesh=MeshConfig(dimension=2, nelx=20, nely=8),
        material=MaterialConfig(e0=1.0, emin=1e-9, penal=3.0),
        bc=BCConfig(preset="mbb_2d", force_magnitude=1.0),
        filter=FilterConfig(filter_type="cone", rmin=1.5),
        projection=ProjectionConfig(enabled=True, beta_init=2.0, eta=0.5),
        continuation=ContinuationConfig(enabled=True, penal_init=2.0, penal_max=3.0, interval=5),
        solver=SolverConfig(backend="scipy"),
        optimizer=OptimizerConfig(algorithm="oc", volfrac=0.5, max_iter=12, tol_change=1e-3),
        post=PostConfig(
            extract_isosurface=True,
            iso_level=0.5,
            taubin_iterations=3,
            export_report_dir=str(report_dir),
        ),
    )

    runner = TopOptRunner(cfg)
    res = runner.run()

    assert len(res.stage_trace) == 13
    assert res.stage_trace[0] == "stage_01_mesh"
    assert res.stage_trace[-1] == "stage_13_report"
    assert res.iterations == 12
    assert res.history[-1].obj < res.history[0].obj
    assert abs(res.final_volume_fraction - 0.5) < 0.05

    assert (report_dir / "convergence.png").exists()
    assert (report_dir / "density_2d.png").exists()
    assert (report_dir / "summary.md").exists()


def test_13_stage_pipeline_3d_mujoco_post_and_pyvista(tmp_path: Path) -> None:
    """Verify full 3D MuJoCo -> Inertia Relief -> TopOptRunner -> STL/STEP/MJCF -> PyVista report."""
    stl_path = tmp_path / "link1_opt.stl"
    step_path = tmp_path / "link1_opt.step"
    xml_out_path = tmp_path / "robot_link_updated.xml"
    report_dir = tmp_path / "report_3d"

    cfg = TopOptConfig(
        name="3D 로봇 링크 MuJoCo 동역학 위상최적화",
        mesh=MeshConfig(dimension=3, nelx=6, nely=3, nelz=3, lx=0.5, ly=0.1, lz=0.1),
        material=MaterialConfig(e0=2000.0, emin=1e-3, nu=0.3, rho0=10.0, penal=3.0),
        bc=BCConfig(preset="mujoco", force_magnitude=5.0),
        filter=FilterConfig(filter_type="cone", rmin=1.4),
        manufacturing=ManufacturingConfig(overhang=True, build_dir="+z"),
        solver=SolverConfig(backend="scipy"),
        optimizer=OptimizerConfig(algorithm="oc", volfrac=0.5, max_iter=8, tol_change=1e-3),
        mujoco=MuJoCoConfig(
            enabled=True,
            xml_string=PENDULUM_MJCF,
            body_name="link1",
            duration=0.2,
            dt=0.002,
            use_inertia_relief=True,
            update_mjcf_inertials=True,
            updated_xml_output_path=str(xml_out_path),
        ),
        post=PostConfig(
            extract_isosurface=True,
            iso_level=0.45,
            taubin_iterations=4,
            export_stl_path=str(stl_path),
            export_step_path=str(step_path),
            export_report_dir=str(report_dir),
        ),
    )

    runner = TopOptRunner(cfg)
    res = runner.run()

    assert res.mujoco_summary is not None
    assert res.mujoco_summary["balanced_load_norm"] > 0.0
    assert xml_out_path.exists()
    assert stl_path.exists()
    assert step_path.exists()

    # Verify PyVista ParaView-style 3D plotter & in-place animation frame update
    reporter = TopOptReport(mesh=res.mesh, history=res.history, rho_phys=res.rho_phys)
    plotter, pv_grid = reporter.create_pyvista_plotter(off_screen=True)

    # Check single scalar bar & dark theme rules
    assert len(plotter.scalar_bars) == 1
    assert DEFAULT_PYVISTA_FONT_SIZE == 12
    assert DEFAULT_FONT_SIZE == 9

    # Test interactive context menu callbacks (XY, YZ, ZX, Perspective toggle)
    callbacks = getattr(plotter, "_wht_view_menu_callbacks", {})
    assert {"xy", "yz", "zx", "toggle_perspective", "right_click_cycle"}.issubset(callbacks.keys())
    callbacks["xy"]()
    callbacks["yz"]()
    callbacks["zx"]()
    callbacks["toggle_perspective"]()
    callbacks["right_click_cycle"]()

    # Test in-place animation scalar update (without recreating actor or adding extra colorbars)
    new_rho = np.clip(res.rho_phys * 0.9, 0.0, 1.0)
    reporter.update_pyvista_animation_frame(plotter, pv_grid, new_rho)
    assert np.allclose(pv_grid.cell_data["density"], new_rho)
    assert len(plotter.scalar_bars) == 1
    plotter.close()


def test_aieng_wht_bridge_compatibility() -> None:
    """Verify aieng.topopt.run_wht_topopt bridge executes cleanly and returns TopOptResult."""
    from aieng.topopt import benchmarks, run_wht_topopt

    prob = benchmarks.zhou_rozvany(volfrac=0.5, max_iters=6)
    res = run_wht_topopt(prob, backend="scipy")

    assert res.iterations == 6
    assert res.xphys.shape == (prob.mesh.n_elems,)
    assert res.compliance > 0.0
    assert abs(res.volume_fraction - prob.volfrac) < 0.05
    assert "stage_trace" in res.extras


def test_post_processing_and_neural_interfaces(tmp_path: Path) -> None:
    """Verify IsosurfaceExtractor, STLExporter, CADReconstructor, GINOTInterface, and TONICInitializer."""
    from wht_topopt.core.boundary import BoundaryConditionManager
    from wht_topopt.neural import GINOTInterface, TONICInitializer
    from wht_topopt.post import CADReconstructor, IsosurfaceExtractor, STLExporter

    mesh2d = StructuredMesh2D(nx=12, ny=6, lx=2.0, ly=1.0)
    bcm2d = BoundaryConditionManager.cantilever_2d(mesh2d, F_mag=1.0)

    # 1. GINOT warm-start prediction
    ginot = GINOTInterface(latent_dim=16, num_fourier_features=8, seed=7)
    ws = ginot.predict_warm_start(mesh2d, bcm2d, target_vol_frac=0.5)
    assert ws.density.shape == (mesh2d.n_elem,)
    assert abs(ws.actual_vol_frac - 0.5) < 1e-3

    # 2. TONIC coordinate MLP warm-start and sensitivity backprop
    tonic = TONICInitializer(ndim=2, hidden_dim=16, num_fourier_features=8, seed=7)
    rho_tonic = tonic.warm_start(mesh2d, bcm2d, target_vol_frac=0.5, num_pretrain_steps=10)
    assert rho_tonic.shape == (mesh2d.n_elem,)
    assert abs(float(np.mean(rho_tonic)) - 0.5) < 1e-2

    grads = tonic.project_sensitivities_to_weights(np.ones(mesh2d.n_elem))
    assert "W1" in grads and grads["W1"].shape == tonic.W1.shape

    # 3. 2D contour extraction, B-spline fitting, DXF & STEP export
    ext2d = IsosurfaceExtractor(mesh2d)
    contours = ext2d.extract_2d(ws.density, tau=0.5)
    assert len(contours) > 0
    _, _, smooth_pts = CADReconstructor.fit_bspline_2d(contours[0], num_eval_points=60)
    assert smooth_pts.shape == (60, 2)

    dxf_path = tmp_path / "contour_2d.dxf"
    CADReconstructor.export_dxf_2d(contours, dxf_path)
    assert dxf_path.exists() and dxf_path.stat().st_size > 0

    verts3d, faces3d = ext2d.extrude_2d_to_3d(contours, thickness=0.5)
    assert len(verts3d) > 0 and len(faces3d) > 0
    quality = STLExporter.verify_mesh(verts3d, faces3d)
    assert quality["volume"] > 0.0

    vtk_path = tmp_path / "mesh2d.vtk"
    STLExporter.export_vtk(vtk_path, mesh=mesh2d, element_fields={"density": ws.density})
    assert vtk_path.exists() and vtk_path.stat().st_size > 0
