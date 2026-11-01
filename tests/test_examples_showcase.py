"""Automated verification tests for the 5 industrial flagship showcases (wht_topopt/examples/)."""
from __future__ import annotations

from pathlib import Path
import pytest

from wht_topopt.examples import (
    BeamsBenchmarkSuite,
    Cantilever3DDfAMShowcase,
    ColumnBucklingShowcase,
    LBracketStressShowcase,
    MetamaterialHomogenizationShowcase,
    RobotLinkMuJoCoShowcase,
)


def test_ex01_lbracket_stress_showcase(tmp_path: Path) -> None:
    """Verify Showcase 01: L-Bracket P-Norm stress-constrained MMA reduces peak von Mises stress."""
    showcase = LBracketStressShowcase(nelx=12, nely=12, volfrac=0.40, max_iter=15)
    res = showcase.run(output_dir=tmp_path / "ex01")
    assert res.peak_vm_stress_constrained < res.peak_vm_compliance_only
    assert res.stress_reduction_pct > 0.0
    assert Path(res.exported_files["comparison_plot"]).exists()
    assert Path(res.exported_files["history_plot"]).exists()
    assert Path(res.exported_files["summary_json"]).exists()


def test_ex02_column_buckling_showcase(tmp_path: Path) -> None:
    """Verify Showcase 02: Compressed column buckling optimization increases critical load factor lambda_1."""
    showcase = ColumnBucklingShowcase(nelx=10, nely=20, volfrac=0.45, max_iter=12)
    res = showcase.run(output_dir=tmp_path / "ex02")
    assert res.final_blf > res.initial_blf
    assert res.blf_improvement_pct > 0.0
    assert Path(res.exported_files["buckling_plot"]).exists()
    assert Path(res.exported_files["summary_json"]).exists()


def test_ex03_cantilever_3d_dfam_showcase(tmp_path: Path) -> None:
    """Verify Showcase 03: 3D Cantilever under DfAM Overhang, CNC Milling, and Split-Mold Casting."""
    showcase = Cantilever3DDfAMShowcase(nelx=8, nely=4, nelz=4, volfrac=0.45, max_iter=8)
    res = showcase.run(output_dir=tmp_path / "ex03")
    assert set(res.variants.keys()) == {"baseline", "overhang", "milling", "casting"}
    for v in res.variants.values():
        assert v.history[-1].obj < v.history[0].obj
    assert Path(res.exported_files["slice_plot"]).exists()
    assert Path(res.exported_files["pyvista_3d_png"]).exists()
    assert Path(res.exported_files["stl"]).exists()
    assert Path(res.exported_files["step"]).exists()


def test_ex04_metamaterial_homogenization_showcase(tmp_path: Path) -> None:
    """Verify Showcase 04: Periodic unit-cell bulk modulus maximization and spinodal de-homogenization."""
    showcase = MetamaterialHomogenizationShowcase(
        cell_nx=12,
        cell_ny=12,
        cell_iters=10,
        macro_nx=16,
        macro_ny=8,
        fine_scale_factor=4,
    )
    res = showcase.run(output_dir=tmp_path / "ex04")
    assert res.final_bulk_modulus > res.initial_bulk_modulus
    assert res.dehomogenized_rho.shape == (res.fine_mesh.n_elem,)
    assert Path(res.exported_files["multiscale_plot"]).exists()
    assert Path(res.exported_files["summary_json"]).exists()


def test_ex05_robot_link_mujoco_closed_loop_showcase(tmp_path: Path) -> None:
    """Verify Showcase 05: Closed-loop 3D robot link MuJoCo dynamics + Inertia Relief + CAD & MJCF re-simulation."""
    showcase = RobotLinkMuJoCoShowcase(nelx=6, nely=3, nelz=3, volfrac=0.45, max_iter=8, duration=0.20)
    res = showcase.run(output_dir=tmp_path / "ex05")
    assert res.optimized_mass_kg < res.initial_mass_kg
    assert res.mass_reduction_pct > 30.0
    assert res.resimulated_peak_joint_force_n < res.initial_peak_joint_force_n
    assert Path(res.exported_files["stl"]).exists()
    assert Path(res.exported_files["step"]).exists()
    assert Path(res.exported_files["mujoco_xml"]).exists()
    assert Path(res.exported_files["closed_loop_plot"]).exists()
    assert Path(res.exported_files["pyvista_3d_png"]).exists()


def test_ex06_numba_acceleration_and_beams_benchmark(tmp_path: Path) -> None:
    """Verify Showcase 06: Numba JIT core parity (< 1e-13), MKL PARDISO solver, and 2D beam benchmark suite."""
    suite = BeamsBenchmarkSuite(output_dir=tmp_path / "beams", solver_backend="pardiso", quick=True)
    sp_res = suite.run_speed_benchmark()
    assert Path(sp_res["png"]).exists()
    assert Path(sp_res["md"]).exists()
    assert sp_res["parity_errors"]["K"] < 1e-13
    assert sp_res["parity_errors"]["c_e0"] < 1e-13
    assert sp_res["parity_errors"]["overhang"] < 1e-13

    mbb_res = suite.run_case(case="mbb", nx=24, ny=8, volfrac=0.4, rmin=1.5)
    assert Path(mbb_res["png"]).exists()
    assert Path(mbb_res["md"]).exists()
    assert len(mbb_res["records"]) == 5
    for rec in mbb_res["records"]:
        assert rec.error is None
        assert rec.compliance > 0.0
        assert rec.common_compliance_pardiso > 0.0
