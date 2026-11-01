"""Showcase 05: Closed-Loop 3D Robot Link MuJoCo Dynamics -> Inertia Relief -> TopOpt -> CAD & Re-Simulation.

Demonstrates the complete closed-loop robotics structural optimization workflow:
1. Forward multibody dynamics trajectory simulation of a 2-DOF robot manipulator in MuJoCo.
2. Peak dynamic wrench + angular/linear acceleration extraction on target link ('link1').
3. RBE3 joint force/moment coupling + D'Alembert nodal inertial forces + Inertia Relief P_IR projection.
4. 13-stage 3D Topology Optimization with DfAM overhang constraint.
5. Taubin-smoothed Marching Cubes surface extraction, Binary STL & ISO-10303-21 STEP B-Rep export.
6. Updated mass, center-of-mass, and 3x3 inertia tensor writeback into MJCF XML and closed-loop
   dynamic re-simulation showing reduced joint reaction wrench under identical actuator torque.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

from wht_topopt.pipeline.config import (
    BCConfig,
    FilterConfig,
    ManufacturingConfig,
    MaterialConfig,
    MeshConfig,
    MuJoCoConfig,
    OptimizerConfig,
    PostConfig,
    SolverConfig,
    TopOptConfig,
)
from wht_topopt.pipeline.report import TopOptReport
from wht_topopt.pipeline.runner import TopOptPipelineResult, TopOptRunner
from wht_topopt.robust.mujoco_loads import MuJoCoLoadProvider

GLOBAL_FONT_SIZE: int = 9
GLOBAL_FIGSIZE_CLOSED_LOOP: Tuple[float, float] = (10.5, 4.0)
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE


def _place_outside_legend(ax: plt.Axes, max_per_col: int = 5) -> None:
    """Place legend outside figure with automatic column scaling."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = max(1, int(np.ceil(len(handles) / max_per_col)))
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=ncol,
        borderaxespad=0.0,
        frameon=True,
    )


ROBOT_ARM_MJCF = """
<mujoco model="industrial_robot_arm_2dof">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 1.0">
      <joint name="shoulder_pitch" type="hinge" axis="0 1 0" damping="0.1"/>
      <inertial pos="0.20 0 0" mass="3.2" fullinertia="0.015 0.095 0.095 0 0 0"/>
      <geom type="box" pos="0.20 0 0" size="0.20 0.05 0.05"/>
      <body name="link2" pos="0.40 0 0">
        <joint name="elbow_pitch" type="hinge" axis="0 1 0" damping="0.05"/>
        <inertial pos="0.15 0 0" mass="1.2" fullinertia="0.005 0.025 0.025 0 0 0"/>
        <geom type="cylinder" fromto="0 0 0 0.30 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder_pitch" ctrlrange="-30 30" gear="1"/>
    <motor joint="elbow_pitch" ctrlrange="-15 15" gear="1"/>
  </actuator>
</mujoco>
""".strip()


@dataclass
class RobotLinkClosedLoopResult:
    """Results container for closed-loop MuJoCo robot link topology optimization."""

    pipeline_result: TopOptPipelineResult
    initial_mass_kg: float
    optimized_mass_kg: float
    mass_reduction_pct: float
    initial_peak_joint_force_n: float
    resimulated_peak_joint_force_n: float
    exported_files: Dict[str, str]


class RobotLinkMuJoCoShowcase:
    """OOP Showcase for Closed-Loop 3D Robot Link MuJoCo Dynamics + Inertia Relief + TopOpt."""

    def __init__(
        self,
        nelx: int = 8,
        nely: int = 4,
        nelz: int = 4,
        volfrac: float = 0.45,
        max_iter: int = 10,
        duration: float = 0.30,
    ) -> None:
        """Initialize RobotLinkMuJoCoShowcase parameters.

        Parameters
        ----------
        nelx, nely, nelz : int
            3D finite element grid resolution for robot link1.
        volfrac : float
            Target volume fraction.
        max_iter : int
            Number of optimization iterations.
        duration : float
            MuJoCo trajectory simulation duration in seconds.
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nelz = int(nelz)
        self.volfrac = float(volfrac)
        self.max_iter = int(max_iter)
        self.duration = float(duration)

    def run(self, output_dir: Optional[Union[str, Path]] = None) -> RobotLinkClosedLoopResult:
        """Execute initial MuJoCo trajectory, 13-stage 3D TopOpt, CAD export, and closed-loop re-simulation."""
        out_dir = Path(output_dir) if output_dir is not None else Path("outputs/ex05_robot_link")
        out_dir.mkdir(parents=True, exist_ok=True)

        stl_path = out_dir / "ex05_link1_optimized.stl"
        step_path = out_dir / "ex05_link1_optimized.step"
        xml_out_path = out_dir / "ex05_robot_arm_updated.xml"
        report_dir = out_dir / "pipeline_report"

        # Measure baseline trajectory and initial mass
        provider_init = MuJoCoLoadProvider(xml_str=ROBOT_ARM_MJCF)
        traj_init = provider_init.simulate_trajectory(t_final=self.duration, dt=0.002)
        peaks_init = provider_init.extract_peak_load_cases("link1", top_k=1)
        init_peak_force = float(peaks_init[0]["metric_value"])
        # Total initial link1 mass from density_mat * domain volume:
        # Domain is 0.40 x 0.10 x 0.10 = 0.004 m^3; with rho0 = 800 kg/m^3 -> initial solid mass = 3.2 kg
        rho0_density = 800.0
        initial_mass_kg = 0.40 * 0.10 * 0.10 * rho0_density

        cfg = TopOptConfig(
            name="3D 로봇 매니퓰레이터 링크1 폐루프 동역학 위상최적화",
            mesh=MeshConfig(
                dimension=3,
                nelx=self.nelx,
                nely=self.nely,
                nelz=self.nelz,
                lx=0.40,
                ly=0.10,
                lz=0.10,
            ),
            material=MaterialConfig(e0=5000.0, emin=1e-3, nu=0.3, rho0=rho0_density, penal=3.0),
            bc=BCConfig(preset="mujoco", force_magnitude=10.0),
            filter=FilterConfig(filter_type="cone", rmin=1.35),
            manufacturing=ManufacturingConfig(overhang=True, build_dir="+z"),
            solver=SolverConfig(backend="scipy"),
            optimizer=OptimizerConfig(algorithm="oc", volfrac=self.volfrac, max_iter=self.max_iter),
            mujoco=MuJoCoConfig(
                enabled=True,
                xml_string=ROBOT_ARM_MJCF,
                body_name="link1",
                duration=self.duration,
                dt=0.002,
                use_inertia_relief=True,
                update_mjcf_inertials=True,
                updated_xml_output_path=str(xml_out_path),
            ),
            post=PostConfig(
                extract_isosurface=True,
                iso_level=0.42,
                taubin_iterations=5,
                export_stl_path=str(stl_path),
                export_step_path=str(step_path),
                export_report_dir=str(report_dir),
            ),
        )

        runner = TopOptRunner(cfg)
        pipe_res = runner.run()

        opt_mass_kg, opt_com, opt_inertia = MuJoCoLoadProvider.compute_optimized_inertial_properties(
            mesh=pipe_res.mesh,
            rho=pipe_res.rho_phys,
            density_mat=rho0_density,
        )
        mass_reduction_pct = float((initial_mass_kg - opt_mass_kg) / initial_mass_kg * 100.0)

        # Closed-loop re-simulation with updated MJCF XML
        provider_updated = MuJoCoLoadProvider(xml_path=str(xml_out_path))
        traj_updated = provider_updated.simulate_trajectory(t_final=self.duration, dt=0.002)
        peaks_updated = provider_updated.extract_peak_load_cases("link1", top_k=1)
        resim_peak_force = float(peaks_updated[0]["metric_value"])

        # Export closed-loop trajectory comparison & ParaView 3D render
        extra_files = self._export_closed_loop_plots(
            out_dir=out_dir,
            pipe_res=pipe_res,
            traj_init=traj_init,
            traj_updated=traj_updated,
            initial_mass_kg=initial_mass_kg,
            opt_mass_kg=opt_mass_kg,
            mass_reduction_pct=mass_reduction_pct,
            init_peak_force=init_peak_force,
            resim_peak_force=resim_peak_force,
            opt_com=opt_com,
            opt_inertia=opt_inertia,
        )

        all_files = dict(pipe_res.exported_files)
        all_files.update(extra_files)

        return RobotLinkClosedLoopResult(
            pipeline_result=pipe_res,
            initial_mass_kg=initial_mass_kg,
            optimized_mass_kg=opt_mass_kg,
            mass_reduction_pct=mass_reduction_pct,
            initial_peak_joint_force_n=init_peak_force,
            resimulated_peak_joint_force_n=resim_peak_force,
            exported_files=all_files,
        )

    def _export_closed_loop_plots(
        self,
        out_dir: Path,
        pipe_res: TopOptPipelineResult,
        traj_init: Dict[str, Any],
        traj_updated: Dict[str, Any],
        initial_mass_kg: float,
        opt_mass_kg: float,
        mass_reduction_pct: float,
        init_peak_force: float,
        resim_peak_force: float,
        opt_com: np.ndarray,
        opt_inertia: np.ndarray,
        figsize_override: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Export closed-loop MuJoCo trajectory comparison plot, PyVista 3D screenshot, and JSON summary."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
        fsize = figsize_override if figsize_override is not None else GLOBAL_FIGSIZE_CLOSED_LOOP
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fsize)

        # Body 1 is 'link1'
        t_arr = traj_init["time"]
        w_init = traj_init["cfrc_int"][:, 1, 3:6] + traj_init["cfrc_ext"][:, 1, 3:6]
        w_upd = traj_updated["cfrc_int"][:, 1, 3:6] + traj_updated["cfrc_ext"][:, 1, 3:6]
        f_norm_init = np.linalg.norm(w_init, axis=1)
        f_norm_upd = np.linalg.norm(w_upd, axis=1)

        ax1.plot(t_arr, f_norm_init, "r--", linewidth=1.4, label=f"초기 솔리드 링크 ({initial_mass_kg:.2f} kg)")
        ax1.plot(t_arr, f_norm_upd, "b-", linewidth=1.5, label=f"위상최적화 링크 ({opt_mass_kg:.2f} kg, -{mass_reduction_pct:.1f}%)")
        ax1.set_xlabel("시뮬레이션 시간 (s)")
        ax1.set_ylabel("관절 반력 크기 ||F_joint|| (N)")
        ax1.set_title("폐루프 MuJoCo 동역학 재시뮬레이션 관절 반력 비교")
        ax1.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax1)

        iters = [s.iteration for s in pipe_res.history]
        objs = [s.obj for s in pipe_res.history]
        ax2.plot(iters, objs, "g-o", markersize=3, linewidth=1.5, label="컴플라이언스 (Compliance)")
        ax2.set_xlabel("반복 횟수 (Iteration)")
        ax2.set_ylabel("컴플라이언스")
        ax2.set_title("관성 릴리프 평형 하중 기반 3D 최적설계 수렴도")
        ax2.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax2)

        fig.tight_layout()
        traj_plot_path = out_dir / "ex05_closed_loop_trajectory.png"
        fig.savefig(traj_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # PyVista 3D ParaView-style dark render
        pv_png_path = out_dir / "ex05_pyvista_robot_link_3d.png"
        rep = TopOptReport(mesh=pipe_res.mesh, history=pipe_res.history, rho_phys=pipe_res.rho_phys)
        plotter, _ = rep.create_pyvista_plotter(
            threshold=0.35,
            off_screen=True,
        )
        plotter.screenshot(str(pv_png_path))
        plotter.close()

        summary_path = out_dir / "ex05_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "showcase": "ex05_robot_link_mujoco_closed_loop",
                    "initial_mass_kg": initial_mass_kg,
                    "optimized_mass_kg": opt_mass_kg,
                    "mass_reduction_pct": mass_reduction_pct,
                    "initial_peak_joint_force_n": init_peak_force,
                    "resimulated_peak_joint_force_n": resim_peak_force,
                    "optimized_com_m": opt_com.tolist(),
                    "optimized_inertia_tensor": opt_inertia.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "closed_loop_plot": str(traj_plot_path),
            "pyvista_3d_png": str(pv_png_path),
            "summary_json": str(summary_path),
        }


if __name__ == "__main__":
    showcase = RobotLinkMuJoCoShowcase()
    res = showcase.run(output_dir=Path(__file__).resolve().parent / "outputs" / "ex05_robot_link")
    print(
        f"[Ex05 Robot Link Closed-Loop] Mass: {res.initial_mass_kg:.3f} kg -> "
        f"{res.optimized_mass_kg:.3f} kg (-{res.mass_reduction_pct:.1f}%), "
        f"Peak Joint Force: {res.initial_peak_joint_force_n:.2f} N -> {res.resimulated_peak_joint_force_n:.2f} N"
    )
