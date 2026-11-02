# Moved from examples/ex03_cantilever_3d_dfam_cnc.py (original in src_bak); not yet rebuilt on src_v1/common, output folder changed.
"""Showcase 03: 3D Cantilever Topology Optimization with DfAM Overhang, CNC Milling & Casting Constraints.

Compares 3D cantilever topology optimization across:
1. Unconstrained baseline (density filter only)
2. Additive Manufacturing (DfAM) 45-degree self-supporting OverhangFilter (Langelaar 2016/2017)
3. Subtractive CNC MillingFilter (Langelaar 2018)
4. Split-mold CastingFilter (Gersborg & Sigmund 2006)
Exports 2D mid-plane comparison plots, ParaView-style 3D PyVista renders, and binary STL / ISO-10303-21 STEP files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.pipeline.config import (
    BCConfig,
    FilterConfig,
    ManufacturingConfig,
    MaterialConfig,
    MeshConfig,
    OptimizerConfig,
    PostConfig,
    SolverConfig,
    TopOptConfig,
)
from wht_topopt.pipeline.report import TopOptReport
from wht_topopt.pipeline.runner import TopOptPipelineResult, TopOptRunner

GLOBAL_FONT_SIZE: int = 9
GLOBAL_FIGSIZE_DFAM: Tuple[float, float] = (12.0, 4.0)
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


@dataclass
class Cantilever3DDfAMResult:
    """Results container for 3D manufacturing-constrained cantilever showcase."""

    mesh: StructuredMesh3D
    variants: Dict[str, TopOptPipelineResult]
    exported_files: Dict[str, str]


class Cantilever3DDfAMShowcase:
    """OOP Showcase comparing 3D cantilever designs under DfAM Overhang, CNC Milling, and Casting constraints."""

    def __init__(
        self,
        nelx: int = 10,
        nely: int = 4,
        nelz: int = 4,
        volfrac: float = 0.45,
        max_iter: int = 10,
    ) -> None:
        """Initialize 3D manufacturing showcase parameters.

        Parameters
        ----------
        nelx, nely, nelz : int
            3D voxel grid dimensions.
        volfrac : float
            Target volume fraction.
        max_iter : int
            Optimization iterations per manufacturing variant.
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.nelz = int(nelz)
        self.volfrac = float(volfrac)
        self.max_iter = int(max_iter)

    def run(self, output_dir: Optional[Union[str, Path]] = None) -> Cantilever3DDfAMResult:
        """Execute baseline, DfAM overhang, CNC milling, and casting 3D optimization runs."""
        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        configs: Dict[str, TopOptConfig] = {
            "baseline": TopOptConfig(
                name="3D 기본 외팔보 (Baseline)",
                mesh=MeshConfig(dimension=3, nelx=self.nelx, nely=self.nely, nelz=self.nelz, lx=2.5, ly=1.0, lz=1.0),
                material=MaterialConfig(e0=1.0, emin=1e-6, penal=3.0),
                bc=BCConfig(preset="cantilever_3d", force_magnitude=1.0),
                filter=FilterConfig(filter_type="cone", rmin=1.3),
                solver=SolverConfig(backend="scipy"),
                optimizer=OptimizerConfig(algorithm="oc", volfrac=self.volfrac, max_iter=self.max_iter),
                post=PostConfig(extract_isosurface=False),
            ),
            "overhang": TopOptConfig(
                name="적층제조 45° 자기지지 오버행 (DfAM Overhang)",
                mesh=MeshConfig(dimension=3, nelx=self.nelx, nely=self.nely, nelz=self.nelz, lx=2.5, ly=1.0, lz=1.0),
                material=MaterialConfig(e0=1.0, emin=1e-6, penal=3.0),
                bc=BCConfig(preset="cantilever_3d", force_magnitude=1.0),
                filter=FilterConfig(filter_type="cone", rmin=1.3),
                manufacturing=ManufacturingConfig(overhang=True, build_dir="+z"),
                solver=SolverConfig(backend="scipy"),
                optimizer=OptimizerConfig(algorithm="oc", volfrac=self.volfrac, max_iter=self.max_iter),
                post=PostConfig(
                    extract_isosurface=out_dir is not None,
                    iso_level=0.42,
                    taubin_iterations=4,
                    export_stl_path=str(out_dir / "ex03_overhang.stl") if out_dir else None,
                    export_step_path=str(out_dir / "ex03_overhang.step") if out_dir else None,
                ),
            ),
            "milling": TopOptConfig(
                name="다축 CNC 밀링 (Subtractive Milling)",
                mesh=MeshConfig(dimension=3, nelx=self.nelx, nely=self.nely, nelz=self.nelz, lx=2.5, ly=1.0, lz=1.0),
                material=MaterialConfig(e0=1.0, emin=1e-6, penal=3.0),
                bc=BCConfig(preset="cantilever_3d", force_magnitude=1.0),
                filter=FilterConfig(filter_type="cone", rmin=1.3),
                manufacturing=ManufacturingConfig(milling=True, tool_dirs=["+z", "-z", "+y"]),
                solver=SolverConfig(backend="scipy"),
                optimizer=OptimizerConfig(algorithm="oc", volfrac=self.volfrac, max_iter=self.max_iter),
                post=PostConfig(extract_isosurface=False),
            ),
            "casting": TopOptConfig(
                name="분할 금형 주조 (Split-Mold Casting)",
                mesh=MeshConfig(dimension=3, nelx=self.nelx, nely=self.nely, nelz=self.nelz, lx=2.5, ly=1.0, lz=1.0),
                material=MaterialConfig(e0=1.0, emin=1e-6, penal=3.0),
                bc=BCConfig(preset="cantilever_3d", force_magnitude=1.0),
                filter=FilterConfig(filter_type="cone", rmin=1.3),
                manufacturing=ManufacturingConfig(casting=True, draw_dir="+z", casting_mode="parting_plane"),
                solver=SolverConfig(backend="scipy"),
                optimizer=OptimizerConfig(algorithm="oc", volfrac=self.volfrac, max_iter=self.max_iter),
                post=PostConfig(extract_isosurface=False),
            ),
        }

        variants: Dict[str, TopOptPipelineResult] = {}
        for key, cfg in configs.items():
            runner = TopOptRunner(cfg)
            variants[key] = runner.run()

        mesh = variants["baseline"].mesh
        assert isinstance(mesh, StructuredMesh3D)

        exported: Dict[str, str] = {}
        if out_dir is not None:
            exported = self._export_comparison(out_dir, mesh, variants)

        return Cantilever3DDfAMResult(
            mesh=mesh,
            variants=variants,
            exported_files=exported,
        )

    def _export_comparison(
        self,
        out_dir: Path,
        mesh: StructuredMesh3D,
        variants: Dict[str, TopOptPipelineResult],
        figsize_override: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Export 2D mid-plane comparison, convergence plot, PyVista 3D screenshot, and JSON summary."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
        fsize = figsize_override if figsize_override is not None else GLOBAL_FIGSIZE_DFAM
        fig, axes = plt.subplots(1, 4, figsize=fsize)

        labels_ko = {
            "baseline": "기본 설계 (Baseline)",
            "overhang": "적층제조 오버행 (DfAM)",
            "milling": "다축 CNC 밀링",
            "casting": "분할 금형 주조",
        }
        mid_y = mesh.nely // 2

        for ax, (key, res) in zip(axes, variants.items()):
            grid3d = res.rho_phys.reshape((mesh.nelz, mesh.nely, mesh.nelx))
            xz_slice = grid3d[:, mid_y, :]
            im = ax.imshow(xz_slice, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_title(f"{labels_ko[key]}\n(C = {res.final_objective:.2f})")
            ax.set_xlabel("X 요소")
            ax.set_ylabel("Z 적층/발형 축")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        slice_plot_path = out_dir / "ex03_dfam_cnc_casting_slices.png"
        fig.savefig(slice_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Convergence plot with outside legend
        fig_c, ax_c = plt.subplots(1, 1, figsize=(8.5, 3.8))
        for key, res in variants.items():
            iters = [s.iteration for s in res.history]
            objs = [s.obj for s in res.history]
            ax_c.plot(iters, objs, "-o", markersize=3, linewidth=1.4, label=labels_ko[key])
        ax_c.set_xlabel("반복 횟수 (Iteration)")
        ax_c.set_ylabel("컴플라이언스 (Compliance)")
        ax_c.set_title("제조 제약조건별 3D 외팔보 수렴 이력 비교")
        ax_c.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax_c)
        fig_c.tight_layout()
        conv_plot_path = out_dir / "ex03_dfam_convergence.png"
        fig_c.savefig(conv_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_c)

        # PyVista 3D ParaView-style dark render of DfAM Overhang design
        overhang_res = variants["overhang"]
        rep = TopOptReport(mesh=mesh, history=overhang_res.history, rho_phys=overhang_res.rho_phys)
        pv_png_path = out_dir / "ex03_pyvista_overhang_3d.png"
        plotter, _ = rep.create_pyvista_plotter(
            threshold=0.35,
            off_screen=True,
        )
        plotter.screenshot(str(pv_png_path))
        plotter.close()

        summary_path = out_dir / "ex03_summary.json"
        summary_data = {
            "showcase": "ex03_cantilever_3d_dfam_cnc",
            "variants": {
                k: {
                    "final_compliance": v.final_objective,
                    "final_volume_fraction": v.final_volume_fraction,
                    "iterations": v.iterations,
                }
                for k, v in variants.items()
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        out_files = {
            "slice_plot": str(slice_plot_path),
            "convergence_plot": str(conv_plot_path),
            "pyvista_3d_png": str(pv_png_path),
            "summary_json": str(summary_path),
        }
        out_files.update(overhang_res.exported_files)
        return out_files


if __name__ == "__main__":
    showcase = Cantilever3DDfAMShowcase()
    res = showcase.run(output_dir=Path(__file__).resolve().parents[2] / "outputs_v1" / "ex03_cantilever_3d_mfg/ex03a_cantilever_dfam_cnc")
    for name, v in res.variants.items():
        print(f"[Ex03 {name}] Final Compliance = {v.final_objective:.4f}, Vol = {v.final_volume_fraction:.4f}")
