"""Showcase 04: Periodic Metamaterial Homogenization & Multiscale Spinodal De-homogenization.

Demonstrates:
1. Inverse homogenization of a 2D periodic unit cell to maximize effective bulk modulus K^H
   using Homogenizer2D and exact self-adjoint homogenization sensitivities.
2. Multiscale de-homogenization of a macroscopic 2D MBB beam density field onto a high-resolution
   fine grid using SpinodalDehomogenizer (Gaussian Random Field quantile thresholding).
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

from wht_topopt.algorithms.oc import OCOptimizer
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter
from wht_topopt.core.mesh import StructuredMesh2D
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.multiscale.dehomogenization import SpinodalDehomogenizer
from wht_topopt.multiscale.homogenization import Homogenizer2D
from wht_topopt.solvers.scipy_solver import ScipySolver

GLOBAL_FONT_SIZE: int = 9
GLOBAL_FIGSIZE_MULTISCALE: Tuple[float, float] = (12.0, 4.2)
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
class MetamaterialShowcaseResult:
    """Results container for periodic metamaterial homogenization & multiscale de-homogenization."""

    unit_cell_mesh: StructuredMesh2D
    unit_cell_rho: np.ndarray
    effective_tensor_EH: np.ndarray
    bulk_modulus_history: List[float]
    initial_bulk_modulus: float
    final_bulk_modulus: float
    macro_mesh: StructuredMesh2D
    macro_rho: np.ndarray
    fine_mesh: StructuredMesh2D
    dehomogenized_rho: np.ndarray
    exported_files: Dict[str, str]


class MetamaterialHomogenizationShowcase:
    """OOP Showcase for Periodic Unit-Cell Inverse Homogenization and Multiscale De-homogenization."""

    def __init__(
        self,
        cell_nx: int = 16,
        cell_ny: int = 16,
        cell_volfrac: float = 0.50,
        cell_iters: int = 15,
        macro_nx: int = 24,
        macro_ny: int = 12,
        fine_scale_factor: int = 5,
    ) -> None:
        """Initialize MetamaterialHomogenizationShowcase parameters.

        Parameters
        ----------
        cell_nx, cell_ny : int
            Unit cell mesh resolution.
        cell_volfrac : float
            Target volume fraction of unit cell microstructure.
        cell_iters : int
            Inverse homogenization iterations for bulk modulus maximization.
        macro_nx, macro_ny : int
            Coarse macroscopic MBB mesh resolution.
        fine_scale_factor : int
            Upsampling factor for fine-scale de-homogenized lattice grid.
        """
        self.cell_nx = int(cell_nx)
        self.cell_ny = int(cell_ny)
        self.cell_volfrac = float(cell_volfrac)
        self.cell_iters = int(cell_iters)
        self.macro_nx = int(macro_nx)
        self.macro_ny = int(macro_ny)
        self.fine_scale_factor = int(fine_scale_factor)

    def run(self, output_dir: Optional[Union[str, Path]] = None) -> MetamaterialShowcaseResult:
        """Execute periodic unit-cell bulk modulus maximization and macro-to-fine de-homogenization."""
        # 1. Unit Cell Inverse Homogenization (Bulk Modulus Maximization)
        cell_mesh = StructuredMesh2D(nx=self.cell_nx, ny=self.cell_ny, lx=1.0, ly=1.0)
        homog = Homogenizer2D(mesh=cell_mesh, e0=1.0, emin=1e-4, nu=0.3, penal=3.0)
        cell_filt = DensityFilter(cell_mesh, r_min=0.12)
        cell_oc = OCOptimizer(volfrac=self.cell_volfrac, move=0.1)

        # Initialize unit cell with central circular hole seed to break translational invariance
        centers = cell_mesh.elem_centers
        r_dist = np.sqrt((centers[:, 0] - 0.5) ** 2 + (centers[:, 1] - 0.5) ** 2)
        rho_cell = np.where(r_dist < 0.25, 0.15, 0.75).astype(np.float64)
        rho_cell *= self.cell_volfrac / np.mean(rho_cell)
        rho_cell = np.clip(rho_cell, 0.05, 0.99)

        bulk_hist: List[float] = []
        for _ in range(self.cell_iters):
            rho_tilde = cell_filt.forward(rho_cell)
            K_H, dKH_dtilde = homog.bulk_modulus(rho_tilde)
            bulk_hist.append(float(K_H))
            # To maximize K_H, minimize -K_H -> sensitivity is -dK_H/drho < 0
            df_dtilde = np.minimum(-dKH_dtilde, -1e-9)
            df_drho = cell_filt.backward(df_dtilde)
            rho_cell = cell_oc.update(rho_cell, df_drho, elem_volumes=cell_mesh.elem_volumes)

        rho_cell_phys = cell_filt.forward(rho_cell)
        EH_final, _ = homog.homogenize(rho_cell_phys)
        final_KH, _ = homog.bulk_modulus(rho_cell_phys)
        bulk_hist.append(float(final_KH))

        # 2. Macroscopic MBB Beam Optimization + Fine-Scale Spinodal De-Homogenization
        macro_mesh = StructuredMesh2D(nx=self.macro_nx, ny=self.macro_ny, lx=2.0, ly=1.0)
        fem = FEMEngine(macro_mesh, nu=0.3)
        bcm = BoundaryConditionManager.mbb_half_2d(macro_mesh, F_mag=1.0)
        solver = ScipySolver(method="direct")
        macro_filt = DensityFilter(macro_mesh, r_min=0.15)
        comp_fn = ComplianceFunction(fem=fem, solver=solver, bcm=bcm, penal=3.0, e0=1.0, emin=1e-6)
        macro_oc = OCOptimizer(volfrac=0.45, move=0.2)

        rho_macro = np.full(macro_mesh.n_elem, 0.45, dtype=np.float64)
        for _ in range(12):
            rt = macro_filt.forward(rho_macro)
            _, dc = comp_fn.value_and_gradient(rt)
            rho_macro = macro_oc.update(rho_macro, macro_filt.backward(dc), elem_volumes=macro_mesh.elem_volumes)
        rho_macro_phys = macro_filt.forward(rho_macro)

        fine_nx = self.macro_nx * self.fine_scale_factor
        fine_ny = self.macro_ny * self.fine_scale_factor
        fine_mesh = StructuredMesh2D(nx=fine_nx, ny=fine_ny, lx=2.0, ly=1.0)
        dehom = SpinodalDehomogenizer(beta_0=38.0, n_waves=120, dim=2, seed=42)
        rho_fine = dehom.dehomogenize(
            macro_rho=rho_macro_phys,
            coarse_mesh=macro_mesh,
            fine_mesh_or_coords=fine_mesh,
            beta=32.0,
            projection_type="smooth",
        )

        exported: Dict[str, str] = {}
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            exported = self._export_report(
                out_dir=out_dir,
                cell_mesh=cell_mesh,
                rho_cell=rho_cell_phys,
                EH=EH_final,
                bulk_hist=bulk_hist,
                macro_mesh=macro_mesh,
                rho_macro=rho_macro_phys,
                fine_mesh=fine_mesh,
                rho_fine=rho_fine,
            )

        return MetamaterialShowcaseResult(
            unit_cell_mesh=cell_mesh,
            unit_cell_rho=rho_cell_phys,
            effective_tensor_EH=EH_final,
            bulk_modulus_history=bulk_hist,
            initial_bulk_modulus=bulk_hist[0],
            final_bulk_modulus=bulk_hist[-1],
            macro_mesh=macro_mesh,
            macro_rho=rho_macro_phys,
            fine_mesh=fine_mesh,
            dehomogenized_rho=rho_fine,
            exported_files=exported,
        )

    def _export_report(
        self,
        out_dir: Path,
        cell_mesh: StructuredMesh2D,
        rho_cell: np.ndarray,
        EH: np.ndarray,
        bulk_hist: List[float],
        macro_mesh: StructuredMesh2D,
        rho_macro: np.ndarray,
        fine_mesh: StructuredMesh2D,
        rho_fine: np.ndarray,
        figsize_override: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Export Korean Matplotlib 9pt figures and JSON summary."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
        fsize = figsize_override if figsize_override is not None else GLOBAL_FIGSIZE_MULTISCALE
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=fsize)

        # Tile 2x2 unit cell for visual periodicity clarity
        cell_grid = rho_cell.reshape((cell_mesh.nely, cell_mesh.nelx))
        tiled_cell = np.tile(cell_grid, (2, 2))
        im0 = ax0.imshow(tiled_cell, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.set_title(f"2x2 주기 단위셀 역설계\n(유효 체적탄성계수 K^H = {bulk_hist[-1]:.3f})")
        ax0.set_xlabel("주기 격자 X")
        ax0.set_ylabel("주기 격자 Y")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(
            rho_macro.reshape((macro_mesh.nely, macro_mesh.nelx)),
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        ax1.set_title(f"거시 MBB 밀도장\n({macro_mesh.nelx}x{macro_mesh.nely})")
        ax1.set_xlabel("X 요소")
        ax1.set_ylabel("Y 요소")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(
            rho_fine.reshape((fine_mesh.nely, fine_mesh.nelx)),
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        ax2.set_title(f"미시 탈균질화 스피노달 격자\n({fine_mesh.nelx}x{fine_mesh.nely})")
        ax2.set_xlabel("미시 X 요소")
        ax2.set_ylabel("미시 Y 요소")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.tight_layout()
        plot_path = out_dir / "ex04_metamaterial_multiscale.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_path = out_dir / "ex04_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "showcase": "ex04_metamaterial_homogenization",
                    "initial_bulk_modulus": bulk_hist[0],
                    "final_bulk_modulus": bulk_hist[-1],
                    "effective_tensor_EH": EH.tolist(),
                    "dehomogenized_mean_density": float(np.mean(rho_fine)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "multiscale_plot": str(plot_path),
            "summary_json": str(summary_path),
        }


if __name__ == "__main__":
    showcase = MetamaterialHomogenizationShowcase()
    res = showcase.run(output_dir=Path(__file__).resolve().parent / "outputs" / "ex04_metamaterial")
    print(
        f"[Ex04 Homogenization] Effective Bulk Modulus K^H: "
        f"{res.initial_bulk_modulus:.4f} -> {res.final_bulk_modulus:.4f}"
    )
