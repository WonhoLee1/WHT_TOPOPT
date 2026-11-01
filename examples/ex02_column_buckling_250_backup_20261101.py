"""Showcase 02: Buckling-Constrained Column Topology Optimization (Ferrari, Sigmund & Guest 2021).

Demonstrates linearized buckling topology optimization using KS aggregation of inverse
buckling load factors mu_i = -1 / lambda_i and the non-self-adjoint stress-stiffness
adjoint equation K * v_i = f_adj,i.
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
from wht_topopt.functions.buckling import BucklingConstraint
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.solvers.scipy_solver import ScipySolver

GLOBAL_FONT_SIZE: int = 9
GLOBAL_FIGSIZE_BUCKLING: Tuple[float, float] = (10.5, 4.2)
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
class ColumnBucklingResult:
    """Results container for compressed column buckling optimization."""

    mesh: StructuredMesh2D
    rho_initial: np.ndarray
    rho_optimized: np.ndarray
    blf_history: List[float]
    compliance_history: List[float]
    initial_blf: float
    final_blf: float
    blf_improvement_pct: float
    exported_files: Dict[str, str]


class ColumnBucklingShowcase:
    """OOP Showcase for Ferrari, Sigmund & Guest (2021) linearized buckling optimization."""

    def __init__(
        self,
        nelx: int = 12,
        nely: int = 24,
        volfrac: float = 0.45,
        max_iter: int = 18,
        n_modes: int = 4,
        buckling_weight: float = 0.65,
    ) -> None:
        """Initialize ColumnBucklingShowcase parameters.

        Parameters
        ----------
        nelx, nely : int
            2D column mesh dimensions.
        volfrac : float
            Target material volume fraction.
        max_iter : int
            Optimization iterations.
        n_modes : int
            Number of lowest buckling modes aggregated via KS function.
        buckling_weight : float
            Weight in [0, 1] blending compliance and buckling sensitivity.
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.volfrac = float(volfrac)
        self.max_iter = int(max_iter)
        self.n_modes = int(n_modes)
        self.buckling_weight = float(buckling_weight)

    def run(self, output_dir: Optional[Union[str, Path]] = None) -> ColumnBucklingResult:
        """Run buckling-driven column topology optimization and export visual reports."""
        mesh = StructuredMesh2D(nx=self.nelx, ny=self.nely, lx=1.0, ly=2.0)
        fem = FEMEngine(mesh, nu=0.3, formulation="FI")
        bcm = BoundaryConditionManager.compressed_column_2d(mesh, F_mag=1.0)
        solver = ScipySolver(method="direct")
        filt = DensityFilter(mesh, r_min=0.16)

        comp_fn = ComplianceFunction(
            fem=fem,
            solver=solver,
            bcm=bcm,
            penal=3.0,
            e0=10.0,
            emin=1e-6,
        )
        buck_fn = BucklingConstraint(
            fem=fem,
            bcm=bcm,
            solver=solver,
            lambda_crit_target=2.0,
            n_modes=self.n_modes,
            rho_ks=20.0,
            penal_k=3.0,
            penal_g=3.0,
            e0=10.0,
            emin=1e-6,
        )

        # Initial slightly non-uniform symmetric seed to break trivial uniform column symmetry
        x_centers = mesh.elem_centers[:, 0]
        edge_bias = 0.08 * np.cos(np.pi * (x_centers - 0.5 * mesh.lx) / mesh.lx)
        rho = np.clip(self.volfrac - edge_bias, 0.1, 0.95)
        rho = rho * (self.volfrac / np.mean(rho))
        rho_initial = filt.forward(rho).copy()

        oc = OCOptimizer(volfrac=self.volfrac, move=0.1)
        blf_hist: List[float] = []
        comp_hist: List[float] = []

        for _ in range(self.max_iter):
            rho_tilde = filt.forward(rho)
            c_val, dc_tilde = comp_fn.value_and_gradient(rho_tilde)
            _, db_tilde = buck_fn.value_and_gradient(rho_tilde)
            blf_cur = buck_fn.get_critical_load_factor(rho_tilde)

            blf_hist.append(float(blf_cur))
            comp_hist.append(float(c_val))

            # Normalize sensitivities and blend compliance + buckling stability
            dc_norm = dc_tilde / (np.linalg.norm(dc_tilde) + 1e-12)
            db_norm = db_tilde / (np.linalg.norm(db_tilde) + 1e-12)
            d_comb_tilde = (1.0 - self.buckling_weight) * dc_norm + self.buckling_weight * db_norm
            # OC expects negative sensitivity for solid promotion
            d_comb_tilde = np.minimum(d_comb_tilde, -1e-8)
            d_comb_rho = filt.backward(d_comb_tilde)

            rho = oc.update(
                rho=rho,
                df_drho=d_comb_rho,
                elem_volumes=mesh.elem_volumes,
            )

        rho_opt = filt.forward(rho)
        final_blf = float(buck_fn.get_critical_load_factor(rho_opt))
        blf_hist.append(final_blf)
        init_blf = blf_hist[0]
        improv_pct = float((final_blf - init_blf) / max(abs(init_blf), 1e-12) * 100.0)

        exported: Dict[str, str] = {}
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            exported = self._export_report(
                out_dir=out_dir,
                mesh=mesh,
                rho_init=rho_initial,
                rho_opt=rho_opt,
                blf_hist=blf_hist,
                comp_hist=comp_hist,
                init_blf=init_blf,
                final_blf=final_blf,
                improv_pct=improv_pct,
            )

        return ColumnBucklingResult(
            mesh=mesh,
            rho_initial=rho_initial,
            rho_optimized=rho_opt,
            blf_history=blf_hist,
            compliance_history=comp_hist,
            initial_blf=init_blf,
            final_blf=final_blf,
            blf_improvement_pct=improv_pct,
            exported_files=exported,
        )

    def _export_report(
        self,
        out_dir: Path,
        mesh: StructuredMesh2D,
        rho_init: np.ndarray,
        rho_opt: np.ndarray,
        blf_hist: List[float],
        comp_hist: List[float],
        init_blf: float,
        final_blf: float,
        improv_pct: float,
        figsize_override: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Export Korean Matplotlib 9pt figures and JSON summary."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
        fsize = figsize_override if figsize_override is not None else GLOBAL_FIGSIZE_BUCKLING
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=fsize)

        im0 = ax0.imshow(rho_init.reshape((mesh.nely, mesh.nelx)), origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax0.set_title(f"초기 밀도 분포\n(λ_1 = {init_blf:.3f})")
        ax0.set_xlabel("X 요소")
        ax0.set_ylabel("Y 요소")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(rho_opt.reshape((mesh.nely, mesh.nelx)), origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        ax1.set_title(f"좌굴 최적화 기둥\n(λ_1 = {final_blf:.3f}, +{improv_pct:.1f}%)")
        ax1.set_xlabel("X 요소")
        ax1.set_ylabel("Y 요소")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        iters_b = np.arange(len(blf_hist))
        ax2.plot(iters_b, blf_hist, "g-o", markersize=3, linewidth=1.5, label="임계 좌굴 계수 (λ_1)")
        ax2.set_xlabel("반복 횟수 (Iteration)")
        ax2.set_ylabel("좌굴 임계하중 계수 (λ_1)")
        ax2.set_title("1차 좌굴 임계하중 수렴 이력")
        ax2.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax2)

        fig.tight_layout()
        plot_path = out_dir / "ex02_column_buckling_report.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_path = out_dir / "ex02_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "showcase": "ex02_column_buckling_250",
                    "initial_blf": init_blf,
                    "final_blf": final_blf,
                    "blf_improvement_pct": improv_pct,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "buckling_plot": str(plot_path),
            "summary_json": str(summary_path),
        }


if __name__ == "__main__":
    showcase = ColumnBucklingShowcase()
    res = showcase.run(output_dir=Path(__file__).resolve().parent / "outputs" / "ex02_buckling")
    print(
        f"[Ex02 Buckling250] Critical Buckling Factor lambda_1: "
        f"{res.initial_blf:.4f} -> {res.final_blf:.4f} (+{res.blf_improvement_pct:.2f}%)"
    )
