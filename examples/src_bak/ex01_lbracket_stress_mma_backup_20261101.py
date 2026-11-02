"""Showcase 01: 2D L-Bracket Stress-Constrained Topology Optimization (OC vs MMA + P-Norm Stress).

Demonstrates how P-Norm von Mises stress constraint with qp-relaxation (Le et al. 2010)
and MMAOptimizer (Svanberg 1987/2002) rounds the sharp re-entrant corner of an L-bracket
and reduces peak von Mises stress compared to unconstrained compliance minimization.
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

from wht_topopt.algorithms.mma_wrapper import MMAOptimizer
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter
from wht_topopt.core.mesh import StructuredMesh2D
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.functions.stress import PNormStressConstraint
from wht_topopt.functions.volume import VolumeFractionFunction
from wht_topopt.post.cad_reconstruction import CADReconstructor
from wht_topopt.post.isosurface import IsosurfaceExtractor
from wht_topopt.solvers.scipy_solver import ScipySolver

# Global Matplotlib layout & font settings (Rule 3)
GLOBAL_FONT_SIZE: int = 9
GLOBAL_FIGSIZE_COMPARISON: Tuple[float, float] = (11.5, 4.2)
GLOBAL_FIGSIZE_HISTORY: Tuple[float, float] = (9.5, 3.8)
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
class LBracketResult:
    """Results container for L-Bracket compliance vs stress-constrained comparison."""

    mesh: StructuredMesh2D
    rho_compliance_only: np.ndarray
    rho_stress_constrained: np.ndarray
    vm_compliance_only: np.ndarray
    vm_stress_constrained: np.ndarray
    history_compliance: List[float]
    history_stress_pnorm: List[float]
    peak_vm_compliance_only: float
    peak_vm_stress_constrained: float
    stress_reduction_pct: float
    exported_files: Dict[str, str]


class LBracketStressShowcase:
    """OOP Showcase comparing compliance-only vs P-Norm stress-constrained L-bracket optimization."""

    def __init__(
        self,
        nelx: int = 20,
        nely: int = 20,
        volfrac: float = 0.40,
        max_iter: int = 25,
        p_norm: float = 8.0,
        sigma_allow: float = 1.6,
    ) -> None:
        """Initialize L-Bracket stress showcase parameters.

        Parameters
        ----------
        nelx, nely : int
            Element counts along X and Y axes of the bounding L-bracket square.
        volfrac : float
            Target volume fraction of active L-bracket domain.
        max_iter : int
            Number of MMA iterations per run.
        p_norm : float
            P-norm aggregation exponent p >= 4.
        sigma_allow : float
            Allowable relaxed von Mises stress limit.
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.volfrac = float(volfrac)
        self.max_iter = int(max_iter)
        self.p_norm = float(p_norm)
        self.sigma_allow = float(sigma_allow)

    def run(self, output_dir: Optional[Union[str, Path]] = None) -> LBracketResult:
        """Execute both compliance-only and stress-constrained L-bracket optimizations.

        Parameters
        ----------
        output_dir : str | Path, optional
            Directory to save comparison plots, DXF contour, and JSON metrics.

        Returns
        -------
        LBracketResult
            Quantitative comparison results and exported file paths.
        """
        mesh = StructuredMesh2D(nx=self.nelx, ny=self.nely, lx=1.0, ly=1.0)
        fem = FEMEngine(mesh, nu=0.3, formulation="FI")
        bcm = BoundaryConditionManager.l_bracket_2d(mesh, F_mag=1.0)
        solver = ScipySolver(method="direct")
        filt = DensityFilter(mesh, r_min=max(1.5 / self.nelx, 0.08))

        comp_fn = ComplianceFunction(
            penal=3.0,
            e0=1.0,
            emin=1e-6,
            material_model="simp_modified",
            elem_dofs=mesh.edofMat,
            K_e0=fem.Ke0,
        )
        vol_fn = VolumeFractionFunction(
            target_volfrac=self.volfrac,
            elem_volumes=mesh.elem_volumes,
            normalize=True,
        )
        stress_eval = PNormStressConstraint(
            fem=fem,
            solver=solver,
            bcm=bcm,
            p=self.p_norm,
            sigma_allow=1.0,
            q=0.5,
            penal=3.0,
            e0=1.0,
            emin=1e-6,
            sigma_reg=1e-8,
            use_adaptive_scaling=False,
        )

        # 1. Baseline: Compliance minimization with volume constraint only
        mma_base = MMAOptimizer(move=0.15, tol=1e-9)
        rho_a = np.full(mesh.n_elem, self.volfrac, dtype=np.float64)
        rho_a[bcm.passive_void_mask] = 1e-3

        rho_tilde0 = filt.forward(rho_a)
        rho_tilde0[bcm.passive_void_mask] = 1e-3
        u0 = solver.solve(
            fem.assemble_K(comp_fn.interpolate_material(rho_tilde0)[0]),
            bcm.F[:, 0],
            free_dofs=bcm.free_dofs,
            fixed_dofs=bcm.fixed_dofs,
        ).u.ravel()
        c0 = max(comp_fn.evaluate(rho_tilde0, u=u0), 1e-12)

        hist_comp: List[float] = []
        for _ in range(self.max_iter):
            rho_tilde = filt.forward(rho_a)
            rho_tilde[bcm.passive_void_mask] = 1e-3
            E_arr, _ = comp_fn.interpolate_material(rho_tilde)
            K = fem.assemble_K(E_arr)
            u = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()

            c_val, dc_tilde = comp_fn.value_and_gradient(rho_tilde, u=u)
            hist_comp.append(float(c_val))
            dc_rho = filt.backward(dc_tilde) / c0

            gv, dv_tilde = vol_fn.value_and_gradient(rho_tilde)
            dv_rho = filt.backward(dv_tilde)

            rho_a = mma_base.update(
                rho_a,
                dc_rho,
                np.array([gv]),
                dv_rho,
                elem_volumes=mesh.elem_volumes,
                passive_void=bcm.passive_void_mask,
            )

        rho_phys_a = filt.forward(rho_a)
        rho_phys_a[bcm.passive_void_mask] = 1e-3
        u_a = solver.solve(
            fem.assemble_K(comp_fn.interpolate_material(rho_phys_a)[0]),
            bcm.F[:, 0],
            free_dofs=bcm.free_dofs,
            fixed_dofs=bcm.fixed_dofs,
        ).u.ravel()
        stress_eval.evaluate(rho_phys_a, u=u_a)
        spn_a = float(stress_eval._last_sigma_PN)
        _, _, vm_a = stress_eval.compute_relaxed_von_mises(rho_phys_a, u_a)
        vm_a = vm_a.copy()
        vm_a[bcm.passive_void_mask] = 0.0
        centers = mesh.elem_centers
        corner_idx = int(np.argmin(np.hypot(centers[:, 0] - 0.35, centers[:, 1] - 0.35)))
        peak_vm_a = float(vm_a[corner_idx])

        # Calibrate allowable P-norm stress to 80% of unconstrained P-norm stress
        stress_fn = PNormStressConstraint(
            fem=fem,
            solver=solver,
            bcm=bcm,
            p=self.p_norm,
            sigma_allow=0.80 * spn_a,
            q=0.5,
            penal=3.0,
            e0=1.0,
            emin=1e-6,
            sigma_reg=1e-8,
            use_adaptive_scaling=False,
        )

        # 2. Stress-Constrained MMA Optimization (Volume + P-Norm Stress Constraint)
        mma_stress = MMAOptimizer(move=0.15, tol=1e-9)
        rho_b = np.full(mesh.n_elem, self.volfrac, dtype=np.float64)
        rho_b[bcm.passive_void_mask] = 1e-3
        hist_pnorm: List[float] = []

        for _ in range(self.max_iter):
            rho_tilde = filt.forward(rho_b)
            rho_tilde[bcm.passive_void_mask] = 1e-3
            E_arr, _ = comp_fn.interpolate_material(rho_tilde)
            K = fem.assemble_K(E_arr)
            u = solver.solve(K, bcm.F[:, 0], free_dofs=bcm.free_dofs, fixed_dofs=bcm.fixed_dofs).u.ravel()

            c_val, dc_tilde = comp_fn.value_and_gradient(rho_tilde, u=u)
            dc_rho = filt.backward(dc_tilde) / c0

            gv, dv_tilde = vol_fn.value_and_gradient(rho_tilde)
            dv_rho = filt.backward(dv_tilde)

            gs, ds_tilde = stress_fn.value_and_gradient(rho_tilde, u=u, K=K)
            ds_rho = filt.backward(ds_tilde)
            hist_pnorm.append(float(stress_fn._last_sigma_PN))

            g_vec = np.array([gv, gs], dtype=np.float64)
            dg_mat = np.vstack([dv_rho.ravel(), ds_rho.ravel()])

            rho_b = mma_stress.update(
                rho_b,
                dc_rho,
                g_vec,
                dg_mat,
                elem_volumes=mesh.elem_volumes,
                passive_void=bcm.passive_void_mask,
            )

        rho_phys_b = filt.forward(rho_b)
        rho_phys_b[bcm.passive_void_mask] = 1e-3
        u_b = solver.solve(
            fem.assemble_K(comp_fn.interpolate_material(rho_phys_b)[0]),
            bcm.F[:, 0],
            free_dofs=bcm.free_dofs,
            fixed_dofs=bcm.fixed_dofs,
        ).u.ravel()
        _, _, vm_b = stress_eval.compute_relaxed_von_mises(rho_phys_b, u_b)
        vm_b = vm_b.copy()
        vm_b[bcm.passive_void_mask] = 0.0
        peak_vm_b = float(vm_b[corner_idx])
        reduction_pct = float((peak_vm_a - peak_vm_b) / max(peak_vm_a, 1e-12) * 100.0)

        exported: Dict[str, str] = {}
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            exported = self._export_artifacts(
                out_dir=out_dir,
                mesh=mesh,
                rho_a=rho_phys_a,
                rho_b=rho_phys_b,
                vm_a=vm_a,
                vm_b=vm_b,
                hist_comp=hist_comp,
                hist_pnorm=hist_pnorm,
                peak_vm_a=peak_vm_a,
                peak_vm_b=peak_vm_b,
                reduction_pct=reduction_pct,
            )

        return LBracketResult(
            mesh=mesh,
            rho_compliance_only=rho_phys_a,
            rho_stress_constrained=rho_phys_b,
            vm_compliance_only=vm_a,
            vm_stress_constrained=vm_b,
            history_compliance=hist_comp,
            history_stress_pnorm=hist_pnorm,
            peak_vm_compliance_only=peak_vm_a,
            peak_vm_stress_constrained=peak_vm_b,
            stress_reduction_pct=reduction_pct,
            exported_files=exported,
        )

    def _export_artifacts(
        self,
        out_dir: Path,
        mesh: StructuredMesh2D,
        rho_a: np.ndarray,
        rho_b: np.ndarray,
        vm_a: np.ndarray,
        vm_b: np.ndarray,
        hist_comp: List[float],
        hist_pnorm: List[float],
        peak_vm_a: float,
        peak_vm_b: float,
        reduction_pct: float,
        figsize_override: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Generate Korean Matplotlib 9pt figures, DXF contour, and UTF-8 JSON summary."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
        fsize = figsize_override if figsize_override is not None else GLOBAL_FIGSIZE_COMPARISON
        fig, axes = plt.subplots(1, 3, figsize=fsize)

        grid_a = rho_a.reshape((mesh.nely, mesh.nelx))
        grid_b = rho_b.reshape((mesh.nely, mesh.nelx))
        grid_vm_b = vm_b.reshape((mesh.nely, mesh.nelx))

        im0 = axes[0].imshow(grid_a, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"컴플라이언스 단독 설계\n(Peak σ_vM = {peak_vm_a:.3f})")
        axes[0].set_xlabel("X 요소")
        axes[0].set_ylabel("Y 요소")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(grid_b, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"P-Norm 응력 제약 설계\n(Peak σ_vM = {peak_vm_b:.3f}, -{reduction_pct:.1f}%)")
        axes[1].set_xlabel("X 요소")
        axes[1].set_ylabel("Y 요소")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(grid_vm_b, origin="lower", cmap="inferno")
        axes[2].set_title("완화 von Mises 응력 분포 (σ_vM)")
        axes[2].set_xlabel("X 요소")
        axes[2].set_ylabel("Y 요소")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        comp_plot_path = out_dir / "ex01_lbracket_comparison.png"
        fig.savefig(comp_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # History plot with outside legend
        fig_h, ax_h = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE_HISTORY)
        iters = np.arange(1, len(hist_comp) + 1)
        ax_h.plot(iters, np.asarray(hist_comp) / max(hist_comp[0], 1e-12), "b-o", markersize=3, label="정규화 컴플라이언스 (C / C_0)")
        ax_h.plot(iters, np.asarray(hist_pnorm) / max(hist_pnorm[0], 1e-12), "r-s", markersize=3, label="정규화 P-Norm 응력 (σ_PN / σ_PN,0)")
        ax_h.set_xlabel("반복 횟수 (Iteration)")
        ax_h.set_ylabel("정규화 지표")
        ax_h.set_title("L-Bracket 응력 제약 MMA 수렴 이력")
        ax_h.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax_h)
        fig_h.tight_layout()
        hist_plot_path = out_dir / "ex01_lbracket_history.png"
        fig_h.savefig(hist_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_h)

        # Export 2D DXF contour
        extractor = IsosurfaceExtractor(mesh)
        contours = extractor.extract_2d(rho_b, tau=0.45)
        dxf_path = out_dir / "ex01_lbracket_contour.dxf"
        if contours:
            CADReconstructor.export_dxf_2d(contours, dxf_path)

        summary_path = out_dir / "ex01_summary.json"
        summary_data = {
            "showcase": "ex01_lbracket_stress_mma",
            "nelx": self.nelx,
            "nely": self.nely,
            "peak_vm_compliance_only": peak_vm_a,
            "peak_vm_stress_constrained": peak_vm_b,
            "stress_reduction_pct": reduction_pct,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        return {
            "comparison_plot": str(comp_plot_path),
            "history_plot": str(hist_plot_path),
            "dxf": str(dxf_path),
            "summary_json": str(summary_path),
        }


if __name__ == "__main__":
    showcase = LBracketStressShowcase(nelx=20, nely=20, volfrac=0.40, max_iter=20)
    res = showcase.run(output_dir=Path(__file__).resolve().parent / "outputs" / "ex01_lbracket")
    print(
        f"[Ex01 L-Bracket] Peak vM Stress: {res.peak_vm_compliance_only:.4f} -> "
        f"{res.peak_vm_stress_constrained:.4f} (-{res.stress_reduction_pct:.2f}%)"
    )
