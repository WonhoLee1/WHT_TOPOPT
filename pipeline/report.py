"""2D Matplotlib (koreanize-matplotlib 9pt) and 3D PyVista (ParaView dark style) report generator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
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

from wht_topopt.algorithms.base import OptState
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh

# Global Matplotlib layout & font defaults
DEFAULT_FONT_SIZE: int = 9
GLOBAL_FONT_SIZE_PT: int = DEFAULT_FONT_SIZE
DEFAULT_FIGSIZE_CONVERGENCE: Tuple[float, float] = (9.0, 4.0)
DEFAULT_FIGSIZE_DENSITY_2D: Tuple[float, float] = (8.0, 4.0)
DEFAULT_FIGSIZE_SUMMARY: Tuple[float, float] = (12.0, 4.8)

# Global PyVista defaults
DEFAULT_PYVISTA_FONT_SIZE: int = 12
DEFAULT_PYVISTA_BG_COLOR: str = "black"
DEFAULT_PYVISTA_FONT_COLOR: str = "white"
DEFAULT_PYVISTA_EDGE_COLOR: str = "#404040"

plt.rcParams["font.size"] = DEFAULT_FONT_SIZE
plt.rcParams["axes.titlesize"] = DEFAULT_FONT_SIZE + 1
plt.rcParams["axes.labelsize"] = DEFAULT_FONT_SIZE
plt.rcParams["xtick.labelsize"] = DEFAULT_FONT_SIZE - 1
plt.rcParams["ytick.labelsize"] = DEFAULT_FONT_SIZE - 1
plt.rcParams["legend.fontsize"] = DEFAULT_FONT_SIZE - 1


def _place_outside_legend(ax: plt.Axes, max_items_per_col: int = 6) -> None:
    """Place legend outside the figure with automatic column scaling so height never exceeds figure."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = max(1, int(np.ceil(len(handles) / max_items_per_col)))
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=ncol,
        borderaxespad=0.0,
        frameon=True,
    )


class TopOptReport:
    """Visualization and engineering report generator for wht_topopt optimization runs.

    Adheres to strict 2D Matplotlib (koreanize-matplotlib, 9pt font, global figsize with
    per-plot override, outside legend, tight_layout) and 3D PyVista (ParaView dark background,
    white font size 12, dark-gray mesh edges, coordinate axes, single colorbar, in-place
    animation scalar updates, and right-click/interactive XY/YZ/ZX/Perspective view toggling)
    standards.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        history: Sequence[OptState],
        rho_phys: np.ndarray,
        title: str = "위상최적화 결과 리포트 (Topology Optimization Report)",
    ) -> None:
        """Initialize TopOptReport.

        Parameters
        ----------
        mesh : StructuredMesh2D | StructuredMesh3D | UnstructuredMesh
            Finite element mesh.
        history : Sequence[OptState]
            Optimization iteration states.
        rho_phys : np.ndarray
            Final physical density field (n_elem,).
        title : str
            Report title.
        """
        self.mesh = mesh
        self.history = list(history)
        self.rho_phys = np.asarray(rho_phys, dtype=np.float64).ravel()
        self.title = title

    def plot_convergence(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """Plot objective function and constraint/volume convergence histories.

        Parameters
        ----------
        save_path : str | Path, optional
            Output PNG/PDF file path.
        figsize : Tuple[float, float], optional
            Custom figure size override (defaults to DEFAULT_FIGSIZE_CONVERGENCE).

        Returns
        -------
        fig : plt.Figure
            Generated Matplotlib figure.
        """
        plt.rcParams["font.size"] = DEFAULT_FONT_SIZE
        fsize = figsize if figsize is not None else DEFAULT_FIGSIZE_CONVERGENCE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fsize)

        if self.history:
            iters = [s.iteration for s in self.history]
            objs = [s.obj for s in self.history]
            vols = [s.vol for s in self.history]
            changes = [s.change for s in self.history]

            ax1.plot(iters, objs, "b-o", markersize=3, linewidth=1.5, label="목적함수 (Objective)")
            ax1.set_xlabel("반복 횟수 (Iteration)")
            ax1.set_ylabel("목적함수 값 (Objective)")
            ax1.set_title("목적함수 수렴 이력")
            ax1.grid(True, linestyle="--", alpha=0.4)
            _place_outside_legend(ax1)

            ax2.plot(iters, vols, "g-s", markersize=3, linewidth=1.5, label="체적비 (Volume Fraction)")
            ax2.plot(iters, changes, "r--^", markersize=3, linewidth=1.2, label="설계변화량 (Max Change)")
            if self.history[0].g is not None and len(self.history[0].g) > 0:
                n_g = len(self.history[0].g)
                for j in range(n_g):
                    g_vals = [float(s.g[j]) for s in self.history if j < len(s.g)]
                    ax2.plot(
                        iters[: len(g_vals)],
                        g_vals,
                        linestyle=":",
                        linewidth=1.2,
                        label=f"제약조건 g_{j+1}",
                    )
            ax2.set_xlabel("반복 횟수 (Iteration)")
            ax2.set_ylabel("비율 / 잔차")
            ax2.set_title("체적비 및 제약조건 수렴 이력")
            ax2.grid(True, linestyle="--", alpha=0.4)
            _place_outside_legend(ax2)

        fig.suptitle(self.title, fontsize=DEFAULT_FONT_SIZE + 2)
        fig.tight_layout()

        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")

        return fig

    def plot_density_2d(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "gray_r",
    ) -> plt.Figure:
        """Plot 2D physical density distribution (or 2D mid-plane slice for 3D meshes).

        Parameters
        ----------
        save_path : str | Path, optional
            Output image path.
        figsize : Tuple[float, float], optional
            Figure size override (defaults to DEFAULT_FIGSIZE_DENSITY_2D).
        cmap : str
            Matplotlib colormap name.

        Returns
        -------
        fig : plt.Figure
            Generated Matplotlib figure.
        """
        plt.rcParams["font.size"] = DEFAULT_FONT_SIZE
        fsize = figsize if figsize is not None else DEFAULT_FIGSIZE_DENSITY_2D
        fig, ax = plt.subplots(1, 1, figsize=fsize)

        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            # Element e = i + j * nx -> reshape (ny, nx)
            grid = self.rho_phys.reshape((ny, nx))
            im = ax.imshow(
                grid,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=[0.0, self.mesh.lx, 0.0, self.mesh.ly],
                aspect="equal",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("물리 밀도 (Physical Density)", fontsize=DEFAULT_FONT_SIZE)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"{self.title} - 2D 밀도 분포")
        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            # Element index e = i + j*nx + k*nx*ny -> reshape (nz, ny, nx)
            grid3d = self.rho_phys.reshape((nz, ny, nx))
            mid_k = nz // 2
            im = ax.imshow(
                grid3d[mid_k],
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                extent=[0.0, self.mesh.lx, 0.0, self.mesh.ly],
                aspect="equal",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("물리 밀도 (Physical Density)", fontsize=DEFAULT_FONT_SIZE)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"{self.title} - 중앙 단면 (Z={mid_k})")
        else:
            centers = self.mesh.elem_centers
            sc = ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c=self.rho_phys,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                s=18,
                label="요소 밀도 (Element Density)",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("물리 밀도 (Physical Density)", fontsize=DEFAULT_FONT_SIZE)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{self.title} - 요소 중심 밀도")
            _place_outside_legend(ax)

        fig.tight_layout()
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")

        return fig

    def create_pyvista_plotter(
        self,
        off_screen: bool = True,
        threshold: float = 0.3,
        window_size: Tuple[int, int] = (1024, 768),
    ) -> Tuple[Any, Any]:
        """Create a ParaView-style 3D PyVista plotter with dark background and context menu.

        Enforces:
            - Background color: black, font color: white, font size: 12
            - Mesh edge color: dark gray (#404040)
            - Coordinate axes enabled
            - Single colorbar only
            - Right-click / key-event context menu for XY, YZ, ZX, and Perspective toggling

        Parameters
        ----------
        off_screen : bool
            Whether to render off-screen (headless/test safe).
        threshold : float
            Minimum density threshold for displaying solid elements.
        window_size : Tuple[int, int]
            Render window resolution.

        Returns
        -------
        plotter : pyvista.Plotter
            Configured PyVista Plotter instance.
        pv_grid : pyvista.DataSet
            Underlying PyVista dataset with ``cell_data["density"]`` bound for in-place animation.
        """
        import pyvista as pv

        pv.global_theme.font.size = DEFAULT_PYVISTA_FONT_SIZE
        pv.global_theme.font.color = DEFAULT_PYVISTA_FONT_COLOR
        pv.global_theme.background = DEFAULT_PYVISTA_BG_COLOR

        plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        plotter.set_background(DEFAULT_PYVISTA_BG_COLOR)

        pv_grid = self._build_pyvista_grid()
        pv_grid.cell_data["density"] = self.rho_phys.copy()

        # Single scalar bar configuration
        sbar_args = {
            "title": "Physical Density",
            "title_font_size": DEFAULT_PYVISTA_FONT_SIZE,
            "label_font_size": DEFAULT_PYVISTA_FONT_SIZE,
            "color": DEFAULT_PYVISTA_FONT_COLOR,
            "n_labels": 5,
            "fmt": "%.2f",
        }

        plotter.add_mesh(
            pv_grid,
            scalars="density",
            clim=[0.0, 1.0],
            cmap="viridis",
            show_edges=True,
            edge_color=DEFAULT_PYVISTA_EDGE_COLOR,
            show_scalar_bar=True,
            scalar_bar_args=sbar_args,
            name="topopt_density_mesh",
        )
        plotter.add_axes(color=DEFAULT_PYVISTA_FONT_COLOR)
        self._register_interactive_view_menu(plotter)

        return plotter, pv_grid

    def update_pyvista_animation_frame(
        self,
        plotter: Any,
        pv_grid: Any,
        new_rho: np.ndarray,
    ) -> None:
        """Update PyVista animation frame in-place without recreating mesh actors or colorbars.

        Parameters
        ----------
        plotter : pyvista.Plotter
            Active PyVista Plotter.
        pv_grid : pyvista.DataSet
            Dataset returned by ``create_pyvista_plotter``.
        new_rho : np.ndarray
            Updated elemental density vector (n_elem,).
        """
        pv_grid.cell_data["density"][:] = np.asarray(new_rho, dtype=np.float64).ravel()
        if hasattr(plotter, "render"):
            plotter.render()

    def _build_pyvista_grid(self) -> Any:
        """Construct a PyVista UnstructuredGrid from self.mesh."""
        import pyvista as pv

        nodes = self.mesh.node_coords
        if nodes.shape[1] == 2:
            pts = np.column_stack([nodes, np.zeros(len(nodes), dtype=np.float64)])
        else:
            pts = nodes.astype(np.float64, copy=False)

        elems = self.mesh.elements
        n_elem, n_nodes_per_elem = elems.shape
        cells = np.column_stack([np.full(n_elem, n_nodes_per_elem, dtype=np.int64), elems]).ravel()

        if n_nodes_per_elem == 4 and self.mesh.ndim == 2:
            celltypes = np.full(n_elem, pv.CellType.QUAD, dtype=np.uint8)
        elif n_nodes_per_elem == 3 and self.mesh.ndim == 2:
            celltypes = np.full(n_elem, pv.CellType.TRIANGLE, dtype=np.uint8)
        elif n_nodes_per_elem == 4 and self.mesh.ndim == 3:
            celltypes = np.full(n_elem, pv.CellType.TETRA, dtype=np.uint8)
        else:
            celltypes = np.full(n_elem, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        return pv.UnstructuredGrid(cells, celltypes, pts)

    @staticmethod
    def _register_interactive_view_menu(plotter: Any) -> Dict[str, Callable[[], None]]:
        """Attach right-click / key-event view callbacks for XY, YZ, ZX, and Perspective toggling."""
        state = {"perspective": True, "view_idx": 0}

        def view_xy() -> None:
            plotter.view_xy()

        def view_yz() -> None:
            plotter.view_yz()

        def view_zx() -> None:
            plotter.view_zx()

        def toggle_perspective() -> None:
            if state["perspective"]:
                plotter.enable_parallel_projection()
                state["perspective"] = False
            else:
                plotter.disable_parallel_projection()
                state["perspective"] = True

        def cycle_context_menu(*_args: Any) -> None:
            actions = [view_xy, view_yz, view_zx, toggle_perspective]
            actions[state["view_idx"] % len(actions)]()
            state["view_idx"] += 1

        try:
            plotter.add_key_event("1", view_xy)
            plotter.add_key_event("2", view_yz)
            plotter.add_key_event("3", view_zx)
            plotter.add_key_event("p", toggle_perspective)
            if hasattr(plotter, "iren") and plotter.iren is not None:
                plotter.iren.add_observer("RightButtonPressEvent", cycle_context_menu)
        except Exception:
            pass

        callbacks = {
            "xy": view_xy,
            "yz": view_yz,
            "zx": view_zx,
            "toggle_perspective": toggle_perspective,
            "right_click_cycle": cycle_context_menu,
        }
        plotter._wht_view_menu_callbacks = callbacks
        return callbacks

    def export_summary_markdown(self, save_path: Union[str, Path]) -> Path:
        """Write a UTF-8 Markdown engineering summary report of the optimization run."""
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        n_iters = len(self.history)
        init_obj = self.history[0].obj if n_iters > 0 else 0.0
        final_obj = self.history[-1].obj if n_iters > 0 else 0.0
        final_vol = self.history[-1].vol if n_iters > 0 else float(np.mean(self.rho_phys))
        final_change = self.history[-1].change if n_iters > 0 else 0.0

        lines = [
            f"# {self.title}",
            "",
            "## 1. 최적화 요약 지표 (Summary Metrics)",
            f"- **총 반복 횟수 (Total Iterations)**: {n_iters}",
            f"- **초기 목적함수 (Initial Objective)**: {init_obj:.6e}",
            f"- **최종 목적함수 (Final Objective)**: {final_obj:.6e}",
            f"- **최종 체적비 (Final Volume Fraction)**: {final_vol:.4f}",
            f"- **최종 설계 변화량 (Final Max Change)**: {final_change:.6e}",
            f"- **물리 밀도 범위 (Physical Density Range)**: [{float(np.min(self.rho_phys)):.4e}, {float(np.max(self.rho_phys)):.4e}]",
            "",
            "## 2. 반복 이력 (Iteration History)",
            "| Iteration | Objective | Volume Fraction | Max Change |",
            "|---:|---:|---:|---:|",
        ]
        for s in self.history[:50]:
            lines.append(f"| {s.iteration} | {s.obj:.6e} | {s.vol:.4f} | {s.change:.4e} |")

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return out


def annotate_beam_bc(ax: Any, nx: int, ny: int, case: str) -> None:
    """Draw boundary condition hatches, support pins, and load arrows for standard 2D benchmarks."""
    arrow = dict(arrowstyle="-|>", color="#c0392b", lw=1.8)
    if case == "cantilever":
        for y in np.linspace(0, ny, 13):
            ax.plot([-0.012 * nx, 0], [y - 0.02 * ny, y + 0.02 * ny], color="#444", lw=0.8, clip_on=False)
        ax.annotate("", xy=(nx, ny / 2 - 0.15 * ny), xytext=(nx, ny / 2 + 0.05 * ny), arrowprops=arrow, annotation_clip=False)
    elif case == "mbb":
        for y in np.linspace(0, ny, 9):
            ax.plot([-0.012 * nx, 0], [y - 0.02 * ny, y + 0.02 * ny], color="#2c7fb8", lw=0.8, clip_on=False)
        ax.plot([nx], [-0.05 * ny], marker="^", color="#444", ms=7, clip_on=False)
        ax.annotate("", xy=(0, ny), xytext=(0, ny + 0.2 * ny), arrowprops=arrow, annotation_clip=False)
    elif case == "lbracket":
        for x in np.linspace(0, 0.4 * nx, 12):
            ax.plot([x - 0.015 * nx, x + 0.015 * nx], [ny, ny + 0.025 * ny], color="#444", lw=0.8, clip_on=False)
        ax.annotate("", xy=(nx, 0.4 * ny - 0.2 * ny), xytext=(nx, 0.4 * ny + 0.1 * ny), arrowprops=arrow, annotation_clip=False)


def plot_2d_density_with_bc(
    ax: Any,
    nx: int,
    ny: int,
    rho_phys: np.ndarray,
    domain_mask: Optional[np.ndarray] = None,
    case: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "gray_r",
) -> None:
    """Plot 2D density field with pale-blue passive void background and BC/load glyphs."""
    img = np.asarray(rho_phys, dtype=float).reshape(ny, nx)
    mask = None if domain_mask is None else ~np.asarray(domain_mask, dtype=bool).reshape(ny, nx)
    show = np.ma.masked_array(img, mask=mask)
    ax.set_facecolor("#dbe9f6")
    ax.imshow(show, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, extent=(0, nx, 0, ny), interpolation="nearest")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if case is not None:
        annotate_beam_bc(ax, nx, ny, case)
    if title:
        ax.set_title(title, fontsize=GLOBAL_FONT_SIZE_PT)


def plot_2d_deformed_mesh(
    ax: Any,
    nodes: np.ndarray,
    elems: np.ndarray,
    nx: int,
    ny: int,
    rho_phys: np.ndarray,
    u: np.ndarray,
    domain_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    threshold: float = 0.5,
    max_shift: float = 0.08,
    cmap: str = "coolwarm",
    scale: Optional[float] = None,
) -> None:
    """Plot deformed Q4 elements (rho_phys > threshold) colored by in-plane displacement magnitude."""
    from matplotlib.collections import PolyCollection

    u_arr = np.asarray(u, dtype=float).ravel()
    n_nodes = nodes.shape[0]
    dof_per_node = len(u_arr) // n_nodes
    disp = u_arr.reshape(n_nodes, dof_per_node)[:, :2]
    xy = np.asarray(nodes, dtype=float)[:, :2]
    quad = np.asarray(elems, dtype=np.int64)[:, :4]
    mag = np.linalg.norm(disp, axis=1)

    if scale is None:
        scale = max_shift * ny / max(float(mag.max()), 1e-30)

    keep = np.asarray(rho_phys, dtype=float).ravel() > threshold
    if domain_mask is not None:
        keep &= np.asarray(domain_mask, dtype=bool).ravel()
    if not np.any(keep):
        keep = np.asarray(rho_phys, dtype=float).ravel() >= float(np.median(rho_phys))
        if domain_mask is not None:
            keep &= np.asarray(domain_mask, dtype=bool).ravel()

    polys = (xy + scale * disp)[quad[keep]]
    color = mag[quad[keep]].mean(axis=1)
    pc = PolyCollection(polys, array=color, cmap=cmap, edgecolors="none", linewidths=0.0)
    ax.add_collection(pc)

    def_xy = xy + scale * disp
    lo, hi = def_xy.min(axis=0), def_xy.max(axis=0)
    ax.set_xlim(lo[0] - 0.02 * nx, hi[0] + 0.02 * nx)
    ax.set_ylim(lo[1] - 0.04 * ny, hi[1] + 0.04 * ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=GLOBAL_FONT_SIZE_PT)

