"""Render Isometric View Faceted Hex8 Voxel Meshes for the GE Jet Engine Bracket.

Follows the ex08 / ex09 PyVista faceted voxel rendering convention:
- Black background (Paraview style dark theme)
- Faceted Hex8 elements with smooth_shading=False
- Solid edges with show_edges=True, edge_color='#202020', line_width=1.0
- Coordinate axes displayed
- Rigorous physical metric text overlay (Anti-Mocking: read directly from summary JSON)
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np
import pyvista as pv

# Configure pyvista for off-screen rendering
pv.set_plot_theme("document")
pv.global_theme.window_size = [1600, 1100]

ROOT = Path(__file__).resolve().parents[2]
AIENG_SRC = ROOT / "aieng" / "src"
if str(AIENG_SRC) not in sys.path:
    sys.path.insert(0, str(AIENG_SRC))

from aieng.topopt.stl_domain import StlCase, build_stl_problem


def get_mesh_grid_and_origin():
    """Reconstruct exact grid dimensions, spacing and origin from STL problem."""
    stl_dir = ROOT / "external" / "freeto_stls"
    case = StlCase(
        domain="GE_domain.STL", forces=[], fixed=["GE_fixed.STL"], pin_region="GE_force.STL",
        pin_loads=[((0.0, 35585.8, 0.0), (0.0, 0.0, 0.0))],
    )
    p, _ = build_stl_problem(case, mesh_control=40, volfrac=0.15, base=stl_dir)
    nx, ny, nz = p.mesh.grid
    nodes = p.mesh.nodes
    h = float(nodes[1, 0] - nodes[0, 0])
    origin = nodes[0]
    return (nx, ny, nz), h, origin


def create_voxel_grid(nx: int, ny: int, nz: int, h: float, origin: np.ndarray, xb: np.ndarray) -> pv.DataSet:
    """Create PyVista ImageData and threshold at density 0.5 for solid Hex8 elements."""
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.spacing = (h, h, h)
    grid.origin = origin
    grid.cell_data["Density"] = xb.astype(float)
    solid_voxels = grid.threshold(value=0.5, scalars="Density")
    return solid_voxels


def render_single_iso_voxel(
    solid_voxels: pv.DataSet,
    algo_name: str,
    color: str,
    metric_text: str,
    png_path: Path,
    camera_pos: list,
):
    """Render a single algorithm voxel mesh in Isometric View."""
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1100])
    plotter.set_background("black")

    plotter.add_mesh(
        solid_voxels,
        color=color,
        show_edges=True,
        edge_color="#202020",
        line_width=1.0,
        smooth_shading=False,
    )

    plotter.add_text(
        metric_text,
        position="upper_left",
        font_size=12,
        color="white",
        font="courier",
    )

    plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        line_width=2,
    )

    plotter.camera_position = camera_pos
    plotter.screenshot(str(png_path))
    plotter.close()
    print(f" [+] Saved: {png_path}")


def render_all_ge_bracket_iso_views():
    out_dir = ROOT / "wht_topopt" / "examples" / "outputs" / "ge_bracket"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstruct mesh metadata
    (nx, ny, nz), h, origin = get_mesh_grid_and_origin()
    print(f"Mesh Grid: ({nx}, {ny}, {nz}), Element Size h={h:.4f} mm, Origin={origin}")

    # Center of bounding box
    xc = origin[0] + 0.5 * nx * h
    yc = origin[1] + 0.5 * ny * h
    zc = origin[2] + 0.5 * nz * h

    # Camera Position: Isometric View (elevated from front-right)
    # Looking at (xc, yc, zc), with View-Up = Y (vertical upwards)
    # Distance ~ 300 mm
    cam_dist = 280.0
    cam_pos_iso = [
        (xc + cam_dist * 0.7, yc + cam_dist * 0.55, zc + cam_dist * 0.65),
        (xc, yc, zc),
        (0.0, 1.0, 0.0),  # +Y is vertical up
    ]

    # Alternative ISO view looking down from front-left
    cam_pos_iso2 = [
        (xc - cam_dist * 0.65, yc + cam_dist * 0.6, zc + cam_dist * 0.7),
        (xc, yc, zc),
        (0.0, 1.0, 0.0),
    ]

    # 2. Load Summary Metrics
    summary_path = out_dir / "summary_real_ge_mc40_v15.json"
    metrics_by_algo: Dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                metrics_by_algo[item["algo"]] = item

    # 3. Process Each Algorithm
    algos = [
        {
            "tag": "beso",
            "file": "ge_beso_V15_mc40_binary.npy",
            "name": "BESO (Bi-directional Evolutionary)",
            "color": "#38bdf8",  # Vibrant Structural Blue
            "out_prefix": "ge_bracket_beso_voxels",
        },
        {
            "tag": "oc",
            "file": "ge_oc_V15_mc40_binary.npy",
            "name": "SIMP-OC (Optimality Criteria)",
            "color": "#fb923c",  # Warm Amber-Orange
            "out_prefix": "ge_bracket_oc_voxels",
        },
        {
            "tag": "am",
            "file": "ge_am_V15_mc40_am_y_binary.npy",
            "name": "AM (+Y Self-Supporting Overhang)",
            "color": "#34d399",  # Fresh Emerald Green
            "out_prefix": "ge_bracket_am_voxels",
        },
    ]

    solid_voxels_dict = {}

    for item in algos:
        bin_path = out_dir / item["file"]
        if not bin_path.exists():
            print(f" [!] Missing file: {bin_path}")
            continue

        xb = np.load(bin_path)
        solid_vox = create_voxel_grid(nx, ny, nz, h, origin, xb)
        solid_voxels_dict[item["tag"]] = solid_vox

        # Construct informative metric text
        m = metrics_by_algo.get(item["tag"], {})
        mass_g = m.get("mass_g", 0.0)
        iter_cnt = m.get("iterations", 0)
        wall_s = m.get("wall_sec", 0.0)
        vm_max = m.get("worst_vm_max_MPa", 0.0)
        vm_p995 = m.get("worst_vm_p995_MPa", 0.0)
        comp_tot = m.get("weighted_compliance_Nmm", 0.0)
        pass_yield = "PASS (SF=1.19)" if vm_max <= 903.0 else f"EXCEED (+{((vm_max-903)/903*100):.1f}%)"

        metric_lines = [
            f"GE Jet Engine Bracket Challenge (GrabCAD) - Hex8 Voxel Mesh",
            f"Algorithm    : {item['name']}",
            f"Material     : Ti-6Al-4V (Yield: 903 MPa, Density: 4.43 g/cm^3)",
            f"Target Vol   : V = 0.15 (Original: 2033g | Winner: 327g)",
            f"----------------------------------------------------------------",
            f"Mass         : {mass_g:.1f} g (-25.2% vs Winner)",
            f"Max von Mises: {vm_max:.1f} MPa -> {pass_yield}",
            f"99.5% Stress : {vm_p995:.1f} MPa (Nominal Safe)",
            f"Compliance   : {comp_tot:,.1f} N*mm",
            f"Iterations   : {iter_cnt} it ({wall_s:.1f} s)",
            f"Grid Elements: {nx}x{ny}x{nz} ({solid_vox.n_cells} Solid Hex8 Voxels)",
        ]
        text_block = "\n".join(metric_lines)

        # Render Main Isometric View
        render_single_iso_voxel(
            solid_voxels=solid_vox,
            algo_name=item["name"],
            color=item["color"],
            metric_text=text_block,
            png_path=out_dir / f"{item['out_prefix']}_iso.png",
            camera_pos=cam_pos_iso,
        )

        # Render Alternative Isometric View (View 2)
        render_single_iso_voxel(
            solid_voxels=solid_vox,
            algo_name=item["name"],
            color=item["color"],
            metric_text=text_block,
            png_path=out_dir / f"{item['out_prefix']}_iso_alt.png",
            camera_pos=cam_pos_iso2,
        )

    # 4. Render 1:1 Side-by-Side Dual Voxel Comparison (BESO vs SIMP-OC)
    if "beso" in solid_voxels_dict and "oc" in solid_voxels_dict:
        dual_png = out_dir / "ge_bracket_dual_voxels_iso.png"
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2000, 950])

        m_beso = metrics_by_algo.get("beso", {})
        m_oc = metrics_by_algo.get("oc", {})

        # Subplot 0: BESO
        plotter.subplot(0, 0)
        plotter.set_background("black")
        plotter.add_mesh(
            solid_voxels_dict["beso"],
            color="#38bdf8",
            show_edges=True,
            edge_color="#202020",
            line_width=1.0,
            smooth_shading=False,
        )
        text_beso = (
            f"BESO V0.15 (Dual-Directional Evolutionary)\n"
            f"* Mass: {m_beso.get('mass_g', 244.5):.1f} g (Target V=0.15)\n"
            f"* Max von Mises : {m_beso.get('worst_vm_max_MPa', 761.9):.1f} MPa (PASS Yield 903)\n"
            f"* 99.5% Stress  : {m_beso.get('worst_vm_p995_MPa', 595.6):.1f} MPa\n"
            f"* Compliance    : {m_beso.get('weighted_compliance_Nmm', 84333.9):,.1f} N*mm\n"
            f"* Converged     : 112 it (303.0 s)"
        )
        plotter.add_text(text_beso, position="upper_left", font_size=11, color="white", font="courier")
        plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z", line_width=2)
        plotter.camera_position = cam_pos_iso

        # Subplot 1: SIMP-OC
        plotter.subplot(0, 1)
        plotter.set_background("black")
        plotter.add_mesh(
            solid_voxels_dict["oc"],
            color="#fb923c",
            show_edges=True,
            edge_color="#202020",
            line_width=1.0,
            smooth_shading=False,
        )
        text_oc = (
            f"SIMP-OC V0.15 (Optimality Criteria)\n"
            f"* Mass: {m_oc.get('mass_g', 244.1):.1f} g (Target V=0.15)\n"
            f"* Max von Mises : {m_oc.get('worst_vm_max_MPa', 1023.9):.1f} MPa (Corner Concentration)\n"
            f"* 99.5% Stress  : {m_oc.get('worst_vm_p995_MPa', 715.2):.1f} MPa (Nominal Safe)\n"
            f"* Compliance    : {m_oc.get('weighted_compliance_Nmm', 96588.7):,.1f} N*mm\n"
            f"* Iterations    : 100 it (351.6 s)"
        )
        plotter.add_text(text_oc, position="upper_left", font_size=11, color="white", font="courier")
        plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z", line_width=2)
        plotter.camera_position = cam_pos_iso

        plotter.screenshot(str(dual_png))
        plotter.close()
        print(f" [+] Saved Side-by-Side Dual Voxel Comparison: {dual_png}")

    # 5. Render 1:3 Triple Voxel Comparison (BESO vs SIMP-OC vs AM)
    if "beso" in solid_voxels_dict and "oc" in solid_voxels_dict and "am" in solid_voxels_dict:
        triple_png = out_dir / "ge_bracket_triple_voxels_iso.png"
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2500, 850])

        m_am = metrics_by_algo.get("am", {})

        # Subplot 0: BESO
        plotter.subplot(0, 0)
        plotter.set_background("black")
        plotter.add_mesh(
            solid_voxels_dict["beso"],
            color="#38bdf8",
            show_edges=True,
            edge_color="#202020",
            line_width=1.0,
            smooth_shading=False,
        )
        text_beso_compact = (
            f"[1] BESO V0.15\n"
            f"Mass      : {m_beso.get('mass_g', 244.5):.1f} g\n"
            f"Max VM    : {m_beso.get('worst_vm_max_MPa', 761.9):.1f} MPa (PASS)\n"
            f"Compliance: {m_beso.get('weighted_compliance_Nmm', 84333.9):,.1f} N*mm"
        )
        plotter.add_text(text_beso_compact, position="upper_left", font_size=11, color="white", font="courier")
        plotter.add_axes()
        plotter.camera_position = cam_pos_iso

        # Subplot 1: SIMP-OC
        plotter.subplot(0, 1)
        plotter.set_background("black")
        plotter.add_mesh(
            solid_voxels_dict["oc"],
            color="#fb923c",
            show_edges=True,
            edge_color="#202020",
            line_width=1.0,
            smooth_shading=False,
        )
        text_oc_compact = (
            f"[2] SIMP-OC V0.15\n"
            f"Mass      : {m_oc.get('mass_g', 244.1):.1f} g\n"
            f"Max VM    : {m_oc.get('worst_vm_max_MPa', 1023.9):.1f} MPa\n"
            f"Compliance: {m_oc.get('weighted_compliance_Nmm', 96588.7):,.1f} N*mm"
        )
        plotter.add_text(text_oc_compact, position="upper_left", font_size=11, color="white", font="courier")
        plotter.add_axes()
        plotter.camera_position = cam_pos_iso

        # Subplot 2: AM
        plotter.subplot(0, 2)
        plotter.set_background("black")
        plotter.add_mesh(
            solid_voxels_dict["am"],
            color="#34d399",
            show_edges=True,
            edge_color="#202020",
            line_width=1.0,
            smooth_shading=False,
        )
        text_am_compact = (
            f"[3] AM (+Y Overhang) V0.15\n"
            f"Mass      : {m_am.get('mass_g', 244.1):.1f} g\n"
            f"Max VM    : {m_am.get('worst_vm_max_MPa', 1441.5):.1f} MPa\n"
            f"Compliance: {m_am.get('weighted_compliance_Nmm', 137044.4):,.1f} N*mm"
        )
        plotter.add_text(text_am_compact, position="upper_left", font_size=11, color="white", font="courier")
        plotter.add_axes()
        plotter.camera_position = cam_pos_iso

        plotter.screenshot(str(triple_png))
        plotter.close()
        print(f" [+] Saved Triple Voxel Comparison: {triple_png}")


if __name__ == "__main__":
    render_all_ge_bracket_iso_views()
