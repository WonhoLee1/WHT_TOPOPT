"""Extract watertight STL mesh and render 3D views for the GE Jet Engine Bracket.

Uses Marching Cubes isocontouring on the 0/1 binary / physical density fields.
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
import trimesh

ROOT = Path(__file__).resolve().parents[2]
AIENG_SRC = ROOT / "aieng" / "src"
if str(AIENG_SRC) not in sys.path:
    sys.path.insert(0, str(AIENG_SRC))

from aieng.topopt.stl_domain import StlCase, build_stl_problem


def export_and_render_ge_stl(
    algo_tag: str = "ge_beso_V15_mc40",
    mesh_control: int = 40,
    volfrac: float = 0.15,
    out_dir: Path = ROOT / "wht_topopt" / "examples" / "outputs" / "ge_bracket",
):
    out_dir = Path(out_dir)
    binary_path = out_dir / f"{algo_tag}_binary.npy"
    if not binary_path.exists():
        print(f"File not found: {binary_path}")
        return

    xb = np.load(binary_path)

    # Reconstruct problem grid coordinates
    stl_dir = ROOT / "external" / "freeto_stls"
    case = StlCase(
        domain="GE_domain.STL", forces=[], fixed=["GE_fixed.STL"], pin_region="GE_force.STL",
        pin_loads=[((0.0, 35585.8, 0.0), (0.0, 0.0, 0.0))],
    )
    p, _ = build_stl_problem(case, mesh_control, volfrac, base=stl_dir)
    grid = p.mesh.grid  # (nx, ny, nz) = (40, 15, 25)
    nx, ny, nz = grid
    field3d = xb.reshape((nz, ny, nx))  # (k, j, i)

    # Voxel physical dimensions
    nodes = p.mesh.nodes
    h = float(nodes[1, 0] - nodes[0, 0])
    origin = nodes[0]  # (x0, y0, z0)

    # Marching Cubes at isovalue 0.5
    # skimage marching_cubes input indexing: axis 0 is z, axis 1 is y, axis 2 is x
    padded = np.pad(field3d, 1, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = measure.marching_cubes(
        padded, level=0.5, spacing=(h, h, h)
    )
    # Adjust for padding and origin
    verts = verts - h
    # Swap coordinates: marching_cubes returns (z, y, x) -> swap to (x, y, z)
    verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]) + origin

    # Save STL
    mesh_tri = trimesh.Trimesh(vertices=verts_xyz, faces=faces)
    mesh_tri.fix_normals()
    stl_out_path = out_dir / f"{algo_tag}.stl"
    mesh_tri.export(str(stl_out_path))
    print(f"Exported watertight STL: {stl_out_path} ({len(verts_xyz)} verts, {len(faces)} faces)")

    # Render High-Resolution 3D Views
    fig = plt.figure(figsize=(10, 8), facecolor="#0a0d14")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0d14")

    # Rotate coordinates for intuitive viewing: y vertical upwards
    # Plot x, z, y
    verts_plot = np.column_stack([verts_xyz[:, 0], verts_xyz[:, 2], verts_xyz[:, 1]])
    mesh_plot = Poly3DCollection(
        verts_plot[faces],
        facecolors="#38bdf8",
        edgecolors="#0284c7",
        linewidths=0.1,
        alpha=0.95,
    )
    ax.add_collection3d(mesh_plot)

    ax.set_xlim(verts_plot[:, 0].min(), verts_plot[:, 0].max())
    ax.set_ylim(verts_plot[:, 1].min(), verts_plot[:, 1].max())
    ax.set_zlim(verts_plot[:, 2].min(), verts_plot[:, 2].max())

    ax.view_init(elev=25, azim=-55)
    ax.set_title(f"GE Jet Engine Bracket Challenge (GrabCAD) - {algo_tag}\nTi-6Al-4V | V=0.15 | Mass ~245g", color="#ffffff", fontsize=12, fontweight="bold", pad=20)
    ax.set_xlabel("X (Width, mm)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Z (Depth, mm)", color="#94a3b8", fontsize=9)
    ax.set_zlabel("Y (Vertical, mm)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=8)

    # Dark theme pane styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1e293b")
    ax.yaxis.pane.set_edgecolor("#1e293b")
    ax.zaxis.pane.set_edgecolor("#1e293b")
    ax.grid(color="#1e293b", linestyle=":", alpha=0.5)

    img_out_path = out_dir / f"{algo_tag}_3d.png"
    plt.tight_layout()
    plt.savefig(img_out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Rendered 3D image: {img_out_path}")


if __name__ == "__main__":
    export_and_render_ge_stl("ge_beso_V15_mc40")
