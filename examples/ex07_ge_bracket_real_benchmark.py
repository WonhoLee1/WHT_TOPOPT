"""Showcase 07 Refined: Real GE Jet Engine Bracket Challenge Master Benchmark.

Reflects the actual GrabCAD 2013 Challenge geometry, material properties, and 4 multi-load cases:
1. Real STLs from external/freeto_stls:
   - GE_domain.STL (176 x 62.5 x 107 mm design envelope)
   - GE_fixed.STL (4 bolt mounting surfaces, all DOFs fixed)
   - GE_force.STL (0.75 in pin lug bores, rigid body interface)
2. Material Properties (Ti-6Al-4V):
   - Young's modulus E = 113.8 GPa = 113,800 MPa
   - Poisson's ratio nu = 0.342
   - Yield strength sigma_y = 131 ksi = 903 MPa
   - Density rho = 4.43 g/cm^3 = 4.43e-3 g/mm^3
3. Four Official Challenge Load Cases:
   - Case 1: Vertical up 8,000 lbf (35,585.8 N, +y)
   - Case 2: Horizontal out 8,500 lbf (37,809.9 N, +z)
   - Case 3: 42 deg outward 9,500 lbf (42,258.1 N, 9500*cos(42) +y, 9500*sin(42) +z)
   - Case 4: Torque 5,000 lb-in (564,924 N-mm, torque about y-axis)
4. Pin Interface:
   - Infinitely stiff pin modeled via RigidBody (CalculiX/Nastran RBE convention)
   - Reference node on pin axis, slave DOFs condensed via T^T * K * T
5. Algorithms Compared:
   - BESO (Bi-directional Evolutionary Structural Optimization)
   - SIMP-OC (Optimality Criteria)
   - AM Overhang Filter (+y build direction, self-supporting)
   - Local Volume Constraint (MMA maximum member size control)
6. Evaluation:
   - Reduced void-free solve (_void_free_system)
   - 0/1 binary re-evaluation at identical volume fraction
   - Element-centre von Mises stress for each load case (max & 99.5th percentile)
   - Live 3-view progress GIF generation
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root and aieng src to path
ROOT = Path(__file__).resolve().parents[2]
AIENG_SRC = ROOT / "aieng" / "src"
if str(AIENG_SRC) not in sys.path:
    sys.path.insert(0, str(AIENG_SRC))

try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

from aieng.topopt import BuiltinAdapter, TopOptEvaluator, run_topology_optimization
from aieng.topopt.elements import _b_matrix, _hex8_dN, isotropic_D, reference_stiffness
from aieng.topopt.mesh import Mesh
from aieng.topopt.progress import ProgressGif
from aieng.topopt.stl_domain import StlCase, build_stl_problem

# Engineering constants
LBF = 4.4482216152605
INCH = 25.4
YIELD_MPA = 903.0
DENSITY_G_MM3 = 4.43e-3
C42 = float(np.cos(np.radians(42.0)))
S42 = float(np.sin(np.radians(42.0)))

# Official Challenge Case Definition
CASE_GE = StlCase(
    domain="GE_domain.STL",
    forces=[],
    fixed=["GE_fixed.STL"],
    pin_region="GE_force.STL",
    pin_loads=[
        # Case 1: Vertical up 8,000 lbf
        ((0.0, 8000.0 * LBF, 0.0), (0.0, 0.0, 0.0)),
        # Case 2: Horizontal out 8,500 lbf
        ((0.0, 0.0, 8500.0 * LBF), (0.0, 0.0, 0.0)),
        # Case 3: 42 deg from vertical, outward, 9,500 lbf
        ((0.0, 9500.0 * LBF * C42, 9500.0 * LBF * S42), (0.0, 0.0, 0.0)),
        # Case 4: Torque 5,000 lb-in about vertical axis
        ((0.0, 0.0, 0.0), (0.0, 5000.0 * LBF * INCH, 0.0)),
    ],
)
LOAD_NAMES = [
    "Case 1: Vertical up 8,000 lbf (35.6 kN)",
    "Case 2: Horizontal out 8,500 lbf (37.8 kN)",
    "Case 3: 42 deg outward 9,500 lbf (42.3 kN)",
    "Case 4: Torque 5,000 lb-in (564.9 N-m)",
]


def calculate_von_mises(mesh: Any, E: np.ndarray, nu: float, u: np.ndarray) -> np.ndarray:
    """Calculate (ncase, ne) element-centre von Mises stress for Hex8 elements."""
    X = mesh.nodes[mesh.elems]
    dN = _hex8_dN(0.0, 0.0, 0.0)
    J = np.einsum("ka,nkb->nab", dN, X)
    B = _b_matrix(np.linalg.solve(J, np.broadcast_to(dN.T, (mesh.n_elems, 3, 8))))
    eps = np.einsum("nsd,cnd->cns", B, u[:, mesh.element_dofs()])
    sig = np.einsum("st,cnt->cns", isotropic_D(1.0, nu), eps) * E[None, :, None]
    sx, sy, sz, txy, tyz, tzx = (sig[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy**2 + tyz**2 + tzx**2))


def binarize_design(problem: Any, inside: np.ndarray, xphys: np.ndarray, vol: np.ndarray, design_fraction: float) -> np.ndarray:
    """Create exact 0/1 solid-void design field matching target volume fraction."""
    order = np.argsort(-np.where(inside, xphys, -np.inf))
    order = order[inside[order]]
    keep_count = int(np.searchsorted(np.cumsum(vol[order]), design_fraction * vol[inside].sum()))
    keep = order[:keep_count]
    xb = np.zeros(problem.mesh.n_elems, dtype=np.float64)
    xb[keep] = 1.0
    xb[problem.passive_solid] = 1.0
    return xb


def evaluate_binary_design(problem: Any, xb: np.ndarray, vol: np.ndarray) -> Dict[str, Any]:
    """Evaluate compliance, mass, and von Mises stress of the 0/1 physical design."""
    inner = replace(
        problem,
        filter_type="none",
        rmin=0.0,
        projection_beta=0.0,
        emin_ratio=1e-6,
        optimizer="oc",
        variable_groups=None,
        am_build_direction=None,
        local_volume=None,
        robust=None,
    )
    e = TopOptEvaluator(inner, BuiltinAdapter(method="direct")).evaluate(xb, gradients=False)
    E = inner.Emin + (inner.E0 - inner.Emin) * xb**3
    vm = calculate_von_mises(problem.mesh, E, problem.nu, e.u)[:, xb > 0.5]

    mass_g = float(vol[xb > 0.5].sum() * DENSITY_G_MM3)
    comp_nmm = [float(c) for c in e.case_compliance]
    vm_max = [float(v) for v in vm.max(axis=1)]
    vm_p995 = [float(v) for v in np.percentile(vm, 99.5, axis=1)]

    return {
        "mass_g": mass_g,
        "compliance_Nmm": comp_nmm,
        "weighted_compliance_Nmm": float(np.sum(comp_nmm)),
        "vm_max_MPa": vm_max,
        "vm_p995_MPa": vm_p995,
        "passes_yield_max": bool(max(vm_max) <= YIELD_MPA),
        "passes_yield_p995": bool(max(vm_p995) <= YIELD_MPA),
    }


def render_3view_figure(problem: Any, xb: np.ndarray, title: str, path: Path) -> None:
    """Render 3-view orthogonal projection (Front: x-y, Side: z-y, Top: x-z) with dark theme."""
    grid = problem.mesh.grid  # (nx, ny, nz)
    nx, ny, nz = grid
    field3d = xb.reshape((nz, ny, nx))  # (k, j, i)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0a0d14")
    fig.suptitle(title, color="#ffffff", fontsize=11, fontweight="bold")

    views = [
        ("Front View (x-y)", np.max(field3d, axis=0), "x (width)", "y (height)"),
        ("Side View (z-y)", np.max(field3d, axis=2), "z (depth)", "y (height)"),
        ("Top View (x-z)", np.max(field3d, axis=1), "x (width)", "z (depth)"),
    ]

    for ax, (vname, proj, xlabel, ylabel) in zip(axes, views):
        ax.set_facecolor("#121824")
        im = ax.imshow(
            proj,
            origin="lower",
            cmap="Blues_r",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_title(vname, color="#94a3b8", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, color="#64748b", fontsize=8)
        ax.set_ylabel(ylabel, color="#64748b", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def run_ge_benchmark_suite(
    mesh_control: int = 40,
    volfrac: float = 0.15,
    algos: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run full GE Bracket challenge benchmark suite."""
    if algos is None:
        algos = ["beso", "oc", "am", "local_volume"]

    out_dir = out_dir or (ROOT / "wht_topopt" / "examples" / "outputs" / "ge_bracket")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = out_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    stl_dir = ROOT / "external" / "freeto_stls"
    if not stl_dir.exists():
        raise FileNotFoundError(f"STL directory not found: {stl_dir}")

    print("=" * 80)
    print(" [RUN] GE Jet Engine Bracket Challenge Master Benchmark Suite (Real STL & 4 Cases)")
    print(f" - MeshControl: {mesh_control} | Target Volume Fraction: {volfrac:.2f} (Target Mass ~245g)")
    print(f" - Material: Ti-6Al-4V (E=113.8 GPa, nu=0.342, Yield=903 MPa, Density=4.43 g/cm^3)")
    print(f" - Algorithms: {algos}")
    print("=" * 80)

    # 1. Build Base Problem
    t_start = time.perf_counter()
    problem_base, inside = build_stl_problem(
        CASE_GE,
        mesh_control=mesh_control,
        volfrac=volfrac,
        base=stl_dir,
        E0=113800.0,
        nu=0.342,
    )
    h_elem = float(problem_base.mesh.nodes[1, 0] - problem_base.mesh.nodes[0, 0])
    problem_base = replace(problem_base, rmin=1.5 * h_elem)
    ref_vol = reference_stiffness(problem_base.mesh, problem_base.nu).volume
    t_prep = time.perf_counter() - t_start

    print(f" [Domain Voxelization Completed in {t_prep:.2f} s]")
    print(f" - Bounding Grid: {problem_base.mesh.grid} ({problem_base.mesh.n_elems} Hex8 elements)")
    print(f" - In-domain (Design Envelope): {inside.sum()} elements ({inside.sum() / problem_base.mesh.n_elems * 100:.1f}%)")
    print(f" - Passive Void: {len(problem_base.passive_void)} | Passive Solid: {len(problem_base.passive_solid)}")
    print(f" - Rigid Pin Slaves: {len(problem_base.rigid_bodies[0].nodes)} nodes")
    print("-" * 80)

    results = []

    for algo in algos:
        print(f"\n>>> Running Algorithm: {algo.upper()} ...")
        t_algo_start = time.perf_counter()

        # Configure specific problem options
        p = problem_base
        tag = f"ge_{algo}_V{int(round(volfrac * 100))}_mc{mesh_control}"

        if algo == "beso":
            p = replace(p, optimizer="beso", max_iters=120)
        elif algo == "oc":
            p = replace(p, optimizer="oc", max_iters=100)
        elif algo == "am":
            # Additive manufacturing build direction +y (base on build plate)
            p = replace(p, optimizer="oc", max_iters=100, am_build_direction="+y")
            tag += "_am_y"
        elif algo == "local_volume":
            # Maximum member size control (MMA with local volume constraint)
            # Local radius 8.0 mm, alpha 0.5
            p = replace(p, optimizer="mma", max_iters=100, local_volume={"radius": 8.0, "alpha": 0.5, "p": 16.0})
            tag += "_local_vol"
        elif algo == "robust":
            # Minimum member size control (eroded/dilated 3-field projection)
            p = replace(p, optimizer="oc", max_iters=120, robust={"delta": 0.15, "beta0": 2.0, "beta_max": 16.0, "beta_every": 30, "volume_every": 5})
            tag += "_robust"

        gif_path = progress_dir / f"{tag}.gif"
        gif = ProgressGif(gif_path, p.mesh.grid, views=("z", "x", "y"), projection="max")

        def progress_callback(iteration: int, eval_res: Any, _x: np.ndarray, _gif=gif, _t=t_algo_start, _name=algo):
            elapsed = time.perf_counter() - _t
            c_val = eval_res.compliance
            v_val = eval_res.volume_fraction
            if iteration % 5 == 0 or iteration <= 5:
                print(f"   [{_name.upper()}] Iter {iteration:3d}: Compliance = {c_val:.4e} N-mm, VolFrac = {v_val:.4f}, Elapsed = {elapsed:.1f} s", flush=True)
            _gif.add(eval_res.xphys, f"{_name.upper()} it {iteration}  C = {c_val:.4g}  V = {v_val:.3f}  {elapsed:.0f} s")

        # Run optimization with BuiltinAdapter (MKL Pardiso & _void_free_system)
        adapter = BuiltinAdapter(method="direct")
        r = run_topology_optimization(p, adapter, callback=progress_callback)
        wall_sec = time.perf_counter() - t_algo_start

        # 0/1 binary re-evaluation at identical volume fraction
        xb = binarize_design(p, inside, r.xphys, ref_vol, volfrac)
        eval_metrics = evaluate_binary_design(p, xb, ref_vol)

        worst_vm = max(eval_metrics["vm_max_MPa"])
        worst_p995 = max(eval_metrics["vm_p995_MPa"])

        record = {
            "algo": algo,
            "volfrac": volfrac,
            "mesh_control": mesh_control,
            "iterations": r.iterations,
            "converged": r.converged,
            "wall_sec": wall_sec,
            "mass_g": eval_metrics["mass_g"],
            "weighted_compliance_Nmm": eval_metrics["weighted_compliance_Nmm"],
            "compliance_per_case_Nmm": eval_metrics["compliance_Nmm"],
            "vm_max_per_case_MPa": eval_metrics["vm_max_MPa"],
            "vm_p995_per_case_MPa": eval_metrics["vm_p995_MPa"],
            "worst_vm_max_MPa": worst_vm,
            "worst_vm_p995_MPa": worst_p995,
            "passes_yield_max": eval_metrics["passes_yield_max"],
            "passes_yield_p995": eval_metrics["passes_yield_p995"],
        }
        results.append(record)

        # Save result JSON and NumPy density
        np.save(out_dir / f"{tag}_xphys.npy", r.xphys)
        np.save(out_dir / f"{tag}_binary.npy", xb)

        # Render 3-view orthogonal projection
        title_str = (
            f"GE Bracket Challenge {algo.upper()} (V={volfrac:.2f}): "
            f"Mass={eval_metrics['mass_g']:.0f}g, Max vM={worst_vm:.0f}MPa (Yield 903), {r.iterations} it"
        )
        render_3view_figure(p, xb, title_str, out_dir / f"{tag}_3view.png")

        print(f" [DONE] {algo.upper()} Completed in {wall_sec:.1f} s ({r.iterations} it):")
        print(f"   - Mass: {eval_metrics['mass_g']:.1f} g (Original: 2,033 g, Winner: 327 g)")
        print(f"   - Max von Mises per case: {[round(v, 1) for v in eval_metrics['vm_max_MPa']]} MPa")
        print(f"   - 99.5% von Mises per case: {[round(v, 1) for v in eval_metrics['vm_p995_MPa']]} MPa")
        print(f"   - Compliance per case: {[round(c, 1) for c in eval_metrics['compliance_Nmm']]} N-mm")
        print(f"   - Passes Yield (Max / 99.5%): {eval_metrics['passes_yield_max']} / {eval_metrics['passes_yield_p995']}")

    # Save summary report
    summary_path = out_dir / f"summary_real_ge_mc{mesh_control}_v{int(round(volfrac * 100))}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GE Jet Engine Bracket Real STL Challenge Benchmark")
    parser.add_argument("--mesh-control", type=int, default=40)
    parser.add_argument("--volfrac", type=float, default=0.15)
    parser.add_argument("--algos", nargs="*", default=["beso", "oc", "am", "local_volume"])
    args = parser.parse_args()

    run_ge_benchmark_suite(
        mesh_control=args.mesh_control,
        volfrac=args.volfrac,
        algos=args.algos,
    )
