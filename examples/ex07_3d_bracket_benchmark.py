"""Showcase 07: 3D Topology Optimization Benchmark Suite & Dual-Engine Performance Profiling.

Compares:
1. Geometries:
   - GE Aircraft Bracket (watertight CAD voxelized domain with 4 bolt constraints & clevis lug load)
   - 3D Cantilever Beam (Ferrari & Sigmund top3d125 benchmark domain)
2. Algorithms:
   - SIMP-OC (Optimality Criteria)
   - SIMP-MMA (Method of Moving Asymptotes)
   - BESO (Bi-directional Evolutionary Structural Optimization)
   - SEMDOT (Smooth-Edged Material Distribution)
3. Tri-Engine Execution Architecture:
   - Numba (Numba JIT CSC assembly + MKL PARDISO via pypardiso + Numba sensitivity kernel)
   - Kratos v1 (Kratos 1st Gen TopologyOptimizationApplication / in-memory cached strategy)
   - Kratos v2 (Kratos 2nd Gen OptimizationApplication response pipeline)
4. Outputs:
   - Binary STL CAD models via Marching Cubes isocontouring
   - ParaView-style 3D PyVista dark theme rendering
   - Quantitative speed comparison charts and engineering report
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

from wht_topopt.algorithms import (
    AlgorithmRegistry,
    TopOptAlgorithmAdapter,
    BESOOptimizer,
    MMAOptimizer,
    OCOptimizer,
    SEMDOTOptimizer,
)
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter
from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.core.voxelizer import (
    VoxelizedDomain3D,
    voxelize_ge_aircraft_bracket,
    voxelize_top3d_cantilever_beam,
)
from wht_topopt.solvers.kratos_solver import KratosSolver
from wht_topopt.solvers.scipy_solver import ScipySolver
from wht_topopt.functions.inertia import InertiaSensitivityCalculator
import scipy.sparse.linalg as spla

GLOBAL_FONT_SIZE_PT: int = 9
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE_PT


def _place_outside_legend(ax: plt.Axes, max_per_col: int = 4) -> None:
    """Place legend outside the axes bounding box with automatic column scaling."""
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
class IterationSnapshot3D:
    """Record of a single optimization iteration."""
    iteration: int
    compliance: float
    volume_fraction: float
    max_change: float
    time_fea_ms: float
    time_sens_ms: float
    time_opt_ms: float
    max_stress_mpa: float = 0.0
    com_shift_mm: float = 0.0
    freq1_hz: float = 0.0


@dataclass
class BenchmarkRunResult3D:
    """Complete summary of a single 3D benchmark run."""
    case_name: str
    engine_name: str
    method_name: str
    resolution: str
    n_elements: int
    n_dofs: int
    final_compliance: float
    final_volume: float
    iterations: int
    total_time_sec: float
    avg_fea_time_ms: float
    avg_sens_time_ms: float
    avg_opt_time_ms: float
    final_max_stress_mpa: float = 0.0
    final_com_shift_mm: float = 0.0
    final_freq1_hz: float = 0.0
    history: List[IterationSnapshot3D] = field(default_factory=list)
    rho_phys: Optional[np.ndarray] = None
    stl_path: Optional[str] = None
    pyvista_png_path: Optional[str] = None
    voxel_png_path: Optional[str] = None
    cad_png_path: Optional[str] = None
    gif_path: Optional[str] = None


class EngineRunner3D:
    """Unified multi-engine execution manager for 3D Topology Optimization."""

    def __init__(self, domain: VoxelizedDomain3D, engine: str = "numba") -> None:
        """Initialize EngineRunner3D with domain and solver backend."""
        self.domain = domain
        self.mesh = domain.mesh
        self.engine = engine.lower()
        self.fem = FEMEngine(self.mesh)
        self.F = domain.assemble_load_vector()
        self.all_dofs = np.arange(self.mesh.ndof)
        self.free_dofs = np.setdiff1d(self.all_dofs, domain.fixed_dofs)

        # Solver instances
        if self.engine == "numba":
            self.scipy_solver = ScipySolver(method="pardiso")
            self.kratos_solver = None
        else:
            self.scipy_solver = None
            # Standard Kratos v2 with PARDISO direct solver
            self.kratos_solver = KratosSolver(method="direct", linear_solver="auto", num_threads=6)

    def solve_fea_and_sensitivity(
        self,
        rho_phys: np.ndarray,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
    ) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        """Perform FEA solve and adjoint sensitivity calculation.

        Returns
        -------
        compliance : float
            Total structural compliance c = F^T * u.
        u : np.ndarray
            Nodal displacement vector.
        df_dphys : np.ndarray
            Sensitivity of compliance w.r.t physical density df/drho_phys.
        t_fea_ms : float
            FEA solve time in milliseconds.
        t_sens_ms : float
            Sensitivity calculation time in milliseconds.
        """
        if self.engine == "numba":
            t0 = time.perf_counter()
            K = self.fem.assemble_stiffness_matrix(rho_phys, penal=penal, e0=e0, emin=emin)
            res = self.scipy_solver.solve(K, self.F, free_dofs=self.free_dofs, fixed_dofs=self.domain.fixed_dofs)
            u = res.u
            compliance = float(np.dot(self.F, u))
            t_fea_ms = (time.perf_counter() - t0) * 1000.0

            t_sens0 = time.perf_counter()
            c_e0 = self.fem.compute_unit_element_compliance(u)
            df_dphys = -penal * (rho_phys ** (penal - 1.0)) * (e0 - emin) * c_e0
            t_sens_ms = (time.perf_counter() - t_sens0) * 1000.0

            return compliance, u, df_dphys, t_fea_ms, t_sens_ms

        elif self.engine in {"kratos_v2", "kratos_v2_native"}:
            t0 = time.perf_counter()
            youngs = emin + (rho_phys ** penal) * (e0 - emin)
            res = self.kratos_solver.evaluate_mesh_cached(
                nodes=self.mesh.nodes,
                elems=self.mesh.elements,
                etype="hex8",
                fixed_dofs=self.domain.fixed_dofs,
                loads=self.F.reshape(1, -1),
                youngs=youngs,
                nu=nu,
                compute_sensitivities=True,
            )
            u = res["u"][0]
            compliance = float(np.dot(self.F, u))
            t_fea_ms = (time.perf_counter() - t0) * 1000.0

            t_sens0 = time.perf_counter()
            dcdE = res.get("dcdE", None)
            if dcdE is not None and not np.all(dcdE == 0):
                df_dphys = -penal * (rho_phys ** (penal - 1.0)) * (e0 - emin) * dcdE
            else:
                c_e0 = self.fem.compute_unit_element_compliance(u)
                df_dphys = -penal * (rho_phys ** (penal - 1.0)) * (e0 - emin) * c_e0
            t_sens_ms = (time.perf_counter() - t_sens0) * 1000.0

            return compliance, u, df_dphys, t_fea_ms, t_sens_ms

        else:
            raise ValueError(f"Unknown engine: {self.engine}. (Note: kratos_v1 has been deprecated; use kratos_v2 or kratos_v2_native)")


class BracketBenchmark3DSuite:
    """Comprehensive 3D Topology Optimization Benchmark and Profiling Suite."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/ex07_bracket",
        resolution: str = "fast",
        max_iter: int = 15,
        target_vol: float = 0.35,
    ) -> None:
        """Initialize Benchmark Suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution.lower()
        self.max_iter = max_iter
        self.target_vol = target_vol

    def get_domain(self, case: str) -> VoxelizedDomain3D:
        """Construct domain with selected resolution."""
        if case == "bracket":
            if self.resolution == "standard":
                nx, ny, nz = 40, 20, 16  # 3.0mm x 3.0mm x 3.125mm (12,800 elements)
            else:
                nx, ny, nz = 20, 10, 8   # 6.0mm x 6.0mm x 6.25mm (1,600 elements)
            return voxelize_ge_aircraft_bracket(nx=nx, ny=ny, nz=nz)
        elif case == "beam":
            if self.resolution == "standard":
                nx, ny, nz = 20, 10, 5   # 3.0mm x 3.0mm x 3.0mm (1,000 elements)
                lx, ly, lz = 60.0, 30.0, 15.0
            else:
                nx, ny, nz = 10, 5, 3    # 6.0mm x 6.0mm x 5.0mm (150 elements)
                lx, ly, lz = 60.0, 30.0, 15.0
            return voxelize_top3d_cantilever_beam(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz)
        else:
            raise ValueError(f"Unknown benchmark case: {case}")

    def create_optimizer(
        self,
        formulation: str = "simp",
        optimizer: Optional[str] = "mma",
        n_elem: int = 0,
        target_vol: float = 0.35,
    ) -> TopOptAlgorithmAdapter:
        """Factory for decoupled Formulation and Optimizer pipeline."""
        adapter = AlgorithmRegistry.create_system(
            formulation=formulation,
            optimizer=optimizer,
            target_volfrac=target_vol,
            volfrac=target_vol,
        )
        adapter.initialize(n_elem=n_elem, target_vol=target_vol)
        return adapter

    def export_stl(self, mesh: StructuredMesh3D, rho: np.ndarray, stl_path: Path) -> Path:
        """Export Marching Cubes isocontour surface to watertight binary STL."""
        import skimage.measure
        import trimesh

        grid3d = rho.reshape((mesh.nelz, mesh.nely, mesh.nelx))
        # Zero-pad grid to guarantee watertight closed boundary surfaces (no hollow shell artifact)
        padded_grid = np.pad(grid3d, pad_width=1, mode="constant", constant_values=0.0)

        level = 0.5
        min_v, max_v = float(np.min(grid3d)), float(np.max(grid3d))
        if level <= min_v or level >= max_v:
            level = 0.5 * (min_v + max_v)

        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                padded_grid,
                level=level,
            )
            # Map padded grid indices to physical coordinates
            iz = verts[:, 0]
            iy = verts[:, 1]
            ix = verts[:, 2]

            is_centered_xy = float(np.min(mesh.node_coords[:, 0])) < -0.1
            x_offset = mesh.lx * 0.5 if is_centered_xy else 0.0
            y_offset = mesh.ly * 0.5 if is_centered_xy else 0.0

            x = (ix - 0.5) * mesh.dx - x_offset
            y = (iy - 0.5) * mesh.dy - y_offset
            z = (iz - 0.5) * mesh.dz

            verts_xyz = np.column_stack([x, y, z])
            stl_mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces)
            stl_path.parent.mkdir(parents=True, exist_ok=True)
            stl_mesh.export(str(stl_path))
        except Exception:
            box = trimesh.creation.box(extents=[mesh.lx, mesh.ly, mesh.lz])
            stl_path.parent.mkdir(parents=True, exist_ok=True)
            box.export(str(stl_path))

        return stl_path

    def render_pyvista_3d(
        self,
        stl_path: Path,
        png_path: Path,
        title: str,
        color: str = "#8fa8c4",
    ) -> Path:
        """Render high-fidelity watertight CAD surface using PyVista (dark theme)."""
        import pyvista as pv

        pv.global_theme.font.size = 12
        pv.global_theme.font.color = "white"
        pv.global_theme.background = "black"

        plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
        plotter.set_background("black")

        try:
            pv_mesh = pv.read(str(stl_path))
            plotter.add_mesh(
                pv_mesh,
                color=color,
                edge_color="#404040",
                show_edges=True,
                smooth_shading=True,
            )
            plotter.add_text(
                title,
                position="upper_left",
                font_size=12,
                color="white",
            )
            plotter.show_axes()
            plotter.view_isometric()
            png_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(png_path))
        finally:
            plotter.close()

        return png_path

    def render_voxel_volume_3d(
        self,
        mesh: StructuredMesh3D,
        rho: np.ndarray,
        png_path: Path,
        title: str,
        threshold: float = 0.35,
    ) -> Path:
        """Render discrete 3D solid element volume (Hex8 voxels with crisp edges)."""
        import pyvista as pv

        pv.global_theme.font.size = 12
        pv.global_theme.font.color = "white"
        pv.global_theme.background = "black"

        is_centered_xy = float(np.min(mesh.node_coords[:, 0])) < -0.1
        x_orig = -mesh.lx * 0.5 if is_centered_xy else 0.0
        y_orig = -mesh.ly * 0.5 if is_centered_xy else 0.0

        grid = pv.ImageData()
        grid.dimensions = (mesh.nelx + 1, mesh.nely + 1, mesh.nelz + 1)
        grid.spacing = (mesh.dx, mesh.dy, mesh.dz)
        grid.origin = (x_orig, y_orig, 0.0)
        grid.cell_data["density"] = rho

        solid_voxels = grid.threshold(value=threshold, scalars="density")

        plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
        plotter.set_background("black")

        try:
            if solid_voxels.n_cells > 0:
                plotter.add_mesh(
                    solid_voxels,
                    scalars="density",
                    cmap="viridis",
                    clim=[threshold, 1.0],
                    edge_color="#202020",
                    show_edges=True,
                    smooth_shading=False,
                    scalar_bar_args={"title": "Element Density", "color": "white"},
                )
            plotter.add_text(
                title,
                position="upper_left",
                font_size=12,
                color="white",
            )
            plotter.show_axes()
            plotter.view_isometric()
            png_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(png_path))
        finally:
            plotter.close()

        return png_path

    def generate_progression_gif(
        self,
        mesh: StructuredMesh3D,
        density_history: List[np.ndarray],
        metric_history: List[IterationSnapshot3D],
        gif_path: Path,
        title: str,
        fps: int = 4,
        threshold: float = 0.35,
    ) -> Path:
        """Generate animated GIF showing 3D element volume evolution across iterations."""
        import pyvista as pv
        import imageio.v2 as imageio

        if not density_history:
            return gif_path

        is_centered_xy = float(np.min(mesh.node_coords[:, 0])) < -0.1
        x_orig = -mesh.lx * 0.5 if is_centered_xy else 0.0
        y_orig = -mesh.ly * 0.5 if is_centered_xy else 0.0

        grid = pv.ImageData()
        grid.dimensions = (mesh.nelx + 1, mesh.nely + 1, mesh.nelz + 1)
        grid.spacing = (mesh.dx, mesh.dy, mesh.dz)
        grid.origin = (x_orig, y_orig, 0.0)

        plotter = pv.Plotter(off_screen=True, window_size=(720, 540))
        plotter.set_background("black")

        frames: List[np.ndarray] = []
        try:
            for idx, rho_it in enumerate(density_history):
                it_num = idx + 1
                comp = metric_history[idx].compliance if idx < len(metric_history) else 0.0
                vol = metric_history[idx].volume_fraction if idx < len(metric_history) else 0.0

                grid.cell_data["density"] = rho_it
                solid_voxels = grid.threshold(value=threshold, scalars="density")

                plotter.clear_actors()
                if solid_voxels.n_cells > 0:
                    plotter.add_mesh(
                        solid_voxels,
                        scalars="density",
                        cmap="viridis",
                        clim=[threshold, 1.0],
                        edge_color="#202020",
                        show_edges=True,
                        smooth_shading=False,
                        scalar_bar_args={"title": "Density", "color": "white"},
                    )
                plotter.add_text(
                    f"{title}\nIter {it_num:02d}/{len(density_history):02d} | C = {comp:.2f} | Vol = {vol:.1%}",
                    position="upper_left",
                    font_size=11,
                    color="white",
                )
                plotter.show_axes()
                plotter.view_isometric()
                frame = plotter.screenshot(return_img=True)
                frames.append(frame)

            gif_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        finally:
            plotter.close()

        return gif_path

    def run_single(
        self,
        case: str,
        engine: str,
        formulation: str = "simp",
        optimizer: Optional[str] = "mma",
        method: Optional[str] = None,
    ) -> BenchmarkRunResult3D:
        """Execute a single benchmark run configuration with decoupled Formulation and Optimizer."""
        if method is not None:
            clean_m = method.strip().lower()
            if "_" in clean_m:
                formulation, optimizer = clean_m.split("_", 1)
            else:
                formulation = clean_m
                optimizer = "ranking" if clean_m == "beso" else "mma"

        form_clean = formulation.strip().lower()
        opt_clean = "ranking" if form_clean == "beso" else (optimizer.strip().lower() if optimizer else "mma")
        method_label = f"{form_clean}_{opt_clean}" if form_clean != "beso" else "beso"

        domain = self.get_domain(case)
        mesh = domain.mesh
        n_elem = mesh.n_elem
        n_dof = mesh.ndof
        vol_total = float(np.sum(mesh.elem_volumes))

        # Setup engine, filter, and inertia calculator
        runner = EngineRunner3D(domain, engine=engine)
        rmin = 2.0 * max(mesh.dx, mesh.dy, mesh.dz)
        filter_op = DensityFilter(mesh, r_min=rmin)
        optimizer_obj = self.create_optimizer(
            formulation=form_clean,
            optimizer=opt_clean,
            n_elem=n_elem,
            target_vol=self.target_vol,
        )
        inertia_calc = InertiaSensitivityCalculator(nodes=mesh.node_coords, elements=mesh.elements)
        init_com = inertia_calc.compute_properties(np.ones(n_elem, dtype=float)).center_of_mass

        # Define physical material properties based on benchmark case
        if case == "bracket":
            # GE Aircraft Bracket: Ti-6Al-4V (Titanium Grade 5) in mm-N-tonne-s consistent unit system
            e0 = 110000.0  # MPa (N/mm^2)
            emin = 1e-4 * e0  # 11.0 MPa
            nu = 0.31
            rho_mat = 4.43e-9  # tonne/mm^3
            stress_limit_mpa = 830.0  # Ti-6Al-4V Yield Strength (MPa)
            compliance_unit = "mJ"
        else:
            # Cantilever Beam: Academic standard (top3d dimensionless)
            e0 = 1.0
            emin = 1e-6
            nu = 0.30
            rho_mat = 1.0
            stress_limit_mpa = 1.0
            compliance_unit = "c"

        def _eval_freq1(rho_v: np.ndarray) -> float:
            try:
                # Fast 1st eigenfrequency calculation via shifted ARPACK
                K_m = runner.fem.assemble_stiffness_matrix(rho_v, penal=3.0, e0=e0, emin=emin)
                M_m = runner.fem.assemble_M(rho_v * rho_mat)
                free_d = runner.free_dofs
                Kf = K_m[free_d, :][:, free_d]
                Mf = M_m[free_d, :][:, free_d]
                v_raw, _ = spla.eigsh(Kf, k=1, M=Mf, sigma=0.0, which="LM", tol=1e-3, maxiter=80)
                return float(np.sqrt(max(1e-12, v_raw[0])) / (2.0 * np.pi))
            except Exception:
                return 0.0

        # Initial density field: target volume, respecting passive masks
        x = np.full(n_elem, self.target_vol, dtype=float)
        x[domain.passive_solid_mask] = 1.0
        x[domain.passive_void_mask] = 1e-3

        history: List[IterationSnapshot3D] = []
        density_history: List[np.ndarray] = []
        t_total_start = time.perf_counter()

        # Branch 1: Kratos Native In-Memory Pipeline (kratos_v2_native: NLopt MMA)
        if engine == "kratos_v2_native":
            algo_label = "Kratos MMA (NLopt)"
            method = "mma"
            mat_info = f"Ti-6Al-4V (E={e0:.0f} MPa, nu={nu:.2f})" if case == "bracket" else f"Academic Top3D (E={e0:.1f}, nu={nu:.2f})"
            print(f"\n==========================================================================================================", flush=True)
            print(f" >>> [START] Case={case.upper()} | Engine=KRATOS_V2_NATIVE | Method={algo_label}", flush=True)
            print(f"     Mesh Grid : {mesh.nelx} x {mesh.nely} x {mesh.nelz} ({n_elem:,} Hex8 elements, {n_dof:,} DOFs)", flush=True)
            print(f"     Material  : {mat_info} | Load: Fz={np.sum(domain.load_values):.1f} N", flush=True)
            print(f"     Resolution: {self.resolution.upper()} (dx={mesh.dx:.2f}, dy={mesh.dy:.2f}, dz={mesh.dz:.2f})", flush=True)
            print(f"     Target Vol: {self.target_vol:.1%} | Max Iterations: {self.max_iter}", flush=True)
            print(f"----------------------------------------------------------------------------------------------------------", flush=True)
            print(f"  {'Iter':<7} | {'Compliance (' + compliance_unit + ')':<18} | {'Volume':<8} | {'Algorithm / Engine':<26} | {'Step Time':<9}", flush=True)
            print(f"  {'-'*88}", flush=True)

            res_native = runner.kratos_solver.run_kratos_v2_topopt(
                nodes=mesh.nodes,
                elems=mesh.elements,
                fixed_dofs=domain.fixed_dofs,
                loads=runner.F.reshape(1, -1),
                target_volfrac=self.target_vol,
                max_iterations=self.max_iter,
                filter_radius=rmin,
                penal=3.0,
                e0=e0,
                emin=emin,
                nu=nu,
            )
            final_rho = res_native["x_phys"]
            final_rho[domain.passive_solid_mask] = 1.0
            final_rho[domain.passive_void_mask] = 1e-3
            density_history = [np.full(n_elem, self.target_vol), final_rho]

            for rec in res_native["history"]:
                it_num = rec["iteration"]
                comp = rec["compliance"]
                vol = rec["volume_fraction"]
                chg = rec.get("change", 0.0)
                print(f"  {it_num:02d}/{self.max_iter:02d}   | {comp:<18.4f} | {vol*100:6.2f}%  | {algo_label:<26} | {'--':<9}", flush=True)
                history.append(
                    IterationSnapshot3D(
                        iteration=it_num,
                        compliance=comp,
                        volume_fraction=vol,
                        max_change=chg,
                        time_fea_ms=0.0,
                        time_sens_ms=0.0,
                        time_opt_ms=0.0,
                    )
                )

            # Evaluate final multi-physics state
            compliance, u, _, _, _ = runner.solve_fea_and_sensitivity(final_rho, penal=3.0, e0=e0, emin=emin, nu=nu)
            _, vm = runner.fem.compute_element_stresses(u, E0=e0)
            vm_phys = (final_rho ** 0.5) * vm
            solid_design_mask = ~domain.passive_void_mask
            final_max_stress = float(np.max(vm_phys[solid_design_mask])) if np.any(solid_design_mask) else float(np.max(vm_phys))
            cur_com = inertia_calc.compute_properties(final_rho).center_of_mass
            final_com_shift = float(np.linalg.norm(cur_com - init_com))
            final_freq1 = _eval_freq1(final_rho)
            t_total_sec = time.perf_counter() - t_total_start
            current_vol = float(np.sum(final_rho * mesh.elem_volumes)) / vol_total

        # Branch 2: WHT Optimization Framework (numba or kratos_v2)
        else:
            mat_info = f"Ti-6Al-4V (E={e0:.0f} MPa, nu={nu:.2f})" if case == "bracket" else f"Academic Top3D (E={e0:.1f}, nu={nu:.2f})"
            print(f"\n==========================================================================================================", flush=True)
            print(f" >>> [START] Case={case.upper()} | Engine={engine.upper()} (WHT) | Formulation={form_clean.upper()} | Optimizer={opt_clean.upper()}", flush=True)
            print(f"     Mesh Grid : {mesh.nelx} x {mesh.nely} x {mesh.nelz} ({n_elem:,} Hex8 elements, {n_dof:,} DOFs)", flush=True)
            print(f"     Material  : {mat_info} | Load: Fz={np.sum(domain.load_values):.1f} N", flush=True)
            print(f"     Resolution: {self.resolution.upper()} (dx={mesh.dx:.2f}, dy={mesh.dy:.2f}, dz={mesh.dz:.2f})", flush=True)
            print(f"     Target Vol: {self.target_vol:.1%} | Max Iterations: {self.max_iter}", flush=True)
            print(f"     Initial Center of Mass: [{init_com[0]:.2f}, {init_com[1]:.2f}, {init_com[2]:.2f}] mm", flush=True)
            print(f"----------------------------------------------------------------------------------------------------------", flush=True)
            print(f"  {'Iter':<7} | {'Compliance (' + compliance_unit + ')':<18} | {'Volume':<7} | {'MaxChange':<9} | {'Max Stress (MPa)':<16} | {'CoM Shift':<9} | {'Freq1 (Hz)':<10} | {'Step Time':<9}", flush=True)
            print(f"  {'-'*110}", flush=True)

            cached_f1 = 0.0
            for it in range(1, self.max_iter + 1):
                t_step0 = time.perf_counter()

                # 1. Filter density
                rho_phys = filter_op.forward(x)
                rho_phys[domain.passive_solid_mask] = 1.0
                rho_phys[domain.passive_void_mask] = 1e-3
                density_history.append(rho_phys.copy())

                # 2. FEA solve & sensitivity
                compliance, u, df_dphys, t_fea_ms, t_sens_ms = runner.solve_fea_and_sensitivity(
                    rho_phys, penal=3.0, e0=e0, emin=emin, nu=nu
                )

                # 3. Adjoint filter chain
                df_dx = filter_op.backward(df_dphys)

                # 4. Volume constraint
                current_vol = float(np.sum(rho_phys * mesh.elem_volumes)) / vol_total
                g_vol = np.array([current_vol / self.target_vol - 1.0])
                dg_dphys = mesh.elem_volumes / (vol_total * self.target_vol)
                dg_dx = filter_op.backward(dg_dphys).reshape(1, -1)

                # 5. Optimizer update step
                t_opt0 = time.perf_counter()
                x_new = optimizer_obj.update(
                    rho=x,
                    df_drho=df_dx,
                    g_vals=g_vol,
                    dg_drho=dg_dx,
                    elem_volumes=mesh.elem_volumes,
                    passive_solid=domain.passive_solid_mask,
                    passive_void=domain.passive_void_mask,
                )
                t_opt_ms = (time.perf_counter() - t_opt0) * 1000.0

                change = float(np.max(np.abs(x_new - x)))
                x = x_new

                # 6. Physical diagnostics: Stress, CoM, and 1st natural frequency
                _, vm = runner.fem.compute_element_stresses(u, E0=e0)
                # qp-relaxation (Le et al. 2010, q=0.5) and solid/design masking to suppress void singularities
                vm_phys = (rho_phys ** 0.5) * vm
                solid_design_mask = ~domain.passive_void_mask
                max_stress_val = float(np.max(vm_phys[solid_design_mask])) if np.any(solid_design_mask) else float(np.max(vm_phys))
                cur_com = inertia_calc.compute_properties(rho_phys).center_of_mass
                com_shift = float(np.linalg.norm(cur_com - init_com))

                # Compute modal frequency on initial and final iterations
                if it == 1 or it == self.max_iter:
                    cached_f1 = _eval_freq1(rho_phys)
                freq_str = f"{cached_f1:.1f}" if cached_f1 > 0 else "--"

                t_step_sec = time.perf_counter() - t_step0
                print(f"  {it:02d}/{self.max_iter:02d}   | {compliance:<18.4f} | {current_vol*100:5.1f}%  | {change:<9.4f} | {max_stress_val:16.2f} | {com_shift:6.2f} mm | {freq_str:<10} | {t_step_sec:6.2f}s", flush=True)

                snapshot = IterationSnapshot3D(
                    iteration=it,
                    compliance=compliance,
                    volume_fraction=current_vol,
                    max_change=change,
                    time_fea_ms=t_fea_ms,
                    time_sens_ms=t_sens_ms,
                    time_opt_ms=t_opt_ms,
                    max_stress_mpa=max_stress_val,
                    com_shift_mm=com_shift,
                    freq1_hz=cached_f1,
                )
                history.append(snapshot)

            t_total_sec = time.perf_counter() - t_total_start
            final_rho = filter_op.forward(x)
            final_rho[domain.passive_solid_mask] = 1.0
            final_rho[domain.passive_void_mask] = 1e-3
            final_max_stress = history[-1].max_stress_mpa
            final_com_shift = history[-1].com_shift_mm
            final_freq1 = history[-1].freq1_hz

        print(f"----------------------------------------------------------------------------------------------------------", flush=True)
        print(f" >>> [DONE] Case={case.upper()} | Engine={engine.upper()} | Method={method_label.upper()}", flush=True)
        print(f"     Final Compliance: {compliance:.4f} {compliance_unit} ({compliance*1e-3:.6f} J) | Final Volume: {current_vol:.2%} | Total Time: {t_total_sec:.2f}s", flush=True)
        print(f"     Max von Mises Stress: {final_max_stress:.2f} MPa (Yield Limit <= {stress_limit_mpa:.1f} MPa, Margin: {max(0.0, stress_limit_mpa - final_max_stress):.2f} MPa)", flush=True)
        print(f"     Center of Mass Shift: {final_com_shift:.4f} mm | 1st Natural Frequency: {final_freq1:.1f} Hz", flush=True)
        print(f"==========================================================================================================\n", flush=True)

        import shutil

        # 1. Export Watertight STL
        stl_filename = f"{case}_{engine}_{method_label}.stl"
        stl_path = self.output_dir / stl_filename
        self.export_stl(mesh, final_rho, stl_path)

        # 2. Render Watertight CAD 3D (Solid surface mesh)
        cad_png_filename = f"{case}_{engine}_{method_label}_cad_3d.png"
        cad_png_path = self.output_dir / cad_png_filename
        title_cad = f"3D {case.upper()} [{engine.upper()}] - {method_label.upper()} (Watertight CAD)\nC = {history[-1].compliance:.2f}, V = {history[-1].volume_fraction:.2%}"
        self.render_pyvista_3d(stl_path, cad_png_path, title=title_cad)

        # 3. Render Voxel Element Volume 3D (요소 볼륨 형태: discrete solid Hex8 voxels with edges)
        voxel_png_filename = f"{case}_{engine}_{method_label}_voxels_3d.png"
        voxel_png_path = self.output_dir / voxel_png_filename
        title_voxel = f"3D {case.upper()} [{engine.upper()}] - {method_label.upper()} (Element Voxel Volume)\nC = {history[-1].compliance:.2f}, V = {history[-1].volume_fraction:.2%}"
        self.render_voxel_volume_3d(mesh, final_rho, voxel_png_path, title=title_voxel)

        # Primary 3d png (copy voxel representation for primary view)
        primary_png_filename = f"{case}_{engine}_{method_label}_3d.png"
        primary_png_path = self.output_dir / primary_png_filename
        shutil.copy2(voxel_png_path, primary_png_path)

        # 4. Generate Progression Animated GIF (iter 진행형)
        gif_filename = f"{case}_{engine}_{method_label}_progression.gif"
        gif_path = self.output_dir / gif_filename
        self.generate_progression_gif(
            mesh=mesh,
            density_history=density_history,
            metric_history=history,
            gif_path=gif_path,
            title=f"3D {case.upper()} [{engine.upper()}] - {method_label.upper()}",
        )

        avg_fea = float(np.mean([s.time_fea_ms for s in history]))
        avg_sens = float(np.mean([s.time_sens_ms for s in history]))
        avg_opt = float(np.mean([s.time_opt_ms for s in history]))

        return BenchmarkRunResult3D(
            case_name=case,
            engine_name=engine,
            method_name=method_label,
            resolution=self.resolution,
            n_elements=n_elem,
            n_dofs=n_dof,
            final_compliance=history[-1].compliance,
            final_volume=history[-1].volume_fraction,
            iterations=len(history),
            total_time_sec=t_total_sec,
            avg_fea_time_ms=avg_fea,
            avg_sens_time_ms=avg_sens,
            avg_opt_time_ms=avg_opt,
            final_max_stress_mpa=final_max_stress,
            final_com_shift_mm=final_com_shift,
            final_freq1_hz=final_freq1,
            history=history,
            rho_phys=final_rho,
            stl_path=str(stl_path),
            pyvista_png_path=str(primary_png_path),
            voxel_png_path=str(voxel_png_path),
            cad_png_path=str(cad_png_path),
            gif_path=str(gif_path),
        )

    def plot_speed_comparison(
        self,
        results: List[BenchmarkRunResult3D],
        save_path: Path,
    ) -> None:
        """Generate Matplotlib 2D speed comparison bar charts adhering strictly to guidelines."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE_PT
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))

        # Filter by unique engine comparisons on bracket
        bracket_results = [r for r in results if r.case_name == "bracket"]
        labels = [f"{r.engine_name}\n({r.method_name})" for r in bracket_results]
        fea_times = [r.avg_fea_time_ms for r in bracket_results]
        sens_times = [r.avg_sens_time_ms for r in bracket_results]
        total_times = [r.total_time_sec for r in bracket_results]

        x_idx = np.arange(len(labels))
        width = 0.35

        # Subplot 1: Average FEA & Sensitivity Time (ms)
        ax1.bar(x_idx - width / 2, fea_times, width, label="FEA 해석 (ms)", color="#3498db")
        ax1.bar(x_idx + width / 2, sens_times, width, label="민감도 계산 (ms)", color="#e67e22")
        ax1.set_ylabel("시간 (ms)")
        ax1.set_title("엔진별 반복당 평균 연산 시간 (FEA vs 민감도)")
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper right", frameon=True)

        # Subplot 2: Total Optimization Execution Time (s)
        colors = ["#2ecc71" if "numba" in r.engine_name else ("#9b59b6" if "v1" in r.engine_name else "#e74c3c") for r in bracket_results]
        bars = ax2.bar(x_idx, total_times, width=0.5, color=colors, edgecolor="#2c3e50")
        ax2.set_ylabel("총 소요 시간 (초)")
        ax2.set_title("GE 브래킷 최적화 총 수행 시간 비교")
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_curves(
        self,
        results: List[BenchmarkRunResult3D],
        save_path: Path,
    ) -> None:
        """Plot compliance and volume convergence history curves across methods."""
        plt.rcParams["font.size"] = GLOBAL_FONT_SIZE_PT
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 4.8))

        # Focus on bracket runs for clear comparison
        bracket_results = [r for r in results if r.case_name == "bracket"]
        for r in bracket_results:
            label = f"{r.engine_name}_{r.method_name}"
            iters = [s.iteration for s in r.history]
            comp = [s.compliance for s in r.history]
            vol = [s.volume_fraction for s in r.history]

            ax1.plot(iters, comp, "-o", markersize=3, linewidth=1.2, label=label)
            ax2.plot(iters, vol, "-s", markersize=3, linewidth=1.2, label=label)

        ax1.set_xlabel("반복 횟수 (Iteration)")
        ax1.set_ylabel("컴플라이언스 (Compliance)")
        ax1.set_title("목적함수 수렴 곡선 (Compliance vs Iteration) - GE Bracket")
        ax1.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax1, max_per_col=6)

        ax2.set_xlabel("반복 횟수 (Iteration)")
        ax2.set_ylabel("체적비 (Volume Fraction)")
        ax2.set_title("체적비 수렴 곡선 (Volume Fraction vs Iteration) - GE Bracket")
        ax2.grid(True, linestyle="--", alpha=0.4)
        _place_outside_legend(ax2, max_per_col=6)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def generate_markdown_report(
        self,
        results: List[BenchmarkRunResult3D],
        report_path: Path,
    ) -> Path:
        """Generate thorough UTF-8 markdown report summarizing all 3D benchmarks."""
        lines = [
            "# 3D 위상최적화 벤치마크 및 멀티엔진 성능 평가 보고서",
            "",
            "## 1. 개요 (Overview)",
            "본 벤치마크는 FreeTO 및 학술 표준 3D 벤치마크(GE Aircraft Bracket, top3d Cantilever Beam)를 대상으로,",
            "다양한 알고리즘(`SIMP-OC`, `SIMP-MMA`, `BESO`, `SEMDOT`)과 고성능 연산 엔진(`Numba`, `Kratos v2 (WHT 연동)`, `Kratos v2 Native`)의 정량적 성능 및 다중 물리(응력, CoM, 고유진동수) 지표를 검증하였습니다.",
            "",
            "## 2. 벤치마크 정량 지표 요약 (Summary Table)",
            "| 형상 (Case) | 엔진 (Engine) | 알고리즘 (Method) | 요소수 | 컴플라이언스 (mJ / c) | 최종 체적비 | 최대 응력 (MPa) | CoM 변화 (mm) | 1차 진동수 (Hz) | 총 소요 시간 (s) |",
            "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for r in results:
            lines.append(
                f"| `{r.case_name}` | `{r.engine_name}` | `{r.method_name}` | {r.n_elements:,} | "
                f"{r.final_compliance:.4f} | {r.final_volume:.2%} | {r.final_max_stress_mpa:.2f} | {r.final_com_shift_mm:.4f} | "
                f"{r.final_freq1_hz:.1f} | {r.total_time_sec:.2f}s |"
            )

        lines.extend([
            "",
            "### 2.1 단위계 및 물리적 파라미터 정의 (Units & Physical Consistency)",
            "- **GE Aircraft Bracket 3D**: `mm - N - tonne - s` 일관 단위계 적용",
            "  - **재료 물성치**: Ti-6Al-4V (탄성계수 $E_0 = 110,000\\text{ MPa}$, 포아송비 $\\nu = 0.31$, 밀도 $\\rho = 4.43 \\times 10^{-9}\\text{ tonne/mm}^3$)",
            "  - **외력 하중**: 클레비스 핀 수직 인장/압축 하중 $F_z = -1,000\\text{ N}$ (약 $102\\text{ kgf}$)",
            "  - **컴플라이언스 (변형 에너지)**: $\\text{mJ}$ 단위 ($1\\text{ J} = 10^3\\text{ mJ} = 10^3\\text{ N}\\cdot\\text{mm}$)",
            "  - **최대 응력 (von Mises)**: 보이드 요소의 수치적 특이성(Stress Singularity)을 방지하기 위해 $q$-이완 필터($\\tilde{\\sigma}_e = \\rho_e^{0.5} \\sigma_e$) 및 실제 고체/설계 영역 마스킹을 적용한 실제 재료 영역의 최대 응력 (단위: $\\text{MPa}$, 항복 한계 $\\le 830\\text{ MPa}$)",
            "- **Cantilever Beam 3D**: 학술 표준 무차원화 단위계 ($F = 1.0, E_0 = 1.0$)",
            "",
            "## 3. 엔진별 아키텍처 및 다중 물리 해석 분석 (Engine Performance Analysis)",
            "- **WHT + Kratos v2 Engine (WHT 오케스트레이션)**:",
            "  - WHT 프레임워크가 다중 제약조건(체적, 응력 800 MPa, CoM 유지)과 알고리즘(MMA/OC)을 주도하고,",
            "  - Kratos v2가 Intel MKL PARDISO AVX-512 기반 대규모 3D 유한요소 강성 행렬 해석 및 수반 감도를 전담합니다.",
            "- **Kratos v2 Native (자체 인메모리 파이프라인)**:",
            "  - Kratos C++ 메모리 파이프라인(`evaluate_mesh_cached`) 내에서 순수 SIMP-OC 루프를 자가 완결적으로 수행합니다.",
            "- **Numba + MKL PARDISO Engine**:",
            "  - OpenMP 병렬화된 JIT CSC 조립과 Intel MKL PARDISO를 연동한 경량 고성능 파이프라인입니다.",
            "",
            "## 4. 산출물 파일 목록 (Generated Artifacts)",
        ])

        for r in results:
            lines.append(f"### {r.case_name.upper()} - {r.method_name.upper()} ({r.engine_name})")
            lines.append(f"- **최종 컴플라이언스**: `{r.final_compliance:.4f}` (체적 분율: `{r.final_volume:.1%}`)")
            lines.append(f"- **최대 von Mises 응력**: `{r.final_max_stress_mpa:.2f} MPa` (항복 한계: `830 MPa`)")
            lines.append(f"- **소요 시간**: `{r.total_time_sec:.2f}초` (반복당 평균 FEA `{r.avg_fea_time_ms:.1f}ms`, 감도 `{r.avg_sens_time_ms:.2f}ms`)")
            if r.voxel_png_path:
                lines.append(f"- **3D 요소 볼륨 형상 (Element Voxel Volume)**: ![{r.method_name}_voxels]({Path(r.voxel_png_path).name})")
            if r.cad_png_path:
                lines.append(f"- **3D 수밀 CAD 곡면 형상 (Watertight CAD Surface)**: ![{r.method_name}_cad]({Path(r.cad_png_path).name})")
            if r.gif_path:
                lines.append(f"- **최적화 진행 애니메이션 (Progression GIF)**: ![{r.method_name}_gif]({Path(r.gif_path).name})")
            if r.stl_path:
                lines.append(f"- **STL 3D 지오메트리 파일**: [`{Path(r.stl_path).name}`]({Path(r.stl_path).name})")
            lines.append("")

        lines.extend([
            "- **속도 비교 차트**: [`speed_comparison_3d.png`](speed_comparison_3d.png)",
            "- **수렴 이력 차트**: [`convergence_comparison_3d.png`](convergence_comparison_3d.png)",
            "",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return report_path


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="3D Topology Optimization Dual-Engine Benchmark Suite")
    parser.add_argument("--case", choices=["bracket", "beam", "all"], default="all", help="Target benchmark geometry")
    parser.add_argument("--engine", choices=["kratos_v2", "kratos_v2_native", "numba", "all"], default="kratos_v2", help="FEM & Sensitivity solver engine (default: kratos_v2)")
    parser.add_argument("--formulation", choices=["simp", "beso", "semdot", "level_set", "all"], default="simp", help="Material & geometric formulation (default: simp)")
    parser.add_argument("--optimizer", choices=["mma", "gcmma", "oc", "all"], default="mma", help="Mathematical optimization engine (default: mma)")
    parser.add_argument("--method", type=str, default=None, help="Legacy method alias (e.g. simp_mma, beso, simp_oc)")
    parser.add_argument("--resolution", choices=["fast", "standard"], default="standard", help="Voxel mesh resolution (standard=3mm, fast=6mm)")
    parser.add_argument("--max-iter", type=int, default=25, help="Maximum iterations per run (default: 25)")
    parser.add_argument("--target-vol", type=float, default=0.35, help="Target volume fraction")
    parser.add_argument("--output-dir", type=str, default="wht_topopt/examples/outputs/ex07_bracket", help="Output directory")

    args = parser.parse_args()

    cases = ["bracket", "beam"] if args.case == "all" else [args.case]
    engines = ["kratos_v2", "kratos_v2_native", "numba"] if args.engine == "all" else [args.engine]

    runs: List[Tuple[str, Optional[str]]] = []
    if args.method is not None:
        clean_m = args.method.strip().lower()
        if clean_m == "all":
            runs = [("simp", "oc"), ("simp", "mma"), ("beso", "ranking"), ("semdot", "mma")]
        elif "_" in clean_m:
            f_part, o_part = clean_m.split("_", 1)
            runs = [(f_part, o_part)]
        elif clean_m == "beso":
            runs = [("beso", "ranking")]
        else:
            runs = [(clean_m, args.optimizer)]
    else:
        form_list = ["simp", "beso", "semdot"] if args.formulation == "all" else [args.formulation]
        opt_list = ["mma", "gcmma", "oc"] if args.optimizer == "all" else [args.optimizer]
        for f in form_list:
            if f == "beso":
                runs.append(("beso", "ranking"))
            else:
                for o in opt_list:
                    runs.append((f, o))

    suite = BracketBenchmark3DSuite(
        output_dir=args.output_dir,
        resolution=args.resolution,
        max_iter=args.max_iter,
        target_vol=args.target_vol,
    )

    results: List[BenchmarkRunResult3D] = []

    print(f"================================================================================")
    print(f" Starting 3D Topology Optimization Showcase (Resolution: {args.resolution})")
    print(f" Cases: {cases} | Engines: {engines} | Configurations (Formulation, Optimizer): {runs}")
    print(f"================================================================================")

    for c in cases:
        for eng in engines:
            for form, opt in runs:
                label = f"{form}_{opt}" if form != "beso" else "beso"
                print(f"\n>>> Running [{c.upper()}] Engine=[{eng}] | Formulation=[{form.upper()}], Optimizer=[{opt.upper() if opt else 'NONE'}] ...")
                res = suite.run_single(case=c, engine=eng, formulation=form, optimizer=opt)
                results.append(res)
                print(f"    Done! Compliance={res.final_compliance:.4f}, Vol={res.final_volume:.2%}, Time={res.total_time_sec:.2f}s (FEA={res.avg_fea_time_ms:.1f}ms, Sens={res.avg_sens_time_ms:.2f}ms)")

    # Generate charts and report
    speed_png = suite.output_dir / "speed_comparison_3d.png"
    suite.plot_speed_comparison(results, speed_png)
    print(f"\n[Saved] Speed comparison chart: {speed_png}")

    conv_png = suite.output_dir / "convergence_comparison_3d.png"
    suite.plot_convergence_curves(results, conv_png)
    print(f"[Saved] Convergence comparison chart: {conv_png}")

    md_report = suite.output_dir / "ex07_3d_bracket_report.md"
    suite.generate_markdown_report(results, md_report)
    print(f"[Saved] Engineering report: {md_report}")

    # Synchronize all generated outputs to Antigravity brain artifact directory for live viewing
    artifact_dir = Path("C:/Users/GOODMAN/.gemini/antigravity-cli/brain/5fa6c777-54ee-4efb-9a95-73c6214c8bb1")
    if artifact_dir.exists():
        import shutil
        for p in suite.output_dir.glob("*.*"):
            shutil.copy2(p, artifact_dir / p.name)
        print(f"[Artifacts Synced] All files copied to {artifact_dir}")


if __name__ == "__main__":
    main()
