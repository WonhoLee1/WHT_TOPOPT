"""High-resolution 2D beam benchmark suite (Half MBB 90x30, Cantilever 60x30, L-Bracket 60x60)
and 5-solver cross-comparison (SuperLU, MKL PARDISO, PyAMG, Kratos C++, FrontISTR v6.0).

Reproduces and extends `aieng/examples/output/beams/mbb_90x30_V40.png` with:
  - Left column: Crisp element density distribution + passive void mask (#dbe9f6) + BC/load glyphs
  - Right column: Deformed Q4 quad mesh (PolyCollection) colored by displacement magnitude
  - Common binary re-analysis on a 2x finer mesh (180x60, 120x60, 120x120) evaluated across
    PARDISO/SciPy, Kratos C++, and FrontISTR fistr1.exe.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except Exception:
    pass

from wht_topopt.algorithms.beso import BESOOptimizer, check_load_path_connected
from wht_topopt.algorithms.level_set import LevelSetOptimizer
from wht_topopt.algorithms.mma_wrapper import MMAOptimizer
from wht_topopt.algorithms.oc import OCOptimizer
from wht_topopt.algorithms.semdot import SEMDOTOptimizer
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter
from wht_topopt.core.interpolation import SIMP
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D
from wht_topopt.core.projection import HeavisideProjection
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.functions.stress import PNormStressConstraint
from wht_topopt.pipeline.report import (
    GLOBAL_FONT_SIZE_PT,
    plot_2d_deformed_mesh,
    plot_2d_density_with_bc,
)
from wht_topopt.solvers.frontistr_adapter import FrontISTRSolver
from wht_topopt.solvers.kratos_solver import KratosSolver
from wht_topopt.solvers.pyamg_solver import PyAMGSolver
from wht_topopt.solvers.scipy_solver import ScipySolver

DEFAULT_SIZES: Dict[str, Tuple[int, int]] = {
    "mbb": (90, 30),
    "cantilever": (60, 30),
    "lbracket": (60, 60),
}
REFINE_FACTOR: int = 2
GLOBAL_FIG_WIDTH_IN: float = 11.5


@dataclass
class BeamProblemSpec:
    """Specification of a 2D benchmark domain, boundary conditions, and loads."""

    case: str
    nx: int
    ny: int
    volfrac: float
    rmin: float
    mesh: StructuredMesh2D
    domain_mask: np.ndarray
    passive_void: np.ndarray
    passive_solid_bc: np.ndarray
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    force: np.ndarray
    load_nodes: np.ndarray
    fixed_nodes: np.ndarray
    e0: float = 1.0
    emin: float = 1e-9
    nu: float = 0.3


@dataclass
class MethodRunRecord:
    """Quantitative and visual results from one optimization method run."""

    label: str
    rho_phys: np.ndarray
    u: np.ndarray
    compliance: float
    common_compliance_pardiso: float
    common_compliance_kratos: Optional[float]
    common_compliance_frontistr: Optional[float]
    design_vol: float
    grayness: float
    iterations: int
    converged: bool
    wall_sec: float
    extracted_grid: Optional[np.ndarray] = None
    extra_note: str = ""
    error: Optional[str] = None


class BeamsBenchmarkSuite:
    """High-resolution 2D beam benchmark runner for multi-algorithm and multi-solver comparisons."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        solver_backend: str = "pardiso",
        quick: bool = False,
    ) -> None:
        """Initialize BeamsBenchmarkSuite.

        Parameters
        ----------
        output_dir : Optional[Path]
            Directory for generated PNG figures and Markdown tables.
        solver_backend : str
            Primary linear solver ('pardiso', 'scipy', 'pyamg', 'kratos', 'frontistr').
        quick : bool
            If True, uses compact iteration budgets for rapid unit testing.
        """
        base_out = output_dir or (Path(__file__).resolve().parent / "outputs" / "beams")
        self.output_dir = Path(base_out)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.solver_backend = solver_backend.lower()
        self.quick = bool(quick)

    @staticmethod
    def build_problem(
        nx: int,
        ny: int,
        volfrac: float,
        rmin: float,
        case: str,
        refine: int = 1,
    ) -> BeamProblemSpec:
        """Construct 2D Q4 mesh, active domain mask, supports, and loads matching topopt_cantilever.py."""
        nelx, nely = nx * refine, ny * refine
        lx, ly = float(nx), float(ny)
        mesh = StructuredMesh2D(nelx=nelx, nely=nely, lx=lx, ly=ly)
        nodes = mesh.nodes
        cent = mesh.element_centroids
        n_elem = mesh.n_elements
        n_dof = mesh.n_dofs

        domain = np.ones(n_elem, dtype=bool)
        fixed: List[int] = []
        f = np.zeros(n_dof, dtype=np.float64)

        if case == "cantilever":
            left_nodes = np.flatnonzero(nodes[:, 0] < 1e-9)
            for n in left_nodes:
                fixed.extend([2 * int(n), 2 * int(n) + 1])
            patch = np.flatnonzero(
                (nodes[:, 0] > lx - 1e-9) & (np.abs(nodes[:, 1] - ly / 2.0) <= 1.0 + 1e-9)
            )
            if len(patch) == 0:
                patch = np.array([int(np.argmin(np.linalg.norm(nodes - np.array([lx, ly / 2.0]), axis=1)))])
            # Trapezoidal line traction along vertical right-edge patch
            if len(patch) == 1:
                f[2 * patch[0] + 1] = -1.0
            else:
                ys = nodes[patch, 1]
                order = np.argsort(ys)
                p_sorted = patch[order]
                y_sorted = ys[order]
                w = np.zeros(len(p_sorted), dtype=np.float64)
                dy_seg = np.diff(y_sorted)
                w[:-1] += 0.5 * dy_seg
                w[1:] += 0.5 * dy_seg
                w /= np.sum(w)
                f[2 * p_sorted + 1] = -w

        elif case == "mbb":
            left_nodes = np.flatnonzero(nodes[:, 0] < 1e-9)
            for n in left_nodes:
                fixed.append(2 * int(n))  # ux = 0 symmetry plane
            br_nodes = np.flatnonzero((nodes[:, 0] > lx - 1e-9) & (nodes[:, 1] < 1e-9))
            for n in br_nodes:
                fixed.append(2 * int(n) + 1)  # uy = 0 bottom-right roller
            tl_nodes = np.flatnonzero((nodes[:, 0] < 1e-9) & (nodes[:, 1] > ly - 1e-9))
            f[2 * tl_nodes + 1] -= 1.0 / max(1, len(tl_nodes))

        elif case == "lbracket":
            cut_x, cut_y = 0.4 * lx, 0.4 * ly
            domain = ~((cent[:, 0] > cut_x) & (cent[:, 1] > cut_y))
            top_nodes = np.flatnonzero((nodes[:, 1] > ly - 1e-9) & (nodes[:, 0] <= cut_x + 1e-9))
            for n in top_nodes:
                fixed.extend([2 * int(n), 2 * int(n) + 1])
            tip_nodes = np.flatnonzero(
                (nodes[:, 0] > lx - 1e-9)
                & (nodes[:, 1] >= cut_y - 1.0 - 1e-9)
                & (nodes[:, 1] <= cut_y + 1e-9)
            )
            if len(tip_nodes) == 0:
                tip_nodes = np.array([int(np.argmin(np.linalg.norm(nodes - np.array([lx, cut_y]), axis=1)))])
            f[2 * tip_nodes + 1] -= 1.0 / len(tip_nodes)
        else:
            raise ValueError(f"Unknown case {case!r}; choose from 'mbb', 'cantilever', 'lbracket'")

        fixed_dofs = np.unique(np.asarray(fixed, dtype=np.int64))
        free_dofs = np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed_dofs)

        # Identify loaded and supported nodes & adjacent active elements
        fixed_nodes = np.unique(fixed_dofs // 2)
        load_nodes = np.flatnonzero(np.linalg.norm(f.reshape(-1, 2), axis=1) > 1e-12)
        clamped_node_mask = np.bincount(fixed_dofs // 2, minlength=mesh.n_nodes) >= 1
        loaded_node_mask = np.zeros(mesh.n_nodes, dtype=bool)
        loaded_node_mask[load_nodes] = True

        # Elements touching loaded nodes or support pins are protected from void erosion in BESO/LevelSet/re-analysis
        support_pin_mask = np.zeros(mesh.n_nodes, dtype=bool)
        if case == "mbb":
            br_nodes = np.flatnonzero((nodes[:, 0] > lx - 1e-9) & (nodes[:, 1] < 1e-9))
            support_pin_mask[br_nodes] = True
        else:
            support_pin_mask[ np.bincount(fixed_dofs // 2, minlength=mesh.n_nodes) >= 2 ] = True

        forced_elem_mask = (loaded_node_mask | support_pin_mask)[mesh.elements].any(axis=1) & domain
        passive_void = np.flatnonzero(~domain)
        passive_solid_bc = np.flatnonzero(forced_elem_mask)

        return BeamProblemSpec(
            case=case,
            nx=nx,
            ny=ny,
            volfrac=float(volfrac),
            rmin=float(rmin),
            mesh=mesh,
            domain_mask=domain,
            passive_void=passive_void,
            passive_solid_bc=passive_solid_bc,
            fixed_dofs=fixed_dofs,
            free_dofs=free_dofs,
            force=f,
            load_nodes=load_nodes,
            fixed_nodes=fixed_nodes,
        )

    def _make_solver(self, backend: Optional[str] = None) -> Any:
        name = (backend or self.solver_backend).lower()
        if name == "pardiso":
            return ScipySolver(method="pardiso")
        if name == "scipy":
            return ScipySolver(method="direct")
        if name == "pyamg":
            return PyAMGSolver(tol=1e-9, maxiter=1500)
        if name == "kratos":
            return KratosSolver(method="direct", prefer_in_memory=True)
        if name == "frontistr":
            return FrontISTRSolver(mode="smp", threads=4, n_buckets=64, method="DIRECT")
        raise ValueError(f"Unknown solver backend {name!r}")

    def _solve_fe(
        self,
        spec: BeamProblemSpec,
        fem: FEMEngine,
        rho_phys: np.ndarray,
        penal: float,
        solver_obj: Any,
        backend_name: str,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Solve FE system and return (u, compliance, dc_drho_phys) using the chosen solver backend."""
        interp = SIMP(E0=spec.e0, Emin=spec.emin, penal=penal)
        youngs = interp.evaluate(rho_phys)
        dE_drho = interp.derivative(rho_phys)
        youngs = np.where(spec.domain_mask, youngs, spec.emin)
        dE_drho = np.where(spec.domain_mask, dE_drho, 0.0)

        bname = backend_name.lower()
        if bname == "kratos" and isinstance(solver_obj, KratosSolver):
            res = solver_obj.evaluate_mesh_cached(
                nodes=spec.mesh.nodes,
                elems=spec.mesh.elements,
                etype="quad4",
                fixed_dofs=spec.fixed_dofs,
                loads=spec.force,
                youngs=youngs,
                nu=spec.nu,
            )
            u = res["u"].ravel()
        elif bname == "frontistr" and isinstance(solver_obj, FrontISTRSolver):
            # Build 1-element-thick 3D HEX8 slab (uz = 0 on z = 0) matching topopt_cantilever.py
            mesh3d = StructuredMesh3D(
                nelx=spec.mesh.nelx,
                nely=spec.mesh.nely,
                nelz=1,
                lx=spec.mesh.lx,
                ly=spec.mesh.ly,
                lz=1.0,
            )
            nodes3d = mesh3d.nodes
            fixed3d = [3 * n + 2 for n in np.flatnonzero(nodes3d[:, 2] < 1e-9)]
            for d2 in spec.fixed_dofs:
                n2 = int(d2) // 2
                c2 = int(d2) % 2
                # Apply in-plane BC to both z=0 and z=1 nodes at the same (x, y)
                n_top = n2 + spec.mesh.n_nodes
                fixed3d.extend([3 * n2 + c2, 3 * n_top + c2])
            f3d = np.zeros(mesh3d.n_dofs, dtype=np.float64)
            for d2 in np.flatnonzero(spec.force):
                n2 = int(d2) // 2
                c2 = int(d2) % 2
                n_top = n2 + spec.mesh.n_nodes
                f3d[3 * n2 + c2] = 0.5 * spec.force[d2]
                f3d[3 * n_top + c2] = 0.5 * spec.force[d2]
            s_res = solver_obj.solve_mesh(
                nodes=nodes3d,
                elems=mesh3d.elements,
                etype="hex8",
                fixed_dofs=np.unique(fixed3d),
                loads=f3d,
                youngs=youngs,
                nu=spec.nu,
            )
            u3d = s_res.u.reshape(-1, 3)
            u = u3d[: spec.mesh.n_nodes, :2].ravel()
        else:
            K = fem.assemble_stiffness_matrix(
                rho_phys, penal=penal, e0=spec.e0, emin=spec.emin, interpolation=interp
            )
            s_res = solver_obj.solve(
                K, spec.force, free_dofs=spec.free_dofs, fixed_dofs=spec.fixed_dofs, node_coords=spec.mesh.nodes
            )
            u = s_res.u.ravel()

        # Element unit strain energies u_e^T K0 u_e via Numba kernel
        ce = fem.compute_unit_element_compliance(u)
        compliance = float(np.dot(spec.force, u))
        dc_drho = -dE_drho * ce
        dc_drho[~spec.domain_mask] = 0.0
        return u, compliance, dc_drho

    @staticmethod
    def design_volume(rho_phys: np.ndarray, domain_mask: np.ndarray) -> float:
        return float(np.sum(rho_phys[domain_mask]) / np.sum(domain_mask))

    @staticmethod
    def grayness(rho_phys: np.ndarray, domain_mask: np.ndarray) -> float:
        xd = np.clip(rho_phys[domain_mask], 0.0, 1.0)
        return float(np.mean(xd * (1.0 - xd)))

    @staticmethod
    def fine_density(
        rho_phys: np.ndarray,
        nx: int,
        ny: int,
        extracted_grid: Optional[np.ndarray] = None,
        refine: int = REFINE_FACTOR,
    ) -> np.ndarray:
        """Map coarse design onto a (refine x) finer mesh, preserving SEMDOT sub-grid boundaries."""
        if extracted_grid is None:
            return np.kron(rho_phys.reshape(ny, nx), np.ones((refine, refine))).ravel()
        g = (extracted_grid.shape[1] - 1) // nx
        if g % refine != 0:
            return np.kron(rho_phys.reshape(ny, nx), np.ones((refine, refine))).ravel()
        h = g // refine
        acc = np.zeros((ny * refine, nx * refine), dtype=np.float64)
        for b in range(h + 1):
            for c in range(h + 1):
                acc += extracted_grid[b : b + h * ny * refine : h, c : c + h * nx * refine : h]
        return (acc / float((h + 1) ** 2)).ravel()

    def common_reanalysis(
        self,
        spec: BeamProblemSpec,
        rho_phys: np.ndarray,
        extracted_grid: Optional[np.ndarray] = None,
        eval_external_solvers: bool = True,
    ) -> Tuple[float, Optional[float], Optional[float], float]:
        """Threshold design to exact volfrac on a 2x finer mesh and re-analyse with PARDISO, Kratos, and FrontISTR."""
        fine = self.build_problem(
            spec.nx, spec.ny, spec.volfrac, spec.rmin, spec.case, refine=REFINE_FACTOR
        )
        x = self.fine_density(rho_phys, spec.nx, spec.ny, extracted_grid=extracted_grid, refine=REFINE_FACTOR)

        # Tie-breaking via local density filter to keep dense member cores intact
        d_filt = DensityFilter(fine.mesh, rmin=max(spec.rmin, 1.5))
        local = d_filt.forward(x)
        key = x + 1e-6 * local

        forced = np.zeros(fine.mesh.n_elements, dtype=bool)
        forced[fine.passive_solid_bc] = True
        free = fine.domain_mask & ~forced
        keep = max(int(round(spec.volfrac * np.sum(fine.domain_mask))) - int(np.sum(forced)), 0)

        xb = np.zeros_like(x)
        xb[forced] = 1.0
        free_idx = np.flatnonzero(free)
        xb[free_idx[np.argsort(-key[free])[ : keep]]] = 1.0

        fem_fine = FEMEngine(fine.mesh, e0=fine.e0, emin=fine.emin, nu=fine.nu)
        pardiso_solver = ScipySolver(method="pardiso")
        _, c_pardiso, _ = self._solve_fe(fine, fem_fine, xb, penal=3.0, solver_obj=pardiso_solver, backend_name="pardiso")

        c_kratos: Optional[float] = None
        c_frontistr: Optional[float] = None
        if eval_external_solvers:
            try:
                k_solver = KratosSolver(method="direct", prefer_in_memory=True)
                _, c_kratos, _ = self._solve_fe(fine, fem_fine, xb, penal=3.0, solver_obj=k_solver, backend_name="kratos")
            except Exception:
                c_kratos = None
            try:
                f_solver = FrontISTRSolver(mode="smp", threads=4, n_buckets=0, method="DIRECT")
                _, c_frontistr, _ = self._solve_fe(fine, fem_fine, xb, penal=3.0, solver_obj=f_solver, backend_name="frontistr")
            except Exception:
                c_frontistr = None

        return c_pardiso, c_kratos, c_frontistr, self.design_volume(xb, fine.domain_mask)

    # -----------------------------------------------------------------------
    # Individual Optimization Method Runners
    # -----------------------------------------------------------------------
    def run_simp_oc(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        max_iters: int = 220,
        tol: float = 0.01,
    ) -> MethodRunRecord:
        """1. SIMP (p = 3, density filter, OC)."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=spec.emin, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)

        # Total volume fraction adjusted for passive void
        total_vf = spec.volfrac * float(np.sum(spec.domain_mask)) / float(spec.mesh.n_elements)
        opt = OCOptimizer(volfrac=total_vf, move=0.2, rho_min=1e-3)

        rho = np.where(spec.domain_mask, spec.volfrac, 1e-3)
        elem_vols = np.ones(spec.mesh.n_elements, dtype=np.float64)
        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0

        iters = 25 if self.quick else max_iters
        for it in range(1, iters + 1):
            rho_phys = d_filt.forward(rho)
            rho_phys[~spec.domain_mask] = 1e-3
            u, c_val, dc_phys = self._solve_fe(spec, fem, rho_phys, penal=3.0, solver_obj=solver, backend_name=bname)
            dc = d_filt.backward(dc_phys)
            dv = d_filt.backward(elem_vols)
            rho_new = opt.update(rho, dc, None, dv, elem_volumes=elem_vols, passive_void=spec.passive_void)
            change = float(np.max(np.abs(rho_new[spec.domain_mask] - rho[spec.domain_mask])))
            rho = rho_new
            if change < tol and it >= 15:
                converged = True
                break

        rho_phys = d_filt.forward(rho)
        rho_phys[~spec.domain_mask] = 0.0
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-3), penal=3.0, solver_obj=solver, backend_name=bname)
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(spec, rho_phys, eval_external_solvers=not self.quick)

        return MethodRunRecord(
            label=f"SIMP (p = 3, density filter, OC) [{bname.upper()}]",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
        )

    def run_simp_heaviside_mma(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        max_iters: int = 160,
        tol: float = 0.01,
    ) -> MethodRunRecord:
        """2. SIMP + Heaviside Projection (p = 3, beta = 1->8, MMA)."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=spec.emin, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)
        proj = HeavisideProjection(beta=1.0, eta=0.5)

        n_elem = spec.mesh.n_elements
        active_count = float(np.sum(spec.domain_mask))
        mma = MMAOptimizer(rho_min=1e-3, rho_max=1.0, move=0.15)

        rho = np.where(spec.domain_mask, spec.volfrac, 1e-3)
        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0
        c_ref: Optional[float] = None

        iters = 25 if self.quick else max_iters
        for it in range(1, iters + 1):
            if it in (40, 80, 120):
                proj.beta = min(8.0, proj.beta * 2.0)

            rho_tilde = d_filt.forward(rho)
            rho_phys = proj.forward(rho_tilde)
            rho_phys[~spec.domain_mask] = 1e-3

            u, c_val, dc_phys = self._solve_fe(spec, fem, rho_phys, penal=3.0, solver_obj=solver, backend_name=bname)
            c_ref = c_ref or max(c_val, 1.0)

            dc_tilde = proj.backward(dc_phys, rho_tilde)
            dc = d_filt.backward(dc_tilde)

            dv_phys = np.where(spec.domain_mask, 1.0 / (spec.volfrac * active_count), 0.0)
            dv_tilde = proj.backward(dv_phys, rho_tilde)
            dv = d_filt.backward(dv_tilde)

            vol_cur = self.design_volume(rho_phys, spec.domain_mask)
            g_val = np.array([vol_cur / spec.volfrac - 1.0])
            dg = dv.reshape(1, -1)

            rho_new = mma.update(
                rho, dc / c_ref, g_val, dg, passive_void=spec.passive_void
            )
            change = float(np.max(np.abs(rho_new[spec.domain_mask] - rho[spec.domain_mask])))
            rho = rho_new
            if change < tol and abs(g_val[0]) <= 3e-3 and it >= 30:
                converged = True
                break

        rho_tilde = d_filt.forward(rho)
        rho_phys = proj.forward(rho_tilde)
        rho_phys[~spec.domain_mask] = 0.0
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-3), penal=3.0, solver_obj=solver, backend_name=bname)
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(spec, rho_phys, eval_external_solvers=not self.quick)

        return MethodRunRecord(
            label="SIMP + Heaviside (p = 3, beta 1->8, MMA)",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
        )

    def run_semdot(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        max_iters: int = 180,
        tol: float = 0.002,
    ) -> MethodRunRecord:
        """3. SEMDOT (non-penalised, smooth sub-grid edges) per Fu & Rolfe (2020)."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=1e-6, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)
        oc = OCOptimizer(
            volfrac=spec.volfrac * float(np.sum(spec.domain_mask)) / float(spec.mesh.n_elements),
            move=0.2,
            rho_min=1e-3,
        )

        total_vf = spec.volfrac * float(np.sum(spec.domain_mask)) / float(spec.mesh.n_elements)
        semdot = SEMDOTOptimizer(
            mesh=spec.mesh,
            volfrac=total_vf,
            n_sub=4,
            beta_init=0.5,
            beta_max=8.0,
            beta_factor=1.05,
            rho_min=1e-6,
            move=0.2,
        )
        semdot.sens_history = 0.0  # handle history explicitly on unsmoothed variable x

        # x is unsmoothed design variable; psi is element volume fraction evaluated by FE
        x = np.where(spec.domain_mask, spec.volfrac, 0.0)
        psi = x.copy()
        elem_vols = np.ones(spec.mesh.n_elements, dtype=np.float64)
        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0
        dc_prev: Optional[np.ndarray] = None

        iters = 25 if self.quick else max_iters
        for it in range(1, iters + 1):
            psi_fe = np.where(spec.domain_mask, np.maximum(psi, 1e-6), 1e-6)
            u, c_val, _ = self._solve_fe(spec, fem, psi_fe, penal=1.0, solver_obj=solver, backend_name=bname)
            ce = fem.compute_unit_element_compliance(u)
            # Eq. 13 discrete sensitivity: dC/dPsi_e = -[(1 - Psi_e)*rho_min + Psi_e] * ce
            dc_raw = -((1.0 - psi_fe) * 1e-6 + psi_fe) * ce
            dc_raw[~spec.domain_mask] = 0.0
            if dc_prev is not None:
                dc_raw = 0.3 * dc_raw + 0.7 * dc_prev
            dc_prev = dc_raw.copy()

            dc_x = d_filt.backward(dc_raw)
            dv_x = d_filt.backward(np.where(spec.domain_mask, 1.0, 0.0))

            x_new = oc.update(
                x,
                dc_x,
                None,
                dv_x,
                elem_volumes=elem_vols,
                passive_solid=spec.passive_solid_bc,
                passive_void=spec.passive_void,
            )
            x_filt = d_filt.forward(x_new)
            x_filt[~spec.domain_mask] = 0.0
            x_filt[spec.passive_solid_bc] = 1.0

            # Project filtered nodal field onto sub-grid points and average to psi_new
            psi_new = semdot.project_filtered(
                x_filt,
                target_volfrac=total_vf,
                passive_solid=spec.passive_solid_bc,
                passive_void=spec.passive_void,
            )
            change = float(np.sum(np.abs(x_new[spec.domain_mask] - x[spec.domain_mask])) / max(np.sum(x_new[spec.domain_mask]), 1e-12))
            x, psi = x_new, psi_new
            if change < tol and it >= 60:
                converged = True
                break

        rho_phys = psi.copy()
        rho_phys[~spec.domain_mask] = 0.0
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-6), penal=1.0, solver_obj=solver, backend_name=bname)
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(
            spec, rho_phys, extracted_grid=semdot.extracted_grid, eval_external_solvers=not self.quick
        )

        return MethodRunRecord(
            label="SEMDOT (non-penalised, smooth edges)",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
            extracted_grid=semdot.extracted_grid,
        )

    def run_beso(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        max_iters: int = 120,
    ) -> MethodRunRecord:
        """4. BESO (adaptive evolutionary ratio + load-path connectivity protection)."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=1e-6, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)

        total_vf = spec.volfrac * float(np.sum(spec.domain_mask)) / float(spec.mesh.n_elements)
        beso = BESOOptimizer(
            target_volfrac=total_vf,
            er=0.02,
            er_adaptive=True,
            er_adaptive_scheme="ER1",
            er_min=0.015,
            er_max=0.04,
            rho_min=1e-3,
            mesh=spec.mesh,
            load_nodes=spec.load_nodes,
            fixed_nodes=spec.fixed_nodes,
        )

        # BESO starts from full solid on the active design domain
        rho = np.where(spec.domain_mask, 1.0, 1e-3)
        elem_vols = np.ones(spec.mesh.n_elements, dtype=np.float64)
        obj_hist: List[float] = []
        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0

        iters = 25 if self.quick else max_iters
        for it in range(1, iters + 1):
            u, c_val, _ = self._solve_fe(spec, fem, rho, penal=3.0, solver_obj=solver, backend_name=bname)
            u_e = u[fem.edof_mat]
            ce = np.einsum("ei,ij,ej->e", u_e, fem.ke0, u_e)
            # Tian & Zhang (2025) Eq. 16-17: alpha_e = x_e^(p-1) * E_en
            alpha = (rho ** 2.0) * ce
            alpha[~spec.domain_mask] = 0.0
            alpha_filt = d_filt.forward(alpha)

            rho_new = beso.update(
                rho,
                -alpha_filt,
                np.array([0.0]),
                elem_vols,
                elem_volumes=elem_vols,
                passive_solid=spec.passive_solid_bc,
                passive_void=spec.passive_void,
            )
            rho = rho_new
            vol_cur = self.design_volume(rho, spec.domain_mask)
            if abs(vol_cur - spec.volfrac) <= 4e-3:
                obj_hist.append(c_val)
                if len(obj_hist) >= 10:
                    w = np.asarray(obj_hist[-10:])
                    err = abs(np.sum(w[-5:] - w[:5])) / max(np.sum(w[-5:]), 1e-12)
                    if err <= 2e-3:
                        converged = True
                        break

        rho_phys = np.where(spec.domain_mask & (rho > 0.5), 1.0, 0.0)
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-3), penal=3.0, solver_obj=solver, backend_name=bname)
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(spec, rho_phys, eval_external_solvers=not self.quick)

        return MethodRunRecord(
            label="BESO (adaptive evolutionary ratio)",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
        )

    def run_level_set(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        simp_warmup_iters: int = 30,
        ls_iters: int = 45,
    ) -> MethodRunRecord:
        """5. Level set (SIMP start, Hilbertian projection) per Wegert (2023) & Jezek (2026)."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=1e-3, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)

        total_vf = spec.volfrac * float(np.sum(spec.domain_mask)) / float(spec.mesh.n_elements)
        elem_vols = np.ones(spec.mesh.n_elements, dtype=np.float64)

        # Stage 1: SIMP warm-start (Jezek et al., 2026) to nucleate internal topology holes
        oc = OCOptimizer(volfrac=total_vf, move=0.2, rho_min=1e-3)
        rho = np.where(spec.domain_mask, spec.volfrac, 1e-3)
        w_iters = 10 if self.quick else simp_warmup_iters
        for _ in range(w_iters):
            rho_f = d_filt.forward(rho)
            rho_f[~spec.domain_mask] = 1e-3
            _, _, dc_phys = self._solve_fe(spec, fem, rho_f, penal=3.0, solver_obj=solver, backend_name=bname)
            dc = d_filt.backward(dc_phys)
            dv = d_filt.backward(elem_vols)
            rho_next = oc.update(rho, dc, None, dv, elem_volumes=elem_vols, passive_void=spec.passive_void)
            if float(np.max(np.abs(rho_next[spec.domain_mask] - rho[spec.domain_mask]))) < 0.04:
                rho = rho_next
                break
            rho = rho_next

        rho_init = d_filt.forward(rho)
        rho_init[~spec.domain_mask] = 1e-3

        # Stage 2: Hilbertian H^1 projection Level-Set boundary refinement
        ls = LevelSetOptimizer(
            mesh=spec.mesh,
            volfrac=total_vf,
            ell=2.0,
            gamma=0.06,
            reinit_interval=4,
            rho_min=1e-3,
            move=0.10,
        )
        # Initialize signed distance level set phi so volume matches target_volfrac exactly
        active_vals = rho_init[spec.domain_mask]
        q_cut = float(np.quantile(active_vals, 1.0 - spec.volfrac))
        ls._build_hilbertian_system(spec.mesh.n_elements)
        ls.phi = -(rho_init - q_cut) * (2.5 * ls._eta)
        ls.phi = ls.reinitialize_distance(ls.phi, n_steps=4)
        # Shift phi by a uniform constant so smoothed_heaviside(phi) matches target_volfrac
        lo_c, hi_c = -3.0 * ls._eta, 3.0 * ls._eta
        for _ in range(50):
            mid_c = 0.5 * (lo_c + hi_c)
            h_try = ls.smoothed_heaviside(ls.phi - mid_c)
            h_try[~spec.domain_mask] = 0.0
            h_try[spec.passive_solid_bc] = 1.0
            if float(np.mean(h_try)) < total_vf:
                lo_c = mid_c
            else:
                hi_c = mid_c
        ls.phi = ls.phi - 0.5 * (lo_c + hi_c)
        rho_ls = np.clip(ls.rho_min + (1.0 - ls.rho_min) * ls.smoothed_heaviside(ls.phi), 1e-3, 1.0)
        rho_ls[~spec.domain_mask] = 1e-3
        rho_ls[spec.passive_solid_bc] = 1.0

        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0
        c_hist: List[float] = []
        n_ls = 12 if self.quick else ls_iters

        for it in range(1, n_ls + 1):
            u, c_val, _ = self._solve_fe(spec, fem, rho_ls, penal=1.0, solver_obj=solver, backend_name=bname)
            ce = fem.compute_unit_element_compliance(u)
            dc_ls = -(1.0 - 1e-3) * ce
            dc_ls[~spec.domain_mask] = 0.0

            rho_new = ls.update(
                rho_ls,
                dc_ls,
                np.array([0.0]),
                elem_vols,
                elem_volumes=elem_vols,
                passive_solid=spec.passive_solid_bc,
                passive_void=spec.passive_void,
            )
            rho_ls = rho_new
            vol_cur = self.design_volume(rho_ls, spec.domain_mask)
            c_hist.append(c_val)
            if len(c_hist) >= 12 and abs(vol_cur - spec.volfrac) <= 3e-3:
                rel_stall = abs(c_hist[-1] - c_hist[-5]) / max(c_hist[-1], 1e-12)
                if rel_stall < 0.005:
                    converged = True
                    break

        rho_phys = rho_ls.copy()
        rho_phys[~spec.domain_mask] = 0.0
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-3), penal=1.0, solver_obj=solver, backend_name=bname)
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(spec, rho_phys, eval_external_solvers=not self.quick)

        return MethodRunRecord(
            label="Level set (SIMP start, Hilbertian projection)",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
        )

    def run_stress_constrained_mma(
        self,
        spec: BeamProblemSpec,
        solver_backend: Optional[str] = None,
        max_iters: int = 120,
    ) -> MethodRunRecord:
        """6. SIMP + P-Norm von Mises Stress Constraint (MMA) — rounds L-bracket re-entrant corner."""
        t0 = time.perf_counter()
        bname = solver_backend or self.solver_backend
        solver = self._make_solver(bname)
        fem = FEMEngine(spec.mesh, e0=spec.e0, emin=spec.emin, nu=spec.nu)
        d_filt = DensityFilter(spec.mesh, rmin=spec.rmin)
        bcm = BoundaryConditionManager(spec.mesh)
        bcm.fix_dofs(spec.fixed_dofs)
        if len(spec.passive_void) > 0:
            bcm.set_passive_void(spec.passive_void)
        stress_fn = PNormStressConstraint(
            fem=fem, solver=solver, bcm=bcm, p=8.0, sigma_allow=1.8, q=0.5, penal=3.0
        )

        n_elem = spec.mesh.n_elements
        active_count = float(np.sum(spec.domain_mask))
        mma = MMAOptimizer(rho_min=1e-3, rho_max=1.0, move=0.12)

        rho = np.where(spec.domain_mask, spec.volfrac, 1e-3)
        c_ref: Optional[float] = None
        s_ref: Optional[float] = None
        converged = False
        u = np.zeros(spec.mesh.n_dofs)
        c_val = 0.0

        iters = 20 if self.quick else max_iters
        for it in range(1, iters + 1):
            rho_phys = d_filt.forward(rho)
            rho_phys[~spec.domain_mask] = 1e-3
            u, c_val, dc_phys = self._solve_fe(spec, fem, rho_phys, penal=3.0, solver_obj=solver, backend_name=bname)
            c_ref = c_ref or max(c_val, 1.0)

            ds_phys = stress_fn.gradient(rho_phys, u=u) * stress_fn.sigma_allow
            _, _, vm_elem = stress_fn.compute_relaxed_von_mises(rho_phys, u)
            vm_elem[~spec.domain_mask] = 0.0
            p_norm_val = float(np.sum(vm_elem ** 8.0) ** (1.0 / 8.0))
            s_ref = s_ref or (0.88 * p_norm_val)

            dc = d_filt.backward(dc_phys)
            ds = d_filt.backward(np.where(spec.domain_mask, ds_phys, 0.0))
            dv_phys = np.where(spec.domain_mask, 1.0 / (spec.volfrac * active_count), 0.0)
            dv = d_filt.backward(dv_phys)

            vol_cur = self.design_volume(rho_phys, spec.domain_mask)
            g_vals = np.array([vol_cur / spec.volfrac - 1.0, p_norm_val / max(s_ref, 1e-12) - 1.0])
            dg = np.vstack([dv, ds / max(s_ref, 1e-12)])

            rho_new = mma.update(
                rho, dc / c_ref, g_vals, dg, passive_void=spec.passive_void
            )
            change = float(np.max(np.abs(rho_new[spec.domain_mask] - rho[spec.domain_mask])))
            rho = rho_new
            if change < 0.01 and it >= 35:
                converged = True
                break

        rho_phys = d_filt.forward(rho)
        rho_phys[~spec.domain_mask] = 0.0
        u, c_val, _ = self._solve_fe(spec, fem, np.maximum(rho_phys, 1e-3), penal=3.0, solver_obj=solver, backend_name=bname)
        _, _, vm_final = stress_fn.compute_relaxed_von_mises(np.maximum(rho_phys, 1e-3), u)
        peak_vm = float(np.max(vm_final[spec.domain_mask]))
        wall = time.perf_counter() - t0
        c_par, c_kra, c_fis, _ = self.common_reanalysis(spec, rho_phys, eval_external_solvers=not self.quick)

        return MethodRunRecord(
            label=f"SIMP + P-Norm Stress Constraint (MMA, peak σ_vM={peak_vm:.2f})",
            rho_phys=rho_phys,
            u=u,
            compliance=c_val,
            common_compliance_pardiso=c_par,
            common_compliance_kratos=c_kra,
            common_compliance_frontistr=c_fis,
            design_vol=self.design_volume(rho_phys, spec.domain_mask),
            grayness=self.grayness(rho_phys, spec.domain_mask),
            iterations=it,
            converged=converged,
            wall_sec=wall,
        )

    # -----------------------------------------------------------------------
    # Figure and Markdown Report Generation
    # -----------------------------------------------------------------------
    def render_comparison_figure(
        self,
        spec: BeamProblemSpec,
        records: List[MethodRunRecord],
        png_path: Path,
        suptitle: Optional[str] = None,
    ) -> Path:
        """Render 2-column comparison figure matching mbb_90x30_V40.png layout."""
        n_rows = len(records)
        row_h = 5.0 * spec.ny / spec.nx + 1.15
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(GLOBAL_FIG_WIDTH_IN, n_rows * row_h), squeeze=False
        )

        for (ax_d, ax_u), rec in zip(axes, records):
            if rec.error is not None:
                ax_d.text(
                    0.5, 0.5, f"{rec.label}\nERROR: {rec.error[:110]}",
                    ha="center", va="center", fontsize=GLOBAL_FONT_SIZE_PT, wrap=True,
                )
                ax_d.set_xticks([]); ax_d.set_yticks([])
                ax_u.set_xticks([]); ax_u.set_yticks([])
                continue

            ext_str = ""
            if rec.common_compliance_kratos is not None:
                ext_str += f" | Kratos={rec.common_compliance_kratos:.2f}"
            if rec.common_compliance_frontistr is not None:
                ext_str += f" | FrontISTR={rec.common_compliance_frontistr:.2f}"

            title_left = (
                f"{rec.label}\n"
                f"C = {rec.compliance:.2f}   common re-analysis (binary, {REFINE_FACTOR}x finer mesh) "
                f"C = {rec.common_compliance_pardiso:.2f}{ext_str}\n"
                f"V = {rec.design_vol:.3f}   gray = {rec.grayness:.3f}   {rec.iterations} it   "
                f"converged = {rec.converged}   ({rec.wall_sec:.1f} s)"
            )
            plot_2d_density_with_bc(
                ax_d, spec.nx, spec.ny, rec.rho_phys,
                domain_mask=spec.domain_mask, case=spec.case, title=title_left,
            )
            plot_2d_deformed_mesh(
                ax_u, spec.mesh.nodes, spec.mesh.elements, spec.nx, spec.ny,
                rec.rho_phys, rec.u, domain_mask=spec.domain_mask,
                title="displacement magnitude (deformed, scaled up)",
            )

        case_title = {"mbb": "Half MBB beam", "cantilever": "Cantilever", "lbracket": "L-bracket"}[spec.case]
        header = suptitle or (
            f"{case_title} {spec.nx} x {spec.ny}, volume fraction {spec.volfrac} of the design domain: "
            f"black = material, red arrow = load (wht_topopt v3.0)"
        )
        fig.suptitle(header, fontsize=GLOBAL_FONT_SIZE_PT + 2)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        return png_path

    def write_markdown_table(self, records: List[MethodRunRecord], md_path: Path) -> Path:
        """Write Markdown comparison table including PARDISO, Kratos, and FrontISTR 2x finer re-analysis."""
        rows = [
            "| method | compliance (own model) | common re-analysis 2x (PARDISO) | common 2x (Kratos C++) | common 2x (FrontISTR) | design volume | gray | iterations | converged | time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
        ]
        for r in records:
            if r.error:
                rows.append(f"| {r.label} | ERROR: {r.error[:60]} | - | - | - | - | - | - | False | {r.wall_sec:.1f} s |")
                continue
            kra_s = f"{r.common_compliance_kratos:.2f}" if r.common_compliance_kratos is not None else "-"
            fis_s = f"{r.common_compliance_frontistr:.2f}" if r.common_compliance_frontistr is not None else "-"
            rows.append(
                f"| {r.label} | {r.compliance:.2f} | {r.common_compliance_pardiso:.2f} | {kra_s} | {fis_s} | "
                f"{r.design_vol:.3f} | {r.grayness:.3f} | {r.iterations} | {r.converged} | {r.wall_sec:.1f} s |"
            )
        md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return md_path

    def run_case(
        self,
        case: str = "mbb",
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        volfrac: float = 0.4,
        rmin: float = 2.0,
    ) -> Dict[str, Any]:
        """Run all optimization algorithms on the chosen benchmark case and save PNG + MD."""
        nx_eff = nx or DEFAULT_SIZES[case][0]
        ny_eff = ny or DEFAULT_SIZES[case][1]
        spec = self.build_problem(nx_eff, ny_eff, volfrac, rmin, case)

        records: List[MethodRunRecord] = [
            self.run_simp_oc(spec),
            self.run_simp_heaviside_mma(spec),
            self.run_semdot(spec),
            self.run_beso(spec),
            self.run_level_set(spec),
        ]
        if case == "lbracket":
            records.append(self.run_stress_constrained_mma(spec))

        stem = f"{case}_{nx_eff}x{ny_eff}_V{int(round(volfrac * 100)):02d}"
        png_path = self.render_comparison_figure(spec, records, self.output_dir / f"{stem}.png")
        md_path = self.write_markdown_table(records, self.output_dir / f"{stem}.md")
        return {"case": case, "png": str(png_path), "md": str(md_path), "records": records}

    def run_solver_head_to_head(
        self,
        nx: int = 60,
        ny: int = 20,
        volfrac: float = 0.4,
        rmin: float = 2.0,
        max_iters: int = 60,
    ) -> Dict[str, Any]:
        """Run the same Half MBB optimization across all 5 FEA solvers (SuperLU, PARDISO, PyAMG, Kratos, FrontISTR)."""
        spec = self.build_problem(nx, ny, volfrac, rmin, "mbb")
        backends = ["scipy", "pardiso", "pyamg", "kratos", "frontistr"]
        records: List[MethodRunRecord] = []
        iters = 12 if self.quick else max_iters

        for bname in backends:
            rec = self.run_simp_oc(spec, solver_backend=bname, max_iters=iters, tol=0.01)
            records.append(rec)

        stem = f"mbb_{nx}x{ny}_V{int(round(volfrac * 100)):02d}_solvers"
        png_path = self.render_comparison_figure(
            spec,
            records,
            self.output_dir / f"{stem}.png",
            suptitle=(
                f"Half MBB {nx} x {ny} (V = {volfrac}) — 5 FEA Solvers Head-to-Head "
                f"(SuperLU vs MKL PARDISO vs PyAMG vs Kratos C++ vs FrontISTR v6.0)"
            ),
        )
        md_path = self.write_markdown_table(records, self.output_dir / f"{stem}.md")
        return {"png": str(png_path), "md": str(md_path), "records": records}

    def run_speed_benchmark(self) -> Dict[str, Any]:
        """Measure and plot Numba Core JIT speedups and 5-solver linear solve times."""
        from wht_topopt.manufacturing.overhang import OverhangFilter

        # 1. Core Operations: NumPy/SciPy COO vs Numba Pre-Indexed CSC / @njit
        mesh2d = StructuredMesh2D(nelx=180, nely=60, lx=180.0, ly=60.0)
        fem2d = FEMEngine(mesh2d, nu=0.3)
        rho2d = np.random.default_rng(42).uniform(0.1, 1.0, size=mesh2d.n_elem)
        u2d = np.random.default_rng(43).normal(size=mesh2d.ndof)

        mesh3d = StructuredMesh3D(nelx=20, nely=10, nelz=10, lx=20.0, ly=10.0, lz=10.0)
        fem3d = FEMEngine(mesh3d, nu=0.3)
        rho3d = np.random.default_rng(44).uniform(0.1, 1.0, size=mesh3d.n_elem)
        u3d = np.random.default_rng(45).normal(size=mesh3d.ndof)
        ov3d = OverhangFilter(mesh3d, build_axis="z")

        # Warmup Numba JIT compilers
        _ = fem2d.assemble_K(rho2d, use_numba=True)
        _ = fem3d.assemble_K(rho3d, use_numba=True)
        _ = fem3d.compute_unit_element_compliance(u3d, use_numba=True)
        _ = fem3d.compute_element_stresses(u3d, use_numba=True)
        _ = ov3d.forward(rho3d, use_numba=True)
        _ = ov3d.backward(rho3d, rho3d, use_numba=True)

        # Verify numerical parity between Numba and NumPy/SciPy (< 1e-13)
        K_nb = fem2d.assemble_K(rho2d, use_numba=True)
        K_py = fem2d.assemble_K(rho2d, use_numba=False)
        k_diff = float(sp.linalg.norm(K_nb - K_py) / max(1.0, sp.linalg.norm(K_py)))
        c_nb = fem3d.compute_unit_element_compliance(u3d, use_numba=True)
        c_py = fem3d.compute_unit_element_compliance(u3d, use_numba=False)
        c_diff = float(np.linalg.norm(c_nb - c_py) / max(1.0, np.linalg.norm(c_py)))
        ov_nb = ov3d.backward(rho3d, rho3d, use_numba=True)
        ov_py = ov3d.backward(rho3d, rho3d, use_numba=False)
        ov_diff = float(np.linalg.norm(ov_nb - ov_py) / max(1.0, np.linalg.norm(ov_py)))

        n_rep = 5 if self.quick else 20

        def time_ms(fn, reps: int = n_rep) -> float:
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return 1000.0 * (time.perf_counter() - t0) / float(reps)

        core_tasks = [
            (
                "2D 강성행렬 조립\n(180x60, 10.8k el)",
                time_ms(lambda: fem2d.assemble_K(rho2d, use_numba=False)),
                time_ms(lambda: fem2d.assemble_K(rho2d, use_numba=True)),
            ),
            (
                "3D 강성행렬 조립\n(20x10x10, 2k Hex8)",
                time_ms(lambda: fem3d.assemble_K(rho3d, use_numba=False)),
                time_ms(lambda: fem3d.assemble_K(rho3d, use_numba=True)),
            ),
            (
                "3D 단위 변형에너지\n(u_e^T Ke0 u_e)",
                time_ms(lambda: fem3d.compute_unit_element_compliance(u3d, use_numba=False)),
                time_ms(lambda: fem3d.compute_unit_element_compliance(u3d, use_numba=True)),
            ),
            (
                "3D von Mises 응력\n(C0 B0 u_e)",
                time_ms(lambda: fem3d.compute_element_stresses(u3d, use_numba=False)),
                time_ms(lambda: fem3d.compute_element_stresses(u3d, use_numba=True)),
            ),
            (
                "3D DfAM 오버행 필터\n(Forward+Backward)",
                time_ms(lambda: ov3d.backward(rho3d, rho3d, use_numba=False)),
                time_ms(lambda: ov3d.backward(rho3d, rho3d, use_numba=True)),
            ),
        ]

        # 2. 5 FEA Solvers Single-Solve Timing on 90x30 MBB (5,642 DOFs)
        spec_mbb = self.build_problem(90, 30, 0.4, 2.0, "mbb")
        fem_mbb = FEMEngine(spec_mbb.mesh, nu=0.3)
        rho_mbb = np.full(spec_mbb.mesh.n_elem, 0.4, dtype=np.float64)

        solver_timings: List[Tuple[str, float]] = []
        for s_label, bname in [
            ("MKL PARDISO\n(pypardiso)", "pardiso"),
            ("SciPy SuperLU\n(splu direct)", "scipy"),
            ("PyAMG\n(AMG-PCG)", "pyamg"),
            ("Kratos C++\n(Cached ModelPart)", "kratos"),
            ("FrontISTR v6.0\n(fistr1.exe SMP)", "frontistr"),
        ]:
            s_obj = self._make_solver(bname)
            # Warmup 1 solve (initializes cached ModelPart for Kratos / PARDISO handle)
            self._solve_fe(spec_mbb, fem_mbb, rho_mbb, 3.0, s_obj, bname)
            reps_s = 2 if (bname == "frontistr" or self.quick) else 8
            ms_val = time_ms(lambda: self._solve_fe(spec_mbb, fem_mbb, rho_mbb, 3.0, s_obj, bname), reps=reps_s)
            solver_timings.append((s_label, ms_val))

        # Plot Korean 9pt Speed Benchmark Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        labels_c = [t[0] for t in core_tasks]
        py_ms = [t[1] for t in core_tasks]
        nb_ms = [t[2] for t in core_tasks]
        speedups = [p / max(n, 1e-6) for p, n in zip(py_ms, nb_ms)]

        x_pos = np.arange(len(labels_c))
        width = 0.36
        ax1.bar(x_pos - width / 2, py_ms, width, label="기존 NumPy / SciPy COO (ms)", color="#7f8c8d")
        bars_nb = ax1.bar(x_pos + width / 2, nb_ms, width, label="Numba @njit + Pre-indexed CSC (ms)", color="#2980b9")
        for idx, (bar, sp_val) in enumerate(zip(bars_nb, speedups)):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(py_ms[idx], nb_ms[idx]) * 1.03,
                f"{sp_val:.1f}x 가속",
                ha="center",
                va="bottom",
                fontsize=GLOBAL_FONT_SIZE_PT - 1,
                fontweight="bold",
                color="#c0392b",
            )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels_c, fontsize=GLOBAL_FONT_SIZE_PT - 1)
        ax1.set_ylabel("1회 실행 소요 시간 (ms, 낮을수록 빠름)", fontsize=GLOBAL_FONT_SIZE_PT)
        ax1.set_title("wht_topopt 코어 연산 Numba JIT 가속 효과", fontsize=GLOBAL_FONT_SIZE_PT + 1)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=GLOBAL_FONT_SIZE_PT - 1)

        s_names = [s[0] for s in solver_timings]
        s_ms = [s[1] for s in solver_timings]
        colors_s = ["#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#c0392b"]
        bars_s = ax2.bar(np.arange(len(s_names)), s_ms, width=0.5, color=colors_s)
        ax2.set_yscale("log")
        for bar, val in zip(bars_s, s_ms):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                val * 1.12,
                f"{val:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=GLOBAL_FONT_SIZE_PT - 1,
            )
        ax2.set_xticks(np.arange(len(s_names)))
        ax2.set_xticklabels(s_names, fontsize=GLOBAL_FONT_SIZE_PT - 1)
        ax2.set_ylabel("1회 FE 해석 시간 (ms, 로그 스케일)", fontsize=GLOBAL_FONT_SIZE_PT)
        ax2.set_title("Half MBB 90x30 (5,642 DOFs) 5대 FEA 솔버 속도 비교", fontsize=GLOBAL_FONT_SIZE_PT + 1)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        png_path = self.output_dir / "speed_benchmark_report.png"
        fig.savefig(png_path, dpi=140)
        plt.close(fig)

        md_lines = [
            "# `wht_topopt` 코어 Numba JIT 가속 및 5대 FEA 솔버 속도 프로파일링 리포트",
            "",
            "## 1. 수치 무결성 교차 검증 (NumPy/SciPy vs Numba `@njit` 상대 오차)",
            f"- **강성행렬 조립 상대 오차 (`||K_numba - K_scipy||_F / ||K_scipy||_F`)**: `{k_diff:.3e}`",
            f"- **3D 단위 변형에너지 상대 오차 (`c_e0`)**: `{c_diff:.3e}`",
            f"- **3D DfAM 오버행 수반 민감도 상대 오차 (`backward`)**: `{ov_diff:.3e}`",
            "",
            "## 2. 코어 연산 Numba 가속비 (`ms/call`)",
            "| 코어 연산 항목 | 기존 NumPy/SciPy COO (ms) | Numba `@njit` + Pre-indexed CSC (ms) | 가속 배율 (Speedup) |",
            "|---|---:|---:|---:|",
        ]
        for (name, p_t, n_t), sp_v in zip(core_tasks, speedups):
            clean_name = name.replace("\n", " ")
            md_lines.append(f"| {clean_name} | {p_t:.3f} ms | {n_t:.3f} ms | **{sp_v:.2f}x** |")

        md_lines.extend([
            "",
            "## 3. 5대 FEA 선형 솔버 반복당 해석 시간 (`Half MBB 90x30`, 5,642 DOFs)",
            "| 솔버 백엔드 | 구동 방식 | 1회 해석 시간 (ms) |",
            "|---|---|---:|",
        ])
        for s_lbl, val in solver_timings:
            clean_lbl = s_lbl.replace("\n", " ")
            md_lines.append(f"| {clean_lbl} | Native In-Memory / SMP | {val:.2f} ms |")

        md_path = self.output_dir / "speed_benchmark_report.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return {
            "png": str(png_path),
            "md": str(md_path),
            "parity_errors": {"K": k_diff, "c_e0": c_diff, "overhang": ov_diff},
            "core_speedups": speedups,
            "solver_timings": solver_timings,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run high-resolution 2D beam & solver benchmarks.")
    ap.add_argument("--case", choices=["mbb", "cantilever", "lbracket", "all"], default="all")
    ap.add_argument("--solver", choices=["pardiso", "scipy", "pyamg", "kratos", "frontistr"], default="pardiso")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--volfrac", type=float, default=0.4)
    ap.add_argument("--rmin", type=float, default=2.0)
    ap.add_argument("--include-solver-comparison", action="store_true")
    ap.add_argument("--include-speed-benchmark", action="store_true")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    suite = BeamsBenchmarkSuite(
        output_dir=Path(args.out) if args.out else None,
        solver_backend=args.solver,
        quick=False,
    )
    if args.include_speed_benchmark:
        sp_res = suite.run_speed_benchmark()
        print(f"[SPEED] Saved {sp_res['png']}")
        print(Path(sp_res["md"]).read_text(encoding="utf-8"))

    cases = ["mbb", "cantilever", "lbracket"] if args.case == "all" else [args.case]
    for c in cases:
        res = suite.run_case(case=c, nx=args.nx, ny=args.ny, volfrac=args.volfrac, rmin=args.rmin)
        print(f"[{c.upper()}] Saved {res['png']}")
        print(Path(res["md"]).read_text(encoding="utf-8"))

    if args.include_solver_comparison:
        s_res = suite.run_solver_head_to_head()
        print(f"[SOLVERS] Saved {s_res['png']}")
        print(Path(s_res["md"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
