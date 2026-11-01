"""13-stage end-to-end topology optimization runner (TopOptRunner)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from wht_topopt.algorithms.base import GradientSource, OptState, Optimizer, TrustRegionController
from wht_topopt.algorithms.beso import BESOOptimizer
from wht_topopt.algorithms.level_set import LevelSetOptimizer
from wht_topopt.algorithms.mma_wrapper import MMAOptimizer
from wht_topopt.algorithms.oc import OCOptimizer
from wht_topopt.algorithms.semdot import SEMDOTOptimizer
from wht_topopt.core.boundary import BoundaryConditionManager
from wht_topopt.core.continuation import ContinuationScheduler
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter, FilterOperator, HelmholtzPDEFilter, SensitivityFilter
from wht_topopt.core.interpolation import (
    ClassicSIMP,
    MaterialInterpolation,
    MultiMaterialSIMP,
    RAMP,
    SIMP,
)
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from wht_topopt.core.projection import HeavisideProjection
from wht_topopt.functions.base import ConstraintFunction, ObjectiveFunction
from wht_topopt.functions.buckling import BucklingConstraint
from wht_topopt.functions.compliance import ComplianceFunction
from wht_topopt.functions.modal import EigenFrequencyConstraint
from wht_topopt.functions.stress import PNormStressConstraint
from wht_topopt.functions.volume import VolumeConstraint
from wht_topopt.manufacturing.casting import CastingFilter
from wht_topopt.manufacturing.extrusion import ExtrusionFilter
from wht_topopt.manufacturing.milling import MillingFilter
from wht_topopt.manufacturing.minimum_length import GeometricLengthScaleConstraint
from wht_topopt.manufacturing.overhang import OverhangFilter
from wht_topopt.pipeline.config import TopOptConfig
from wht_topopt.pipeline.report import TopOptReport
from wht_topopt.robust.mujoco_loads import MuJoCoLoadProvider
from wht_topopt.robust.robust_projection import RobustFormulation
from wht_topopt.solvers.base import FEASolver
from wht_topopt.solvers.frontistr_adapter import FrontISTRSolver
from wht_topopt.solvers.kratos_solver import KratosSolver
from wht_topopt.solvers.pyamg_solver import PyAMGSolver
from wht_topopt.solvers.scipy_solver import ScipySolver


@dataclass
class TopOptPipelineResult:
    """Result container returned by TopOptRunner.run()."""

    config: TopOptConfig
    mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]
    rho_design: np.ndarray
    rho_tilde: np.ndarray
    rho_phys: np.ndarray
    u: np.ndarray
    final_objective: float
    final_volume_fraction: float
    converged: bool
    iterations: int
    history: List[OptState] = field(default_factory=list)
    stage_trace: List[str] = field(default_factory=list)
    exported_files: Dict[str, str] = field(default_factory=dict)
    mujoco_summary: Optional[Dict[str, Any]] = None
    isosurface_vertices: Optional[np.ndarray] = None
    isosurface_faces: Optional[np.ndarray] = None


class TopOptRunner:
    """13-stage end-to-end topology optimization execution engine.

    Stages:
        1. Mesh Construction
        2. Material Interpolation Setup
        3. FEM Engine & Linear Solver Setup
        4. Boundary Conditions & MuJoCo Dynamics / Inertia Relief Setup
        5. Spatial Filter Construction
        6. Threshold Projection & Continuation Scheduler Setup
        7. Manufacturing Constraint Filter Chain Setup
        8. Objective & Physics Constraints (Compliance/Robust, Volume, Stress, Buckling, Modal) Setup
        9. Mathematical Optimizer & Trust-Region Controller Setup
       10. Iterative Forward Pass (Filter -> Projection -> Manufacturing -> FEA -> Evaluation)
       11. Iterative Adjoint Sensitivity Backward Pass & Design Update
       12. Post-Processing (Isosurface Extraction, Taubin Smoothing, STL/STEP Export, MJCF Inertial Update)
       13. Report Generation (2D Korean Matplotlib & 3D ParaView-Style PyVista)
    """

    def __init__(
        self,
        config: Union[TopOptConfig, Dict[str, Any]],
        mesh: Optional[Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]] = None,
        bcm: Optional[BoundaryConditionManager] = None,
    ) -> None:
        """Initialize TopOptRunner.

        Parameters
        ----------
        config : TopOptConfig | dict
            Declarative pipeline configuration.
        mesh : StructuredMesh2D | StructuredMesh3D | UnstructuredMesh, optional
            Optional pre-constructed mesh override.
        bcm : BoundaryConditionManager, optional
            Optional pre-configured boundary condition manager override.
        """
        if isinstance(config, dict):
            self.config = TopOptConfig.from_dict(config)
        else:
            self.config = config
        self._custom_mesh = mesh
        self._custom_bcm = bcm
        self.stage_trace: List[str] = []

    def _build_mesh(self) -> Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]:
        """Stage 1: Build computational mesh."""
        self.stage_trace.append("stage_01_mesh")
        if self._custom_mesh is not None:
            return self._custom_mesh

        mc = self.config.mesh
        if mc.dimension == 2:
            lx = mc.lx if mc.lx is not None else float(mc.nelx)
            ly = mc.ly if mc.ly is not None else float(mc.nely)
            return StructuredMesh2D(nx=mc.nelx, ny=mc.nely, lx=lx, ly=ly)
        else:
            lx = mc.lx if mc.lx is not None else float(mc.nelx)
            ly = mc.ly if mc.ly is not None else float(mc.nely)
            lz = mc.lz if mc.lz is not None else float(mc.nelz)
            return StructuredMesh3D(nx=mc.nelx, ny=mc.nely, nz=mc.nelz, lx=lx, ly=ly, lz=lz)

    def _build_material(self) -> MaterialInterpolation:
        """Stage 2: Build material interpolation law."""
        self.stage_trace.append("stage_02_material")
        mat = self.config.material
        if mat.model == "simp":
            return ClassicSIMP(E0=mat.e0, Emin=mat.emin, penal=mat.penal)
        elif mat.model in ("simp_modified", "hs_upper", "halpin_tsai"):
            return SIMP(E0=mat.e0, Emin=mat.emin, penal=mat.penal)
        elif mat.model == "ramp":
            return RAMP(E0=mat.e0, Emin=mat.emin, q=mat.ramp_q)
        raise ValueError(f"Unknown material model: {mat.model}")

    def _build_fem_and_solver(
        self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]
    ) -> Tuple[FEMEngine, FEASolver]:
        """Stage 3: Build FEMEngine and FEASolver."""
        self.stage_trace.append("stage_03_fem_and_solver")
        mat = self.config.material
        fem = FEMEngine(
            mesh=mesh,
            nu=mat.nu,
            plane="plane_stress" if mat.plane_stress else "plane_strain",
        )
        sc = self.config.solver
        if sc.backend == "scipy":
            solver: FEASolver = ScipySolver()
        elif sc.backend == "pyamg":
            solver = PyAMGSolver(tol=sc.tol, maxiter=sc.maxiter)
        elif sc.backend == "kratos":
            solver = KratosSolver(mode=sc.mode)
        elif sc.backend == "frontistr":
            solver = FrontISTRSolver(mode=sc.mode)
        else:
            raise ValueError(f"Unsupported solver backend: {sc.backend}")
        return fem, solver

    def _build_boundary_conditions(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        fem: FEMEngine,
    ) -> Tuple[BoundaryConditionManager, Optional[MuJoCoLoadProvider], Optional[Dict[str, Any]]]:
        """Stage 4: Configure boundary conditions, loads, and optional MuJoCo inertia relief."""
        self.stage_trace.append("stage_04_boundary_conditions")
        if self._custom_bcm is not None:
            return self._custom_bcm, None, None

        bc = self.config.bc
        mj_cfg = self.config.mujoco
        mj_provider: Optional[MuJoCoLoadProvider] = None
        mj_summary: Optional[Dict[str, Any]] = None

        if mj_cfg.enabled or bc.preset == "mujoco":
            bcm = BoundaryConditionManager(mesh)
            if bc.passive_solid:
                bcm.set_passive_solid(np.asarray(bc.passive_solid, dtype=np.int64))
            if bc.passive_void:
                bcm.set_passive_void(np.asarray(bc.passive_void, dtype=np.int64))

            if mj_cfg.xml_path is not None:
                mj_provider = MuJoCoLoadProvider(xml_path=mj_cfg.xml_path)
            elif mj_cfg.xml_string is not None:
                mj_provider = MuJoCoLoadProvider(xml_str=mj_cfg.xml_string)
            else:
                raise ValueError("MuJoCo mode requires either mujoco.xml_path or mujoco.xml_string")

            mj_provider.simulate_trajectory(t_final=mj_cfg.duration, dt=mj_cfg.dt)
            peak_cases = mj_provider.extract_peak_load_cases(body_name=mj_cfg.body_name, top_k=1)
            peak_case = peak_cases[0]

            coords = mesh.node_coords
            min_x = float(np.min(coords[:, 0]))
            max_x = float(np.max(coords[:, 0]))
            left_nodes = np.flatnonzero(np.isclose(coords[:, 0], min_x, atol=1e-6))
            right_nodes = np.flatnonzero(np.isclose(coords[:, 0], max_x, atol=1e-6))

            fe_cases = mj_provider.map_to_fe_loads(
                mesh=mesh,
                peak_cases=peak_cases,
                attachment_node_indices=right_nodes,
                density_mat=self.config.material.rho0,
                apply_inertia_relief=mj_cfg.use_inertia_relief,
            )
            f_bal = fe_cases[0].force_vector

            # Fix left support nodes
            bcm.fix_nodes(left_nodes)

            # Inject balanced nodal load vector into bcm._forces[0]
            bcm._forces[0] = {}
            nz_dofs = np.flatnonzero(np.abs(f_bal) > 1e-16)
            for dof_idx in nz_dofs:
                bcm._forces[0][int(dof_idx)] = float(f_bal[dof_idx])
            if len(nz_dofs) == 0:
                tip_dof = int(right_nodes[0] * mesh.ndof_per_node + 1)
                bcm._forces[0][tip_dof] = -abs(bc.force_magnitude)

            mj_summary = {
                "peak_time": float(peak_case["time"]),
                "peak_force_norm": float(np.linalg.norm(peak_case["force"])),
                "linear_accel_norm": float(np.linalg.norm(peak_case["linear_accel"])),
                "balanced_load_norm": float(np.linalg.norm(f_bal)),
            }
            return bcm, mj_provider, mj_summary

        if bc.preset == "mbb_2d" and isinstance(mesh, StructuredMesh2D):
            bcm = BoundaryConditionManager.mbb_half_2d(mesh, F_mag=bc.force_magnitude)
        elif bc.preset == "cantilever_2d" and isinstance(mesh, StructuredMesh2D):
            bcm = BoundaryConditionManager.cantilever_2d(mesh, F_mag=bc.force_magnitude)
        elif bc.preset == "column_2d" and isinstance(mesh, StructuredMesh2D):
            bcm = BoundaryConditionManager.compressed_column_2d(mesh, F_mag=bc.force_magnitude)
        elif bc.preset == "cantilever_3d" and isinstance(mesh, StructuredMesh3D):
            bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=bc.force_magnitude)
        elif bc.preset == "custom":
            bcm = BoundaryConditionManager(mesh)
            if bc.fixed_dofs:
                bcm.fix_dofs(bc.fixed_dofs)
            if bc.loads:
                bcm._forces[0] = {}
                for d, val in bc.loads:
                    bcm._forces[0][int(d)] = bcm._forces[0].get(int(d), 0.0) + float(val)
        else:
            if isinstance(mesh, StructuredMesh2D):
                bcm = BoundaryConditionManager.cantilever_2d(mesh, F_mag=bc.force_magnitude)
            else:
                bcm = BoundaryConditionManager.cantilever_3d(mesh, F_mag=bc.force_magnitude)

        if bc.passive_solid:
            bcm.set_passive_solid(np.asarray(bc.passive_solid, dtype=np.int64))
        if bc.passive_void:
            bcm.set_passive_void(np.asarray(bc.passive_void, dtype=np.int64))

        return bcm, None, None

    def _build_filter(
        self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]
    ) -> Optional[FilterOperator]:
        """Stage 5: Build spatial density filter."""
        self.stage_trace.append("stage_05_filter")
        fc = self.config.filter
        if fc.filter_type == "none":
            return None
        elif fc.filter_type in ("cone", "gaussian"):
            return DensityFilter(mesh, r_min=fc.rmin)
        elif fc.filter_type == "helmholtz":
            return HelmholtzPDEFilter(mesh, r_min=fc.rmin)
        raise ValueError(f"Unsupported filter_type: {fc.filter_type}")

    def _build_projection_and_continuation(
        self,
    ) -> Tuple[Optional[HeavisideProjection], Optional[ContinuationScheduler]]:
        """Stage 6: Build Heaviside projection and continuation scheduler."""
        self.stage_trace.append("stage_06_projection_and_continuation")
        pc = self.config.projection
        cc = self.config.continuation

        proj = (
            HeavisideProjection(beta=pc.beta_init, eta=pc.eta)
            if pc.enabled
            else None
        )
        sched = (
            ContinuationScheduler(
                penal_init=cc.penal_init,
                penal_max=cc.penal_max,
                penal_step=cc.penal_step,
                penal_interval=cc.interval,
                beta_init=cc.beta_init,
                beta_max=cc.beta_max,
                beta_factor=cc.beta_factor,
                beta_interval=cc.interval,
            )
            if cc.enabled
            else None
        )
        return proj, sched

    def _build_manufacturing_chain(
        self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]
    ) -> List[FilterOperator]:
        """Stage 7: Build manufacturing filter chain."""
        self.stage_trace.append("stage_07_manufacturing")
        mfg = self.config.manufacturing
        chain: List[FilterOperator] = []
        if mfg.extrusion:
            chain.append(ExtrusionFilter(mesh, extrusion_axis=mfg.extrude_axis.lstrip("+-")))
        if mfg.casting:
            chain.append(
                CastingFilter(
                    mesh,
                    draw_axis=mfg.draw_dir.lstrip("+-"),
                    mode="parting_plane" if mfg.casting_mode == "split" else mfg.casting_mode,
                )
            )
        if mfg.milling:
            chain.append(MillingFilter(mesh, directions=mfg.tool_dirs))
        if mfg.overhang:
            chain.append(OverhangFilter(mesh, build_axis=mfg.build_dir.lstrip("+-")))
        return chain

    def _build_functions(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        fem: FEMEngine,
        solver: FEASolver,
        bcm: BoundaryConditionManager,
        mat_interp: MaterialInterpolation,
    ) -> Tuple[Union[ObjectiveFunction, RobustFormulation], List[ConstraintFunction]]:
        """Stage 8: Build objective function and physics constraints."""
        self.stage_trace.append("stage_08_functions")
        mat = self.config.material
        opt = self.config.optimizer
        phys = self.config.physics

        comp_fn = ComplianceFunction(
            fem=fem,
            solver=solver,
            bcm=bcm,
            penal=mat.penal,
            e0=mat.e0,
            emin=mat.emin,
            material_model=mat.model if mat.model in ("simp", "simp_modified", "ramp") else "simp_modified",
            ramp_q=mat.ramp_q,
        )

        if self.config.robust.enabled:
            rc = self.config.robust
            obj_fn: Union[ObjectiveFunction, RobustFormulation] = RobustFormulation(
                base_objective=comp_fn,
                beta=self.config.projection.beta_init if self.config.projection.enabled else 2.0,
                eta_erode=rc.eta_erode,
                eta_inter=rc.eta_inter,
                eta_dilate=rc.eta_dilate,
                aggregation=rc.aggregation,
                elem_volumes=mesh.elem_volumes,
            )
        else:
            obj_fn = comp_fn

        constraints: List[ConstraintFunction] = [
            VolumeConstraint(mesh=mesh, volfrac=opt.volfrac)
        ]

        if phys.stress_enabled:
            constraints.append(
                PNormStressConstraint(
                    fem=fem,
                    solver=solver,
                    bcm=bcm,
                    p=phys.stress_p,
                    sigma_allow=phys.sigma_allow,
                    q=phys.stress_q,
                    penal=mat.penal,
                    e0=mat.e0,
                    emin=mat.emin,
                )
            )
        if phys.buckling_enabled:
            constraints.append(
                BucklingConstraint(
                    fem=fem,
                    solver=solver,
                    bcm=bcm,
                    lambda_min=phys.lambda_min,
                    n_modes=phys.n_buckling_modes,
                    rho_ks=phys.buckling_ks_rho,
                    penal=mat.penal,
                    e0=mat.e0,
                    emin=mat.emin,
                )
            )
        if phys.modal_enabled:
            constraints.append(
                EigenFrequencyConstraint(
                    fem=fem,
                    solver=solver,
                    bcm=bcm,
                    omega_min=phys.omega_min,
                    n_modes=phys.n_modal_modes,
                    penal=mat.penal,
                    e0=mat.e0,
                    emin=mat.emin,
                )
            )
        if self.config.manufacturing.min_length_scale is not None:
            constraints.append(
                GeometricLengthScaleConstraint(
                    mesh=mesh,
                    rmin=self.config.manufacturing.min_length_scale,
                )
            )

        return obj_fn, constraints

    def _build_optimizer(
        self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]
    ) -> Tuple[Optimizer, Optional[TrustRegionController]]:
        """Stage 9: Build mathematical optimizer and optional TrustRegionController."""
        self.stage_trace.append("stage_09_optimizer")
        oc = self.config.optimizer
        tr = TrustRegionController(initial_move=oc.move) if oc.use_trust_region else None
        gs = GradientSource(oc.gradient_source)

        has_extra_constraints = (
            self.config.physics.stress_enabled
            or self.config.physics.buckling_enabled
            or self.config.physics.modal_enabled
            or self.config.manufacturing.min_length_scale is not None
        )
        alg = "mma" if (oc.algorithm == "oc" and has_extra_constraints) else oc.algorithm

        if alg == "oc":
            optimizer: Optimizer = OCOptimizer(
                volfrac=oc.volfrac,
                move=oc.move,
                rho_min=oc.rho_min,
                trust_region=tr,
                gradient_source=gs,
            )
        elif alg == "mma":
            optimizer = MMAOptimizer(
                move=oc.move,
                rho_min=oc.rho_min,
                trust_region=tr,
                gradient_source=gs,
            )
        elif alg == "beso":
            optimizer = BESOOptimizer(
                target_volfrac=oc.volfrac,
                rho_min=oc.rho_min,
                move=oc.move,
                trust_region=tr,
                gradient_source=gs,
            )
        elif alg == "level_set":
            optimizer = LevelSetOptimizer(
                mesh=mesh,
                volfrac=oc.volfrac,
                rho_min=oc.rho_min,
                move=oc.move,
                trust_region=tr,
                gradient_source=gs,
            )
        elif alg == "semdot":
            optimizer = SEMDOTOptimizer(
                mesh=mesh,
                volfrac=oc.volfrac,
                rho_min=oc.rho_min,
                move=oc.move,
                trust_region=tr,
                gradient_source=gs,
            )
        else:
            raise ValueError(f"Unsupported optimizer algorithm: {alg}")

        return optimizer, tr

    @staticmethod
    def _apply_passive_domains(
        rho_arr: np.ndarray, bcm: BoundaryConditionManager, rho_min: float
    ) -> None:
        """Apply passive solid (1.0) and passive void (rho_min) masks in-place."""
        if np.any(bcm.passive_solid_mask):
            rho_arr[bcm.passive_solid_mask] = 1.0
        if np.any(bcm.passive_void_mask):
            rho_arr[bcm.passive_void_mask] = rho_min

    def run(self) -> TopOptPipelineResult:
        """Execute the full 13-stage topology optimization pipeline."""
        self.stage_trace.clear()

        # Stages 1 - 9: Initialization
        mesh = self._build_mesh()
        mat_interp = self._build_material()
        fem, solver = self._build_fem_and_solver(mesh)
        bcm, mj_provider, mj_summary = self._build_boundary_conditions(mesh, fem)
        spatial_filter = self._build_filter(mesh)
        projection, continuation = self._build_projection_and_continuation()
        mfg_chain = self._build_manufacturing_chain(mesh)
        obj_fn, constraints = self._build_functions(mesh, fem, solver, bcm, mat_interp)
        optimizer, tr_ctrl = self._build_optimizer(mesh)

        n_elem = mesh.n_elem
        volfrac = self.config.optimizer.volfrac
        rho_min = self.config.optimizer.rho_min
        rho = np.full(n_elem, volfrac, dtype=np.float64)
        self._apply_passive_domains(rho, bcm, rho_min)

        history: List[OptState] = []
        converged = False
        u_last = np.zeros(fem.ndof, dtype=np.float64)
        rho_tilde = rho.copy()
        rho_phys = rho.copy()

        # Stages 10 & 11: Forward and Adjoint Optimization Loop
        self.stage_trace.append("stage_10_forward_loop")
        self.stage_trace.append("stage_11_adjoint_and_update_loop")

        max_iter = self.config.optimizer.max_iter
        tol_change = self.config.optimizer.tol_change

        for it in range(1, max_iter + 1):
            if continuation is not None:
                c_state = continuation.step(it)
                if hasattr(obj_fn, "penal"):
                    obj_fn.penal = c_state["penal"]
                if projection is not None:
                    projection.beta = c_state["beta"]
                if isinstance(obj_fn, RobustFormulation):
                    obj_fn.set_beta(c_state["beta"])

            # 10.1 Forward filtering chain: rho -> rho_tilde -> rho_proj -> rho_phys
            if spatial_filter is not None:
                rho_tilde = spatial_filter.forward(rho)
            else:
                rho_tilde = rho.copy()

            if isinstance(obj_fn, RobustFormulation):
                rho_proj = rho_tilde.copy()
            elif projection is not None:
                rho_proj = projection.forward(rho_tilde)
            else:
                rho_proj = rho_tilde.copy()

            rho_mfg = rho_proj
            for mfg_op in mfg_chain:
                rho_mfg = mfg_op.forward(rho_mfg)

            rho_phys = rho_mfg.copy()
            self._apply_passive_domains(rho_phys, bcm, rho_min)

            # 10.2 Objective & Gradient evaluation w.r.t. rho_phys
            f_val, df_dphys = obj_fn.value_and_gradient(rho_phys)
            if hasattr(obj_fn, "_last_u") and obj_fn._last_u is not None:
                u_last = obj_fn._last_u.copy()
            elif isinstance(obj_fn, RobustFormulation) and hasattr(obj_fn.base_objective, "_last_u"):
                if obj_fn.base_objective._last_u is not None:
                    u_last = obj_fn.base_objective._last_u.copy()

            # 10.3 Constraint evaluation & gradients w.r.t. rho_phys
            g_vals_list: List[float] = []
            dg_dphys_list: List[np.ndarray] = []
            for c_fn in constraints:
                g_v, dg_v = c_fn.value_and_gradient(rho_phys)
                g_vals_list.append(float(g_v))
                dg_dphys_list.append(np.asarray(dg_v, dtype=np.float64).ravel())

            g_vals = np.asarray(g_vals_list, dtype=np.float64)
            dg_dphys = np.vstack(dg_dphys_list) if dg_dphys_list else np.zeros((0, n_elem))

            # 11.1 Backward chain rule: d/d(rho_phys) -> mfg -> projection -> spatial_filter -> d/d(rho)
            df_dcurr = df_dphys.copy()
            dg_dcurr = dg_dphys.copy()

            for mfg_op in reversed(mfg_chain):
                df_dcurr = mfg_op.backward(df_dcurr)
                for j in range(len(dg_dcurr)):
                    dg_dcurr[j] = mfg_op.backward(dg_dcurr[j])

            if not isinstance(obj_fn, RobustFormulation) and projection is not None:
                df_dcurr = projection.backward(df_dcurr, rho_tilde)
                for j in range(len(dg_dcurr)):
                    dg_dcurr[j] = projection.backward(dg_dcurr[j], rho_tilde)

            if spatial_filter is not None:
                df_drho = spatial_filter.backward(df_dcurr)
                dg_drho = np.vstack([spatial_filter.backward(dg_dcurr[j]) for j in range(len(dg_dcurr))])
            else:
                df_drho = df_dcurr
                dg_drho = dg_dcurr

            # 11.2 Design update
            rho_new = optimizer.update(
                rho=rho,
                df_drho=df_drho,
                g_vals=g_vals,
                dg_drho=dg_drho,
                elem_volumes=mesh.elem_volumes,
                passive_solid=bcm.passive_solid_mask if np.any(bcm.passive_solid_mask) else None,
                passive_void=bcm.passive_void_mask if np.any(bcm.passive_void_mask) else None,
            )
            self._apply_passive_domains(rho_new, bcm, rho_min)

            change = float(np.max(np.abs(rho_new - rho)))
            vol_cur = float(np.sum(rho_phys * mesh.elem_volumes) / np.sum(mesh.elem_volumes))

            state = OptState(
                iteration=it,
                rho=rho_new.copy(),
                rho_tilde=rho_tilde.copy(),
                rho_phys=rho_phys.copy(),
                obj=float(f_val),
                g=g_vals.copy(),
                vol=vol_cur,
                change=change,
                penal=float(getattr(obj_fn, "penal", self.config.material.penal)),
                beta=float(projection.beta if projection is not None else 1.0),
            )
            history.append(state)
            rho = rho_new

            if change < tol_change and it >= 5:
                converged = True
                break

        # Stage 12: Post-processing (Isosurface, Taubin smoothing, STL/STEP export, MJCF update)
        self.stage_trace.append("stage_12_post_processing")
        exported_files: Dict[str, str] = {}
        iso_verts: Optional[np.ndarray] = None
        iso_faces: Optional[np.ndarray] = None

        post_cfg = self.config.post
        if post_cfg.extract_isosurface:
            from wht_topopt.post.isosurface import IsosurfaceExtractor
            from wht_topopt.post.stl_export import STLExporter
            from wht_topopt.post.cad_reconstruction import CADReconstructor

            extractor = IsosurfaceExtractor(mesh)
            verts, faces = extractor.extract(rho_phys, level=post_cfg.iso_level)
            if post_cfg.taubin_iterations > 0 and len(verts) > 0 and len(faces) > 0:
                verts = extractor.taubin_smooth(
                    verts,
                    faces,
                    iterations=post_cfg.taubin_iterations,
                    lam=post_cfg.taubin_lam,
                    mu=post_cfg.taubin_mu,
                )
            iso_verts, iso_faces = verts, faces

            if post_cfg.export_stl_path is not None and len(verts) > 0 and len(faces) > 0:
                stl_path = Path(post_cfg.export_stl_path)
                stl_path.parent.mkdir(parents=True, exist_ok=True)
                STLExporter.write_binary_stl(stl_path, verts, faces)
                exported_files["stl"] = str(stl_path)

            if post_cfg.export_step_path is not None and len(verts) > 0 and len(faces) > 0:
                step_path = Path(post_cfg.export_step_path)
                step_path.parent.mkdir(parents=True, exist_ok=True)
                CADReconstructor.reconstruct_to_step(verts, faces, step_path)
                if step_path.exists():
                    exported_files["step"] = str(step_path)

        if (
            mj_provider is not None
            and self.config.mujoco.update_mjcf_inertials
            and self.config.mujoco.updated_xml_output_path is not None
        ):
            out_xml = Path(self.config.mujoco.updated_xml_output_path)
            out_xml.parent.mkdir(parents=True, exist_ok=True)
            mass_opt, com_opt, inertia_opt = mj_provider.compute_optimized_inertial_properties(
                mesh=mesh,
                rho=rho_phys,
                density_mat=self.config.material.rho0,
            )
            updated_xml_str = mj_provider.update_mjcf_inertial(
                xml_str=mj_provider.xml_str,
                body_name=self.config.mujoco.body_name,
                mass=mass_opt,
                com=com_opt,
                inertia_tensor=inertia_opt,
            )
            out_xml.write_text(updated_xml_str, encoding="utf-8")
            exported_files["mujoco_xml"] = str(out_xml)

        # Stage 13: Report Generation
        self.stage_trace.append("stage_13_report")
        if post_cfg.export_report_dir is not None:
            rep_dir = Path(post_cfg.export_report_dir)
            rep_dir.mkdir(parents=True, exist_ok=True)
            reporter = TopOptReport(mesh=mesh, history=history, rho_phys=rho_phys, title=self.config.name)

            conv_path = rep_dir / "convergence.png"
            fig_c = reporter.plot_convergence(save_path=conv_path)
            import matplotlib.pyplot as plt
            plt.close(fig_c)
            exported_files["convergence_plot"] = str(conv_path)

            dens_path = rep_dir / "density_2d.png"
            fig_d = reporter.plot_density_2d(save_path=dens_path)
            plt.close(fig_d)
            exported_files["density_plot"] = str(dens_path)

            md_path = rep_dir / "summary.md"
            reporter.export_summary_markdown(md_path)
            exported_files["summary_markdown"] = str(md_path)

        final_obj = history[-1].obj if history else 0.0
        final_vol = history[-1].vol if history else float(np.mean(rho_phys))

        return TopOptPipelineResult(
            config=self.config,
            mesh=mesh,
            rho_design=rho,
            rho_tilde=rho_tilde,
            rho_phys=rho_phys,
            u=u_last,
            final_objective=final_obj,
            final_volume_fraction=final_vol,
            converged=converged,
            iterations=len(history),
            history=history,
            stage_trace=list(self.stage_trace),
            exported_files=exported_files,
            mujoco_summary=mj_summary,
            isosurface_vertices=iso_verts,
            isosurface_faces=iso_faces,
        )
