"""Kratos Multiphysics FEA and native Topology Optimization solver adapter.

Dual-Mode Architecture:
1. In-Memory Mode: Directly builds KM.ModelPart in memory (zero disk I/O) on Python 3.12,
   supports 2D/3D solid and shell elements, and native SIMP/OC optimization loops.
2. Subprocess Worker Fallback: Automatically delegates to a standalone Kratos Python environment
   (e.g. `kratos-playground/.venvsrc/Scripts/python.exe`) via `_kratos_worker.py`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import FEASolver, SolverResult


_WORKER_SCRIPT = Path(__file__).with_name("_kratos_worker.py")

_ELEMENT_MAP = {
    "tet4": "SmallDisplacementElement3D4N",
    "tet10": "SmallDisplacementElement3D10N",
    "hex8": "SmallDisplacementElement3D8N",
    "hex8_bbar": "SmallDisplacementBbarElement3D8N",
    "hex20": "SmallDisplacementElement3D20N",
    "quad4": "SmallDisplacementElement2D4N",
    "tri3": "SmallDisplacementElement2D3N",
    "shell_thin": "ShellThinElementCorotational3D3N",
    "shell_thick": "ShellThickElementCorotational3D4N",
    "simp_hex8": "SmallDisplacementSIMPElement3D8N",
    "simp_tet4": "SmallDisplacementSIMPElement3D4N",
}


def ensure_kratos_windows_dlls() -> List[str]:
    """Ensure that all requisite DLL directories for KratosMultiphysics are registered on Windows.

    Windows Python 3.8+ does not search PATH for DLL dependencies of extension modules (.pyd).
    Explicit calls to os.add_dll_directory() are required for:
    - Official wheels: site-packages/KratosMultiphysics/.libs
    - Fallback custom build: E:/kratos-build/install/libs, condadeps/Library/bin
    """
    added_dirs: List[str] = []

    # Prioritize custom build with MKL PARDISO if present, else fallback to site-packages
    kratos_install_root = Path(r"E:\kratos-build\install")
    if kratos_install_root.is_dir() and (kratos_install_root / "KratosMultiphysics").is_dir():
        if str(kratos_install_root) not in sys.path:
            sys.path.insert(0, str(kratos_install_root))

    if os.name != "nt":
        return added_dirs

    candidate_dll_dirs = [
        Path(sys.prefix) / "Lib" / "site-packages" / "KratosMultiphysics" / ".libs",
        Path(r"E:\kratos-build\install\libs"),
        Path(r"D:\PythonCodeStudy\kratos-deps\condadeps\Library\bin"),
        Path(r"D:\PythonCodeStudy\kratos-deps\mmg-5.8-novtk\bin"),
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "Scripts",
    ]

    # Also detect mkl package path in site-packages if installed via pip
    try:
        import mkl
        mkl_dir = Path(mkl.__file__).resolve().parent
        candidate_dll_dirs.extend([mkl_dir, mkl_dir / ".." / ".." / "Library" / "bin"])
    except Exception:
        pass

    for d in candidate_dll_dirs:
        try:
            resolved = d.resolve()
            if resolved.is_dir() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(resolved))
                added_dirs.append(str(resolved))
        except Exception:
            pass

    if added_dirs:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(added_dirs + [existing_path])

    return added_dirs


# Automatically register Windows DLL paths and MKL PARDISO library roots upon module import
ensure_kratos_windows_dlls()


def _check_in_memory_kratos() -> bool:
    """Check if KratosMultiphysics and StructuralMechanicsApplication are available in-process."""
    ensure_kratos_windows_dlls()
    try:
        import KratosMultiphysics as KM  # noqa: F401
        import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401
        return True
    except Exception:
        return False


def _check_in_memory_topopt() -> bool:
    """Check if either OptimizationApplication or TopologyOptimizationApplication is available in-process."""
    ensure_kratos_windows_dlls()
    try:
        import KratosMultiphysics.OptimizationApplication as OA  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import KratosMultiphysics.TopologyOptimizationApplication as TOA  # noqa: F401
        return True
    except Exception:
        return False


def resolve_kratos_python_path() -> Optional[str]:
    """Find a Python executable capable of executing Kratos scripts."""
    # 1. Environment variables
    for env_var in ("WHT_KRATOS_PYTHON", "KRATOS_PYTHON", "AIENG_KRATOS_PYTHON"):
        val = os.environ.get(env_var, "").strip().strip('"')
        if val and (shutil.which(val) or Path(val).is_file()):
            return str(Path(val).resolve())

    # 2. Known local project environments
    candidates = [
        Path(r"D:\PythonCodeStudy\kratos-playground\.venvsrc\Scripts\python.exe"),
        Path(r"D:\PythonCodeStudy\kratos-deps\condadeps\python.exe"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())

    # 3. Check system PATH
    found = shutil.which("python")
    return found


def _build_kratos_env(python_path: str) -> Dict[str, str]:
    """Build environment variables with requisite DLL paths for Kratos subprocess execution."""
    prefix = Path(python_path).resolve().parent
    extra = [
        prefix,
        prefix / "Library" / "bin",
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Scripts",
        prefix / "bin",
        Path(r"E:\kratos-build\install\libs"),
        Path(r"D:\PythonCodeStudy\kratos-deps\condadeps\Library\bin"),
    ]
    env = dict(os.environ)
    existing = env.get("PATH", "")
    dll_paths = [str(p) for p in extra if p.is_dir()]
    env["PATH"] = os.pathsep.join(dll_paths + [existing])
    return env


class KratosSolver(FEASolver):
    """Kratos Multiphysics FEA Solver with in-memory execution and fallback worker."""

    solver_name: str = "kratos"

    def __init__(
        self,
        method: str = "direct",
        linear_solver: Optional[str] = "pardiso_lu",
        num_threads: int = 6,
        python_executable: Optional[str] = None,
        timeout_sec: float = 900.0,
        prefer_in_memory: bool = True,
    ) -> None:
        """Initialize KratosSolver.
        
        Args:
            method: 'direct' (SparseLU/Pardiso) or 'cg' / 'amgcl'.
            linear_solver: 'auto', 'pardiso_lu' / 'pardiso', 'sparse_lu', or 'amgcl'.
            num_threads: Number of parallel threads for OpenMP & MKL PARDISO (default 6).
            python_executable: Path to Kratos Python executable for subprocess fallback.
            timeout_sec: Timeout for subprocess execution.
            prefer_in_memory: If True, uses in-memory Python API when available.
        """
        self.method = method.lower()
        self.linear_solver = (linear_solver or "auto").lower()
        self.num_threads = int(num_threads)
        self.timeout_sec = timeout_sec
        self.prefer_in_memory = prefer_in_memory

        # Configure OpenMP and MKL thread environment
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)

        self._in_memory_available = _check_in_memory_kratos()
        self._topopt_available = _check_in_memory_topopt()

        if self._in_memory_available:
            try:
                import KratosMultiphysics as KM
                KM.ParallelUtilities.SetNumThreads(self.num_threads)
            except Exception:
                pass

        if python_executable is not None:
            self.python_executable = python_executable
        else:
            self.python_executable = resolve_kratos_python_path()

        # Cache for solve_cached
        self._last_lu: Optional[spla.SuperLU] = None
        self._last_free_dofs: Optional[np.ndarray] = None
        self._last_n_dof: int = 0
        self._cached_mp_bundle: Optional[Dict[str, Any]] = None

    def reset_cache(self) -> None:
        """Reset cached Kratos in-memory ModelPart bundle."""
        self._cached_mp_bundle = None

    @property
    def is_in_memory(self) -> bool:
        """True if the solver can execute inside the current Python process."""
        return self.prefer_in_memory and self._in_memory_available

    @property
    def is_topopt_available(self) -> bool:
        """True if OptimizationApplication or TopologyOptimizationApplication is available."""
        return self._topopt_available

    def solve(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        free_dofs: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
        M: Optional[sp.spmatrix] = None,
        node_coords: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve global system K u = F.
        
        When invoked with global sparse matrix K and vector F, partitions K_ff and solves
        via direct sparse LU (or AMG) with full caching for subsequent adjoint solves.
        """
        t0 = time.perf_counter()
        n_dof = K.shape[0]

        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if len(free_dofs) == n_dof and node_coords is not None:
            dim = 2 if node_coords.shape[1] == 2 else 3
            F = self.apply_inertia_relief(F, M, node_coords, free_dofs=free_dofs, dim=dim)

        K_ff, F_f, free_dofs, fixed_dofs = self.partition_system(K, F, free_dofs, fixed_dofs, u_fixed)
        K_ff_csc = K_ff.tocsc()
        n_free = K_ff_csc.shape[0]

        # Use splu with caching for fast adjoint solves
        lu = spla.splu(K_ff_csc)
        self._last_lu = lu
        self._last_free_dofs = free_dofs
        self._last_n_dof = n_dof

        u_free = lu.solve(F_f)
        u_full = self.reconstruct_displacement(u_free, n_dof, free_dofs, fixed_dofs, u_fixed)
        solve_time = time.perf_counter() - t0

        if F_f.ndim == 1:
            res = K_ff_csc @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))
        else:
            res = K_ff_csc @ u_free - F_f
            res_norm = float(np.linalg.norm(res) / max(1.0, float(np.linalg.norm(F_f))))

        extra = {
            "mode": "in_memory" if self.is_in_memory else "sparse_algebra",
            "method": self.method,
            "n_free": int(n_free),
            "n_dof": int(n_dof),
        }

        return SolverResult(
            u=u_full,
            residual_norm=res_norm,
            iterations=1,
            solve_time_sec=solve_time,
            solver_name=self.solver_name,
            extra=extra,
        )

    def solve_cached(self, rhs: np.ndarray) -> np.ndarray:
        """Solve adjoint equation K_ff u_f_adj = rhs_f reusing the cached LU factorization."""
        if self._last_free_dofs is None or self._last_lu is None:
            raise RuntimeError("solve_cached called before any system was factorized with solve()")

        rhs_arr = np.asarray(rhs, dtype=float)
        free_dofs = self._last_free_dofs
        n_free = len(free_dofs)
        n_dof = self._last_n_dof

        if rhs_arr.shape[0] == n_dof:
            rhs_f = rhs_arr[free_dofs]
        elif rhs_arr.shape[0] == n_free:
            rhs_f = rhs_arr
        elif rhs_arr.ndim == 2 and rhs_arr.shape[1] == n_dof:
            rhs_f = rhs_arr[:, free_dofs].T
        elif rhs_arr.ndim == 2 and rhs_arr.shape[0] == n_dof:
            rhs_f = rhs_arr[free_dofs, :]
        else:
            raise ValueError(f"rhs shape {rhs_arr.shape} is incompatible with n_dof={n_dof}")

        u_free_adj = self._last_lu.solve(rhs_f)
        return self.reconstruct_displacement(u_free_adj, n_dof, free_dofs)

    def evaluate_mesh(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype: str,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        youngs: Optional[np.ndarray] = None,
        nu: float = 0.3,
        x_phys: Optional[np.ndarray] = None,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        use_simp_elements: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate displacement and sensitivities from FE mesh definition in memory or via worker."""
        if self.is_in_memory:
            return self._evaluate_mesh_in_memory(
                nodes=nodes,
                elems=elems,
                etype=etype,
                fixed_dofs=fixed_dofs,
                loads=loads,
                youngs=youngs,
                nu=nu,
                x_phys=x_phys,
                penal=penal,
                e0=e0,
                emin=emin,
                use_simp_elements=use_simp_elements,
            )
        else:
            return self._evaluate_mesh_subprocess(
                nodes=nodes,
                elems=elems,
                etype=etype,
                fixed_dofs=fixed_dofs,
                loads=loads,
                youngs=youngs,
                nu=nu,
                x_phys=x_phys,
                penal=penal,
                e0=e0,
                emin=emin,
                use_simp_elements=use_simp_elements,
            )

    def _evaluate_mesh_in_memory(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype: str,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        youngs: Optional[np.ndarray] = None,
        nu: float = 0.3,
        x_phys: Optional[np.ndarray] = None,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        use_simp_elements: bool = False,
    ) -> Dict[str, Any]:
        """In-memory Kratos ModelPart creation and solution."""
        import KratosMultiphysics as KM
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
        import KratosMultiphysics.LinearSolversApplication  # noqa: F401
        from KratosMultiphysics import python_linear_solver_factory

        TOA = None
        if use_simp_elements and self._topopt_available:
            import KratosMultiphysics.TopologyOptimizationApplication as TOA

        t0 = time.perf_counter()
        n_nodes = nodes.shape[0]
        dim = 2 if ("2d" in etype.lower() or nodes.shape[1] == 2 or "quad4" in etype.lower() or "tri3" in etype.lower()) else 3
        dof_per_node = dim

        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.ndim == 1:
            loads_arr = loads_arr.reshape(1, -1)
        n_cases = loads_arr.shape[0]

        if youngs is None:
            youngs = np.full(len(elems), e0)

        # Map element name
        use_simp = bool(use_simp_elements and TOA is not None)
        if use_simp:
            element_name = "SmallDisplacementSIMPElement3D8N" if "hex" in etype.lower() else "SmallDisplacementSIMPElement3D4N"
        else:
            element_name = _ELEMENT_MAP.get(etype.lower(), "SmallDisplacementElement3D8N")

        model = KM.Model()
        mp = model.CreateModelPart("Structure")
        mp.SetBufferSize(1)

        mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
        mp.AddNodalSolutionStepVariable(KM.REACTION)
        mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)

        for i in range(n_nodes):
            x = float(nodes[i, 0])
            y = float(nodes[i, 1])
            z = float(nodes[i, 2]) if nodes.shape[1] >= 3 else 0.0
            mp.CreateNewNode(i + 1, x, y, z)

        KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
        if dim == 3:
            KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)

        if use_simp:
            prop = mp.CreateNewProperties(1)
            prop.SetValue(KM.YOUNG_MODULUS, e0)
            prop.SetValue(KM.POISSON_RATIO, nu)
            prop.SetValue(KM.DENSITY, 1.0)
            prop.SetValue(TOA.YOUNGS_MODULUS_0, e0)
            prop.SetValue(TOA.YOUNGS_MODULUS_MIN, emin)
            prop.SetValue(TOA.PENAL, penal)
            prop.SetValue(TOA.MAT_INTERP, "simp_modified")
            prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElastic3DLaw())

            x_phys_arr = np.asarray(x_phys, dtype=float) if x_phys is not None else np.ones(len(elems))
            for e_idx, conn in enumerate(elems):
                elem_id = e_idx + 1
                node_ids = [int(n) + 1 for n in conn]
                el = mp.CreateNewElement(element_name, elem_id, node_ids, prop)
                el.SetValue(TOA.X_PHYS, float(x_phys_arr[e_idx]))
                el.SetValue(TOA.PENAL, penal)
                el.SetValue(TOA.MAT_INTERP, "simp_modified")
                el.SetValue(KM.YOUNG_MODULUS, e0)
        else:
            uniq, inverse = np.unique(youngs, return_inverse=True)
            inverse = np.asarray(inverse).ravel()
            for pid, E in enumerate(uniq, 1):
                prop = mp.CreateNewProperties(pid)
                prop.SetValue(KM.YOUNG_MODULUS, float(E))
                prop.SetValue(KM.POISSON_RATIO, nu)
                prop.SetValue(KM.DENSITY, 1.0)
                law = SMA.LinearElasticPlaneStress2DLaw() if dim == 2 else SMA.LinearElastic3DLaw()
                prop.SetValue(KM.CONSTITUTIVE_LAW, law)

            for e_idx, conn in enumerate(elems):
                elem_id = e_idx + 1
                node_ids = [int(n) + 1 for n in conn]
                prop_id = int(inverse[e_idx]) + 1
                mp.CreateNewElement(element_name, elem_id, node_ids, mp.GetProperties()[prop_id])

        comps = [KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y]
        if dim == 3:
            comps.append(KM.DISPLACEMENT_Z)

        for d in np.unique(fixed_dofs):
            node_idx = int(d) // dof_per_node
            comp_idx = int(d) % dof_per_node
            mp.GetNode(node_idx + 1).Fix(comps[comp_idx])

        load_props = mp.CreateNewProperties(999999)
        cond_name = "PointLoadCondition2D1N" if dim == 2 else "PointLoadCondition3D1N"
        active_load_nodes = sorted({int(d) // dof_per_node for lc in loads_arr for d in np.flatnonzero(lc)})
        for cid, n_idx in enumerate(active_load_nodes, 1):
            mp.CreateNewCondition(cond_name, cid, [n_idx + 1], load_props)

        has_mkl = False
        try:
            import KratosMultiphysics.LinearSolversApplication as LSA
            has_mkl = bool(LSA.HasMKL())
        except Exception:
            has_mkl = False

        if self.linear_solver in ("pardiso", "pardiso_lu"):
            solver_type = "LinearSolversApplication.pardiso_lu" if has_mkl else "LinearSolversApplication.sparse_lu"
        elif self.linear_solver in ("sparse_lu", "direct"):
            solver_type = "LinearSolversApplication.sparse_lu"
        elif self.linear_solver in ("amgcl", "cg"):
            solver_type = "amgcl"
        else:  # auto
            solver_type = "LinearSolversApplication.pardiso_lu" if (has_mkl and self.method == "direct") else ("LinearSolversApplication.sparse_lu" if self.method == "direct" else "amgcl")

        params = {"solver_type": solver_type}
        if solver_type == "amgcl" or self.method in ("cg", "amgcl"):
            params.update({"krylov_type": "cg", "tolerance": 1e-10, "max_iteration": 2000, "verbosity": 0})
        linear_solver = python_linear_solver_factory.ConstructSolver(KM.Parameters(json.dumps(params)))

        if use_simp:
            scheme = TOA.ResidualBasedIncrementalUpdateStaticSIMPScheme()
        else:
            scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()

        builder = KM.ResidualBasedEliminationBuilderAndSolver(linear_solver) if solver_type == "LinearSolversApplication.pardiso_lu" else KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
        strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
        strategy.SetEchoLevel(0)

        u_out = np.zeros((n_cases, n_nodes * dof_per_node), dtype=float)
        dcdx_out = np.zeros(len(elems), dtype=float) if use_simp else None

        for c in range(n_cases):
            load_vec = loads_arr[c]
            for node in mp.Nodes:
                i = node.Id - 1
                node.SetSolutionStepValue(KM.DISPLACEMENT, [0.0] * 3)
                fx = float(load_vec[dim * i])
                fy = float(load_vec[dim * i + 1])
                fz = float(load_vec[dim * i + 2]) if dim == 3 else 0.0
                node.SetSolutionStepValue(SMA.POINT_LOAD, [fx, fy, fz])

            if c == 0:
                strategy.Initialize()
            strategy.InitializeSolutionStep()
            strategy.Predict()
            strategy.SolveSolutionStep()
            strategy.FinalizeSolutionStep()

            for node in mp.Nodes:
                i = node.Id - 1
                disp = node.GetSolutionStepValue(KM.DISPLACEMENT)
                u_out[c, dim * i] = disp[0]
                u_out[c, dim * i + 1] = disp[1]
                if dim == 3:
                    u_out[c, dim * i + 2] = disp[2]

        if use_simp:
            sens_strategy = TOA.StructureAdjointSensitivityStrategy(mp, linear_solver, dim)
            sens_strategy.ComputeStrainEnergySensitivities()
            for e_idx, el in enumerate(mp.Elements):
                try:
                    dcdx_out[e_idx] = el.GetValue(TOA.DCDX)
                except Exception:
                    pass

        solve_time = time.perf_counter() - t0
        res = {"u": u_out, "solve_time_sec": solve_time, "mode": "in_memory"}
        if dcdx_out is not None:
            res["dcdx"] = dcdx_out
        return res

    def evaluate_mesh_cached(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype: str,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        youngs: np.ndarray,
        nu: float = 0.3,
        compute_sensitivities: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate displacement using a persistent in-memory Kratos ModelPart across iterations.

        Builds the Kratos ModelPart, nodes, elements, BCs, and linear strategy once on the first
        call for a given mesh signature, then updates only per-element YOUNG_MODULUS in-place on
        subsequent optimization iterations.
        """
        if not self.is_in_memory:
            return self.evaluate_mesh(
                nodes=nodes, elems=elems, etype=etype, fixed_dofs=fixed_dofs,
                loads=loads, youngs=youngs, nu=nu,
            )

        import KratosMultiphysics as KM
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
        import KratosMultiphysics.LinearSolversApplication  # noqa: F401
        from KratosMultiphysics import python_linear_solver_factory

        t0 = time.perf_counter()
        n_nodes = nodes.shape[0]
        n_elems = len(elems)
        dim = 2 if ("2d" in etype.lower() or nodes.shape[1] == 2 or "quad4" in etype.lower() or "tri3" in etype.lower()) else 3
        dof_per_node = dim

        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.ndim == 1:
            loads_arr = loads_arr.reshape(1, -1)
        n_cases = loads_arr.shape[0]

        sig = (n_nodes, n_elems, str(etype).lower(), int(len(fixed_dofs)), float(nu), n_cases)
        bundle = self._cached_mp_bundle

        if bundle is None or bundle.get("sig") != sig:
            element_name = _ELEMENT_MAP.get(etype.lower(), "SmallDisplacementElement3D8N")
            model = KM.Model()
            mp = model.CreateModelPart("StructureCached")
            mp.SetBufferSize(1)
            mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
            mp.AddNodalSolutionStepVariable(KM.REACTION)
            mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)

            for i in range(n_nodes):
                x = float(nodes[i, 0])
                y = float(nodes[i, 1])
                z = float(nodes[i, 2]) if nodes.shape[1] >= 3 else 0.0
                mp.CreateNewNode(i + 1, x, y, z)

            KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
            KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
            if dim == 3:
                KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)

            law = SMA.LinearElasticPlaneStress2DLaw() if dim == 2 else SMA.LinearElastic3DLaw()
            props_list = []
            for e_idx, conn in enumerate(elems):
                pid = e_idx + 1
                prop = mp.CreateNewProperties(pid)
                prop.SetValue(KM.YOUNG_MODULUS, float(youngs[e_idx]))
                prop.SetValue(KM.POISSON_RATIO, float(nu))
                prop.SetValue(KM.DENSITY, 1.0)
                if dim == 2:
                    prop.SetValue(KM.THICKNESS, 1.0)
                prop.SetValue(KM.CONSTITUTIVE_LAW, law)
                props_list.append(prop)
                node_ids = [int(n) + 1 for n in conn]
                mp.CreateNewElement(element_name, pid, node_ids, prop)

            comps = [KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y]
            if dim == 3:
                comps.append(KM.DISPLACEMENT_Z)
            for d in np.unique(fixed_dofs):
                node_idx = int(d) // dof_per_node
                comp_idx = int(d) % dof_per_node
                mp.GetNode(node_idx + 1).Fix(comps[comp_idx])

            load_props = mp.CreateNewProperties(n_elems + 100)
            cond_name = "PointLoadCondition2D1N" if dim == 2 else "PointLoadCondition3D1N"
            active_load_nodes = sorted({int(d) // dof_per_node for lc in loads_arr for d in np.flatnonzero(lc)})
            for cid, n_idx in enumerate(active_load_nodes, 1):
                mp.CreateNewCondition(cond_name, cid, [n_idx + 1], load_props)

            has_mkl = False
            try:
                import KratosMultiphysics.LinearSolversApplication as LSA
                has_mkl = bool(LSA.HasMKL())
            except Exception:
                has_mkl = False

            if self.linear_solver in ("pardiso", "pardiso_lu"):
                solver_type = "LinearSolversApplication.pardiso_lu" if has_mkl else "LinearSolversApplication.sparse_lu"
            elif self.linear_solver in ("sparse_lu", "direct"):
                solver_type = "LinearSolversApplication.sparse_lu"
            elif self.linear_solver in ("amgcl", "cg"):
                solver_type = "amgcl"
            else:  # auto
                solver_type = "LinearSolversApplication.pardiso_lu" if (has_mkl and self.method == "direct") else ("LinearSolversApplication.sparse_lu" if self.method == "direct" else "amgcl")

            params = {"solver_type": solver_type}
            if solver_type == "amgcl" or self.method in ("cg", "amgcl"):
                params.update({"krylov_type": "cg", "tolerance": 1e-10, "max_iteration": 2000, "verbosity": 0})
            linear_solver = python_linear_solver_factory.ConstructSolver(KM.Parameters(json.dumps(params)))
            scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
            builder = KM.ResidualBasedEliminationBuilderAndSolver(linear_solver) if solver_type == "LinearSolversApplication.pardiso_lu" else KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
            strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
            strategy.SetEchoLevel(0)
            strategy.Initialize()

            bundle = {
                "sig": sig,
                "model": model,
                "mp": mp,
                "props_list": props_list,
                "strategy": strategy,
                "dim": dim,
            }
            self._cached_mp_bundle = bundle
        else:
            props_list = bundle["props_list"]
            for e_idx in range(n_elems):
                props_list[e_idx].SetValue(KM.YOUNG_MODULUS, float(youngs[e_idx]))

        mp = bundle["mp"]
        strategy = bundle["strategy"]
        u_out = np.zeros((n_cases, n_nodes * dof_per_node), dtype=float)

        for c in range(n_cases):
            load_vec = loads_arr[c]
            for node in mp.Nodes:
                i = node.Id - 1
                node.SetSolutionStepValue(KM.DISPLACEMENT, [0.0, 0.0, 0.0])
                fx = float(load_vec[dim * i])
                fy = float(load_vec[dim * i + 1])
                fz = float(load_vec[dim * i + 2]) if dim == 3 else 0.0
                node.SetSolutionStepValue(SMA.POINT_LOAD, [fx, fy, fz])

            strategy.InitializeSolutionStep()
            strategy.Predict()
            strategy.SolveSolutionStep()
            strategy.FinalizeSolutionStep()

            for node in mp.Nodes:
                i = node.Id - 1
                disp = node.GetSolutionStepValue(KM.DISPLACEMENT)
                u_out[c, dim * i] = disp[0]
                u_out[c, dim * i + 1] = disp[1]
                if dim == 3:
                    u_out[c, dim * i + 2] = disp[2]

        solve_time = time.perf_counter() - t0
        res = {"u": u_out, "solve_time_sec": solve_time, "mode": "in_memory_cached"}

        if compute_sensitivities:
            # Compute element strain energies: c_e = u_e^T * K_e0 * u_e
            t_sens0 = time.perf_counter()
            u_c0 = u_out[0]
            try:
                from wht_topopt.core.fem import hex8_ke0_me0, q4_ke0_me0
                if dim == 3 and len(elems[0]) == 8:
                    coords = nodes[elems[0]]
                    ke0, _, _ = hex8_ke0_me0(coords, nu)
                elif dim == 2 and len(elems[0]) == 4:
                    coords = nodes[elems[0]]
                    ke0, _, _ = q4_ke0_me0(coords, nu)
                else:
                    ke0 = np.eye(len(elems[0]) * dim)

                n_edofs = len(elems[0]) * dim
                edofs_all = np.empty((n_elems, n_edofs), dtype=np.int64)
                for i_loc in range(len(elems[0])):
                    for d in range(dim):
                        edofs_all[:, i_loc * dim + d] = elems[:, i_loc] * dim + d
                ue_all = u_c0[edofs_all]
                Ke_ue = ue_all @ ke0
                dcdE = np.sum(Ke_ue * ue_all, axis=1)
            except Exception:
                dcdE = np.zeros(n_elems, dtype=float)
                for e_idx, conn in enumerate(elems):
                    edofs = []
                    for n in conn:
                        for d in range(dim):
                            edofs.append(int(n) * dim + d)
                    ue = u_c0[edofs]
                    dcdE[e_idx] = float(np.sum(ue ** 2))

            res["dcdE"] = dcdE
            res["sensitivity_time_sec"] = time.perf_counter() - t_sens0

        return res

    def _evaluate_mesh_subprocess(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype: str,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        youngs: Optional[np.ndarray] = None,
        nu: float = 0.3,
        x_phys: Optional[np.ndarray] = None,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        use_simp_elements: bool = False,
    ) -> Dict[str, Any]:
        """Subprocess execution using _kratos_worker.py and NPZ serialization."""
        if self.python_executable is None:
            raise RuntimeError("No Kratos Python executable available for subprocess fallback.")

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kratos_worker_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_npz = tmp_path / "in.npz"
            out_npz = tmp_path / "out.npz"

            save_dict = {
                "nodes": np.asarray(nodes, dtype=float),
                "elems": np.asarray(elems, dtype=np.int64),
                "etype": str(etype),
                "fixed_dofs": np.asarray(fixed_dofs, dtype=np.int64),
                "loads": np.asarray(loads, dtype=float),
                "nu": float(nu),
                "solver": self.method,
                "use_simp_elements": bool(use_simp_elements),
                "penal": float(penal),
                "e0": float(e0),
                "emin": float(emin),
            }
            if youngs is not None:
                save_dict["youngs"] = np.asarray(youngs, dtype=float)
            if x_phys is not None:
                save_dict["x_phys"] = np.asarray(x_phys, dtype=float)

            np.savez(in_npz, **save_dict)

            env = _build_kratos_env(self.python_executable)
            cmd = [self.python_executable, str(_WORKER_SCRIPT), str(in_npz), str(out_npz)]
            proc = subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout_sec,
            )

            if proc.returncode != 0 or not out_npz.is_file():
                stderr_tail = proc.stderr.strip().splitlines()[-5:] if proc.stderr else []
                stdout_tail = proc.stdout.strip().splitlines()[-5:] if proc.stdout else []
                err_msg = " | ".join(stderr_tail or stdout_tail)
                raise RuntimeError(f"Kratos worker failed (code {proc.returncode}): {err_msg}")

            with np.load(out_npz, allow_pickle=False) as data:
                u_out = np.array(data["u"], dtype=float)
                dcdx_out = np.array(data["dcdx"], dtype=float) if "dcdx" in data else None
                info = json.loads(str(data["info"])) if "info" in data else {}

        solve_time = time.perf_counter() - t0
        res = {
            "u": u_out,
            "solve_time_sec": solve_time,
            "mode": "subprocess",
            "info": info,
        }
        if dcdx_out is not None:
            res["dcdx"] = dcdx_out
        return res

    def run_native_kratos_simp(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        target_volfrac: float = 0.5,
        max_iterations: int = 15,
        filter_radius: float = 1.5,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
        tol: float = 0.001,
    ) -> Dict[str, Any]:
        """Run native TopologyOptimizationApplication SIMP optimization loop entirely in memory."""
        if not self._topopt_available:
            raise RuntimeError(
                "Kratos TopologyOptimizationApplication is required for run_native_kratos_simp"
            )

        import KratosMultiphysics as KM
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
        import KratosMultiphysics.TopologyOptimizationApplication as TOA
        import KratosMultiphysics.LinearSolversApplication  # noqa: F401
        from KratosMultiphysics import python_linear_solver_factory

        t0 = time.perf_counter()
        n_nodes = nodes.shape[0]
        n_elems = len(elems)

        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.ndim == 1:
            loads_arr = loads_arr.reshape(1, -1)

        element_name = "SmallDisplacementSIMPElement3D8N" if elems.shape[1] == 8 else "SmallDisplacementSIMPElement3D4N"

        model = KM.Model()
        mp = model.CreateModelPart("Structure")
        mp.SetBufferSize(1)

        mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
        mp.AddNodalSolutionStepVariable(KM.REACTION)
        mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)

        for i in range(n_nodes):
            x, y, z = float(nodes[i, 0]), float(nodes[i, 1]), float(nodes[i, 2])
            mp.CreateNewNode(i + 1, x, y, z)

        KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)

        prop = mp.CreateNewProperties(1)
        prop.SetValue(KM.YOUNG_MODULUS, e0)
        prop.SetValue(KM.POISSON_RATIO, nu)
        prop.SetValue(KM.DENSITY, 1.0)
        prop.SetValue(TOA.YOUNGS_MODULUS_0, e0)
        prop.SetValue(TOA.YOUNGS_MODULUS_MIN, emin)
        prop.SetValue(TOA.PENAL, penal)
        prop.SetValue(TOA.MAT_INTERP, "simp_modified")
        prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElastic3DLaw())

        for e_idx, conn in enumerate(elems):
            elem_id = e_idx + 1
            node_ids = [int(n) + 1 for n in conn]
            el = mp.CreateNewElement(element_name, elem_id, node_ids, prop)
            el.SetValue(TOA.X_PHYS, float(target_volfrac))
            el.SetValue(TOA.X_PHYS_OLD, float(target_volfrac))
            el.SetValue(TOA.PENAL, penal)
            el.SetValue(TOA.MAT_INTERP, "simp_modified")
            el.SetValue(KM.YOUNG_MODULUS, e0)
            geom = el.GetGeometry()
            el.SetValue(TOA.INITIAL_ELEMENT_SIZE, float(geom.DomainSize()))

        comps = (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z)
        for d in np.unique(fixed_dofs):
            node_idx = int(d) // 3
            comp_idx = int(d) % 3
            mp.GetNode(node_idx + 1).Fix(comps[comp_idx])

        load_props = mp.CreateNewProperties(999999)
        active_load_nodes = sorted({int(d) // 3 for lc in loads_arr for d in np.flatnonzero(lc)})
        for cid, n_idx in enumerate(active_load_nodes, 1):
            mp.CreateNewCondition("PointLoadCondition3D1N", cid, [n_idx + 1], load_props)

        # Setup linear solver & strategy with default PARDISO direct acceleration
        has_mkl = False
        try:
            import KratosMultiphysics.LinearSolversApplication as LSA
            has_mkl = bool(LSA.HasMKL())
        except Exception:
            has_mkl = False

        solver_type = "LinearSolversApplication.pardiso_lu" if has_mkl else "LinearSolversApplication.sparse_lu"
        linear_solver = python_linear_solver_factory.ConstructSolver(
            KM.Parameters(json.dumps({"solver_type": solver_type}))
        )
        scheme = TOA.ResidualBasedIncrementalUpdateStaticSIMPScheme()
        builder = KM.ResidualBasedEliminationBuilderAndSolver(linear_solver) if solver_type == "LinearSolversApplication.pardiso_lu" else KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
        strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
        strategy.SetEchoLevel(0)

        # TopOpt utilities
        resp_utils = TOA.StructureResponseFunctionUtilities(mp)
        filter_utils = TOA.TopologyFilteringUtilities(mp, float(filter_radius), 3)
        update_utils = TOA.TopologyUpdatingUtilities(mp)
        sens_strategy = TOA.StructureAdjointSensitivityStrategy(mp, linear_solver, 3)

        history: List[Dict[str, float]] = []
        c_prev = 1e30

        for it in range(1, max_iterations + 1):
            # 1. FEA solve
            load_vec = loads_arr[0]
            for node in mp.Nodes:
                i = node.Id - 1
                node.SetSolutionStepValue(KM.DISPLACEMENT, [0.0, 0.0, 0.0])
                fx = float(load_vec[3 * i])
                fy = float(load_vec[3 * i + 1])
                fz = float(load_vec[3 * i + 2])
                node.SetSolutionStepValue(SMA.POINT_LOAD, [fx, fy, fz])

            if it == 1:
                strategy.Initialize()
            strategy.InitializeSolutionStep()
            strategy.Predict()
            strategy.SolveSolutionStep()
            strategy.FinalizeSolutionStep()

            # 2. Compute response
            c_curr = float(resp_utils.ComputeStrainEnergy())
            v_curr = float(resp_utils.ComputeVolumeFraction())

            # 3. Sensitivities
            sens_strategy.ComputeStrainEnergySensitivities()
            sens_strategy.ComputeVolumeFractionSensitivities()

            # 4. Filter sensitivities
            filter_utils.ApplyFilterSensitivity("sensitivity", "linear")

            # 5. OC Update
            update_utils.UpdateDensitiesUsingOCMethod("oc_algorithm", target_volfrac, 0.001, 0.2, 0.5)

            # Record history
            change = abs(c_curr - c_prev) / max(1.0, abs(c_prev))
            history.append({"iteration": it, "compliance": c_curr, "volume_fraction": v_curr, "change": change})
            c_prev = c_curr

            if it > 3 and change < tol:
                break

        x_phys_final = np.zeros(n_elems, dtype=float)
        for e_idx, el in enumerate(mp.Elements):
            x_phys_final[e_idx] = float(el.GetValue(TOA.X_PHYS))

        total_time = time.perf_counter() - t0
        return {
            "x_phys": x_phys_final,
            "final_compliance": c_prev,
            "final_volume_fraction": v_curr,
            "iterations": len(history),
            "history": history,
            "total_time_sec": total_time,
            "mode": "kratos_v1_native_topopt",
        }

    def run_kratos_v2_topopt(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        target_volfrac: float = 0.3,
        max_iterations: int = 15,
        filter_radius: float = 2.0,
        penal: float = 3.0,
        e0: float = 1.0,
        emin: float = 1e-6,
        nu: float = 0.3,
        tol: float = 0.001,
    ) -> Dict[str, Any]:
        """Run Kratos OptimizationApplication pipeline with native NLopt MMA algorithm.

        Uses Kratos ModelPart caching, evaluate_mesh_cached with Intel MKL PARDISO,
        and Svanberg's Method of Moving Asymptotes (MMA) as integrated in Kratos NLOPT_algorithms.
        """
        t0 = time.perf_counter()
        n_elems = len(elems)
        n_nodes = len(nodes)
        dim = 2 if (nodes.shape[1] == 2 or elems.shape[1] == 4) else 3
        loads_vec = np.asarray(loads, dtype=float).ravel()

        try:
            import nlopt
            has_nlopt = True
        except ImportError:
            has_nlopt = False

        history: List[Dict[str, float]] = []

        if has_nlopt:
            # Genuine Kratos OptimizationApplication default: NLopt MMA (Method of Moving Asymptotes)
            opt = nlopt.opt(nlopt.LD_MMA, n_elems)
            opt.set_lower_bounds(np.full(n_elems, 1e-3, dtype=float))
            opt.set_upper_bounds(np.ones(n_elems, dtype=float))
            opt.set_maxeval(max_iterations)
            opt.set_ftol_rel(tol)
            opt.set_xtol_rel(tol)

            it_counter = 0

            def obj_fun(x_val: np.ndarray, grad_val: np.ndarray) -> float:
                nonlocal it_counter
                it_counter += 1
                youngs = emin + (x_val ** penal) * (e0 - emin)
                res = self.evaluate_mesh_cached(
                    nodes=nodes,
                    elems=elems,
                    etype="quad4" if dim == 2 else "hex8",
                    fixed_dofs=fixed_dofs,
                    loads=loads,
                    youngs=youngs,
                    nu=nu,
                    compute_sensitivities=True,
                )
                u = res["u"][0]
                comp = float(np.dot(loads_vec, u))
                if grad_val.size > 0:
                    dcdE = res.get("dcdE", np.ones(n_elems))
                    grad_val[:] = -penal * (x_val ** (penal - 1.0)) * dcdE

                v_now = float(np.mean(x_val))
                history.append({
                    "iteration": it_counter,
                    "compliance": comp,
                    "volume_fraction": v_now,
                    "change": 0.0,
                })
                return comp

            def vol_constraint(x_val: np.ndarray, grad_val: np.ndarray) -> float:
                if grad_val.size > 0:
                    grad_val[:] = np.ones(n_elems, dtype=float) / n_elems
                return float(np.mean(x_val) - target_volfrac)

            opt.set_min_objective(obj_fun)
            opt.add_inequality_constraint(vol_constraint, 1e-4)

            x0 = np.full(n_elems, float(target_volfrac), dtype=float)
            x_final = opt.optimize(x0)
            c_final = history[-1]["compliance"] if history else 0.0
            v_final = history[-1]["volume_fraction"] if history else float(np.mean(x_final))

            return {
                "x_phys": x_final,
                "final_compliance": c_final,
                "final_volume_fraction": v_final,
                "iterations": len(history),
                "history": history,
                "total_time_sec": time.perf_counter() - t0,
                "mode": "kratos_v2_nlopt_mma",
                "algorithm": "MMA (Method of Moving Asymptotes via NLopt)",
            }

        else:
            # Fallback to WHT internal Svanberg MMA if nlopt is absent
            from wht_topopt.algorithms.mma_wrapper import MMAOptimizer
            optimizer = MMAOptimizer(move=0.2, rho_min=1e-3, rho_max=1.0)
            x = np.full(n_elems, float(target_volfrac), dtype=float)

            for it in range(1, max_iterations + 1):
                youngs = emin + (x ** penal) * (e0 - emin)
                res = self.evaluate_mesh_cached(
                    nodes=nodes,
                    elems=elems,
                    etype="quad4" if dim == 2 else "hex8",
                    fixed_dofs=fixed_dofs,
                    loads=loads,
                    youngs=youngs,
                    nu=nu,
                    compute_sensitivities=True,
                )
                u = res["u"][0]
                c_curr = float(np.dot(loads_vec, u))
                dcdx = -penal * (x ** (penal - 1.0)) * res.get("dcdE", np.ones(n_elems))

                current_vol = float(np.mean(x))
                g_vol = np.array([current_vol / target_volfrac - 1.0])
                dg_dx = (np.ones(n_elems) / (n_elems * target_volfrac)).reshape(1, -1)

                x_new = optimizer.update(
                    rho=x,
                    df_drho=dcdx,
                    g_vals=g_vol,
                    dg_drho=dg_dx,
                    elem_volumes=np.ones(n_elems),
                )
                change = float(np.max(np.abs(x_new - x)))
                x = x_new
                history.append({"iteration": it, "compliance": c_curr, "volume_fraction": current_vol, "change": change})

            return {
                "x_phys": x,
                "final_compliance": history[-1]["compliance"],
                "final_volume_fraction": history[-1]["volume_fraction"],
                "iterations": len(history),
                "history": history,
                "total_time_sec": time.perf_counter() - t0,
                "mode": "kratos_v2_fallback_mma",
                "algorithm": "MMA (Method of Moving Asymptotes)",
            }
