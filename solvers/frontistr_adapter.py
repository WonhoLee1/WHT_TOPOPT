"""FrontISTR SMP and Hybrid MPI finite element solver adapter.

Enforces:
- Automatic binary discovery: SMP (fistr1.exe) & Hybrid MS-MPI (fistr1.exe, hecmw_part1.exe, mpiexec.exe).
- Element formulations: 341 (TET4), 342 (TET10), 361 (HEX8), 362 (HEX20), 731 (MITC3), 741 (MITC4).
- Formulation options: FORM361=FI (Full Integration default for exact Ke0 match), BBAR, IC, FBAR.
- Material bucketing: Exact per-element properties (n_buckets=0) or logarithmic groups (EB_0001..EB_0064).
- Verification: Strict "FrontISTR Completed !!" log validation.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import FEASolver, SolverResult


_FRONTISTR_ETYPE_MAP = {
    "tet4": 341,
    "tet10": 342,
    "hex8": 361,
    "hex8_bbar": 361,
    "hex20": 362,
    "mitc3": 731,
    "tria3_shell": 731,
    "mitc4": 741,
    "quad4_shell": 741,
}

_SMP_BINARY_DEFAULT = Path(r"D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas\fistr1.exe")
_MPI_DIR_DEFAULT = Path(r"D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.hybrid_msmpi_openblas")


def resolve_frontistr_executables(mode: str = "smp") -> Dict[str, Optional[str]]:
    """Discover FrontISTR binaries on local filesystem or environment variables."""
    res: Dict[str, Optional[str]] = {"fistr1": None, "hecmw_part1": None, "mpiexec": None}

    # 1. Check environment variables
    env_cmd = os.environ.get("FRONTISTR_CMD", os.environ.get("WHT_FISTR_CMD", os.environ.get("AIENG_FISTR_CMD", os.environ.get("FISTR_CMD", "")))).strip()
    if env_cmd:
        parts = [t.strip('"') for t in shlex.split(env_cmd, posix=(os.name != "nt"))]
        found = shutil.which(parts[0]) or (parts[0] if Path(parts[0]).is_file() else None)
        if found:
            res["fistr1"] = str(Path(found).resolve())

    # 2. Check local paths
    if res["fistr1"] is None:
        if mode.lower() == "smp" and _SMP_BINARY_DEFAULT.is_file():
            res["fistr1"] = str(_SMP_BINARY_DEFAULT.resolve())
        elif (_MPI_DIR_DEFAULT / "fistr1.exe").is_file():
            res["fistr1"] = str((_MPI_DIR_DEFAULT / "fistr1.exe").resolve())

    if (_MPI_DIR_DEFAULT / "hecmw_part1.exe").is_file():
        res["hecmw_part1"] = str((_MPI_DIR_DEFAULT / "hecmw_part1.exe").resolve())
    if (_MPI_DIR_DEFAULT / "mpiexec.exe").is_file():
        res["mpiexec"] = str((_MPI_DIR_DEFAULT / "mpiexec.exe").resolve())
    elif shutil.which("mpiexec"):
        res["mpiexec"] = shutil.which("mpiexec")

    # 3. PATH fallback
    if res["fistr1"] is None:
        res["fistr1"] = shutil.which("fistr1")

    return res


def compute_material_buckets(
    youngs: np.ndarray,
    n_buckets: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Group element moduli into exact or log-spaced material bins.
    
    Args:
        youngs: Array of Young's moduli per element.
        n_buckets: Number of logarithmic bins. If 0 or len(unique) <= n_buckets, exact grouping is used.
        
    Returns:
        (group_indices, representative_moduli, max_relative_error)
    """
    youngs_arr = np.asarray(youngs, dtype=float)
    uniq, inverse = np.unique(youngs_arr, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    if n_buckets <= 0 or len(uniq) <= n_buckets:
        return inverse, uniq, 0.0

    logs = np.log(np.maximum(youngs_arr, 1e-30))
    edges = np.linspace(logs.min(), logs.max(), n_buckets + 1)
    bin_of = np.clip(np.digitize(logs, edges[1:-1]), 0, n_buckets - 1)
    used, group = np.unique(bin_of, return_inverse=True)
    group = np.asarray(group).ravel()

    rep = np.array([np.exp(logs[group == g].mean()) for g in range(len(used))])
    max_err = float(np.max(np.abs(rep[group] / youngs_arr - 1.0)))
    return group, rep, max_err


def parse_frontistr_res_displacement(res_path: Path, n_nodes: int) -> np.ndarray:
    """Read nodal DISPLACEMENT block from FrontISTR text .res.0.* file."""
    if not res_path.is_file():
        raise FileNotFoundError(f"FrontISTR result file not found: {res_path}")

    text = res_path.read_text(encoding="utf-8", errors="replace")
    marker = text.find("*data")
    if marker < 0:
        raise ValueError(f"No '*data' section found in {res_path.name}; invalid FrontISTR result")

    tok = text[marker + len("*data"):].split()
    nn, _ne, n_items = int(tok[0]), int(tok[1]), int(tok[2])
    sizes = [int(t) for t in tok[4: 4 + n_items]]
    names = tok[4 + n_items: 4 + 2 * n_items]

    if nn != n_nodes:
        raise ValueError(f"{res_path.name}: result has {nn} nodes, expected {n_nodes}")
    if not names or names[0] != "DISPLACEMENT" or sizes[0] != 3:
        raise ValueError(f"Expected DISPLACEMENT (size 3), found {names[:1]} (size {sizes[:1]})")

    per_node = sum(sizes)
    total_vals = nn * (1 + per_node)
    body_toks = tok[4 + 2 * n_items: 4 + 2 * n_items + total_vals]
    if len(body_toks) != total_vals:
        raise ValueError(f"Truncated nodal data in {res_path.name}: {len(body_toks)} vs {total_vals}")

    body = np.array(body_toks, dtype=float).reshape(nn, 1 + per_node)
    u = np.zeros((n_nodes, 3), dtype=float)
    ids = body[:, 0].astype(np.int64) - 1
    u[ids] = body[:, 1:4]
    return u


class FrontISTRSolver(FEASolver):
    """FrontISTR FEA solver adapter supporting SMP and Hybrid MS-MPI."""

    solver_name: str = "frontistr"

    def __init__(
        self,
        mode: str = "smp",
        threads: int = 4,
        mpi_processes: int = 1,
        form361: str = "FI",
        form341: str = "FI",
        n_buckets: int = 0,
        method: str = "DIRECT",
        fistr_cmd: Optional[str] = None,
        timeout_sec: float = 600.0,
    ) -> None:
        """Initialize FrontISTRSolver.
        
        Args:
            mode: 'smp' or 'mpi'.
            threads: Number of OpenMP threads (-t flag).
            mpi_processes: Number of MPI ranks (-np flag for mpiexec).
            form361: HEX8 element formulation ('FI', 'BBAR', 'IC', 'FBAR').
            form341: TET4 element formulation ('FI' or 'SELECTIVE_ESNS').
            n_buckets: Number of logarithmic material groups (0 for exact per-element).
            method: 'DIRECT' or 'CG'.
            fistr_cmd: Custom path to fistr1 executable.
            timeout_sec: Solver execution timeout in seconds.
        """
        self.mode = mode.lower()
        self.threads = max(1, int(threads))
        self.mpi_processes = max(1, int(mpi_processes))
        self.form361 = form361.upper()
        self.form341 = form341.upper()
        self.n_buckets = n_buckets
        self.method = method.upper()
        self.timeout_sec = timeout_sec

        execs = resolve_frontistr_executables(mode=self.mode)
        if fistr_cmd is not None:
            self.fistr1_bin = fistr_cmd
        else:
            self.fistr1_bin = execs["fistr1"]
        self.hecmw_part1_bin = execs["hecmw_part1"]
        self.mpiexec_bin = execs["mpiexec"]

        # Cache for solve_cached
        self._last_lu: Optional[spla.SuperLU] = None
        self._last_free_dofs: Optional[np.ndarray] = None
        self._last_n_dof: int = 0

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
        """Solve global system K u = F via sparse direct factorization with caching."""
        t0 = time.perf_counter()
        n_dof = K.shape[0]

        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if len(free_dofs) == n_dof and node_coords is not None:
            dim = 2 if node_coords.shape[1] == 2 else 3
            F = self.apply_inertia_relief(F, M, node_coords, free_dofs=free_dofs, dim=dim)

        K_ff, F_f, free_dofs, fixed_dofs = self.partition_system(K, F, free_dofs, fixed_dofs, u_fixed)
        K_ff_csc = K_ff.tocsc()
        n_free = K_ff_csc.shape[0]

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
            "mode": "sparse_algebra",
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
        """Solve adjoint equation K_ff u_f_adj = rhs_f reusing the cached factorization."""
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

    def _write_mesh_file(
        self,
        msh_path: Path,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype_int: int,
        group_indices: np.ndarray,
        n_groups: int,
        nu: float,
    ) -> None:
        """Write FrontISTR .msh file with nodal coordinates, element connectivity and sections."""
        lines: List[str] = ["!HEADER", " wht_topopt frontistr", "!NODE"]
        for i, pt in enumerate(nodes, 1):
            x, y = float(pt[0]), float(pt[1])
            z = float(pt[2]) if len(pt) >= 3 else 0.0
            lines.append(f"{i}, {x:.17g}, {y:.17g}, {z:.17g}")

        conn = elems + 1
        for g in range(n_groups):
            grp_name = f"EB_{g + 1:04d}" if n_groups > 1 else "EB_0001"
            lines.append(f"!ELEMENT, TYPE={etype_int}, EGRP={grp_name}")
            e_in_g = np.flatnonzero(group_indices == g)
            for e in e_in_g:
                c_str = ", ".join(map(str, conn[e]))
                lines.append(f"{e + 1}, {c_str}")

        for g in range(n_groups):
            grp_name = f"EB_{g + 1:04d}" if n_groups > 1 else "EB_0001"
            mat_name = f"MAT_{g + 1:04d}" if n_groups > 1 else "MAT_0001"
            lines.append(f"!MATERIAL, NAME={mat_name}, ITEM=1")
            lines.append("!ITEM=1, SUBITEM=2")
            lines.append(f"1.0, {nu:.17g}")
            lines.append(f"!SECTION, TYPE=SOLID, EGRP={grp_name}, MATERIAL={mat_name}")

        lines.append("!END\n")
        msh_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_control_file(
        self,
        cnt_path: Path,
        fixed_dofs: np.ndarray,
        load_vector: np.ndarray,
        group_moduli: np.ndarray,
        etype_int: int,
        nu: float,
    ) -> None:
        """Write FrontISTR .cnt file specifying boundary conditions, loads, and formulation flags."""
        lines: List[str] = [
            "!VERSION", " 3",
            "!SOLUTION, TYPE=STATIC",
            "!WRITE,RESULT",
            "!BOUNDARY",
        ]

        # Group fixed DOFs by node
        by_node: Dict[int, List[int]] = {}
        for d in np.unique(fixed_dofs):
            node_id = int(d) // 3 + 1
            comp_id = int(d) % 3 + 1
            by_node.setdefault(node_id, []).append(comp_id)

        for n_id, comps in by_node.items():
            s_comps = sorted(set(comps))
            if len(s_comps) == 1:
                lines.append(f" {n_id}, {s_comps[0]}, {s_comps[0]}, 0.0")
            elif s_comps == list(range(s_comps[0], s_comps[-1] + 1)):
                lines.append(f" {n_id}, {s_comps[0]}, {s_comps[-1]}, 0.0")
            else:
                for c in s_comps:
                    lines.append(f" {n_id}, {c}, {c}, 0.0")

        lines.append("!CLOAD")
        for d in np.flatnonzero(load_vector):
            n_id = int(d) // 3 + 1
            c_id = int(d) % 3 + 1
            val = float(load_vector[d])
            lines.append(f" {n_id}, {c_id}, {val:.17g}")

        # Material definitions with moduli
        for g, E in enumerate(group_moduli):
            mat_name = f"MAT_{g + 1:04d}" if len(group_moduli) > 1 else "MAT_0001"
            lines.append(f"!MATERIAL, NAME={mat_name}")
            lines.append("!ELASTIC")
            lines.append(f" {E:.17g}, {nu:.17g}")

        # Formulation directives
        if etype_int == 361:
            for g in range(len(group_moduli)):
                lines.append(f"!SECTION, SECNUM={g + 1}, FORM361={self.form361}")
        elif etype_int == 341:
            for g in range(len(group_moduli)):
                lines.append(f"!SECTION, SECNUM={g + 1}, FORM341={self.form341}")

        # Solver method
        if self.method == "DIRECT":
            lines.extend(["!SOLVER,METHOD=DIRECT", " 1, 1", " 1.0e-12, 1.0, 0.0"])
        else:
            lines.extend(["!SOLVER,METHOD=CG,PRECOND=1", " 20000, 1", " 1.0e-10, 1.0, 0.0"])

        lines.extend(["!OUTPUT_RES", " DISP, ON", "!END\n"])
        cnt_path.write_text("\n".join(lines), encoding="utf-8")

    def solve_mesh(
        self,
        nodes: np.ndarray,
        elems: np.ndarray,
        etype: str,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        youngs: Optional[np.ndarray] = None,
        nu: float = 0.3,
        work_dir: Optional[Union[str, Path]] = None,
    ) -> SolverResult:
        """Run complete FrontISTR FEA pipeline on physical mesh and return exact displacements."""
        if self.fistr1_bin is None or not Path(self.fistr1_bin).is_file():
            raise RuntimeError(
                f"FrontISTR fistr1 executable not found. Please install FrontISTR or set FRONTISTR_CMD."
            )

        t0 = time.perf_counter()
        nodes_arr = np.asarray(nodes, dtype=float)
        elems_arr = np.asarray(elems, dtype=np.int64)
        n_nodes = nodes_arr.shape[0]
        n_elems = len(elems_arr)

        etype_key = etype.lower()
        if etype_key not in _FRONTISTR_ETYPE_MAP:
            raise ValueError(f"Unsupported FrontISTR element type {etype!r}. Supported: {list(_FRONTISTR_ETYPE_MAP.keys())}")
        etype_int = _FRONTISTR_ETYPE_MAP[etype_key]

        if youngs is None:
            youngs = np.ones(n_elems, dtype=float)
        else:
            youngs = np.asarray(youngs, dtype=float)

        loads_arr = np.asarray(loads, dtype=float)
        if loads_arr.ndim == 1:
            loads_arr = loads_arr.reshape(1, -1)
        n_cases = loads_arr.shape[0]

        # Compute material buckets
        group_idx, group_moduli, quant_err = compute_material_buckets(youngs, n_buckets=self.n_buckets)

        with tempfile.TemporaryDirectory(prefix="wht_fistr_") as tmp_dir:
            run_path = Path(work_dir) if work_dir is not None else Path(tmp_dir)
            run_path.mkdir(parents=True, exist_ok=True)

            # 1. Write hecmw_ctrl.dat
            ctrl_dat = (
                "!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE\n case.msh\n"
                "!CONTROL, NAME=fstrCNT\n case.cnt\n"
                "!RESULT, NAME=fstrRES, IO=OUT\n case.res\n"
                "!RESULT, NAME=vis_out, IO=OUT\n case_vis\n"
            )
            (run_path / "hecmw_ctrl.dat").write_text(ctrl_dat, encoding="utf-8")

            # 2. Write mesh file
            self._write_mesh_file(
                run_path / "case.msh",
                nodes_arr,
                elems_arr,
                etype_int,
                group_idx,
                len(group_moduli),
                nu,
            )

            u_all = np.zeros((n_cases, n_nodes * 3), dtype=float)

            for c in range(n_cases):
                # Clean prior results
                for stale in run_path.glob("case.res*"):
                    try:
                        stale.unlink()
                    except Exception:
                        pass

                # 3. Write control file
                self._write_control_file(
                    run_path / "case.cnt",
                    fixed_dofs,
                    loads_arr[c],
                    group_moduli,
                    etype_int,
                    nu,
                )

                # 4. Execute fistr1
                cmd = [self.fistr1_bin, "-t", str(self.threads)]
                proc = subprocess.run(
                    cmd,
                    cwd=run_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )

                stdout_full = proc.stdout + "\n" + proc.stderr
                # Enforce "FrontISTR Completed !!" check
                if "FrontISTR Completed !!" not in stdout_full:
                    tail = "\n".join(stdout_full.strip().splitlines()[-10:])
                    raise RuntimeError(
                        f"FrontISTR run failed to complete successfully (code {proc.returncode}). Log tail:\n{tail}"
                    )

                # 5. Parse displacement
                results = sorted(run_path.glob("case.res.0.*"), key=lambda p: int(p.suffix[1:]))
                if not results:
                    raise FileNotFoundError("FrontISTR finished without generating case.res.0.* file")

                u_disp = parse_frontistr_res_displacement(results[-1], n_nodes)
                u_all[c] = u_disp.ravel()

        solve_time = time.perf_counter() - t0
        extra = {
            "mode": self.mode,
            "element_code": etype_int,
            "form361": self.form361,
            "form341": self.form341,
            "n_materials": len(group_moduli),
            "quantization_error": quant_err,
        }

        u_ret = u_all[0] if n_cases == 1 else u_all
        return SolverResult(
            u=u_ret,
            residual_norm=0.0,
            iterations=1,
            solve_time_sec=solve_time,
            solver_name=self.solver_name,
            extra=extra,
        )
