#!/usr/bin/env python3
"""Cross Performance Evaluator: Static & Dynamic Evaluations for Static & Dynamic TopOpt Models.

Strict Anti-Mocking: Runs real FEA static solve and dynamic time-integration
to build the 2x2 Cross Evaluation Matrix:
              | Static Evaluation (Compliance) | Dynamic Evaluation (Transient Deflection/Vibration)
--------------+--------------------------------+---------------------------------------------------
Static Model  | Real K_stat * u = F            | Real Chung-Hulbert Gen-Alpha 70ms
Dynamic Model | Real K_dyn * u = F             | Real Chung-Hulbert Gen-Alpha 70ms
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import scipy.sparse as sp
import pypardiso

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Enforce UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wht_topopt.core.fem import FEMEngine
from wht_topopt.examples.ex09_esl_dynamic_robot_arm import RobotArmDomain3D
from wht_topopt.examples.ex09_compare_dynamic_vs_static import (
    RobotArmStaticOptimizer3D,
    DynamicVsStaticCrossEvaluator,
)


def evaluate_cross_matrix(output_dir_str: str = "wht_topopt/examples/outputs/ex09_rev1") -> Dict[str, Any]:
    print("=" * 80)
    print(f" [*] RUNNING RIGOROUS 2x2 STATIC & DYNAMIC CROSS EVALUATION (Anti-Mocking: {output_dir_str})")
    print("=" * 80, flush=True)

    domain = RobotArmDomain3D()
    output_dir = Path(output_dir_str)

    # 1. Load density fields
    static_rho_path = output_dir / "static" / "robot_arm_static_rho.npy"
    dynamic_rho_path = output_dir / "builtin" / "robot_arm_ex09_builtin_rho.npy"

    if not static_rho_path.exists() or not dynamic_rho_path.exists():
        raise FileNotFoundError("Density files missing!")

    rho_static = np.load(static_rho_path)
    rho_dynamic = np.load(dynamic_rho_path)

    # 2. Material & FEM engine
    e0 = 71000.0   # MPa
    emin = 71.0
    nu = 0.33
    rho_mat = 2.81e-9 # ton/mm^3
    penal = 3.0
    free = domain.free_dofs
    elem_vol = domain.mesh.elem_volumes
    vol_total = float(np.sum(elem_vol))

    fem = FEMEngine(domain.mesh, nu=nu, e0=e0, emin=emin)
    pardiso = pypardiso.PyPardisoSolver(mtype=11)

    # Setup 4 Peak Static Load Cases
    static_opt = RobotArmStaticOptimizer3D(domain=domain, target_vol=0.25, max_iter=1, output_dir=output_dir / "temp")
    f_static_cases, case_names = static_opt._setup_peak_static_load_cases()
    n_cases = len(f_static_cases)
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    # -------------------------------------------------------------------------
    # PART A: PURE STATIC EVALUATION (정해석 평가)
    # -------------------------------------------------------------------------
    print("\n[PART A] Evaluating Pure Static Equilibrium (4 Peak Load Cases)...", flush=True)

    def evaluate_static_model(rho_field: np.ndarray, label: str):
        youngs = emin + (rho_field ** penal) * (e0 - emin)
        K = fem.assemble_K(youngs)
        K_ff = K[free, :][:, free].tocsr()
        F_ff = np.asfortranarray(f_static_cases[:, free].T, dtype=np.float64)

        u_ff = pardiso.solve(K_ff, F_ff)
        u_cases = np.zeros((n_cases, domain.ndof), dtype=float)
        u_cases[:, free] = u_ff.T

        compliance_per_case = []
        tip_disp_per_case = []
        tot_comp = 0.0

        tip_nodes = domain.loaded_nodes

        for c in range(n_cases):
            fc = f_static_cases[c]
            uc = u_cases[c]
            comp_c = float(np.dot(fc, uc))  # F(N) * u(mm) = N*mm = mJ
            tot_comp += weights[c] * comp_c
            compliance_per_case.append(comp_c)

            # Tip displacement norm in mm
            tip_disp_mag = np.mean([np.linalg.norm(uc[3*nid:3*nid+3]) for nid in tip_nodes])
            tip_disp_per_case.append(float(tip_disp_mag))

        vol_frac = float(np.sum(rho_field * elem_vol) / vol_total)

        print(f"  * {label}: Weighted Compliance = {tot_comp:8.2f} mJ | Volume = {vol_frac*100:5.2f}%", flush=True)
        for c in range(n_cases):
            print(f"      {case_names[c]}: Comp = {compliance_per_case[c]:8.2f} mJ, Tip Disp = {tip_disp_per_case[c]:.4f} mm", flush=True)

        return {
            "volume_fraction": vol_frac,
            "weighted_compliance_mJ": tot_comp,
            "compliance_per_case_mJ": compliance_per_case,
            "tip_disp_per_case_mm": tip_disp_per_case,
        }

    static_res_for_static_model = evaluate_static_model(rho_static, "Static Model (정적 최적화 형상)")
    static_res_for_dynamic_model = evaluate_static_model(rho_dynamic, "Dynamic Model (동적 F-ESL 형상)")

    # -------------------------------------------------------------------------
    # PART B: TRANSIENT DYNAMIC EVALUATION (동해석 평가)
    # -------------------------------------------------------------------------
    print("\n[PART B] Running Transient Dynamic & Modal Analysis (70ms S-Curve + Free Vibration)...", flush=True)
    dyn_evaluator = DynamicVsStaticCrossEvaluator(domain=domain, rho_static=rho_static, rho_dynamic=rho_dynamic, output_dir=output_dir)
    dyn_summary = dyn_evaluator.evaluate_and_plot()

    # Combine into comprehensive report data
    report_data = {
        "metadata": {
            "date": "2026-11-01",
            "domain": "Robot Arm Link (160 x 48 x 40 mm)",
            "elements": 24000,
            "dofs": 80325,
            "material": "Al 7075-T6 (E=71,000 MPa, nu=0.33, rho=2.81 g/cm^3)",
            "target_volume": 0.25,
        },
        "static_evaluation": {
            "case_names": case_names,
            "static_model": static_res_for_static_model,
            "dynamic_model": static_res_for_dynamic_model,
            "compliance_diff_pct": (static_res_for_dynamic_model["weighted_compliance_mJ"] - static_res_for_static_model["weighted_compliance_mJ"]) / static_res_for_static_model["weighted_compliance_mJ"] * 100.0,
        },
        "dynamic_evaluation": dyn_summary,
    }

    report_json_path = output_dir / "comprehensive_cross_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n [+] Comprehensive Cross Evaluation JSON saved: {report_json_path}", flush=True)
    return report_data


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "wht_topopt/examples/outputs/ex09_rev1"
    evaluate_cross_matrix(out_dir)
