"""GE bracket SE-AHCA study: Heaviside projection + single signal smoothing, signal exponent rho^p vs rho^(p-1).

Variants (same STL domain, loads, volume fraction and iteration count):
    base  : previous ex07 SE-AHCA (rho^p signal, no density filter, 2-step moving average, no projection)
    p     : projection + density-filter smoothing, signal ~ rho^p * u0
    pm1   : projection + density-filter smoothing, signal ~ rho^(p-1) * u0

The table lists, per variant, mass, weighted compliance, per-case max von Mises (not only 99.5%) and per-case 99.5% von Mises
of the 0/1 design (``evaluate_binary_design``), plus grayness of the final physical density and run time.

Output: docs/benchmarks/04_ge_se_ahca_signal_study_20261102.md (+ .json)

Usage:
    python wht_topopt/examples/ex07_se_ahca_signal_study.py --mesh_control 60 --iters 100
    python wht_topopt/examples/ex07_se_ahca_signal_study.py --mesh_control 40 --iters 4 --tag smoke   # quick check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ex07_ge_bracket_advanced_algorithms as ex07  # noqa: E402
from ex07_ge_bracket_advanced_algorithms import (  # noqa: E402
    CASE_GE,
    ROOT,
    SeAhcaOptions,
    BuiltinAdapter,
    build_stl_problem,
    reference_stiffness,
    run_single_algorithm,
)

DOCS_DIR = ROOT / "wht_topopt" / "docs" / "benchmarks"

VARIANTS: Dict[str, SeAhcaOptions] = {
    "base": SeAhcaOptions(signal_exponent="p", projection=False, smoothing="moving_average", rmin_factor=0.0),
    "p": SeAhcaOptions(signal_exponent="p"),
    "pm1": SeAhcaOptions(signal_exponent="p_minus_1"),
}


def run_variant(key: str, opts: SeAhcaOptions, mesh_control: int, volfrac: float, iters: int, out_root: Path) -> Dict[str, Any]:
    """Run one SE-AHCA variant on the GE bracket and collect metrics.

    Args:
        key: Variant id.
        opts: SE-AHCA options.
        mesh_control: Elements along the longest bounding-box side.
        volfrac: Target volume fraction of the design domain.
        iters: SE-AHCA iterations.
        out_root: Output root directory (a sub directory per variant is created).

    Returns:
        Metrics dict (binary evaluation + grayness + timing).
    """
    p_base, inside = build_stl_problem(CASE_GE, mesh_control, volfrac, base=ROOT / "external" / "freeto_stls")
    h = float(p_base.mesh.nodes[1, 0] - p_base.mesh.nodes[0, 0])
    vol = reference_stiffness(p_base.mesh, p_base.nu).volume
    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_single_algorithm(
        tag="se_ahca",
        name=f"SE-AHCA [{key}]",
        p_base=p_base,
        adapter=BuiltinAdapter(method="direct"),
        inside=inside,
        vol=vol,
        h=h,
        max_iters=iters,
        volfrac=volfrac,
        out_dir=out_dir,
        force_rerun=True,
        se_opts=opts,
    )
    m = dict(res["metrics"])
    x = np.clip(res["xphys"][inside], 0.0, 1.0)
    m["variant"] = key
    m["options"] = opts.label()
    m["grayness_inside"] = round(float(np.mean(x * (1.0 - x))), 4)
    m["final_weighted_comp_history"] = round(float(res["history_comp"][-1]), 1)
    return m


def write_markdown(rows: List[Dict[str, Any]], mesh_control: int, volfrac: float, iters: int, path: Path) -> None:
    """Write the comparison tables.

    Args:
        rows: Metrics dicts from ``run_variant``.
        mesh_control: Mesh control used.
        volfrac: Volume fraction used.
        iters: Iterations used.
        path: Output markdown path.
    """
    lines = [
        "# GE 브라켓 SE-AHCA: 투영 + 단일 평활화, 신호 지수 rho^p vs rho^(p-1)",
        "",
        f"- 격자 MC{mesh_control}, 설계 체적비 {volfrac}, 반복 {iters}회, 해법 PARDISO(직접법)",
        "- 비교 대상 `base` = 기존 ex07 SE-AHCA (rho^p, 밀도 필터 없음, 이동평균 2, 투영 없음). `p`, `pm1` = 투영(beta 1->8, 20회마다 2배) + 신호 밀도 필터 단독 평활화.",
        "- 모든 응력은 최종 0/1 설계를 4개 하중으로 다시 풀어 요소 중심 von Mises로 계산 (`evaluate_binary_design`). 항복 903 MPa.",
        "",
        "## 1. 질량 / 강성 / 회색도 / 시간",
        "| 변형 | 설정 | 질량(g) | 가중 컴플라이언스(N*mm) | 케이스별 컴플라이언스 (1/2/3/4) | 회색도(설계 요소) | 시간(s) |",
        "|:---|:---|---:|---:|:---|---:|---:|",
    ]
    for r in rows:
        cc = " / ".join(f"{c:,.0f}" for c in r["compliance_per_case_Nmm"])
        lines.append(f"| {r['variant']} | {r['options']} | {r['mass_g']:.1f} | {r['weighted_compliance_Nmm']:,.0f} | {cc} | {r['grayness_inside']:.4f} | {r['t_total_sec']:.0f} |")
    lines += [
        "",
        "## 2. 케이스별 최대 von Mises vs 99.5% von Mises (MPa)",
        "| 변형 | 최대 1 | 최대 2 | 최대 3 | 최대 4 | 99.5% 1 | 99.5% 2 | 99.5% 3 | 99.5% 4 | 통과(최대) | 통과(99.5%) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        vm, p9 = r["vm_max_per_case_MPa"], r["vm_p995_per_case_MPa"]
        lines.append(
            f"| {r['variant']} | " + " | ".join(f"{v:.0f}" for v in vm) + " | " + " | ".join(f"{v:.0f}" for v in p9)
            + f" | {'통과' if r['passes_yield_max'] else '실패'} | {'통과' if r['passes_yield_p995'] else '실패'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the three variants and write markdown/json into docs/benchmarks."""
    parser = argparse.ArgumentParser(description="GE bracket SE-AHCA signal-exponent study")
    parser.add_argument("--mesh_control", "-m", type=int, default=60)
    parser.add_argument("--volfrac", "-v", type=float, default=0.15)
    parser.add_argument("--iters", "-i", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--tag", default="", help="Suffix for output names (e.g. smoke)")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    stem = "04_ge_se_ahca_signal_study_20261102" + (f"_{args.tag}" if args.tag else "")
    out_name = f"ex07_se_ahca_study_mc{args.mesh_control}" + (f"_{args.tag}" if args.tag else "")
    out_root = ROOT / "wht_topopt" / "examples" / "outputs_v1" / out_name

    rows: List[Dict[str, Any]] = []
    for key in args.variants:
        rows.append(run_variant(key, VARIANTS[key], args.mesh_control, args.volfrac, args.iters, out_root))
        # Persist after every variant so a long run keeps finished results.
        write_markdown(rows, args.mesh_control, args.volfrac, args.iters, DOCS_DIR / f"{stem}.md")
        (DOCS_DIR / f"{stem}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {DOCS_DIR / (stem + '.md')}")


if __name__ == "__main__":
    main()
