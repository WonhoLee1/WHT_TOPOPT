"""SE-AHCA stage-by-stage ablation on the 2D half-MBB beam (180x60, V = 0.4, rmin = 2.4).

Stages (each stage keeps the previous change):
    S0  legacy      : unit-energy signal, no projection, signal filter + moving average, dimensional sigma_allow
    S1  signal      : rho_phys^(p-1) * u0 / v, relative sigma_allow (unit independent)
    S2  projection  : S1 + Heaviside(filter(xi)) in ``hca.step`` (beta 1 -> 8, doubled every 20 it), bisection on projected rho
    S3a smoothing   : S2 + only the density (spatial) filter on the signal
    S3b smoothing   : S2 + only the 2-step moving average on the signal
Reference rows: previous documented SE-AHCA (35 it, C = 595.29, grayness 0.210) and MMA run for the same iteration count.

Output: docs/benchmarks/03_se_ahca_mbb_ablation_20261102.md (+ .json, + .png)

Usage:
    python wht_topopt/examples/ex06_se_ahca_ablation.py                 # full 100-iteration ablation
    python wht_topopt/examples/ex06_se_ahca_ablation.py --iters 20 --tag smoke   # quick smoke run
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ex06_beams_benchmark_suite import (  # noqa: E402
    MethodRunRecord,
    OptimizationSuite,
    SolverSuite,
    build_2d_beam_problem,
)

PREV_C = 595.29       # documented previous SE-AHCA compliance (35 it)
PREV_GRAY = 0.210     # documented previous SE-AHCA grayness
DOCS_DIR = ROOT / "wht_topopt" / "docs" / "benchmarks"


@dataclass
class Stage:
    """One ablation stage.

    Attributes:
        key: Short id used in tables.
        description: Human readable change description.
        options: Keyword arguments forwarded to ``OptimizationSuite.run_se_ahca``.
    """

    key: str
    description: str
    options: Dict[str, Any]


STAGES: List[Stage] = [
    Stage("S0", "기준(기존): 단위 에너지 신호, 투영 없음, 신호 필터+이동평균, sigma_allow=250e6",
          dict(signal_mode="unit", projection=False, smoothing="both", sigma_mode="legacy")),
    Stage("S1", "1) 신호 rho^(p-1)*u0/v, sigma_allow 상대(90퍼센타일)",
          dict(signal_mode="rho_pm1", projection=False, smoothing="both", sigma_mode="relative")),
    Stage("S2", "2) + Heaviside 투영(beta 1->8, 20회마다 2배), 투영 밀도로 체적 이분법",
          dict(signal_mode="rho_pm1", projection=True, smoothing="both", sigma_mode="relative")),
    Stage("S3a", "3a) + 평활화는 신호 밀도 필터만",
          dict(signal_mode="rho_pm1", projection=True, smoothing="density_filter", sigma_mode="relative")),
    Stage("S3b", "3b) + 평활화는 이동평균(2)만",
          dict(signal_mode="rho_pm1", projection=True, smoothing="moving_average", sigma_mode="relative")),
]


def run_stage(suite: OptimizationSuite, stage: Stage, iters: int) -> MethodRunRecord:
    """Run one SE-AHCA stage on a fresh half-MBB problem.

    Args:
        suite: Optimization suite with the configured FEM solver.
        stage: Ablation stage to run.
        iters: Number of SE-AHCA iterations (all executed: tol = 0).

    Returns:
        MethodRunRecord evaluated on the returned physical density.
    """
    spec = build_2d_beam_problem(case="mbb", nelx=180, nely=60, volfrac=0.40, rmin=2.4)
    return suite.run_se_ahca(spec, max_iters=iters, tol=0.0, **stage.options)


def run_mma_reference(suite: OptimizationSuite, iters: int) -> MethodRunRecord:
    """Run MMA with the same iteration budget as a reference row.

    Args:
        suite: Optimization suite.
        iters: Maximum MMA iterations.
    """
    spec = build_2d_beam_problem(case="mbb", nelx=180, nely=60, volfrac=0.40, rmin=2.4)
    return suite.run_mma(spec, max_iters=iters)


def record_to_row(label: str, desc: str, rec: MethodRunRecord, prev_c: Optional[float]) -> Dict[str, Any]:
    """Convert a run record to a table row dict.

    Args:
        label: Row id.
        desc: Row description.
        rec: Run record.
        prev_c: Compliance of the previous stage (None for the first row).
    """
    return {
        "stage": label,
        "description": desc,
        "compliance": round(float(rec.compliance), 3),
        "volume": round(float(rec.design_vol), 4),
        "grayness": round(float(rec.grayness), 4),
        "iterations": int(rec.iterations),
        "time_s": round(float(rec.timing.t_total), 1),
        "dC_vs_prev_pct": None if prev_c is None else round((rec.compliance / prev_c - 1.0) * 100.0, 1),
        "dC_vs_doc595_pct": round((rec.compliance / PREV_C - 1.0) * 100.0, 1),
    }


def render_png(records: List[Any], path: Path) -> None:
    """Save density maps (left) and compliance histories (right) for all rows.

    Args:
        records: List of (label, MethodRunRecord) tuples.
        path: Output PNG path.
    """
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(12.0, 2.6 * n), squeeze=False)
    for i, (label, rec) in enumerate(records):
        rho = rec.rho_phys.reshape(60, 180)
        axes[i, 0].imshow(1.0 - rho, cmap="gray", origin="lower", extent=[0, 180, 0, 60])
        axes[i, 0].set_title(f"{label} | C={rec.compliance:.2f} | V={rec.design_vol*100:.1f}% | gray={rec.grayness:.3f}")
        axes[i, 1].plot(rec.history_c, color="#2563eb")
        axes[i, 1].set_yscale("log")
        axes[i, 1].set_ylabel("C (log)")
        axes[i, 1].grid(True, linestyle=":", alpha=0.4)
    axes[-1, 1].set_xlabel("iteration")
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_markdown(rows: List[Dict[str, Any]], iters: int, path: Path) -> None:
    """Write the comparison table to markdown.

    Args:
        rows: Table rows produced by ``record_to_row``.
        iters: Iteration count used.
        path: Output markdown path.
    """
    lines = [
        "# SE-AHCA MBB 단계별 개선 비교 (ex06 `run_se_ahca`)",
        "",
        f"- 문제: half-MBB 2D 180x60 Q4, V=0.4, rmin=2.4, SIMP p=3, 반복 {iters}회 고정(tol=0), PARDISO",
        f"- 이전 값(기록): C = {PREV_C}, 회색도 = {PREV_GRAY} (SE-AHCA 35회, `outputs_v1/beams/mbb_benchmark_suite.png`)",
        "- C, V, 회색도는 반환된 최종 물리 밀도로 한 번 더 해석해 구했다 (루프 마지막 해석은 마지막 갱신 전 밀도에 해당).",
        "- 회색도 = mean(x(1-x)). C의 dC는 직전 단계 대비, dC(595)는 이전 기록 대비.",
        "",
        "| 단계 | 변경 | C | V | 회색도 | 반복 | 시간(s) | dC 직전 | dC 595 대비 |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        dp = "-" if r["dC_vs_prev_pct"] is None else f"{r['dC_vs_prev_pct']:+.1f}%"
        lines.append(
            f"| {r['stage']} | {r['description']} | {r['compliance']:.2f} | {r['volume']*100:.1f}% | "
            f"{r['grayness']:.3f} | {r['iterations']} | {r['time_s']:.1f} | {dp} | {r['dC_vs_doc595_pct']:+.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run all stages, then write markdown/json/png into docs/benchmarks."""
    parser = argparse.ArgumentParser(description="SE-AHCA MBB ablation")
    parser.add_argument("--iters", type=int, default=100, help="SE-AHCA iterations per stage (default 100)")
    parser.add_argument("--solver", default="pardiso", choices=["pardiso", "scipy"])
    parser.add_argument("--tag", default="", help="Suffix for output names (e.g. smoke) to avoid overwriting")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    suite = OptimizationSuite(SolverSuite(backend=args.solver))
    stem = "03_se_ahca_mbb_ablation_20261102" + (f"_{args.tag}" if args.tag else "")

    rows: List[Dict[str, Any]] = []
    records: List[Any] = []
    prev_c: Optional[float] = None
    for st in STAGES:
        print(f"[{st.key}] {st.description}", flush=True)
        rec = run_stage(suite, st, args.iters)
        rows.append(record_to_row(st.key, st.description, rec, prev_c))
        records.append((st.key, rec))
        prev_c = rec.compliance
        print(f"    C={rec.compliance:.3f} V={rec.design_vol:.4f} gray={rec.grayness:.4f} t={rec.timing.t_total:.1f}s", flush=True)

    print("[MMA] reference", flush=True)
    mma = run_mma_reference(suite, args.iters)
    rows.append(record_to_row("MMA", f"MMA 기준(최대 {args.iters}회, 투영 없음)", mma, None))
    records.append(("MMA", mma))

    write_markdown(rows, args.iters, DOCS_DIR / f"{stem}.md")
    (DOCS_DIR / f"{stem}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    render_png(records, DOCS_DIR / f"{stem}.png")
    print(f"Saved: {DOCS_DIR / (stem + '.md')}")


if __name__ == "__main__":
    main()
