"""A pillar study of one example: what the example defines, this runs and writes.

    outputs_v1/<family>/<name>/manifest.json, pillar1_algorithms/ ... pillar6_deliverables/

A pillar the example does not run must be declared with its reason (``not_applicable``); otherwise the study refuses to start.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .pillar1_algorithms import ALGORITHMS, DesignRun, LoadSet, run_algorithms, run_mma
from .pillar2_static_dynamic import run_static_vs_dynamic
from .pillar3_volume_relaxed import dominance_table, run_volume_relaxed
from .pillar5_audit import TestRig, audit_design
from .pillar6_deliverables import OUTPUT_ROOT, PILLARS, export_stl, pillar_status, write_design_figure, write_load_peaks_plot, write_progression_2d, write_report, write_transient_plot, write_voxel_gif
from .structure import Structure


@dataclass
class Study:
    name: str
    family: str
    structure: Structure
    loads: LoadSet
    rig: TestRig
    volfrac: float
    rmin: float
    delta_v: float = 0.0
    make_map: Optional[Callable[[], object]] = None  # design map of Pillars 2 and 3
    manufacturing: Dict[str, Callable[[], object]] = field(default_factory=dict)  # label -> design map of Pillar 4
    bending_key: str = ""
    torsion_key: str = ""
    not_applicable: Dict[int, str] = field(default_factory=dict)
    smoke: bool = False
    iters: Optional[int] = None
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        needs = {2: self.rig.dynamic is not None and self.make_map is not None, 3: self.rig.dynamic is not None and self.make_map is not None and self.delta_v > 0.0, 4: bool(self.manufacturing)}
        for p, runnable in needs.items():
            if not runnable and p not in self.not_applicable:
                raise ValueError(f"{self.name}: pillar {p} has nothing to run and is not declared not applicable")
        if not 0.0 < self.volfrac < 1.0:
            raise ValueError(f"{self.name}: volfrac must be in (0, 1), got {self.volfrac}")
        passive = self.structure.volume_fraction(self.structure.solid_mask.astype(float))
        if passive >= self.volfrac:
            raise ValueError(f"{self.name}: the passive solid alone is {passive:.3f} of the volume, above the target {self.volfrac}; refine the mesh or lower the passive share")


def _save_density(folder: Path, name: str, runs: Sequence[DesignRun]) -> List[Path]:
    out = []
    for r in runs:
        out.append(folder / f"{name}_{r.label}_density.npy")
        np.save(out[-1], r.rho)
    return out


def _fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def _run_row(s: Structure, r: DesignRun, a: Dict) -> List[str]:
    return [r.label, str(r.iterations), "예" if r.converged else "아니오", f"{r.t_total:.1f}", f"{r.t_fem:.1f}", f"{r.t_sens:.1f}", f"{r.t_opt:.1f}", f"{s.grayness(r.rho):.3f}", _fmt(a["mass_g"], ".1f"),
            f"{a['static']['compliance_mJ']:.4g}", f"{a['static']['von_mises_max_MPa']:.4g}", f"{a['static']['von_mises_p995_MPa']:.4g}", "연결" if a["load_path_connected"] else "끊김"]


def run_study(st: Study, algorithms: Sequence[str] = ALGORITHMS, root: Optional[Path] = None) -> Dict:
    s = st.structure
    out = Path(root or OUTPUT_ROOT) / st.family / st.name
    folders = {1: "pillar1_algorithms", 2: "pillar2_static_dynamic", 3: "pillar3_volume_relaxed", 4: "pillar4_manufacturing", 5: "pillar5_audit", 6: "pillar6_deliverables"}
    p = {k: out / d for k, d in folders.items() if k not in st.not_applicable}
    for d in p.values():
        d.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    audit = lambda r, v: audit_design(s, st.rig, r.rho, v, st.max_steps)

    print(f"[{st.name}] Pillar 1", flush=True)
    algo_runs, skipped = run_algorithms(s, st.rmin, st.loads, st.volfrac, algorithms, st.iters)
    files += _save_density(p[1], st.name, algo_runs)
    algo_audits = {r.label: audit(r, st.volfrac) for r in algo_runs}

    models: List[DesignRun] = []
    audits: Dict[str, Dict] = {}
    model1 = model2 = model3 = None
    if 2 in p:
        print(f"[{st.name}] Pillar 2", flush=True)
        model1, model2 = run_static_vs_dynamic(s, st.make_map, st.loads, st.rig.dynamic, st.volfrac, st.iters, st.max_steps)
        files += _save_density(p[2], st.name, [model1, model2])
        audits["model1_static"], audits["model2_dynamic"] = audit(model1, st.volfrac), audit(model2, st.volfrac)
        models += [model1, model2]
    if 3 in p:
        print(f"[{st.name}] Pillar 3", flush=True)
        model3 = run_volume_relaxed(s, st.make_map, st.rig.dynamic, st.volfrac, st.delta_v, st.iters, st.max_steps)
        files += _save_density(p[3], st.name, [model3])
        audits["model3_relaxed"] = audit(model3, st.volfrac + st.delta_v)
        models.append(model3)
    mfg_runs: List[DesignRun] = []
    if 4 in p:
        print(f"[{st.name}] Pillar 4", flush=True)
        mfg_runs = [run_mma(s, make(), st.loads, st.volfrac, st.iters, label=f"mfg_{label}") for label, make in st.manufacturing.items()]
        files += _save_density(p[4], st.name, mfg_runs)
        for r in mfg_runs:
            audits[r.label] = audit(r, st.volfrac)
    dominance = dominance_table(audits["model1_static"], audits["model2_dynamic"], audits["model3_relaxed"], st.bending_key, st.torsion_key) if model3 is not None else None

    print(f"[{st.name}] Pillar 5, 6", flush=True)
    every = {**algo_audits, **audits}
    report_audits = audits or algo_audits
    stl_files: Dict[str, Path] = {}
    stl_info: Dict[str, Dict] = {}
    notes: Dict[int, str] = {}
    figure: Optional[Path] = None
    if s.dim == 3:
        for label, a in every.items():
            path = p[6] / f"{st.name}_{label}.stl"
            stl_info[label] = export_stl(s, a["xb"], path)
            stl_files[label] = path
            files.append(path)
        frames = (model2 or algo_runs[0]).frames or [(model2 or algo_runs[0]).rho]
        gif, reason = write_voxel_gif(s, frames, [f"frame {i + 1}/{len(frames)}  V target {st.volfrac}" for i in range(len(frames))], p[6] / f"{st.name}_progression.gif")
        if gif is not None:
            files.append(gif)
        if reason:
            notes[6] = reason
    else:
        figure = write_design_figure(s, algo_runs, algo_audits, p[1] / f"{st.name}_algorithms.png")
        files += [figure, write_progression_2d(s, algo_runs, p[1] / f"{st.name}_progression.gif")]
    if model3 is not None:
        figure = write_transient_plot({k: audits[k] for k in ("model1_static", "model2_dynamic", "model3_relaxed")}, st.rig.dynamic.formula, p[5] / f"{st.name}_transient_vibration.png")
        files.append(figure)
    elif model2 is not None and s.dim == 2:
        figures = [
            write_transient_plot({k: audits[k] for k in ("model1_static", "model2_dynamic")}, st.rig.dynamic.name, p[5] / f"{st.name}_transient_response.png"),
            write_load_peaks_plot(st.rig.dynamic, model2.extras["esl_peaks_last_iteration"], p[5] / f"{st.name}_load_history_peaks.png"),
            write_design_figure(s, [model1, model2], audits, p[2] / f"{st.name}_static_vs_dynamic.png"),
        ]
        files += figures + [write_progression_2d(s, [model1, model2], p[2] / f"{st.name}_static_vs_dynamic.gif")]
        figure = figures

    if s.dim == 3 or st.rig.dynamic is not None:
        notes[1] = f"Kratos 미실행: {next(iter(skipped.values()))}"
    status = pillar_status([k for k in PILLARS if k in p], st.not_applicable, notes)
    conditions = {"elements": s.n_elem, "design_elements": int(s.design_mask.sum()), "E0": s.e0, "nu": s.nu, "density": s.density, "yield": s.yield_stress, "volfrac": st.volfrac, "delta_v": st.delta_v, "rmin": st.rmin, "smoke": st.smoke}
    public = lambda a: {k: v for k, v in a.items() if k != "xb"}
    metrics = {"example": st.name, "conditions": conditions, "algorithms": {k: public(v) for k, v in algo_audits.items()}, "models": {k: public(v) for k, v in audits.items()}, "dominance": dominance, "skipped": skipped, "stl": stl_info,
               "timing": {r.label: {"iterations": r.iterations, "converged": r.converged, "volume_fraction": s.volume_fraction(r.rho), "grayness": s.grayness(r.rho), "t_total_s": r.t_total, "t_fem_s": r.t_fem, "t_sens_s": r.t_sens, "t_opt_s": r.t_opt, "history_c": r.history_c, "notes": r.notes, **r.extras} for r in [*algo_runs, *models, *mfg_runs]}}
    metrics_path = p[5] / f"{st.name}_cross_stiffness_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else o.tolist()), encoding="utf-8")
    files.append(metrics_path)

    algo_rows = [["방법", "반복", "수렴", "시간 (s)", "FEM", "감도", "갱신", "회색도", "0/1 질량 (g)", "0/1 C", "최대 vM", "99.5% vM", "하중 경로"]] + [_run_row(s, r, algo_audits[r.label]) for r in algo_runs]
    report = p[6] / f"{st.name}_comprehensive_master_report.html"
    write_report(report, st.name, conditions, report_audits, algo_rows, dominance, status, stl_files, figure, st.smoke)
    files.append(report)
    lines = [f"# {st.name}", "", f"- 조건: {conditions}", "- 표의 C와 응력은 0/1 설계를 같은 모델로 다시 푼 값이다.", "", "| " + " | ".join(algo_rows[0]) + " |", "|" + "---|" * len(algo_rows[0])] + ["| " + " | ".join(r) + " |" for r in algo_rows[1:]]
    lines += ["", "## Pillar 상태", ""] + [f"- Pillar {k}: {v['status']}" + (f" ({v['reason']})" if "reason" in v else "") for k, v in status.items()]
    if dominance:
        lines += ["", "## Pillar 3 목표", ""] + [f"- {d['target']}: {d['model1']:.4g} -> {d['model3']:.4g} ({d['change_percent']:+.1f} %), {'달성' if d['passed'] else '미달'}" for d in dominance]
    if model2 is not None and model3 is None:
        d1, d2 = audits["model1_static"]["dynamic"], audits["model2_dynamic"]["dynamic"]
        change = lambda a, b: f"{100.0 * (b - a) / a:+.1f}" if a else "-"
        lines += ["", "## 정적 vs 동적 (같은 하중 이력)", "", f"- 하중: {st.rig.dynamic.formula}", "", "| 지표 | model1_static | model2_dynamic | 변화 (%) |", "|---|---|---|---|"]
        lines += [f"| 피크 abs {k} | {d1['peak'][k]:.5g} | {d2['peak'][k]:.5g} | {change(d1['peak'][k], d2['peak'][k])} |" for k in d1["peak"]]
        lines.append(f"| 동적 최대 vM | {d1['von_mises_max_MPa']:.5g} | {d2['von_mises_max_MPa']:.5g} | {change(d1['von_mises_max_MPa'], d2['von_mises_max_MPa'])} |")
        lines += [f"| f{k + 1} (Hz) | {a:.4g} | {b:.4g} | {change(a, b)} |" for k, (a, b) in enumerate(zip(d1["frequencies_hz"], d2["frequencies_hz"]))]
        lines += ["", "## ESL 피크 (마지막 반복)", "", "| t (s) | u^T K u | w | 하중 합력 |", "|---|---|---|---|"]
        lines += [f"| {q['t_s']:.2f} | {q['energy']:.4g} | {q['weight']:.3f} | {', '.join(format(v, '.3f') for v in q['force'])} |" for q in model2.extras["esl_peaks_last_iteration"]]
        lines += ["", "- 0차 ESL: 과도 수반해석이 없어 KKT 보장이 없다.", f"- 수렴: model1 {'예' if model1.converged else '아니오'} ({model1.iterations} 반복), model2 {'예' if model2.converged else '아니오'} ({model2.iterations} 반복)"]
        notes_all = sorted({n for a in report_audits.values() for n in a["notes"]})
        if notes_all:
            lines += ["", "## 주의", ""] + [f"- {n}" for n in notes_all]
    if st.smoke:
        lines += ["", "**스모크 실행: 수치를 결과로 쓰지 말 것.**"]
    summary = p[6] / f"{st.name}_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(summary)
    manifest = {"example": st.name, "family": st.family, "smoke": st.smoke, "pillars": status, "algorithms": list(algorithms), "models": list(audits), "files": [str(f.relative_to(out)).replace("\\", "/") for f in files]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {out}", flush=True)
    return manifest


def main(build: Callable[..., Study], full: Dict, smoke: Dict, argv: Optional[Sequence[str]] = None, default_algos: Sequence[str] = ALGORITHMS) -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small mesh and few iterations: checks the chain, the numbers mean nothing")
    ap.add_argument("--algos", nargs="*", default=list(default_algos))
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("-o", "--output-root", type=Path, default=None)
    args = ap.parse_args(argv)
    study = build(**(smoke if args.smoke else full))
    if args.iters:
        study.iters = args.iters
    return run_study(study, args.algos, args.output_root)
