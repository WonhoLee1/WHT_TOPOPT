"""Pillar 6: density files, metrics, figures, progression GIF, STL (3D) and a standalone HTML report with a Three.js viewer (3D).

    outputs_v1/<family>/<name>/
      manifest.json
      pillar1_algorithms/  pillar2_static_dynamic/  pillar3_volume_relaxed/  pillar4_manufacturing/   density .npy per design
      pillar5_audit/       <name>_cross_stiffness_metrics.json, <name>_transient_vibration.png
      pillar6_deliverables/ <name>_<design>.stl, <name>_progression.gif, <name>_summary.md, <name>_comprehensive_master_report.html
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure

PILLARS = (1, 2, 3, 4, 5, 6)

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs_v1"
FONT_PT = 9
VIEWER_MODELS = {"model1_static": "#d4a017", "model2_dynamic": "#2f7fff", "model3_relaxed": "#10b981"}


def export_stl(s: Structure, xb: np.ndarray, path: Path) -> Dict:
    """Closed surface of the 0/1 voxel design by marching cubes on the padded grid."""
    import trimesh
    from skimage.measure import marching_cubes

    nx, ny, nz = s.grid
    grid = np.pad(np.asarray(xb, dtype=np.float32).reshape(nz, ny, nx), 1)
    h = s.size
    verts, faces, _, _ = marching_cubes(grid, level=0.5, spacing=(float(h[2]), float(h[1]), float(h[0])))
    origin = s.nodes[: int(s.conn.max()) + 1].min(axis=0)
    # cell centres of the padded grid sit half a cell inside the first node row, one padding cell away
    verts = verts[:, ::-1] + origin - 0.5 * h
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    tm.export(str(path))
    return {"path": path.name, "triangles": int(len(tm.faces)), "watertight": bool(tm.is_watertight)}


def write_design_figure(s: Structure, runs, audits: Dict[str, Dict], path: Path) -> Path:
    """Density and convergence of every method; 2D problems."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = s.grid
    lx, ly = nx * s.size[0], ny * s.size[1]
    fig, axes = plt.subplots(len(runs), 2, figsize=(11.0, 2.9 * len(runs)), squeeze=False, gridspec_kw={"width_ratios": [1.6, 1.0]})
    for (ax_d, ax_c), r in zip(axes, runs):
        ax_d.imshow(1.0 - r.rho.reshape(ny, nx), cmap="gray", origin="lower", vmin=0.0, vmax=1.0, extent=[0, lx, 0, ly])
        ax_d.set_title(f"{r.label} | V = {s.volume_fraction(r.rho):.3f} | gray = {s.grayness(r.rho):.3f} | {r.iterations} it | 0/1 C = {audits[r.label]['static']['compliance_mJ']:.4g}", fontsize=FONT_PT)
        ax_d.tick_params(labelsize=FONT_PT - 1)
        ax_c.plot(r.history_c, color="#2563eb", linewidth=1.5)
        ax_c.set_yscale("log")
        ax_c.set_xlabel("iteration", fontsize=FONT_PT)
        ax_c.set_ylabel("compliance", fontsize=FONT_PT)
        ax_c.tick_params(labelsize=FONT_PT - 1)
        ax_c.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_progression_2d(s: Structure, runs, path: Path, scale: int = 2, frame_ms: int = 150) -> Path:
    """One row per method, the density advancing together."""
    from PIL import Image, ImageDraw

    nx, ny = s.grid
    n_frames = max(len(r.frames) for r in runs)
    label_h = 14
    images = []
    for k in range(n_frames):
        canvas = np.full((len(runs) * (ny * scale + label_h), nx * scale), 255, dtype=np.uint8)
        for row, r in enumerate(runs):
            frame = r.frames[min(k, len(r.frames) - 1)]
            tile = np.clip(255.0 * (1.0 - frame.reshape(ny, nx)), 0, 255).astype(np.uint8)[::-1]
            top = row * (ny * scale + label_h) + label_h
            canvas[top : top + ny * scale, : nx * scale] = np.kron(tile, np.ones((scale, scale), dtype=np.uint8))
        im = Image.fromarray(canvas, mode="L")
        draw = ImageDraw.Draw(im)
        for row, r in enumerate(runs):
            draw.text((2, row * (ny * scale + label_h) + 1), f"{r.label}  frame {min(k, len(r.frames) - 1) + 1}/{len(r.frames)}", fill=0)
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0)
    return path


def write_transient_plot(audits: Dict[str, Dict], load_formula: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k in next(iter(audits.values()))["dynamic"]["response"] if k != "t_ms"]
    long_window = next(iter(audits.values()))["dynamic"]["load_profile"]["duration_s"] > 1.0  # 20 s reads as seconds, 50 ms as milliseconds
    scale, t_label = (1e-3, "t (s)") if long_window else (1.0, "t (ms)")
    fig, axes = plt.subplots(1, len(labels), figsize=(11.0, 3.6), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        for name, a in audits.items():
            ax.plot(scale * np.asarray(a["dynamic"]["response"]["t_ms"]), a["dynamic"]["response"][lab], label=name, linewidth=1.0 if long_window else 1.3)
        ax.set_xlabel(t_label, fontsize=FONT_PT)
        ax.set_ylabel(f"{lab} (mm)", fontsize=FONT_PT)
        ax.tick_params(labelsize=FONT_PT - 1)
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0][-1].legend(fontsize=FONT_PT - 1, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle(load_formula, fontsize=FONT_PT)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_load_peaks_plot(load, peaks: Sequence[Dict], path: Path, samples: int = 4401) -> Path:
    """The 2D load history (magnitude, direction from -y, components) with the ESL instants marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0.0, load.duration, samples)
    f = np.array([load.force_at(float(v)).reshape(-1, 2).sum(axis=0) for v in t])
    mag = np.hypot(f[:, 0], f[:, 1])
    angle = np.degrees(np.arctan2(f[:, 0], -f[:, 1]))
    angle[mag == 0.0] = np.nan
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 6.4), sharex=True)
    for ax, y, name in zip(axes, (mag, angle, None), ("|F|", "angle from -y (deg)", "Fx, Fy")):
        if y is None:
            ax.plot(t, f[:, 0], linewidth=0.9, label="Fx", color="#2563eb")
            ax.plot(t, f[:, 1], linewidth=0.9, label="Fy", color="#dc2626")
        else:
            ax.plot(t, y, linewidth=0.9, color="#334155")
        for k, p in enumerate(peaks):
            ax.axvline(p["t_s"], color="#16a34a", linewidth=1.0, linestyle="--", label="ESL peak" if y is None and k == 0 else None)
        if y is None:
            ax.legend(fontsize=FONT_PT - 1, loc="upper right")
        if load.window_start > 0.0:
            ax.axvspan(0.0, load.window_start, color="#94a3b8", alpha=0.25)
        ax.set_ylabel(name, fontsize=FONT_PT)
        ax.tick_params(labelsize=FONT_PT - 1)
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("t (s), shaded: start-up ramp, dashed: ESL instants of the last iteration", fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_voxel_gif(s: Structure, frames: Sequence[np.ndarray], captions: Sequence[str], path: Path, threshold: float = 0.5) -> Tuple[Optional[Path], Optional[str]]:
    """PyVista voxel frames in the spec's style (black background, viridis, edges). Without PyVista: no GIF and the reason, no substitute."""
    try:
        import pyvista as pv
    except ImportError as exc:
        return None, f"PyVista를 불러올 수 없어 GIF를 만들지 않았다: {exc}"
    from PIL import Image

    pv.OFF_SCREEN = True
    nx, ny, nz = s.grid
    h = s.size
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=tuple(float(v) for v in h), origin=tuple(float(v) for v in s.nodes[: int(s.conn.max()) + 1].min(axis=0)))
    images = []
    for frame, caption in zip(frames, captions):
        grid.cell_data["density"] = np.asarray(frame, dtype=np.float64)
        solid = grid.threshold(threshold, scalars="density")
        plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        plotter.set_background("black")
        if solid.n_cells:
            plotter.add_mesh(solid, scalars="density", cmap="viridis", clim=[threshold, 1.0], show_edges=True, edge_color="#202020", show_scalar_bar=False)
        plotter.show_axes()
        plotter.add_text(caption, font="courier", font_size=12, color="white")
        plotter.view_isometric()
        images.append(Image.fromarray(plotter.screenshot(return_img=True)))
        plotter.close()
    images[0].save(path, save_all=True, append_images=images[1:], duration=350, loop=0)
    return path, None


def _html_table(rows: List[List[str]]) -> str:
    head = "".join(f"<th>{c}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table><tr>{head}</tr>{body}</table>"


def static_table(audits: Dict[str, Dict]) -> List[List[str]]:
    keys = list(next(iter(audits.values()))["static"]["stiffness"])
    units = next(iter(audits.values()))["static"]["stiffness_units"]
    fmt = lambda v, spec: "-" if v is None else format(v, spec)
    rows = [["모델", "질량 (g)", "V"] + [f"{k} ({units[k]})" for k in keys] + ["C", "최대 vM", "99.5% vM", "안전율", "하중 경로"]]
    for name, a in audits.items():
        st = a["static"]
        rows.append([name, fmt(a["mass_g"], ".1f"), f"{a['volume_fraction']:.3f}"] + [f"{st['stiffness'][k]:.4g}" for k in keys]
                    + [f"{st['compliance_mJ']:.4g}", f"{st['von_mises_max_MPa']:.4g}", f"{st['von_mises_p995_MPa']:.4g}", fmt(st["safety_factor"], ".2f"), "연결" if a["load_path_connected"] else "끊김"])
    return rows


def dynamic_table(audits: Dict[str, Dict]) -> List[List[str]]:
    first = next(iter(audits.values()))["dynamic"]
    labels = list(first["peak"])
    residual = list(first["settling_time_ms"])  # empty when the load lasts to the end of the window
    rows = [["모델", "f1 (Hz)", "f2 (Hz)", "f3 (Hz)", "스텝"] + [f"피크 |{k}| (mm)" for k in labels] + [f"잔류 RMS {k} (mm)" for k in residual] + [f"안정화 {k} (ms)" for k in residual] + ["동적 vM (MPa)", "SF_dyn"]]
    for name, a in audits.items():
        d = a["dynamic"]
        rows.append([name] + [f"{v:.4g}" for v in d["frequencies_hz"][:3]] + [str(d["time_steps"])] + [f"{d['peak'][k]:.4g}" for k in labels]
                    + [f"{d['residual_rms'].get(k, float('nan')):.4g}" for k in residual]
                    + ["구간 끝까지 미안정" if d["settling_time_ms"][k] is None else f"{d['settling_time_ms'][k]:.2f}" for k in residual]
                    + [f"{d['von_mises_max_MPa']:.4g}", "-" if d["safety_factor"] is None else f"{d['safety_factor']:.2f}"])
    return rows


VIEWER_JS = """
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import {STLLoader} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js';
const host=document.getElementById('viewer');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x111418);
const cam=new THREE.PerspectiveCamera(45,host.clientWidth/host.clientHeight,0.1,1e5);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(host.clientWidth,host.clientHeight);ren.localClippingEnabled=true;host.appendChild(ren.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1.3));const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(1,2,3);scene.add(dl);
const ctl=new OrbitControls(cam,ren.domElement);const loader=new STLLoader();const meshes={};const clip=new THREE.Plane(new THREE.Vector3(0,0,-1),1e9);
const box=new THREE.Box3();
for(const [name,m] of Object.entries(MODELS)){
  const bytes=Uint8Array.from(atob(m.stl),c=>c.charCodeAt(0));
  const geo=loader.parse(bytes.buffer);geo.computeVertexNormals();
  const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:m.color,metalness:.2,roughness:.55,side:THREE.DoubleSide,transparent:true,clippingPlanes:[]}));
  scene.add(mesh);meshes[name]=mesh;box.expandByObject(mesh);
}
const c=box.getCenter(new THREE.Vector3());const r=box.getSize(new THREE.Vector3()).length();
ctl.target.copy(c);cam.position.copy(c).add(new THREE.Vector3(.7,.6,.9).multiplyScalar(r));ctl.update();
function show(sel){for(const [n,m] of Object.entries(meshes)){m.visible=sel==='all'||sel===n;m.material.opacity=sel==='all'?.45:1;}}
document.querySelectorAll('[data-show]').forEach(b=>b.onclick=()=>show(b.dataset.show));
show(Object.keys(meshes)[0]);
const cb=document.getElementById('cut'),sl=document.getElementById('cutz');
sl.min=box.min.z;sl.max=box.max.z;sl.step=(box.max.z-box.min.z)/200;sl.value=box.max.z;
function cut(){const on=cb.checked;sl.disabled=!on;clip.constant=+sl.value;for(const m of Object.values(meshes))m.material.clippingPlanes=on?[clip]:[];}
cb.onchange=cut;sl.oninput=cut;cut();
(function loop(){requestAnimationFrame(loop);ren.render(scene,cam);})();
"""


def write_report(path: Path, name: str, conditions: Dict, audits: Dict[str, Dict], algorithm_rows: List[List[str]], dominance: Optional[List[Dict]], status: Dict, stl_files: Dict[str, Path], figure, smoke: bool) -> None:
    """``figure``: one path, several paths, or None."""
    models = {k: {"color": VIEWER_MODELS.get(k, "#c0c0c0"), "stl": base64.b64encode(p.read_bytes()).decode("ascii")} for k, p in stl_files.items() if k in VIEWER_MODELS}
    viewer = script = ""
    if models:
        buttons = "".join(f"<button data-show='{k}' style='border-color:{VIEWER_MODELS[k]}'>{k}</button>" for k in models) + ("<button data-show='all' style='border-color:#a855f7'>겹쳐 보기</button>" if len(models) > 1 else "")
        viewer = f"<h2>3D 뷰어</h2><div>{buttons}<label><input type='checkbox' id='cut'> 내부 단면 절단</label> <input type='range' id='cutz' disabled></div><div id='viewer'></div>"
        script = f'<script>globalThis.__models={json.dumps(models)};</script><script type="module">{VIEWER_JS.replace("MODELS", "globalThis.__models")}</script>'
    figures = [figure] if isinstance(figure, Path) else list(figure or [])
    img = "".join(f"<img alt='results' src='data:image/png;base64,{base64.b64encode(f.read_bytes()).decode('ascii')}'>" for f in figures)
    dynamic = ""
    first = next(iter(audits.values()))
    if first["dynamic"] is not None:
        profile = first["dynamic"]["load_profile"]
        dynamic = f"<h2>동적 성능 평가표</h2><p>하중: {profile['name']}: {profile['formula']}, 피크 {profile['peak_force_N']:.1f} N, 구간 {profile['duration_s'] * 1e3:.0f} ms</p><div class='scroll'>{_html_table(dynamic_table(audits))}</div>"
    dom = ""
    if dominance:
        rows = [["목표", "Model 1", "Model 3", "변화 (%)", "달성"]] + [[d["target"], f"{d['model1']:.4g}", f"{d['model3']:.4g}", f"{d['change_percent']:+.1f}", "달성" if d["passed"] else "미달"] for d in dominance]
        dom = "<h2>Pillar 3 목표 달성 여부</h2>" + _html_table(rows)
    pillars = "".join(f"<li>Pillar {p}: {s['status']}{' (' + s['reason'] + ')' if 'reason' in s else ''}</li>" for p, s in status.items())
    notes = sorted({n for a in audits.values() for n in a["notes"]})
    notes_html = "<h2>주의</h2><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>" if notes else ""
    banner = "<p><strong>스모크 실행: 수치를 결과로 쓰지 말 것.</strong></p>" if smoke else ""
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title>
<style>body{{font:14px sans-serif;margin:24px;max-width:1200px}}table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #bbb;padding:4px 8px;text-align:right;white-space:nowrap}}
.scroll{{overflow-x:auto}}img{{max-width:100%}}#viewer{{width:100%;height:520px}}button{{margin:4px;padding:6px 12px;border:2px solid;background:#fff;cursor:pointer}}</style></head><body>
<h1>{name}</h1>{banner}<p>{conditions}</p>{viewer}
<h2>정적 강성 평가표</h2><div class="scroll">{_html_table(static_table(audits))}</div>{dynamic}{img}{dom}
<h2>Pillar 1 알고리즘 비교</h2><div class="scroll">{_html_table(algorithm_rows)}</div>
<h2>Pillar 상태</h2><ul>{pillars}</ul>{notes_html}{script}</body></html>"""
    path.write_text(html, encoding="utf-8")


def pillar_status(done: Sequence[int], not_applicable: Dict[int, str], partial: Optional[Dict[int, str]] = None) -> Dict[str, Dict[str, str]]:
    """Status per pillar; every pillar must be accounted for."""
    partial = partial or {}
    out: Dict[str, Dict[str, str]] = {}
    for p in PILLARS:
        if p in not_applicable:
            out[str(p)] = {"status": "not_applicable", "reason": not_applicable[p]}
        elif p in partial:
            out[str(p)] = {"status": "partial", "reason": partial[p]}
        elif p in done:
            out[str(p)] = {"status": "done"}
        else:
            raise ValueError(f"pillar {p} neither ran nor is declared not applicable")
    return out
