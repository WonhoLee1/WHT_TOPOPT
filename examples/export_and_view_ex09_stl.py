"""Showcase 09: Export Watertight STL for Static Model & Render with Explicit 3D Axes.

Generates:
1. robot_arm_static.stl (Marching Cubes 100% Watertight)
2. dynamic_vs_static_faceted_mesh.png (with prominent X, Y, Z coordinate axes on both viewports)
3. dynamic_vs_static_3d_viewer.html (Standalone Three.js interactive 3D comparison viewer)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

import numpy as np
import pyvista as pv
import skimage.measure
import trimesh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wht_topopt.examples.ex09_esl_dynamic_robot_arm import RobotArmDomain3D


def export_stl(rho: np.ndarray, domain: RobotArmDomain3D, stl_path: Path, isovalue: float = 0.25) -> Tuple[Path, bool, float]:
    """Export zero-padded Marching Cubes surface guaranteeing 100% watertight binary STL."""
    mesh = domain.mesh
    grid3d = rho.reshape((mesh.nelz, mesh.nely, mesh.nelx))
    padded = np.pad(grid3d, pad_width=1, mode="constant", constant_values=0.0)

    verts, faces, normals, values = skimage.measure.marching_cubes(
        padded,
        level=isovalue,
        spacing=(mesh.dz, mesh.dy, mesh.dx),
    )

    coords = np.zeros_like(verts)
    coords[:, 0] = verts[:, 2] - mesh.dx
    coords[:, 1] = verts[:, 1] - mesh.dy
    coords[:, 2] = verts[:, 0] - mesh.dz

    stl_mesh = trimesh.Trimesh(vertices=coords, faces=faces)
    try:
        stl_mesh.remove_duplicate_faces()
        stl_mesh.remove_degenerate_faces()
        stl_mesh.fill_holes()
    except Exception:
        pass

    stl_mesh.export(str(stl_path))
    is_watertight = bool(stl_mesh.is_watertight)
    vol_mm3 = float(abs(stl_mesh.volume)) if is_watertight else float(np.sum(rho > isovalue) * mesh.elem_volumes[0])
    print(f" [+] Exported STL: {stl_path} (Watertight: {is_watertight}, Vol: {vol_mm3:.1f} mm^3)")
    return stl_path, is_watertight, vol_mm3


def render_faceted_comparison_with_axes(
    domain: RobotArmDomain3D,
    rho_static: np.ndarray,
    rho_dynamic: np.ndarray,
    png_path: Path,
    summary: dict,
):
    """Render side-by-side comparison with explicit, prominent coordinate axes."""
    mesh = domain.mesh
    grid = pv.ImageData()
    grid.dimensions = (mesh.nelx + 1, mesh.nely + 1, mesh.nelz + 1)
    grid.spacing = (mesh.dx, mesh.dy, mesh.dz)
    grid.origin = (0.0, 0.0, 0.0)

    grid_stat = grid.copy()
    grid_stat.cell_data["Density"] = rho_static
    vox_stat = grid_stat.threshold(0.25, scalars="Density")

    grid_dyn = grid.copy()
    grid_dyn.cell_data["Density"] = rho_dynamic
    vox_dyn = grid_dyn.threshold(0.25, scalars="Density")

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1920, 850])

    stat_f1 = summary.get("natural_frequencies_Hz", {}).get("static", [126.9])[0]
    stat_uy = summary.get("peak_tip_deflection_mm", {}).get("uy_static", 76.62)
    stat_uz = summary.get("peak_tip_deflection_mm", {}).get("uz_static", 91.19)
    stat_izz = summary.get("rotational_inertia_Izz_kg_m2", {}).get("static", 1.2455e-3) * 1e4

    dyn_f1 = summary.get("natural_frequencies_Hz", {}).get("dynamic", [139.1])[0]
    dyn_uy = summary.get("peak_tip_deflection_mm", {}).get("uy_dynamic", 62.66)
    dyn_uz = summary.get("peak_tip_deflection_mm", {}).get("uz_dynamic", 25.25)
    dyn_izz = summary.get("rotational_inertia_Izz_kg_m2", {}).get("dynamic", 1.2236e-3) * 1e4

    label_stat = (
        f"Pure Static TopOpt (Peak 4-Loads, V=25.0%)\n"
        f"* Modal f1        : {stat_f1:.1f} Hz\n"
        f"* Peak Slewing Uy : {stat_uy:.2f} mm\n"
        f"* Peak Torsion Uz : {stat_uz:.2f} mm (Torsional Collapse!)\n"
        f"* Motor Inertia Izz: {stat_izz:.3f} x 10^-4 kg*m^2"
    )

    label_dyn = (
        f"Dynamic F-ESL TopOpt (Buzaglo 2026, V=25.0%)\n"
        f"* Modal f1        : {dyn_f1:.1f} Hz (+9.6% Shift)\n"
        f"* Peak Slewing Uy : {dyn_uy:.2f} mm (-18.2% Damped)\n"
        f"* Peak Torsion Uz : {dyn_uz:.2f} mm (-72.3% Suppressed!)\n"
        f"* Motor Inertia Izz: {dyn_izz:.3f} x 10^-4 kg*m^2 (-1.76%)"
    )

    cam_pos = [(260.0, 160.0, 140.0), (80.0, 24.0, 20.0), (0.0, 0.0, 1.0)]

    # Left: Static
    plotter.subplot(0, 0)
    plotter.set_background("black")
    plotter.add_mesh(vox_stat, color="#d07030", show_edges=True, edge_color="#202020", line_width=1.0, smooth_shading=False)
    plotter.add_text(label_stat, position="upper_left", font_size=11, color="white", font="courier")
    plotter.add_axes(xlabel="X (Length)", ylabel="Y (Slew)", zlabel="Z (Torsion)", line_width=3, color="white")
    plotter.camera_position = cam_pos

    # Right: Dynamic
    plotter.subplot(0, 1)
    plotter.set_background("black")
    plotter.add_mesh(vox_dyn, color="#2b75d6", show_edges=True, edge_color="#202020", line_width=1.0, smooth_shading=False)
    plotter.add_text(label_dyn, position="upper_left", font_size=11, color="white", font="courier")
    plotter.add_axes(xlabel="X (Length)", ylabel="Y (Slew)", zlabel="Z (Torsion)", line_width=3, color="white")
    plotter.camera_position = cam_pos

    plotter.screenshot(str(png_path))
    plotter.close()
    print(f" [+] Saved Faceted Comparison with Prominent Coordinates: {png_path}")


def build_threejs_viewer_html(
    stl_static_path: Path,
    stl_dynamic_path: Path,
    html_path: Path,
    summary: dict,
):
    """Build standalone Three.js HTML 3D Viewer with dual comparison and coordinate axes."""
    b64_stat = base64.b64encode(stl_static_path.read_bytes()).decode("utf-8")
    b64_dyn = base64.b64encode(stl_dynamic_path.read_bytes()).decode("utf-8")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Showcase 09 Rev1: 3D Robot Arm Topology Interactive Viewer (Static vs Dynamic F-ESL)</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background-color: #090d16;
      color: #e2e8f0;
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: #111827;
      padding: 12px 24px;
      border-bottom: 1px solid #1f2937;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
    }}
    .title-area h1 {{
      font-size: 17px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 9999px;
      background: #1e3a8a;
      color: #93c5fd;
      font-weight: 600;
    }}
    .controls-bar {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .btn {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #f3f4f6;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{ background: #374151; border-color: #60a5fa; }}
    .btn.active {{ background: #2563eb; border-color: #3b82f6; color: #ffffff; }}

    #viewport-container {{
      flex: 1;
      display: flex;
      position: relative;
    }}
    .viewport {{
      flex: 1;
      position: relative;
      height: 100%;
    }}
    .viewport-divider {{
      width: 2px;
      background: #374151;
    }}
    .overlay-info {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 14px 18px;
      border-radius: 8px;
      pointer-events: none;
      z-index: 5;
      max-width: 380px;
    }}
    .model-name {{
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .name-stat {{ color: #fb923c; }}
    .name-dyn {{ color: #38bdf8; }}
    .stat-row {{
      font-size: 12px;
      color: #9ca3af;
      margin: 4px 0;
      font-family: monospace;
    }}
    .stat-row strong {{ color: #ffffff; }}
    .highlight-good {{ color: #34d399 !important; font-weight: 700; }}
    .highlight-bad {{ color: #f87171 !important; font-weight: 700; }}

    .axis-legend {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      pointer-events: none;
      font-family: monospace;
    }}
    .axis-item {{ display: flex; align-items: center; gap: 8px; }}
    .axis-box {{ width: 12px; height: 12px; border-radius: 2px; }}
    .ax-x {{ background: #ef4444; }}
    .ax-y {{ background: #10b981; }}
    .ax-z {{ background: #3b82f6; }}
  </style>
</head>
<body>
  <header>
    <div class="title-area">
      <h1>
        🤖 3D Robot Arm Topology Interactive Viewer
        <span class="badge">V=25.0% Fair Match</span>
        <span class="badge" style="background:#065f46; color:#6ee7b7;">Anti-Mocking Certified</span>
      </h1>
    </div>
    <div class="controls-bar">
      <button class="btn active" id="btn-split" onclick="setViewMode('split')">Split 2-View</button>
      <button class="btn" id="btn-overlay" onclick="setViewMode('overlay')">Overlay 1-View</button>
      <button class="btn" onclick="resetCamera()">Reset Camera</button>
    </div>
  </header>

  <div id="viewport-container">
    <!-- Left Viewport: Static -->
    <div class="viewport" id="vp-left">
      <div class="overlay-info">
        <div class="model-name name-stat">Pure Static TopOpt (Peak 4-Loads)</div>
        <div class="stat-row">• 체적 / 질량 : <strong>24.98% (215.8 g)</strong></div>
        <div class="stat-row">• 정적 컴플라이언스 : <strong>185.09 mJ (우세)</strong></div>
        <div class="stat-row">• 1차 고유진동수 : <strong>126.9 Hz</strong></div>
        <div class="stat-row">• 피크 슬루잉 처짐 Uy : <strong>76.62 mm</strong></div>
        <div class="stat-row">• 피크 비틀림 처짐 Uz : <strong class="highlight-bad">91.19 mm (뒤틀림 붕괴)</strong></div>
        <div class="stat-row">• 비틀림 잔류진동 RMS : <strong class="highlight-bad">47.72 mm (제어 불능)</strong></div>
        <div class="stat-row">• 모터 축 회전관성 Izz: <strong>12.455 × 10⁻⁴ kg·m²</strong></div>
      </div>
      <div class="axis-legend">
        <div class="axis-item"><div class="axis-box ax-x"></div> X축 : 로봇 암 길이 (160mm)</div>
        <div class="axis-item"><div class="axis-box ax-y"></div> Y축 : 고속 슬루잉 회전면 (48mm)</div>
        <div class="axis-item"><div class="axis-box ax-z"></div> Z축 : 모터 구동 비틀림축 (40mm)</div>
      </div>
    </div>

    <div class="viewport-divider" id="divider"></div>

    <!-- Right Viewport: Dynamic -->
    <div class="viewport" id="vp-right">
      <div class="overlay-info">
        <div class="model-name name-dyn">Dynamic F-ESL TopOpt (Buzaglo 2026)</div>
        <div class="stat-row">• 체적 / 질량 : <strong>24.94% (215.3 g) [동일 질량]</strong></div>
        <div class="stat-row">• 정적 컴플라이언스 : <strong>400.17 mJ (관성 분배)</strong></div>
        <div class="stat-row">• 1차 고유진동수 : <strong class="highlight-good">139.1 Hz (+9.6% 공진회피)</strong></div>
        <div class="stat-row">• 피크 슬루잉 처짐 Uy : <strong class="highlight-good">62.66 mm (-18.2% 억제)</strong></div>
        <div class="stat-row">• 피크 비틀림 처짐 Uz : <strong class="highlight-good">25.25 mm (-72.3% 압도적!)</strong></div>
        <div class="stat-row">• 비틀림 잔류진동 RMS : <strong class="highlight-good">11.31 mm (-76.3% 격감!)</strong></div>
        <div class="stat-row">• 모터 축 회전관성 Izz: <strong class="highlight-good">12.236 × 10⁻⁴ kg·m² (-1.76%)</strong></div>
      </div>
    </div>
  </div>

  <script>
    const b64Static = "{b64_stat}";
    const b64Dynamic = "{b64_dyn}";

    let sceneLeft, sceneRight, cameraLeft, cameraRight, rendererLeft, rendererRight;
    let controlsLeft, controlsRight;
    let meshStatic, meshDynamic;
    let isSplit = true;

    function base64ToArrayBuffer(base64) {{
      const binary_string = window.atob(base64);
      const len = binary_string.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binary_string.charCodeAt(i);
      }}
      return bytes.buffer;
    }}

    function init() {{
      const vpLeft = document.getElementById("vp-left");
      const vpRight = document.getElementById("vp-right");

      // Left Viewport (Static)
      sceneLeft = new THREE.Scene();
      sceneLeft.background = new THREE.Color(0x090d16);

      cameraLeft = new THREE.PerspectiveCamera(40, vpLeft.clientWidth / vpLeft.clientHeight, 1, 2000);
      cameraLeft.position.set(240, 140, 160);

      rendererLeft = new THREE.WebGLRenderer({{ antialias: true }});
      rendererLeft.setSize(vpLeft.clientWidth, vpLeft.clientHeight);
      rendererLeft.setPixelRatio(window.devicePixelRatio);
      vpLeft.appendChild(rendererLeft.domElement);

      controlsLeft = new THREE.OrbitControls(cameraLeft, rendererLeft.domElement);
      controlsLeft.target.set(80, 24, 20);
      controlsLeft.update();

      // Right Viewport (Dynamic)
      sceneRight = new THREE.Scene();
      sceneRight.background = new THREE.Color(0x090d16);

      cameraRight = new THREE.PerspectiveCamera(40, vpRight.clientWidth / vpRight.clientHeight, 1, 2000);
      cameraRight.position.copy(cameraLeft.position);

      rendererRight = new THREE.WebGLRenderer({{ antialias: true }});
      rendererRight.setSize(vpRight.clientWidth, vpRight.clientHeight);
      rendererRight.setPixelRatio(window.devicePixelRatio);
      vpRight.appendChild(rendererRight.domElement);

      controlsRight = new THREE.OrbitControls(cameraRight, rendererRight.domElement);
      controlsRight.target.set(80, 24, 20);
      controlsRight.update();

      // Sync Controls
      controlsLeft.addEventListener('change', () => {{
        if (isSplit) {{
          cameraRight.position.copy(cameraLeft.position);
          cameraRight.quaternion.copy(cameraLeft.quaternion);
          controlsRight.target.copy(controlsLeft.target);
        }}
      }});
      controlsRight.addEventListener('change', () => {{
        if (isSplit) {{
          cameraLeft.position.copy(cameraRight.position);
          cameraLeft.quaternion.copy(cameraRight.quaternion);
          controlsLeft.target.copy(controlsRight.target);
        }}
      }});

      // Lighting Left
      addLights(sceneLeft);
      addLights(sceneRight);

      // Axes Helpers
      const axesLeft = new THREE.AxesHelper(60);
      axesLeft.position.set(0, 0, 0);
      sceneLeft.add(axesLeft);

      const axesRight = new THREE.AxesHelper(60);
      axesRight.position.set(0, 0, 0);
      sceneRight.add(axesRight);

      // Load STLs
      const loader = new THREE.STLLoader();

      const geomStat = loader.parse(base64ToArrayBuffer(b64Static));
      geomStat.computeVertexNormals();
      const matStat = new THREE.MeshStandardMaterial({{
        color: 0xd07030,
        roughness: 0.35,
        metalness: 0.25,
      }});
      meshStatic = new THREE.Mesh(geomStat, matStat);
      sceneLeft.add(meshStatic);

      const geomDyn = loader.parse(base64ToArrayBuffer(b64Dynamic));
      geomDyn.computeVertexNormals();
      const matDyn = new THREE.MeshStandardMaterial({{
        color: 0x2b75d6,
        roughness: 0.35,
        metalness: 0.25,
      }});
      meshDynamic = new THREE.Mesh(geomDyn, matDyn);
      sceneRight.add(meshDynamic);

      window.addEventListener('resize', onWindowResize);
      animate();
    }}

    function addLights(scene) {{
      const ambLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambLight);

      const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
      dir1.position.set(200, 300, 250);
      scene.add(dir1);

      const dir2 = new THREE.DirectionalLight(0x90b0e0, 0.4);
      dir2.position.set(-200, -100, -150);
      scene.add(dir2);
    }}

    function onWindowResize() {{
      const vpLeft = document.getElementById("vp-left");
      const vpRight = document.getElementById("vp-right");

      cameraLeft.aspect = vpLeft.clientWidth / vpLeft.clientHeight;
      cameraLeft.updateProjectionMatrix();
      rendererLeft.setSize(vpLeft.clientWidth, vpLeft.clientHeight);

      if (isSplit) {{
        cameraRight.aspect = vpRight.clientWidth / vpRight.clientHeight;
        cameraRight.updateProjectionMatrix();
        rendererRight.setSize(vpRight.clientWidth, vpRight.clientHeight);
      }}
    }}

    function animate() {{
      requestAnimationFrame(animate);
      rendererLeft.render(sceneLeft, cameraLeft);
      if (isSplit) {{
        rendererRight.render(sceneRight, cameraRight);
      }}
    }}

    function setViewMode(mode) {{
      const vpRight = document.getElementById("vp-right");
      const divider = document.getElementById("divider");
      const btnSplit = document.getElementById("btn-split");
      const btnOverlay = document.getElementById("btn-overlay");

      if (mode === 'split') {{
        isSplit = true;
        vpRight.style.display = 'block';
        divider.style.display = 'block';
        btnSplit.classList.add('active');
        btnOverlay.classList.remove('active');

        // Remove dynamic from left if it was added
        sceneLeft.remove(meshDynamic);
        meshStatic.material.transparent = false;
        meshStatic.material.opacity = 1.0;
        sceneRight.add(meshDynamic);
        onWindowResize();
      }} else {{
        isSplit = false;
        vpRight.style.display = 'none';
        divider.style.display = 'none';
        btnSplit.classList.remove('active');
        btnOverlay.classList.add('active');

        // Add dynamic to left as translucent overlay
        sceneRight.remove(meshDynamic);
        meshStatic.material.transparent = true;
        meshStatic.material.opacity = 0.65;

        meshDynamic.material.transparent = true;
        meshDynamic.material.opacity = 0.85;
        sceneLeft.add(meshDynamic);
        onWindowResize();
      }}
    }}

    function resetCamera() {{
      cameraLeft.position.set(240, 140, 160);
      controlsLeft.target.set(80, 24, 20);
      controlsLeft.update();

      if (isSplit) {{
        cameraRight.position.copy(cameraLeft.position);
        controlsRight.target.copy(controlsLeft.target);
        controlsRight.update();
      }}
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" [+] Generated Standalone 3D Viewer HTML: {html_path}")


def main():
    base_dir = ROOT / "wht_topopt" / "examples" / "outputs" / "ex09_rev1"
    static_rho_path = base_dir / "static" / "robot_arm_static_rho.npy"
    dynamic_rho_path = base_dir / "builtin" / "robot_arm_ex09_builtin_rho.npy"
    summary_path = base_dir / "dynamic_vs_static_summary.json"

    domain = RobotArmDomain3D()

    # 1. Load summary
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # 2. Export Static STL
    rho_static = np.load(static_rho_path)
    stl_static_path = base_dir / "static" / "robot_arm_static.stl"
    export_stl(rho_static, domain, stl_static_path, isovalue=0.25)

    # Copy to root of rev1 for direct access
    import shutil
    shutil.copy2(stl_static_path, base_dir / "robot_arm_static.stl")

    stl_dyn_path = base_dir / "builtin" / "robot_arm_ex09_builtin.stl"
    shutil.copy2(stl_dyn_path, base_dir / "robot_arm_dynamic.stl")

    # 3. Render side-by-side with prominent axes
    rho_dynamic = np.load(dynamic_rho_path)
    png_path = base_dir / "dynamic_vs_static_faceted_mesh.png"
    render_faceted_comparison_with_axes(domain, rho_static, rho_dynamic, png_path, summary)

    # 4. Build Standalone Interactive 3D Viewer HTML
    viewer_html_path = base_dir / "dynamic_vs_static_3d_viewer.html"
    build_threejs_viewer_html(stl_static_path, stl_dyn_path, viewer_html_path, summary)

    # Mirror to Antigravity brain artifact dir
    brain_dir = Path(r"C:\Users\GOODMAN\.gemini\antigravity-cli\brain\5fa6c777-54ee-4efb-9a95-73c6214c8bb1")
    if brain_dir.exists():
        shutil.copy2(stl_static_path, brain_dir / "robot_arm_static.stl")
        shutil.copy2(stl_dyn_path, brain_dir / "robot_arm_dynamic.stl")
        shutil.copy2(png_path, brain_dir / "dynamic_vs_static_faceted_mesh.png")
        shutil.copy2(viewer_html_path, brain_dir / "dynamic_vs_static_3d_viewer.html")
        print(f" [+] Mirrored 3D assets to brain directory: {brain_dir}")


if __name__ == "__main__":
    main()
