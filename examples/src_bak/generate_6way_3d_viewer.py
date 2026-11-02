"""Generates an all-in-one standalone interactive HTML 3D Viewer for the 6 Humanoid Robot Arm Models:
1. Static 3D Model (Slate Silver)
2. Free 3D Dynamic Model (Sky Blue)
3. Y-Extrusion Dynamic Model (Emerald Green)
4. Z-Extrusion Dynamic Model (Amber Gold)
5. Z-Extrusion with -Z Bottom Skin Plate Model (Coral Pink)
6. Outer Shell Hull + Interior Extruded Z-Ribs Model (Vibrant Purple)

Embeds all 6 binary STLs as Base64 so the HTML opens instantly in any browser without CORS issues.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


def build_6way_viewer() -> None:
    base_dir = Path("wht_topopt/examples/outputs/ex09_rev1")
    metrics_path = base_dir / "all_6_models_comparison_metrics.json"

    if not metrics_path.exists():
        print(f"Error: Metrics not found at {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Encode STLs to Base64
    stl_files = {
        "static": base_dir / "static" / "robot_arm_static.stl",
        "builtin": base_dir / "builtin" / "robot_arm_ex09_builtin.stl",
        "ext_y": base_dir / "ext_y" / "robot_arm_ex09_ext_y.stl",
        "ext_z": base_dir / "ext_z" / "robot_arm_ex09_ext_z.stl",
        "ext_z_plate": base_dir / "ext_z_plate" / "robot_arm_ex09_ext_z_plate.stl",
        "ext_shell_rib": base_dir / "ext_shell_rib" / "robot_arm_ex09_ext_shell_rib.stl",
    }

    base64_models = {}
    for mid, path in stl_files.items():
        if not path.exists():
            print(f"Warning: STL not found: {path}")
            continue
        with open(path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            base64_models[mid] = b64_str
        print(f" [+] Encoded {mid}: {len(b64_str):,} Base64 chars")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>휴머노이드 로봇 암 3D 동적 위상최적화 6-모델 마스터 3D 비교 뷰어</title>
  <style>
    :root {{
      --bg-dark: #090d16;
      --card-bg: rgba(15, 23, 42, 0.90);
      --border-color: #1e293b;
      --text-main: #f8fafc;
      --text-sub: #94a3b8;
      --accent-static: #94a3b8;
      --accent-dynamic: #38bdf8;
      --accent-exty: #34d399;
      --accent-extz: #fbbf24;
      --accent-extz-plate: #ec4899;
      --accent-shell-rib: #a855f7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Malgun Gothic", sans-serif;
      background: var(--bg-dark);
      color: var(--text-main);
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: rgba(15, 23, 42, 0.95);
      border-bottom: 1px solid var(--border-color);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
    }}
    .title-group h1 {{
      font-size: 16px;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .title-group p {{
      font-size: 11px;
      color: var(--text-sub);
      margin-top: 2px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn:hover {{
      background: #334155;
      border-color: #475569;
    }}
    .btn.active {{
      background: #0284c7;
      border-color: #38bdf8;
      color: #ffffff;
    }}
    .main-container {{
      position: relative;
      flex: 1;
      width: 100%;
      height: 100%;
    }}
    #viewer-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .hud-panel {{
      position: absolute;
      top: 15px;
      left: 15px;
      width: 420px;
      background: var(--card-bg);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      z-index: 5;
      font-size: 12px;
    }}
    .hud-header {{
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: #cbd5e1;
      border-bottom: 1px solid #334155;
      padding-bottom: 6px;
    }}
    .model-selector-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
      margin-bottom: 12px;
    }}
    .model-btn {{
      padding: 7px 10px;
      border-radius: 6px;
      border: 1px solid #334155;
      background: #1e293b;
      color: #e2e8f0;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      text-align: left;
      transition: all 0.2s;
    }}
    .model-btn:hover {{ background: #273549; }}
    .model-btn.active {{
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
    }}
    .model-btn.active.purple {{
      border-color: #a855f7;
      background: rgba(168, 85, 247, 0.18);
      color: #a855f7;
    }}
    .model-btn.active.pink {{
      border-color: #ec4899;
      background: rgba(236, 72, 153, 0.18);
      color: #ec4899;
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 6px;
      font-size: 11px;
    }}
    .metrics-table th {{
      text-align: left;
      color: var(--text-sub);
      padding: 4px 6px;
      font-weight: 500;
      border-bottom: 1px solid #1e293b;
    }}
    .metrics-table td {{
      padding: 5px 6px;
      border-bottom: 1px solid #1e293b;
    }}
    .metrics-table tr:hover {{
      background: rgba(255,255,255,0.03);
    }}
    .tag {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
    }}
    .view-controls {{
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 30px;
      padding: 6px 14px;
      display: flex;
      gap: 8px;
      z-index: 5;
    }}
    .axes-legend {{
      position: absolute;
      bottom: 20px;
      right: 20px;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      z-index: 5;
      font-size: 11px;
      line-height: 1.6;
    }}
    .axis-item {{ display: flex; align-items: center; gap: 8px; }}
    .axis-color {{ width: 12px; height: 3px; border-radius: 2px; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>🚀 휴머노이드 로봇 암 3D 동적 위상최적화 6-모델 마스터 뷰어</h1>
      <p>Showcase 09 Master | Static vs Free 3D vs Y-Extrude vs Z-Extrude vs Z-Plate vs Outer Shell + Interior Ribs</p>
    </div>
    <div class="toolbar">
      <button class="btn active" id="btn-mode-single" onclick="setViewMode('single')">단일 모델 검토</button>
      <button class="btn" id="btn-mode-overlay" onclick="setViewMode('overlay')">6-모델 동시 중첩</button>
      <button class="btn" id="btn-wireframe" onclick="toggleWireframe()">와이어프레임 ON/OFF</button>
    </div>
  </header>

  <div class="main-container">
    <div id="viewer-canvas"></div>

    <div class="hud-panel">
      <div class="hud-header">
        <span>선택 모델 및 다물리 성능 지표</span>
        <span id="active-model-tag" class="tag" style="background:#a855f7; color:white;">Model 6</span>
      </div>

      <div class="model-selector-grid">
        <button class="model-btn" id="btn-model-static" onclick="selectModel('static')">
          <div style="color:var(--accent-static)">● Model 1: Static 3D</div>
          <div style="font-size:9px; color:#94a3b8">정적 전용 (기준)</div>
        </button>
        <button class="model-btn" id="btn-model-builtin" onclick="selectModel('builtin')">
          <div style="color:var(--accent-dynamic)">● Model 2: Free 3D Dyn</div>
          <div style="font-size:9px; color:#94a3b8">자유 동적 F-ESL</div>
        </button>
        <button class="model-btn" id="btn-model-ext_y" onclick="selectModel('ext_y')">
          <div style="color:var(--accent-exty)">● Model 3: Y-Extrude</div>
          <div style="font-size:9px; color:#94a3b8">Y축 압출 (슬루잉면)</div>
        </button>
        <button class="model-btn" id="btn-model-ext_z" onclick="selectModel('ext_z')">
          <div style="color:var(--accent-extz)">● Model 4: Z-Extrude</div>
          <div style="font-size:9px; color:#94a3b8">Z축 압출 (모터축)</div>
        </button>
        <button class="model-btn" id="btn-model-ext_z_plate" onclick="selectModel('ext_z_plate')">
          <div style="color:var(--accent-extz-plate)">● Model 5: Z-Ext + Plate</div>
          <div style="font-size:9px; color:#94a3b8">바닥판 + Z-수직리브</div>
        </button>
        <button class="model-btn active purple" id="btn-model-ext_shell_rib" onclick="selectModel('ext_shell_rib')">
          <div style="color:var(--accent-shell-rib)">★ Model 6: Shell + Ribs</div>
          <div style="font-size:9px; color:#94a3b8">외형 쉘(바닥+양측벽) + 내부 리브</div>
        </button>
      </div>

      <table class="metrics-table">
        <tbody>
          <tr>
            <th>체적률 / 실측 질량</th>
            <td id="m-volume" style="font-weight:700">25.0% (215.8 g)</td>
          </tr>
          <tr>
            <th>1차 고유진동수 (f1)</th>
            <td id="m-f1" style="font-weight:700; color:#38bdf8">-</td>
          </tr>
          <tr>
            <th>모터축 관성모멘트 (Izz)</th>
            <td id="m-izz">-</td>
          </tr>
          <tr>
            <th>피크 접선 처짐 (Uy)</th>
            <td id="m-peak-uy" style="font-weight:700">-</td>
          </tr>
          <tr>
            <th>피크 비틀림 처짐 (Uz)</th>
            <td id="m-peak-uz" style="font-weight:700">-</td>
          </tr>
          <tr>
            <th>잔류 진동 RMS (Uz)</th>
            <td id="m-rms-uz" style="font-weight:700">-</td>
          </tr>
          <tr>
            <th>2.5D CNC 밀링 적합도</th>
            <td id="m-cnc">-</td>
          </tr>
          <tr>
            <th>제조성 제약 구조</th>
            <td id="m-ext-axis">-</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="view-controls">
      <button class="btn" onclick="setCameraView('iso')">Isometric</button>
      <button class="btn" onclick="setCameraView('top')">Top (XY: 슬루잉면)</button>
      <button class="btn" onclick="setCameraView('side')">Side (XZ: 주 구조면)</button>
      <button class="btn" onclick="setCameraView('front')">Front (YZ: 모터단면)</button>
    </div>

    <div class="axes-legend">
      <div style="font-weight:700; margin-bottom:4px; color:#cbd5e1">3D 좌표축 기즈모</div>
      <div class="axis-item">
        <div class="axis-color" style="background:#ef4444"></div>
        <span>X축 (Red): 암 길이 방향 (160mm)</span>
      </div>
      <div class="axis-item">
        <div class="axis-color" style="background:#22c55e"></div>
        <span>Y축 (Green): 슬루잉 회전면 (48mm)</span>
      </div>
      <div class="axis-item">
        <div class="axis-color" style="background:#3b82f6"></div>
        <span>Z축 (Blue): 모터 회전/비틀림축 (40mm)</span>
      </div>
    </div>
  </div>

  <script>
    const METRICS = {json.dumps(metrics_data, ensure_ascii=False)};
    const STL_B64 = {json.dumps(base64_models)};

    let scene, camera, renderer, controls;
    let meshes = {{}};
    let currentMode = 'single';
    let activeModelId = 'ext_shell_rib';
    let isWireframe = false;

    function init() {{
      const container = document.getElementById('viewer-canvas');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090d16);

      camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 2000);
      setCameraView('iso');

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(window.innerWidth, window.innerHeight - 50);
      renderer.shadowMap.enabled = true;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.2;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(80, 24, 20);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      // Lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
      dirLight1.position.set(200, 300, 400);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.8);
      dirLight2.position.set(-200, -200, -200);
      scene.add(dirLight2);

      // 3D Axes Gizmo
      const axesHelper = new THREE.AxesHelper(60);
      axesHelper.position.set(0, 0, 0);
      scene.add(axesHelper);

      // Domain Bounding Box wireframe
      const boxGeo = new THREE.BoxGeometry(160, 48, 40);
      boxGeo.translate(80, 24, 20);
      const boxWire = new THREE.LineSegments(
        new THREE.EdgesGeometry(boxGeo),
        new THREE.LineBasicMaterial({{ color: 0x334155, linewidth: 1 }})
      );
      scene.add(boxWire);

      loadModels();

      window.addEventListener('resize', onWindowResize, false);
      animate();
    }}

    function base64ToArrayBuffer(base64) {{
      const binaryString = window.atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}
      return bytes.buffer;
    }}

    function loadModels() {{
      const loader = new THREE.STLLoader();
      const colors = {{
        'static': 0x94a3b8,
        'builtin': 0x38bdf8,
        'ext_y': 0x34d399,
        'ext_z': 0xfbbf24,
        'ext_z_plate': 0xec4899,
        'ext_shell_rib': 0xa855f7
      }};

      for (const [id, b64] of Object.entries(STL_B64)) {{
        const buffer = base64ToArrayBuffer(b64);
        const geometry = loader.parse(buffer);
        geometry.computeVertexNormals();

        const material = new THREE.MeshStandardMaterial({{
          color: colors[id],
          metalness: 0.35,
          roughness: 0.35,
          wireframe: isWireframe,
          transparent: true,
          opacity: 0.95
        }});

        const mesh = new THREE.Mesh(geometry, material);
        mesh.visible = (id === activeModelId);
        scene.add(mesh);
        meshes[id] = mesh;
      }}
      updateHUD(activeModelId);
    }}

    function selectModel(id) {{
      activeModelId = id;
      document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active', 'purple', 'pink'));
      const activeBtn = document.getElementById('btn-model-' + id);
      activeBtn.classList.add('active');
      if (id === 'ext_shell_rib') activeBtn.classList.add('purple');
      if (id === 'ext_z_plate') activeBtn.classList.add('pink');

      if (currentMode === 'single') {{
        for (const [mid, mesh] of Object.entries(meshes)) {{
          mesh.visible = (mid === id);
          mesh.material.opacity = 0.95;
        }}
      }}
      updateHUD(id);
    }}

    function setViewMode(mode) {{
      currentMode = mode;
      document.getElementById('btn-mode-single').classList.toggle('active', mode === 'single');
      document.getElementById('btn-mode-overlay').classList.toggle('active', mode === 'overlay');

      if (mode === 'single') {{
        selectModel(activeModelId);
      }} else if (mode === 'overlay') {{
        for (const [mid, mesh] of Object.entries(meshes)) {{
          mesh.visible = true;
          mesh.material.opacity = 0.35;
        }}
      }}
    }}

    function toggleWireframe() {{
      isWireframe = !isWireframe;
      for (const mesh of Object.values(meshes)) {{
        mesh.material.wireframe = isWireframe;
      }}
      document.getElementById('btn-wireframe').classList.toggle('active', isWireframe);
    }}

    function setCameraView(preset) {{
      if (preset === 'iso') {{
        camera.position.set(280, 140, 180);
        camera.up.set(0, 0, 1);
      }} else if (preset === 'top') {{
        camera.position.set(80, 24, 320);
        camera.up.set(0, 1, 0);
      }} else if (preset === 'side') {{
        camera.position.set(80, -280, 20);
        camera.up.set(0, 0, 1);
      }} else if (preset === 'front') {{
        camera.position.set(380, 24, 20);
        camera.up.set(0, 0, 1);
      }}
      if (controls) controls.target.set(80, 24, 20);
    }}

    function updateHUD(id) {{
      const met = METRICS.find(m => m.id === id);
      if (!met) return;

      document.getElementById('active-model-tag').textContent = met.model_name.split(' (')[0];
      document.getElementById('active-model-tag').style.background = met.color;
      document.getElementById('m-volume').textContent = `${{(met.volume_fraction * 100).toFixed(1)}}% (${{met.mass_g.toFixed(1)}} g)`;
      document.getElementById('m-f1').textContent = `${{met.f1_hz.toFixed(1)}} Hz`;
      document.getElementById('m-izz').textContent = `${{(met.i_zz_arm_kg_m2 * 1e4).toFixed(2)}} × 10⁻⁴ kg·m²`;
      document.getElementById('m-peak-uy').textContent = `${{met.peak_uy_mm.toFixed(2)}} mm`;
      document.getElementById('m-peak-uz').textContent = `${{met.peak_uz_mm.toFixed(2)}} mm`;
      document.getElementById('m-rms-uz').textContent = `${{met.res_uz_rms_mm.toFixed(2)}} mm`;
      document.getElementById('m-cnc').textContent = met.cnc_score;
      document.getElementById('m-ext-axis').textContent = met.ext_axis;
    }}

    function onWindowResize() {{
      camera.aspect = window.innerWidth / (window.innerHeight - 50);
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight - 50);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      if (controls) controls.update();
      renderer.render(scene, camera);
    }}

    window.onload = init;
  </script>
</body>
</html>
"""

    out_html = base_dir / "dynamic_extrusion_master_viewer.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n [SUCCESS] Built Interactive Master 6-Way 3D Viewer: {out_html}", flush=True)


if __name__ == "__main__":
    build_6way_viewer()
