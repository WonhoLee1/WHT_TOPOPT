from pathlib import Path

report_path = Path("outputs/ge_bracket/ge_bracket_challenge_report.html")
content = report_path.read_text(encoding="utf-8")

old_sec3 = """    <!-- Section 3: Visualizations (3D STL & 3-Views) -->
    <div class="card">
      <div class="card-title">
        <span>🧊</span> 3. 3D 수밀 표면 메쉬 (Watertight STL) 및 3차원 투시도
      </div>
      <div class="media-box">
        <img src="ge_beso_V15_mc40_3d.png" alt="GE Bracket BESO 3D Rendering">
        <div class="media-caption">
          <strong>GE Jet Engine Bracket BESO V=0.15 3D Isosurface (Marching Cubes)</strong> — 4개 볼트 고정면에서 2개 러그 핀을 지탱하는 유선형 티타늄 트러스 케이지가 자발적으로 형성됨. 핀 주위의 비틀림 토크(Case 4)를 지지하기 위해 러그 보어 외곽에 두꺼운 원통 쉘이 구축됨.
        </div>
      </div>
    </div>"""

new_sec3 = """    <!-- Section 3: Isometric Faceted Hex8 Voxel Mesh (ex08/ex09 Convention) -->
    <div class="card">
      <div class="card-title">
        <span>🧊</span> 3. ex08 / ex09 규격 Iso View 이산 요소 화면 (Faceted Hex8 Voxel Mesh)
      </div>
      <div class="media-box" style="margin-bottom: 24px;">
        <img src="ge_bracket_dual_voxels_iso.png" alt="GE Bracket Dual Faceted Voxels Iso View">
        <div class="media-caption">
          <strong>[Side-by-Side 1:1 비교] BESO (좌, 파랑) vs SIMP-OC (우, 주황) Iso View 이산 Hex8 요소 화면</strong><br>
          • <strong>PyVista Faceted Mesh</strong>: <code>smooth_shading=False</code> 및 <code>show_edges=True (edge_color=#202020)</code>를 적용하여 연속 곡면 보간 이전의 순수 4.4 mm 고체 Hex8 복셀 요소를 정밀 시각화.<br>
          • <strong>위상 차이</strong>: BESO는 4대 다중하중 하에서 중앙 러그 핀으로부터 바닥 볼트 보스 4개로 연결되는 아치형 입체 케이지를 명확한 0/1 이산 경계로 형성하여 응력 집중을 균일 분산(최대 761.9 MPa). 반면 SIMP-OC는 불연속 단면으로 인해 코너 부위에 국소 응력 집중(최대 1023.9 MPa) 발생.
        </div>
      </div>

      <div class="media-box" style="margin-bottom: 24px;">
        <img src="ge_bracket_beso_voxels_iso.png" alt="GE Bracket BESO Faceted Voxels Iso View">
        <div class="media-caption">
          <strong>[BESO V0.15 단독 고해상도 Iso View]</strong> 244.5 g (우승작 대비 -25.2% 경량화), 최대 von Mises 761.9 MPa (항복 903 MPa 대비 안전계수 1.19 합격). 러그 보어 외주부의 토크 저항 원통 쉘과 4점 볼트 고정면을 연결하는 3차원 브레이싱 트러스의 Hex8 요소망 구조가 선명하게 확인됨.
        </div>
      </div>

      <div class="media-box" style="margin-bottom: 24px;">
        <img src="ge_bracket_triple_voxels_iso.png" alt="GE Bracket Triple Faceted Voxels Iso View">
        <div class="media-caption">
          <strong>[3대 알고리즘 1:3 종합 비교] BESO vs SIMP-OC vs AM (+Y Overhang)</strong><br>
          AM 적층제조 알고리즘(녹색)은 +Y 방향 45도 자립 경사각 필터를 엄격히 적용함에 따라 오버행이 금지되어 하단 지지 기둥이 수직으로 두껍게 보강되었으며, 이로 인해 유효 하중 경로가 제한되어 컴플라이언스와 응력이 상승하는 공학적 트레이드오프를 보여줌.
        </div>
      </div>
    </div>

    <!-- Section 4: 3D STL & 3-Views -->
    <div class="card">
      <div class="card-title">
        <span>🌐</span> 4. 3D 수밀 표면 메쉬 (Watertight STL) 및 3차원 투시도
      </div>
      <div class="media-box">
        <img src="ge_beso_V15_mc40_3d.png" alt="GE Bracket BESO 3D Rendering">
        <div class="media-caption">
          <strong>GE Jet Engine Bracket BESO V=0.15 3D Isosurface (Marching Cubes)</strong> — 4개 볼트 고정면에서 2개 러그 핀을 지탱하는 유선형 티타늄 트러스 케이지가 자발적으로 형성됨. 핀 주위의 비틀림 토크(Case 4)를 지지하기 위해 러그 보어 외곽에 두꺼운 원통 쉘이 구축됨.
        </div>
      </div>
    </div>"""

# Normalize newlines
c_norm = content.replace("\r\n", "\n")
o_norm = old_sec3.replace("\r\n", "\n")
n_norm = new_sec3.replace("\r\n", "\n")

if o_norm in c_norm:
    c_norm = c_norm.replace(o_norm, n_norm)
    report_path.write_text(c_norm, encoding="utf-8")
    print("Successfully updated HTML report!")
else:
    print("Target block not found in HTML report.")
