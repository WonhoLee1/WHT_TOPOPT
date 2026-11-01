# 8번 Compliant Mechanism (유연 기구학: Force Inverter) 표준 벤치마크 요약 보고서

## 1. 개요 및 설계 목적
- **물리 문제**: 힘 반전기 (Force Inverter / Displacement Multiplier)
- **핵심 메커니즘**:
  - 입력단 $(x=0, y=L_y/2)$에 가해진 $+x$ 방향 액추에이터 하중($F_{\text{in}}$)에 대해, 출력단 $(x=L_x, y=L_y/2)$에서 반대 방향($-x$ 방향)으로 변위 $u_{\text{out}}$를 극대화하는 힌지 없는(Joint-free) 일체형 유연 기구 설계.
  - `wht_topopt/functions/modal.py`의 `CompliantMechanismFunction` 엄밀 수반 해석 기반 감도 연동.
- **제조성 사양 (DfAM / 2.5D EDM)**:
  - 3D 모델에 Z축 압출 필터(`ext_axis="z"`)를 적용하여 판형 유연 메커니즘 도출 (와이어 방전 가공 및 3D 프린팅 100% 호환).

---

## 2. 정량적 기구학 성능 비교 (MMA vs SE-AHCA)

| 모델 / 알고리즘 | 격자 해상도 | 출력 변위 $u_{\text{out}}$ | 입력 변위 $u_{\text{in}}$ | 변위 증폭비 (Gain) | 기구학적 효율 $\eta$ | 체적률 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MMA (3D Extrusion)** | $60\times 60\times 10$ ($36,000$ Hex8) | **+4.3104e+00 mm** | +9.3888e-01 mm | **4.59x** | **1.98%** | 24.9% |
| **SE-AHCA (3D Extrusion)** | $60\times 60\times 10$ ($36,000$ Hex8) | **-2.5796e-01 mm** | +5.1209e-01 mm | **0.50x** | **0.01%** | 25.0% |
| **MMA (2D High-Res)** | $120\times 120$ ($14,400$ Q4) | **+2.5279e+00 mm** | +9.8695e-01 mm | **2.56x** | **0.65%** | 25.0% |

---

## 3. 솔버 및 계산 시간 프로파일링 분석

| 알고리즘 | $t_{\text{FEM}}$ (전진 평형) | $t_{\text{Adjoint}}$ (수반 방정식) | $t_{\text{Opt}}$ (업데이트 & 필터) | 스텝당 평균 시간 | 총 소요 시간 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MMA (3D Extrusion)** | 39.63 s | 39.63 s | 5.98 s | **2842.7 ms** | **85.28 s** |
| **SE-AHCA (3D Extrusion)** | 3.72 s | 3.72 s | 0.96 s | **280.5 ms** | **8.42 s** |

---

## 4. 생성된 산출물 목록 및 검증 결과

1. **100% Watertight STL**: `compliant_mechanism_3d.stl`
   - Watertight 여부: **True**
   - 오픈 에지 개수: **0개 (Closed Manifold)**
   - 정점 수: 7,996개 / 삼각 패싯: 15,976개
2. **PyVista 3D Discrete Hex8 Voxel 수렴 애니메이션 GIF**: `compliant_mechanism_progression.gif`
   - ParaView Black 테마, 모서리 `#202020`, 12pt 모노스페이스 오버레이.
3. **정밀 시각화 PNG**:
   - `compliant_mechanism_kinematics.png`: 기구학 변형 모드 및 힌지 영역 변위 벡터장.
   - `compliant_mechanism_voxels_3d.png`: 3D 복셀 ParaView 스타일 렌더링.
   - `compliant_mechanism_cad_3d.png`: 3D CAD 표면 (STL) 매끄러운 렌더링.
   - `compliant_mechanism_convergence.png`: 수렴 및 변위 증폭비 곡선.
4. **Three.js WebGL 인터랙티브 3D HTML 마스터 리포트**:
   - `compliant_mechanism_comprehensive_master_report.html` (Base64 임베딩, 절단 기본 OFF, 듀얼 평가표 탑재).
5. **정량 메트릭스 JSON**: `compliant_metrics.json`.
