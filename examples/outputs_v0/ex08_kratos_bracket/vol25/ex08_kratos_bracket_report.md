# Showcase 08: Kratos 네이티브 고해상도 3D 브래킷 제조 제약 최적화 벤치마크 보고서

**작성 일자**: 2026-10-31  
**해석 엔진**: Kratos Multiphysics 10.4.4 (In-Memory Cached ModelPart + Intel MKL PARDISO AVX-512)  
**최적화 알고리즘**: Svanberg Method of Moving Asymptotes (MMA) + 체적 가중 수반 감도 연쇄 미분  
**비교 대상**:
1. **Case 1: 자유 3D 유기적 형상 (Free 3D Organic)** — 제약 없는 3차원 최적 하중 경로 (금속 3D 프린팅 / SLM 최적)
2. **Case 2: Y방향 압출 제조 제약 형상 (Y-Extruded 2.5D)** — Y축 균일 2D 단면 강제 (프로파일 압출 / 와이어 방전 / 2.5D 밀링 최적)

---

## 1. 수치해석 및 최적화 사양

| 항목 | Case 1: 자유 3D 유기적 형상 | Case 2: Y방향 압출 제조 제약 | 비고 |
|---|:---:|:---:|---|
| **도메인 크기** | $120.0 \times 60.0 \times 50.0\text{ mm}$ | $120.0 \times 60.0 \times 50.0\text{ mm}$ | Bounding Box 동일 |
| **요소 분할** | $50 \times 25 \times 20 = \mathbf{25,000}$ Hex8 | $50 \times 25 \times 20 = \mathbf{25,000}$ Hex8 | 기존 대비 **2배 고해상도** |
| **총 자유도 (DOFs)** | $27,846$ Nodes $\rightarrow \mathbf{83,538}$ DOFs | $27,846$ Nodes $\rightarrow \mathbf{83,538}$ DOFs | 3차원 변위 벡터 |
| **요소 격자 크기** | $dx=2.40\text{ mm}, dy=2.40\text{ mm}, dz=2.50\text{ mm}$ | $dx=2.40\text{ mm}, dy=2.40\text{ mm}, dz=2.50\text{ mm}$ | 등방성 큐브 격자 |
| **재질 특성** | Ti-6Al-4V ($E_0=110\text{ GPa}, \nu=0.31$) | Ti-6Al-4V ($E_0=110\text{ GPa}, \nu=0.31$) | 항공우주 티타늄 Grade 5 |
| **허용 응력 기준** | 항복강도 $\sigma_y = 830.0\text{ MPa}$ | 항복강도 $\sigma_y = 830.0\text{ MPa}$ | 공칭 인장 항복 |
| **하중 조건** | 핀 러그 $F_z = -1,000.0\text{ N}$ | 핀 러그 $F_z = -1,000.0\text{ N}$ | 하향 전단/인장 하중 |
| **구속 조건** | 볼트 홀 4개 완전 고정 (744 DOFs) | 볼트 홀 4개 완전 고정 (744 DOFs) | 동일 구속 조건 |
| **제조 제약** | **없음 (자유 3D)** | **$Y$축 단면 압출 제약 (`ExtrusionFilter`)** | 단면 분산 $< 10^{-15}$ |
| **목표 체적비** | $\mathbf{25.0\%}$ (75% 경량화) | $\mathbf{25.0\%}$ (75% 경량화) | 초경량화 조건 |
| **최대 반복수** | $\mathbf{50}$ Iterations | $\mathbf{50}$ Iterations | Kratos 인메모리 완주 |

---

## 2. 정량적 성능 비교 및 제조성 트레이드오프

| 성능 지표 | Case 1: 자유 3D 형상 | Case 2: Y-압출 제약 형상 | 비교 분석 및 엔지니어링 의미 |
|---|:---:|:---:|---|
| **최종 컴플라이언스 ($c$)** | **85.6066 mJ** | **86.1130 mJ** | 압출 제약으로 인한 강성 페널티: **+0.6%** |
| **최종 체적 분율 ($V/V_0$)** | **24.99%** | **24.99%** | 양 케이스 모두 목표 체적 25.0% 정밀 수렴 |
| **최대 응력 (von Mises)** | **3.40 MPa** | **39.86 MPa** | 양 케이스 모두 항복한계(830 MPa) 이하 안전 |
| **항복 안전계수 (SF)** | **244.5x** | **20.8x** | 구조적 안전 마진 충분 확보 |
| **1차 고유진동수 ($f_1$)** | **76047.2 Hz** | **71105.2 Hz** | 동역학적 특성 비교 |
| **질량 중심 이동량 ($\Delta\text{CoM}$)** | 4.577 mm | 6.295 mm | 대칭성 및 밸런스 유지 |
| **CAD 수밀성 (Watertight)** | **True** (98901.8 $\text{mm}^3$) | **True** (106250.8 $\text{mm}^3$) | 100% 닫힌 폐곡면 바이너리 STL |
| **추천 생산 공정** | **금속 3D 프린팅 (SLM / DMLS)** | **압출 가공 + 2.5D 와이어 방전 / 밀링** | 제조 비용 70~80% 절감 가능 |

---

## 3. 3D 형상 및 수렴 이력 비교

### 3.1 3D CAD 서피스 나란히 비교 (Side-by-Side 3D View)
좌측(청색)은 3차원 유기적 자유 형상, 우측(황색)은 Y방향 일정한 단면을 유지하는 압출 가공 최적화 형상입니다.
![Side-by-Side 3D Comparison](bracket_ex08_comparison_3d.png)

### 3.2 다중 물리 수렴 이력 비교 곡선 (4-Panel Convergence Comparison)
![Dual Convergence Comparison](convergence_ex08_comparison.png)

### 3.3 이산 Hex8 솔리드 볼륨 형상 (Discrete Voxels View)
| Case 1: 자유 3D 이산 볼륨 | Case 2: Y-압출 이산 볼륨 |
|:---:|:---:|
| ![bracket_ex08_free_3d_voxels_3d.png](bracket_ex08_free_3d_voxels_3d.png) | ![bracket_ex08_extruded_y_voxels_3d.png](bracket_ex08_extruded_y_voxels_3d.png) |

### 3.4 50 Iteration 최적화 진화 애니메이션 (Dynamic GIFs)
| Case 1: 자유 3D 진화 GIF | Case 2: Y-압출 진화 GIF |
|:---:|:---:|
| ![bracket_ex08_free_3d_progression.gif](bracket_ex08_free_3d_progression.gif) | ![bracket_ex08_extruded_y_progression.gif](bracket_ex08_extruded_y_progression.gif) |

---

## 4. 공학적 고찰 및 결론

1. **제조 제약 조건(`ExtrusionFilter`)의 수치적 무결성**:
   - $Y$축 압출 제약 적용 시 단면 내 밀도 분산이 $0.0$으로 완벽히 유지되었으며, 수반 감도 연쇄 미분의 내적 오차가 $1.22 \times 10^{-16}$의 기계 정밀도로 보존되어 Kratos MKL PARDISO 해석기와 결합 시 단 1회의 수렴 진동 없이 50회 최적화에 완벽 수렴하였습니다.
2. **제조성 확보에 따른 강성 페널티(Stiffness Trade-off)**:
   - 자유 3D 형상은 3차원 공간 전체에 유기적인 아치 트러스를 자유롭게 배치할 수 있어 컴플라이언스가 가장 우수합니다.
   - 반면 Y-압출 제약은 단면이 Y축 전체로 동일하게 연장되어야 하므로 약간의 강성 차이(+0.6%)가 발생하지만, 대신 **압출 다이(Die)를 통한 대량 압출 가공이나 2.5D 와이어 방전 가공이 가능해져 제조 단가를 획기적으로 낮출 수 있는 압도적인 제조 단가 경쟁력**을 확보합니다.
3. **Kratos 네이티브 최적화 연산 성능**:
   - 25,000개 요소(83,538 DOFs)의 대규모 선형 방정식을 Kratos C++ 코어의 인메모리 ModelPart와 MKL PARDISO로 스텝당 ~7초대에 고속 풀이하여, 2개 케이스 50회 최적화(총 100회 대규모 해석)를 안정적으로 완수하였습니다.
