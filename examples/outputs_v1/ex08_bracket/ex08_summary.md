# Showcase 08: GE Aircraft Bracket 3D Native Kratos Topology Optimization Benchmark

**Run Date**: 2026-11-02 00:48:14  
**FEA Solver**: Kratos Multiphysics Native C++ (OpenMP 16-threads + Intel MKL PARDISO AVX-512)  
**Acceleration Factor**: **4.35x** speedup over SciPy sparse direct solver  
**Target Mesh**: 50 × 25 × 20 = **25,000 Hex8 elements** (83,538 DOFs), $dx=2.4\text{ mm}, dy=2.4\text{ mm}, dz=2.5\text{ mm}$  
**Material**: Ti-6Al-4V Grade 5 ($E_0=110\text{ GPa}, \nu=0.31, \sigma_y=830\text{ MPa}$)  

---

## 1. 정량적 종합 성능 평가표 (Dual Evaluation Matrices)

### (1) 정적 강성 및 동역학 성능 지표
| 최적화 모드 | 목표 체적비 | 최종 컴플라이언스 ($c$) | 상대 강성비 | 1차 고유진동수 ($f_1$) | CoM 변위 |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Free 3D Organic (MMA)** | 25.0% | **85.6066 mJ** | 기준 (0.0%) | **76,047.2 Hz** | 4.577 mm |
| **2. Y-Extruded 2.5D (MMA)** | 25.0% | **86.1130 mJ** | **+0.6%** | **71,105.2 Hz** | 6.295 mm |
| **3. SE-AHCA Organic (HCA)** | 25.0% | **85.7785 mJ** | +0.2% | **71,188.5 Hz** | 4.564 mm |

### (2) 응력, 질량 및 제조성 지표
| 최적화 모드 | 최대 응력 (von Mises) | 항복 안전계수 (SF) | 예상 질량 | 권장 제조 공정 | 수밀성 (Watertight STL) |
|---|:---:|:---:|:---:|---|:---:|
| **1. Free 3D Organic (MMA)** | 3.40 MPa | **244.5x** | 438.1 g | 금속 3D 프린팅 (SLM / DMLS) | **100% Closed (98,902 $\text{mm}^3$)** |
| **2. Y-Extruded 2.5D (MMA)** | 39.86 MPa | **20.8x** | 470.7 g | 프로파일 압출 + 2.5D 와이어 EDM | **100% Closed (106,251 $\text{mm}^3$)** |
| **3. SE-AHCA Organic (HCA)** | 3.66 MPa | **226.7x** | 456.7 g | DMLS / SLM 복합 정밀가공 | **100% Closed (103,092 $\text{mm}^3$)** |

---

## 2. Kratos Multiphysics Native C++ 계산 시간 프로파일링
- **Kratos In-Memory ModelPart + Intel MKL PARDISO 16-threads** 직접 계측치:
  - 1-Step 평균 소요 시간: **26994.0 ms**
  - Kratos FEM 강성행렬 조립 및 PARDISO 직접 풀이: **25743.3 ms**
  - 수반 감도 연쇄 미분 해석: **433.2 ms**
  - 최적화기 갱신 (MMA/HCA): **741.3 ms**
  - 총 계산 시간: **1414.1 s** (50 Iterations 완주)
  - SciPy sparse 직접 풀이 대비 가속비: **4.35배 고속화**

---

## 3. 핵심 산출물 목록 (Deliverables)
1. **Three.js WebGL 인터랙티브 3D 리포트**: [`ex08_comprehensive_master_report.html`](ex08_comprehensive_master_report.html)
2. **Watertight STL 형상**:
   - Free 3D: [`ex08_bracket_3d.stl`](ex08_bracket_3d.stl)
   - Y-Extruded: [`ex08_bracket_extruded_3d.stl`](ex08_bracket_extruded_3d.stl)
   - SE-AHCA: [`ex08_bracket_hca_3d.stl`](ex08_bracket_hca_3d.stl)
3. **PyVista 3D Voxel 수렴 애니메이션**: [`ex08_bracket_progression.gif`](ex08_bracket_progression.gif)
4. **마스터 비교 3D 뷰**: [`ex08_bracket_master_comparison_3d.png`](ex08_bracket_master_comparison_3d.png)
5. **수렴 및 응력 곡선**: [`ex08_bracket_convergence.png`](ex08_bracket_convergence.png)
6. **복셀 및 CAD 스크린샷**: [`ex08_bracket_voxels_3d.png`](ex08_bracket_voxels_3d.png), [`ex08_bracket_cad_3d.png`](ex08_bracket_cad_3d.png)
