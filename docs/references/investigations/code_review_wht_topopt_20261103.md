# wht_topopt 코드 리뷰: 이론, 계획, 실행 대조 결과와 수정 순서

작성: 2026-11-03. 작성 주체: aieng/topopt 담당 에이전트 (`/code-review` 실행 결과를 정리). 기준 커밋: `eb90349` + 작업트리.
이 문서는 수정 작업의 입력이었고, 수정 결과는 8절에 있다. 수정은 사용자 지시로 같은 에이전트가 직접 했으며 커밋하지 않았다.

**작업 범위: 수정과 단위 테스트까지. 벤치마크 재실행은 하지 않는다** (7절 참고). 벤치마크는 사용자가 따로 지시한다.

## 1. 근거 수준

| 수준 | 해당 항목 |
|---|---|
| 리뷰어가 수치로 재현 | R1, R2, R3, R9, R10 |
| 작성자가 현재 코드에서 줄을 직접 확인 | R1, R2, R3, R4, R10 |
| 코드 독해만 (재현 없음) | R5, R6, R7, R8 및 4절 |

벤치마크는 리뷰에서 다시 돌리지 않았다. 줄 번호는 작성 시점 기준이며, 파일이 바뀌었으면 식별자로 찾는다.

## 2. 먼저 고칠 결함 (이것을 고치기 전까지 ex07, ex09의 `docs/benchmarks` 비교 결론은 믿을 수 없다)

### R1. Generalized-alpha 우변의 항 누락
위치: `examples/ex09_esl_dynamic_robot_arm.py:413`, 같은 복사본 `examples/ex09_compare_dynamic_vs_static.py:442`.

```
현재  M @ (... + (1-am)*(1-2b)/(2b) * a)      C @ (... + (1-af)*(g/b - 1) * v + ...)
정답  M @ (... + ((1-am)/(2b) - 1)  * a)      C @ (... + ((1-af)*g/b - 1) * v + ...)
```
차이는 정확히 `-am * a_n` 과 `-af * v_n` 이다 (이론 02 §2의 Chung-Hulbert 식).

재현: 무감쇠 SDOF (omega=1, u0=1, rho_inf=0.85), t=5T, 정답 1.0.

| 스텝 수 | 현재 코드 | 두 항 추가 |
|---|---:|---:|
| 100 | 7.79 | 0.966 |
| 400 | -1.01 | 0.99986 |
| 1600 | -0.65 | 0.9999994 |

현재 식은 스텝을 줄여도 수렴하지 않는다. 영향: ex09 계열의 과도 응답, 임계 시점, ESL, "Uz -66%, All-Dominant" 주장 전부.
`verification/01` §3의 "에너지 변동 < 0.002%"는 이를 측정하는 코드를 찾지 못했다.

수정 시 복사본 두 곳을 한 함수로 합칠 것을 권한다. 또 복사되면 같은 결함이 다시 생긴다.
테스트: SDOF에서 위 수렴 표를 재현하고 2차 수렴을 확인한다.

### R2. optimizer 값이 무시됨
위치: `core/stl_domain.py:616` 부근 `else: # Default fallback to MMA`.

재현: 12x4x4 캔틸레버에서 다섯 옵티마이저의 compliance가 모두 191.0100354031105, xphys 완전 동일.
영향: ex07의 GCMMA, SEMDOT, BESO 비교 행과 `benchmarks/02`의 "5대 알고리즘 비교"는 MMA 재실행에 라벨만 다르다 (SEMDOT은 rmin만 다름).

수정: 실제 옵티마이저로 연결하거나, 미지원 값은 `ValueError`로 거부한다. 조용한 대체는 없앤다.
테스트: 미지원 값이 예외를 내는지, 지원 값끼리 결과가 달라지는지 확인한다.

### R3. variable_groups가 무시됨
위치: `core/stl_domain.py:603` (`x`를 항상 `n_elems` 길이로 초기화), 분기 628, 642, 674 (`len(x) == n_groups`일 때만 동작).

재현: y 압출 그룹(48개, 요소 192개)을 줘도 `x` 길이 192, 압출 방향 편차 0.116, 결과가 무제약 MMA와 동일.
영향: ex07의 `kratos_ext_y`, `kratos_ext_z`와 `StlCase.symmetric_in`의 제조 제약이 적용되지 않았는데 "Extruded"로 보고된다.

수정: `x0`가 없으면 `n_groups` 길이로 초기화한다. `x0` 길이가 어느 쪽도 아니면 예외를 낸다.
테스트: 압출 방향 밀도 편차가 0인지 확인한다.

### R10. 감도가 실제 기울기의 절반
위치: `core/stl_domain.py:553` `ce_all = 0.5 * np.sum(...)`. 목적함수는 `c = sum(w f.u)`인데 감도에 0.5가 곱해진다.

재현: adjoint -7.012, 수치 미분 -14.024 (정확히 2배). 단일 목적 MMA에서는 스케일만 달라 드러나지 않지만 목적과 제약의 균형이 왜곡된다.
수정: 0.5 제거. 테스트: 수치 미분과 비교하는 테스트를 추가한다 (현재 이 경로의 검증 테스트가 없다).

## 3. 이어서 고칠 결함

### R4. 응력이 두 번 완화됨 (SE-AHCA 응력 항에 직접 영향)
위치: `examples/ex07_ge_bracket_advanced_algorithms.py:498` 부근, ex07의 hybrid 분기, `ex11_ge_bracket_hca.py:199,337,487,538`.

`E_field`(rho^p 반영)로 계산한 응력을 `von_mises_unpenalized_cases`로 넘기면 `hca.py:120`에서 `rho^q`가 또 곱해진다.
유효 응력이 이론 03 §2의 `rho^0.5 * sigma0`가 아니라 약 `rho^3.5 * sigma0`가 된다. rho=0.5에서 0.71 대신 약 0.088이다.
ex09와 ex11_robot_arm은 E0 응력을 넘겨서 예제끼리 물리량이 다르다.

수정: 페널티 없는 `E0` 응력을 넘기도록 통일한다. `ex07_se_ahca_signal_study.py`도 같은 경로를 쓰는지 확인한다.

### R5. Peak von Mises가 피크 시점이 아님
위치: `ex09_esl_dynamic_robot_arm.py:889` (`u_static_cases[0]`만 사용, 324행에서 시간순 정렬하므로 0번은 가장 이른 시점). 764행 가중치 `[0.30,0.25,0.30,0.15]`도 정렬 위치에 붙어 반복마다 의미가 바뀐다.
수정: 모든 케이스의 최대를 쓰고, 가중치는 계획서 S06의 `w_k ∝ SE(t_k) dt`로 계산한다.

### R6. 밀도 필터를 신호에 재사용
위치: `ex09_esl_dynamic_robot_arm.py:786`. `rib_filter.forward`가 신호에 Heaviside와 `[1e-4, 1]` clip을 적용해 1을 넘는 과응력 구분이 사라진다.
수정: 신호에는 `H_2d` 평균만 적용한다.

### R7. F-ESL 명칭과 구현 불일치
위치: `ex09_esl_dynamic_robot_arm.py:828`. 과도 수반 해석이 없고 `kkt_residual`도 코드에 없다. 이론 02 §3은 0차 식을 "True F-ESL"로 정의한다.
또 757행 정적 해는 `f = K u`라 `u_dyn`과 항등이어서 매 반복 LU가 낭비된다.
수정: 정식화대로 구현하거나, 구현 전까지 명칭을 낮춰 휴리스틱으로 표기한다. 문서도 같이 고친다.

### R8. 시간 스텝과 질량 보간이 계획서 위반
위치: `ex09_esl_dynamic_robot_arm.py:222` (`n_steps=30`, dt 약 1.72 ms), 721행 (`rho * m0` 선형 질량).
문서의 f1 266~353 Hz 기준 omega*dt 약 2.9~3.8로 S09의 `<= 0.5`를 어긴다. S07은 선형 질량 보간을 금지하는데 이론 02 §1은 그 식을 표준으로 적었다.
수정: dt를 f1 기준으로 정하고 omega*dt 검사를 넣는다. Du-Olhoff 질량 보간을 적용하고 이론 문서를 고친다.

### R9. TET10 경로
위치: `core/tetra_mesher.py:117`, `fe_builders.py:632`. gmsh TET10 절점 순서(8=2-3, 9=1-3)가 fe_builders 순서(8=1-3, 9=2-3)로 변환되지 않아 "Inverted TET10 element, detJ=-1.34" 예외가 난다.
정상 순서에서도 4점 적분으로는 `Me0` 랭크가 12/30이다 (질량행렬에 필요한 적분 차수는 4).
수정: 순서 변환 추가, 질량행렬 적분 차수 상향. 테스트: `Me0` 랭크 30, 패치 테스트.

## 4. 그 밖의 항목 (상한 때문에 위에서 뺀 것)

| 위치 | 문제 |
|---|---|
| `algorithms/semdot.py:244` | 3D는 절점 보간 없이 `rho+offset` 샘플만 사용. 이론 01 §3.3의 Hex8 하위 셀 적분은 2D 외 미구현 |
| `algorithms/hca.py:341` | `update`가 투영된 `rho_next`를 반환해 다음 반복의 설계변수로 쓰임 (필터 누적). `sigma_allow` 기본값 250e6은 Pa인데 예제는 MPa |
| `algorithms/hca.py:196` | 이론 03 §4의 과응력 게인 부스트와 §2의 최대값 정규화가 코드에 없음. 코드는 PID와 85퍼센타일 사용. 어느 쪽이 맞는지 정해 문서를 맞출 것 |
| `examples/ex07_ge_bracket_advanced_algorithms.py:634` | t_FEM, t_Sens, t_Opt를 `dur*0.70/0.20/0.10`으로 고정 배분해 측정값처럼 보고. verification/01의 무결성 선언과 충돌 |
| `core/tetra_assembler.py:148` | numba 경로에 inverted 검사가 없어 음의 강성이 조용히 조립됨 (이 환경은 numba가 없어 미실행) |
| `core/tetra_mesher.py:162` | BC 태깅이 꼭짓점, 면, 변 중점만 샘플링해 큰 삼각형 내부 절점이 구속에서 빠질 수 있음 |
| `robust/mujoco_loads.py:20` | optional인 mujoco를 최상위에서 import해 mujoco가 없으면 `import wht_topopt`가 실패 |
| `tests/` | SE-AHCA는 `test_core.py`에서만 언급, Generalized-alpha 테스트 없음. 위 결함들이 테스트를 통과한 이유로 보임 |

## 5. GE 컴플라이언스 값의 크기 (원인 확인됨)

`benchmarks/04`의 케이스별 컴플라이언스는 약 3.5e9 N*mm로, aieng 결과(약 3.7e4)보다 약 10만 배 크다.

원인: `ex07`과 `ex07_se_ahca_signal_study.py`가 `build_stl_problem`에 재료를 넘기지 않아 기본값 **E0 = 1, nu = 0.3**으로 실행되었다. `E_TI_MPA`, `NU_TI`는 정의만 되고 쓰이지 않았다.
힘이 같으면 응력은 E0와 거의 무관해서 응력 수치만 보면 드러나지 않는다. 04의 `pm1` 0/1 설계를 다시 푼 값:

| 재료 | 컴플라이언스 케이스 1/2/3/4 (N*mm) | 최대 von Mises (MPa) |
|---|---|---|
| E0 = 1, nu = 0.3 (실행된 값) | 3.71e9 / 2.81e9 / 2.08e9 / 1.01e9 | 1147 / 641 / 817 / 449 |
| E0 = 113800, nu = 0.342 | 32328 / 24720 / 18067 / 8820 | 1119 / 634 / 793 / 439 |

## 6. 수정 순서

1. R1, R2, R3, R10 + 각 테스트. 커밋을 항목별로 나눈다.
2. R4, R6 (SE-AHCA 입력 정리). `se_ahca_improvement_study_from_aieng_20261102.md`의 개선안과 함께 본다.
3. R5, R7, R8 (ex09 동역학).
4. R9, 5절의 컴플라이언스 이상, 4절 항목.

## 7. 벤치마크 전까지의 완료 조건

- `pytest`와 위의 소형 테스트(SDOF, 캔틸레버 12x4x4, 수치 미분, TET10 랭크)는 실행한다.
- 다음은 **실행하지 않는다**: ex06, ex07, ex09, ex11 전체 실행, `docs/benchmarks/` 재생성, `ex07_se_ahca_signal_study.py` 재실행.
- 숫자를 고치지 않은 기존 보고서(`benchmarks/02`, 03, 04, `verification/01`, 관련 html)에는 상단에 "코드 리뷰 R1~R4 수정 전 결과, 재생성 필요"라고 표시만 한다.
- 끝나면 항목별로 수정함, 보류함, 이유를 표로 정리하고 벤치마크 재실행이 필요한 예제 목록을 남긴 뒤 멈춘다.

## 8. 수정 결과 (2026-11-03, 벤치마크 전까지)

테스트: `tests/test_review_fixes_20261103.py` 24개 추가. 수정 전 80 통과, 2 실패 -> 수정 후 104 통과, 2 실패.
남은 2 실패는 수정 전과 같다: `test_kratos_native_simp_application`(Kratos 없음), `test_13_stage_pipeline_3d_mujoco_post_and_pyvista`(원인은 조사하지 않음, 이 환경에 mujoco와 pyvista 없음).
`test_examples_showcase.py`, `test_homogenization.py`는 수정 전부터 수집되지 않아 제외했다 (`examples/__init__.py`가 없는 이름 `LBracketStressShowcase`를 import, mujoco 직접 import).
scikit-image와 gmsh는 임시 경로에 설치해 돌렸고, 프로젝트 환경에는 설치하지 않았다.

스모크 실행(벤치마크 아님, 결과는 임시 폴더): ex09 `HYBRID_SE_AHCA`와 `FESL_MMA`(16x8x6, 2~3회), ex07의 7개 태그(MC24, 4회)가 오류 없이 끝났다. pyvista와 Kratos는 목(mock)으로 대체했다.
4회로는 수렴하지 않아 스모크의 수치는 결과로 해석하지 않는다. `ex09_compare_dynamic_vs_static.py`와 `ex11_ge_bracket_hca.py`는 컴파일만 확인했다 (ex09 비교의 적분기는 SDOF 테스트로 공용 함수를 검증).

| 항목 | 상태 | 내용 |
|---|---|---|
| R1 | 수정 | `core/time_integration.py`로 공용화, 두 예제 교체. SDOF 2차 수렴 확인 |
| R2 | 수정 | 실제 옵티마이저로 연결, 미지원 값은 `ValueError` |
| R3 | 수정 | `n_groups` 길이로 초기화, 압출 방향 편차 0 확인. 그룹은 MMA, GCMMA만 허용 |
| R4 | 수정 | ex07, ex11의 6곳이 E0 응력을 넘김 |
| R5 | 수정 | 모든 임계 시점의 최대 응력, 가중치 `w_k ∝ SE(t_k)` |
| R6 | 수정 | 신호용 `PlanarRibExtrusionFilter.smooth_signal` 추가 |
| R7 | 일부 | 수반 해석은 구현하지 않음. 명칭과 문서를 휴리스틱으로 고치고 불필요한 정적 풀이 제거. 보고서 HTML 안의 "F-ESL" 표기 다수는 그대로 |
| R8 | 수정 | `n_steps` 240, 첫 반복에 omega*dt 검사, Du-Olhoff 질량과 그 미분. Kratos 해석기 경로의 질량은 고치지 않음 |
| R9 | 수정 | gmsh 순서 변환, 질량행렬 36점 적분 (랭크 30) |
| R10 | 수정 | 0.5 제거, 수치 미분 일치 |
| 5절 | 수정 | ex07, 신호 연구에 재료 전달 |
| 4절 semdot | 수정 | 구조 격자 Hex8의 3선형 하위 격자, 드라이버 전용 SEMDOT 루프, 표면 추출 구현 (아래 10절) |
| 4절 hca.py | 수정, 문서 | `update`가 설계변수 반환, `sigma_allow` 기본값 0(자동). 이론 03에 코드와 다른 점 표시 (부스트 미구현) |
| 4절 ex07 시간 | 수정 | 고정 비율 제거, `measured_phase_times` |
| 4절 tetra | 수정 | numba 경로 inverted 검사 (이 환경에 numba 없어 미실행), BC 태깅 표면 조밀 샘플링 |
| 4절 mujoco | 수정 | optional import |

리뷰 밖에서 추가로 고친 것:
- `GCMMAAdapter`: `eval_fn` 값의 형태 오류로 내부 루프가 예외를 냈다 (R2 연결 전에는 실행된 적이 없음).
- BESO 드라이버 경로: 전체 솔리드에서 시작, 하중 경로 검사와 하중/지지 요소 보호를 연결. 튜닝은 하지 않았다. 캔틸레버 12x4x4에서 30회에는 일시적 경로 단절(C=1838), 60회에는 V 0.394, C 141.5, 100회에는 V 0.426으로 진동한다.
- `BuiltinAdapter`: 비유한 변위를 조용히 돌려주지 않고 예외를 낸다. MC16에서는 재료와 무관하게 NaN이 나왔다 (원인 미추적, 격자가 거칠어 생기는 문제로 추정).

추가로 발견했지만 고치지 않은 것:
- `ex09_esl_dynamic_robot_arm.py`는 KratosMultiphysics를 최상위에서 import해 Kratos 없이는 실행되지 않는다.
- `examples/__init__.py`의 깨진 import (위).
- `*_backup*.py` 예제에는 R1 등의 결함이 그대로 있다.

## 9. 벤치마크 재실행 시 달라질 점과 대상

재실행 대상: `ex07_ge_bracket_advanced_algorithms.py`, `ex07_se_ahca_signal_study.py`, `ex09_esl_dynamic_robot_arm.py`, `ex09_compare_dynamic_vs_static.py`, `ex11_ge_bracket_hca.py`, `compare_dynamic_extrusion_all.py`. 보고서 `benchmarks/01`, `02`, `04`, `verification/01`에 재생성 표시를 달았다. `benchmarks/03`(ex06, MBB)은 수정 경로를 쓰지 않아 표시하지 않았다.

주의:
- ex09는 `n_steps`가 8배가 되어 과도 해석이 느려진다.
- R4로 회색 요소의 응력 신호가 커진다. SE-AHCA의 `eta`, `sigma_allow` 조정이 다시 필요할 수 있다.
- R10으로 목적과 체적 제약의 상대 크기가 달라져 MMA 궤적이 바뀐다.

## 10. SEMDOT 3D 구현 (2026-11-03)

수정 전: 3D는 요소 밀도에 offset을 더한 샘플만 썼고 요소 내부 경계가 없었다. `ex06`의 "Free 3D" SEMDOT과 ex07의 SEMDOT(리뷰 R2 이전에는 MMA로 대체됨)이 모두 그렇다. 수정 내용:

- `algorithms/semdot.py`: 구조 격자 Hex8에 3선형 하위 격자. 절점 번호 체계와 무관하고 수동 공백 요소는 절점 평균에서 뺀다. 샘플은 활성 요소만 계산한다.
- `fine_density_grid`, `extract_grid`, `extract_isosurface` (닫힌 표면). 비구조 격자는 요소 샘플 방식으로 두고 경고를 낸다.
- `core/stl_domain.py`의 `_run_semdot`: x를 OC로 갱신하고 필터한 뒤 하위 격자 절단, 해석은 페널티 없는 선형 모듈러스. `result.extras["semdot"]`로 표면을 꺼낸다.

3D 캔틸레버 48x16x8, V 0.3 (하중부 수동 솔리드, 2배 세밀 격자 재해석, 같은 체적):

| 방법 | 30회 | 60회 | 회색도 |
|---|---|---|---|
| MMA | 33.92 | 34.48 | 0.106 |
| SEMDOT, 요소 내부 경계 | 43.58 | 40.57 | 0.055 |
| SEMDOT, 요소 단위 0/1 | 46.51 | 43.21 | 0.055 |

읽는 법: 내부 경계가 요소 단위보다 약 6 % 낫고 회색도는 MMA의 절반이지만, 컴플라이언스는 MMA보다 약 18 % 나쁘다. 파라미터는 튜닝하지 않았다 (`n_sub` 4, beta 0.5 -> 8, 민감도 이력 0.65).

주의: SEMDOT은 부재가 얇아 하중 절점이 공백에 남기 쉽다. 하중부를 수동 솔리드로 두지 않으면 같은 문제의 재해석 컴플라이언스가 1955(내부 경계), 11763(요소 단위)으로 폭증했다. GE는 하중과 볼트 영역이 이미 수동 솔리드다.
GE 전체 메시 결과는 아직 없다. 벤치마크 전까지의 범위라 돌리지 않았다.

격자가 거칠면 약하다: 두께 방향 4요소(24x8x4)에서는 beta가 오르는 동안 컴플라이언스가 104 -> 180으로 올랐고(40회), 100회에 138이었다. 두께 방향 8요소(48x16x8)에서는 48 -> 34로 내려갔다.
테스트: `tests/test_semdot_3d.py` 8개. 전체 결과는 112 통과, 2 실패(8절의 같은 2건). `ex06`의 3D SEMDOT도 이제 이 경로를 타며, 거기 설정은 `n_sub=2`다. GE MC40 6회 스모크는 오류 없이 끝났다.
