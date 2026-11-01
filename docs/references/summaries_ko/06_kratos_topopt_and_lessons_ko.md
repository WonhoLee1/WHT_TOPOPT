# Kratos TopologyOptimization + 플레이그라운드 레슨 (소스코드 직접 분석)

> 소스 기준: `D:\PythonCodeStudy\kratos-src`, 플레이그라운드: `D:\PythonCodeStudy\kratos-playground\KRATOS_LESSONS_LEARNED.md`.
> 논문 다운로드 불필요. 아래 파일:행 인용은 전부 직접 읽어 확인한 것이다.

---

## 1. 구(legacy) TopologyOptimizationApplication — SIMP 정식화 전체

### 1.1 등록 변수 11종 (반드시 알아야 할 이름)

`applications/TopologyOptimizationApplication/topology_optimization_application.cpp:26-36` 생성,
`:64-74` 등록:

```cpp
KRATOS_CREATE_VARIABLE(double, YOUNGS_MODULUS_MIN)
KRATOS_CREATE_VARIABLE(double, YOUNGS_MODULUS_0)
KRATOS_CREATE_VARIABLE(double, PENAL)
KRATOS_CREATE_VARIABLE(std::string, MAT_INTERP)
KRATOS_CREATE_VARIABLE(double, X_PHYS)
KRATOS_CREATE_VARIABLE(double, X_PHYS_OLD)
KRATOS_CREATE_VARIABLE(double, DCDX)
KRATOS_CREATE_VARIABLE(double, DVDX)
KRATOS_CREATE_VARIABLE(double, SOLID_VOID)
KRATOS_CREATE_VARIABLE(double, LOCAL_STRAIN_ENERGY)
KRATOS_CREATE_VARIABLE(double, INITIAL_ELEMENT_SIZE)
```

의미: `E_MIN`/`E_0` = 재료 하한/기준 탄성계수, `PENAL` = SIMP 벌칙지수 p,
`X_PHYS` = 물리 밀도(필터 통과 후), `X_PHYS_OLD` = 이전 반복 밀도(OC 이동제한용),
`DCDX` = 컴플라이언스 민감도 ∂C/∂x, `DVDX` = 체적 민감도 ∂V/∂x,
`SOLID_VOID` = 0 일반 / 1 강제솔리드 / 2 강제보이드,
`LOCAL_STRAIN_ENERGY` = 요소 변형에너지, `INITIAL_ELEMENT_SIZE` = 요소 체적(= `DomainSize`).

등록 요소는 3종 (`:45-47`, `:59-61`):
`SmallDisplacementSIMPElement3D3N`(표면 표현용 더미), `...3D4N`(TET4), `...3D8N`(HEX8).
즉 **구 앱의 SIMP 요소는 1차 요소 전용**이다.

### 1.2 재료 보간 3종 — `small_displacement_simp_element.cpp:121-155`

```cpp
const double E_min     = this->GetProperties()[YOUNGS_MODULUS_MIN];   // :126
const double E_initial = this->GetProperties()[YOUNGS_MODULUS_0];     // :127
const double E_current = this->GetValue(YOUNG_MODULUS);               // :130
const double penalty   = this->GetValue(PENAL);                       // :131
const double x_phys    = this->GetValue(X_PHYS);                      // :132
```

| `MAT_INTERP` | `E_new` 식 | 소스 |
|---|---|---|
| `simp` | `E_new = E_0 · x^p` | `:137-141` |
| `simp_modified` | **`E_new = E_min + x^p·(E_0 − E_min)`** | `:142-146` |
| `ramp` | `E_new = E_min + x/(1+p(1−x))·(E_0−E_min)` | `:147-151` |

민감도(해석 미분, 같은 블록):

```cpp
simp:          interp_sens = p·E_0·x^(p−1)            // :140
simp_modified: interp_sens = p·(E_0−E_min)·x^(p−1)    // :145
ramp:          interp_sens = (E_0−E_min)(p+1)/(1−p(x−1))²  // :150
```

### 1.3 Scheme가 강성을 스케일한다 — `residualbased_incrementalupdate_static_simp_scheme.h:107-165`

매 조립마다 기준 강성행렬에 보간 비율을 곱한다:

```cpp
const double factor = E_new / E_current;   // :154
rLHSContribution *= factor;                // :159
rRHSContribution *= factor;                // :160
```

주석 `:157-158` 명시: 이 시점에 RHS는 `−K·u_init`만 있으므로 LHS/RHS에 직접 곱해도 정당하다.

### 1.4 DCDX / 변형에너지 / DVDX — `small_displacement_simp_element.cpp:158-209`

핵심 트릭: **`Ke0`를 `CalculateLeftHandSide`로 복원** (`:161-167`) —

```cpp
MatrixType Ke0 = Matrix();
this->CalculateLeftHandSide(Ke0, const_cast<ProcessInfo&>(rCurrentProcessInfo)); // :163
double factor = E_new / E_current;   // :166
MatrixType Ke = Ke0 * factor;        // :167
```

즉 "현재 재료의 강성 = 기준 재료 강성 × 보간비"를 요소 차원에서 재구성한다.
변위는 절점 `DISPLACEMENT`를 모아 `ue`(요소당 `NumNodes×3`) 벡터로 조립 (`:169-183`):

```cpp
ue[3*node_i+0] = CurrentDisplacement[0]; ... // :180-182
intermediateVector = prod(trans(ue), Ke0);   // :188
double ue_Ke0_ue = inner_prod(intermediateVector, ue); // :189
```

최종식:

```
DCDX:               dcdx = −interp_sens · (uᵉᵀ·Ke0·uᵉ)        // :191-195
LOCAL_STRAIN_ENERGY:        = factor · (uᵉᵀ·Ke0·uᵉ)          // :197-203
DVDX:               dvdx = INITIAL_ELEMENT_SIZE · 1           // :205-209
```

`DVDX`는 단순히 요소 체적이다 — 체적 민감도가 밀도와 무관한 상수인 이유.
`X_PHYS`의 적분점 출력은 전부 요소 대표값으로 채운다 (`CalculateOnIntegrationPoints`, `:99-116`).

### 1.5 어드조인트 민감도 루프 — `structure_adjoint_sensitivity_strategy.h:92-132`

```cpp
block_for_each(Elements, [&](Element& e){ e.Calculate(DCDX, Out, ...); }); // :102-106
block_for_each(Elements, [&](Element& e){ e.Calculate(DVDX, Out, ...); }); // :124-127
```

자기수반(self-adjoint) 정적 컴플라이언스 문제라 별도 어드조인트 해 없이
위 §1.4 요소 계산을 전 요소에 병렬 실행하는 구조다.

### 1.6 응답함수 — `structure_response_function_utilities.h:99-160`

```
C_global = Σ LOCAL_STRAIN_ENERGY                              // :111-119
V_global = Σ (INITIAL_ELEMENT_SIZE · X_PHYS) / Σ INITIAL_ELEMENT_SIZE  // :143-157
```

### 1.7 필터 — `topology_filtering_utilities.h` + `filter_function.h`

가중함수는 선형 하나뿐 (`filter_function.h:96-116`):

```
w(d) = max(0, R − d)     // :107
```

- **민감도 필터** (`ApplyFilterSensitivity`, `:158-266`):
  요소 중심 KD-tree(`Bucket`/`tree`, `:120-121`) 반경 탐색 후
  ```
  dcdx_filt[i] = Σ H·x_j·dcdx_j / (ΣH · max(0.001, x_i))   // :227-250
  ```
  (`:222` 주석: `ResultingSquaredDistances` 버그로 거리를 직접 재계산한다.)
- **밀도 필터 + Heaviside 투영** (`ApplyFilterDensity`, `:268-385`):
  ```
  x_try = ΣH·x / ΣH                                        // :366-367
  β = min(150, 1·2^((iter−1)/50))                          // :368
  x_filt = (tanh(βν)+tanh(β(x_try−ν))) / (tanh(βν)+tanh(β(1−ν))), ν=0.5  // :369
  ```

### 1.8 OC 업데이트 — `topology_updating_utilities.h:104-207`

이분법으로 체적제약 만족 라그랑지 승수 `lmid` 탐색 (`:139-197`), 요소 갱신식 (`:162`):

```
x_new = max(0, max(x_old−m, min(1, (x_old·√(−dcdx/dvdx/λ))^q))),  m = 0.2  // move limit, :131
```

`SOLID_VOID==1 → x=1, ==2 → x=0` (비설계영역, `:165-176`).
`greyscale==1`이면 q를 10회 이후 `min(qmax, 1.1q)`로 키운다 (`:113-122`).

### 1.9 실행 순서와 I/O — `topology_optimizer_factory.py`

요소 초기화 (`:125-144`): `PENAL`, `MAT_INTERP`, `X_PHYS=X_PHYS_OLD=초기체적률`,
`YOUNG_MODULUS=재료값`, **`INITIAL_ELEMENT_SIZE = geom.DomainSize`** (`:143-144`).
반복 1회 = 해석(`Analyzer`) → 민감도 필터 (`:319-324`) → OC 갱신 (`:326-334`) →
선택적 밀도 필터 (`:336-341`) → `X_PHYS_OLD` 동기화 (`:467-471`).
수렴: 상대변화 `< relative_tolerance` 또는 최대반복 (`:416-465`).

출력 (`io_utilities.h`): 재시작 = 입력 mdpa 복사 후 `Begin ElementalData X_PHYS` 블록만 교체
(`:93-139`); STL = 조건(condition) 표면메시 수기 기록, 삼각형 아니면 에러 (`:146-183`).

### 1.10 검증값 — `tests/Small_Cantilever/run_TopOpt.py:195-199`

```python
iterations == 13
compliance ≈ 278.493 (3자리)
volume_fraction ≈ 0.50010 (5자리)
```

`Analyzer` (`:139-185`): `strain_energy`→해석+`ComputeStrainEnergy`,
`volume_fraction`→`ComputeVolumeFraction − target`,
각 grad→대응 sensitivity solver 호출. 해석 조건
(`ProjectParameters.json:9-44`): `solver_type=topology_optimization_simp_static_solver`,
`domain_size=3`, `analysis_type=linear`.
최적화 설정 (`Optimization_Parameters.json:2-17`):
`material_interpolation=simp_modified, penalty=3, initial_volume_fraction=0.5,
filter_type=sensitivity, filter_kernel=linear, filter_radius=1.5,
optimization_algorithm=oc_algorithm, max_opt_iterations=100, relative_tolerance=0.001`.

---

## 2. 현행 OptimizationApplication 구조 (신규 작업은 여기)

구 TopoOpt 앱은 요소·유틸리티 직조 방식이라 확장성이 낮다.
현행 `OptimizationApplication`은 **Control / Response / ExecutionPolicy / Filter** 분리형이다.

- **ExecutionPolicy** (추상기반, `execution_policies/execution_policy.py:4-46`):
  `Initialize/Check/Finalize/GetAnalysisModelPart/Execute` 계약.
  구현: `KratosAnalysisExecutionPolicy`, `SteppingAnalysisExecutionPolicy`,
  `IndependentAnalysisExecutionPolicy`.
- **SimpControl** (`controls/material/simp_control.py:58-230`):
  물리량 `DENSITY + YOUNG_MODULUS` (`:87`), 다중재료 `φ→(ρ,E)` 투영 (`:120-121`),
  요소별 property 분리 (`:111-114`), 필터는 `FilterFactory(... DataLocation.Element ...)` 로
  **요소 컨테이너** 기준 (`:103`).
  구 앱의 `X_PHYS` 스칼라와 달리 설계변수는 `DoubleTensorAdaptor(control_phi)` 필드이며
  (`:142`), 매 갱신마다 forward-filter → project → property 저장 (`:200-221`).
- **ImplicitFilter** (`filtering/implicit_filter.py`):
  Helmholtz PDE 기반. 기본 `filter_radius=0.2` (`:14-22`),
  지원 위치 = NodeHistorical/NodeNonHistorical/Condition/**Element** (`:35-42`),
  변수 자동 판별 scalar/vector/shape (`:49-60`).
  연산 4종: `ForwardFilterField / BackwardFilterField / BackwardFilterIntegratedField /
  UnfilterField` (`:96-124`) — 구 앱의 명시적 KD-tree 합과 달리 PDE 해이다.
  구 앱식 명시 필터가 필요하면 `explicit_filter.py`의 `DataLocation.Element` 분기 (`:79`) 참조.
- **Response**: `MassResponseFunction` (`responses/mass_response_function.py:17-68`)은
  영향 변수로 `SHAPE, DENSITY, THICKNESS, CROSS_AREA` (`:39-40`)를 선언 —
  밀도 기반 SIMP와 두께/형상 최적화를 같은 인터페이스로 취급한다.
- **ContainerExpression 후계**: 구 `ContainerExpression` 대신 코어
  `Kratos.TensorAdaptors.DoubleTensorAdaptor / DoubleCombinedTensorAdaptor`가
  컨테이너 필드 운반을 담당 (`utilities/optimization_problem_utilities.py:106-118`,
  `utilities/model_part_utilities.py:97-103`).

## 3. ShapeOptimizationApplication 개요 (형상 최적화, 위상 아님)

`custom_python/add_custom_utilities_to_python.cpp:101-171` 등록 기준:

- **VertexMorphing 매퍼족**: `MapperVertexMorphing`,
  `MapperVertexMorphingMatrixFree`, `MapperVertexMorphingImprovedIntegration`,
  `MapperVertexMorphingSymmetric` + 각각의 `AdaptiveRadius` 변형.
  `Initialize/Update/Map/InverseMap` 인터페이스 — 설계변수↔형상기울기의
  필터링/매핑 연산자다 (SIMP 밀도 필터와 역할은 유사, 대상이 절점 좌표).
- **Damping** (`custom_utilities/damping/damping_utilities.cpp:47-168`):
  `damping_function_type=cosine, damping_radius>0` 필수 (`:47-55`),
  감쇠영역 SubModelPart 기준 KD 탐색 후 이웃 절점에 `DAMPING_FACTOR` 부여 (`:115-140`),
  형상 업데이트에 곱한다 (`:163-168`).

## 4. MeshingApplication MmgProcess3D 연동

`meshing_application/python_scripts/mmg_process.py:390-397`:

```python
if self.domain_size == 2:
    self.mmg_process = MeshingApplication.MmgProcess2D(...)
else:
    if self.is_surface:
        self.mmg_process = MeshingApplication.MmgProcess3DSurfaces(...)
    else:
        self.mmg_process = MeshingApplication.MmgProcess3D(...)   # 체적 리메시
```

C++ 바인딩 `add_processes_to_python.cpp:142` (`MmgProcess3D`),
`:151` (`MmgProcess3DSurfaces`), MPI판 `ParMmgProcess3D` (`:163`).
위상최적 후 경계 추출·스무딩, 형상최적 반복 중 메시 품질 유지에 사용.
구 TopoOpt STL 익스포트(§1.9)와 달리 실제 리메시 경로다.

## 5. MDPA 규격 — Begin Geometries vs Elements, SubModelPartNodes

### 5.1 구 TopoOpt 예제는 구방식 (직접 확인)

`tests/Small_Cantilever/Small_Cantilever_Hex.mdpa`:

```
Begin Elements SmallDisplacementSIMPElement3D8N ...   // :215 (요소 직접 선언)
Begin SubModelPartNodes                               // :324 (Parts_Solid_solid)
Begin SubModelPartElements                            // :524
Begin SubModelPart DISPLACEMENT_disp                  // :629 (고정 BC)
Begin SubModelPart PointLoad3D_load                   // :655 (하중)
```

재료: `Begin Properties 1: DENSITY 1 / YOUNG_MODULUS 1 / POISSON_RATIO 0.29` (`:5-11`).
`run_TopOpt.py:46-54`에서 `YOUNGS_MODULUS_MIN=1e-9, YOUNGS_MODULUS_0=1`을 덮어쓴다
(mdpa 재료 ≠ SIMP 보간 상한 — 코드 주입값이 실제 사용값).

### 5.2 현행 방식 (플레이그라운드 검증, LESSONS §2)

`KRATOS_LESSONS_LEARNED.md:29-37`:

1. MDPA는 `Begin ModelPartData → Begin Properties → Begin Nodes →
   Begin Geometries <Type> → Begin SubModelPart ...` 순서 (`:29-30`).
2. **요소는 MDPA `Elements` 블록이 아니라 `Geometries` 블록 +
   `CreateEntitiesFromGeometriesModeler`로 생성** (`:31-32`).
   Geometries 없으면 "노드 20 / 요소 0"으로 조용히 읽히고 에러도 없다 (`:32-33`).
3. SubModelPart에 **`SubModelPartNodes`와 `SubModelPartGeometries` 둘 다** 있어야
   modeler가 요소를 만든다 (`:33-34`).
4. 재료 JSON 키는 `properties: [{model_part_name, properties_id,
   Material: {constitutive_law: {name}, Variables, Tables}}]` —
   `element_type` 같은 임의 키는 무의미 (`:34-35`).
5. `model_part_name`은 `root.SubModelPart` 형식 (`:37`).

> 구 앱 예제(mdpa `Elements` + `SubModelPartElements`)를 그대로 쓰면 신 solver/modeler
> 경로에서는 요소가 안 생긴다. 신 OA 경로로 옮길 때는 Geometries 방식으로 변환할 것.

## 6. 1차 요소 전단잠김 — 오차식과 대책

### 6.1 검증된 오차식 (LESSONS §3, `test_nlgeom_rollup.py`)

80×0.5 mm 평면변형 패널, 끝단 원호 처방 0→180°, Q4 vs Q8 (`:39-53`):

| 요소 | 메시 | M/M_theory |
|---|---|---|
| Q4 | 161×3 | **1.409** |
| Q4 | 321×9 | 1.093 |
| Q8 | 41×2 | **1.018** |
| Q8 | 161×2 | 1.003 |

결론 (`:56-61`):

```
Q4 모멘트 오차 ≈ (1−ν)/2 · (Δx/H)²        (Δx=요소 길이, H=전체 두께)
```

실측: 예측 1.350/1.0875 vs 실측(ny=9) 1.357/1.093. **두께 방향 분할(ny)은 무효** —
결정 인자는 요소 길이 대 전체 두께비뿐. 오차 <1% 조건은 `Δx ≈ 0.1H`.
다층(PET/PSA)에서도 동일식이 성립 (`:156-159`):
`1+(1−ν)/2·(Δx/H)²`, H=전체 두께 0.21 — 층 수·얇은 층 두께가 아니라 전체 두께가 지배.

### 6.2 대책 (소스 근거)

- **Kratos 2D 솔리드에 EAS Q4는 없다.** `custom_elements/solid_elements/`에는
  `small_displacement_bbar.*` (소변형 B-bar, 체적잠김용)와
  `small_displacement_mixed_volumetric_strain*` (혼합체적, 체적잠김용),
  `total_lagrangian_q1p0_mixed_element.*` (Q1P0, 체적잠김용)만 있고
  **굽힘 잠김용 EAS/ANS는 없음** (`LESSONS:60-61`, 디렉토리 직접 확인).
  Q1P0 혼합은 Q4 전용이며 체적잠김용이지 굽힘 잠김과 무관 (`:61`).
- **굽힘 지배 문제는 2차 요소(Q8, Q9)가 정답** (`:61`, `:245-246`).
  3D로 치면 TET10 / HEX20(27절점 포함 2차 육면체) 계열 —
  선형 TET4/HEX8(SIMP 구 앱 기본)은 같은 잠김을 겪는다.
- B-bar 근거: `SmallDisplacementBbar::CalculateBbar`가 정수차 B행렬을
  대체한다 (`solid_elements/small_displacement_bbar.cpp:222-260`,
  `CalculateBbar:340-...`, 선언 `small_displacement_bbar.h:299-316`).
  단, B-bar는 **체적잠김** 대책이지 굽힘 잠김 대책이 아니다 — 혼동 금지.

> SIMP 구 앱 요소가 1차(TET4/HEX8) 전용(§1.1)이라는 점과 맞물린다:
> 굽힘 지배 구조를 구 앱으로 풀면 잠김 오차가 밀도 업데이트(DCDX)에 그대로 들어간다.
> 정확도가 필요하면 신 OA + 2차 요소 조합으로 갈 것.

## 7. Windows h5py → HDF5Application import 순서 (필수)

`KRATOS_LESSONS_LEARNED.md:103-105`:

> **`h5py`를 먼저 import하면 정상. Kratos HDF5를 먼저 하면
> `ImportError: DLL load failed while importing _errors`.**
> 변환/후처리 스크립트는 Kratos를 import하지 말고 h5py만 쓸 것.

근본 원인 3점 (소스 확인):

1. `hdf5_file.cpp:108-113` — Kratos `File` 생성자가 HDF5 에러 핸들러를 떼어낸다.
   주석 명시: "h5py 같은 의존성이 legacy HDF5 API로 콜백을 설정하면 Kratos HDF5 호출이 깨진다."
2. Python 기본 드라이버 분기 (`HDF5Application/python_scripts/core/file_io.py:57-69`):
   ```python
   if 'parallel' in io_type: file_driver = 'mpio'   # :63-64
   elif os.name == 'nt':    file_driver = 'windows' # :65-66
   else:                    file_driver = 'sec2'     # :67-68
   ```
   저장은 Windows에서 정상 동작한다 (`LESSONS:99`).
3. **XDMF 연결정보 작성기는 드라이버를 바꿀 방법이 없다.**
   `xdmf_utils.py:99-100` 호출은 `(path, h5path)` 2인자,
   바인딩 `add_custom_processes_to_python.cpp:18-21`도
   `py::init<const std::string&, const std::string&>()` 2인자만 노출.
   반면 C++ `File::SetFileDriver`는 Windows에서 `"windows"`만 허용
   (`hdf5_file.cpp:879-906`, `:882-885` 에러 분기) —
   즉 **Kratos 자체의 Windows 문제**이며 메시·요소차수와 무관 (`LESSONS:102`).
   테스트도 Windows 기본값을 못박는다 (`tests/test_hdf5_core.py:109,263,276`).

실무 규칙: 동일 프로세스에서 둘 다 필요하면 **반드시 `import h5py`를 먼저**.
ParaView 확인 경로는 레거시 `.vtk` (`vtk_output_process`)가 검증됨
(`LESSONS:106-109`, Q8 203점/40셀, 최대변위 95.1 mm 일치).

## 8. 기타 검증된 함정 (재발 방지)

- `output_processes`는 JSON **최상위 키** (`processes` 안이 아님).
  안에 넣으면 `Initialize()`에서 "Size can only be queried..." 실패 (`LESSONS:85-86`).
- HDF5 한 파일 다프레임: `file_name` 고정 + 결과 프리픽스 `/ResultsData/step_<step>`,
  `file_access_mode=read_write` + 실행 전 빈 파일 생성. `<time>`은 `.` 포함이라
  경로 거부 → `<step>` 정수 사용 (`LESSONS:121-135`).
- `line_search` 전략은 가짜 완주 (반복 1회/스텝, `|R|=nan`, 반력 0) — 후처리에서
  반력·평형(ΣF≈0)을 반드시 확인할 것 (`LESSONS:232-236`).
- 예측자·스텝제어 등 비선형 러너 노하우는 `LESSONS §7` 전문 참조
  (증분 예측자 `u_prev+[u_arc(t)−u_arc(t_prev)]`, `GeometricStepController`,
  `ReduceTimeStep` 후 좌표 복원 병행).

---

*작성: kratos-src 직접 판독 + KRATOS_LESSONS_LEARNED.md 검증값 인용. 코드·수식 생략 없음.*
