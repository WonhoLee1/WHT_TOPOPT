# 위상최적화 표준 예제 설계 사양 및 다물리 벤치마크 평가 기준표
## (Standard Benchmark Specification Matrix for Topology Optimization)

---

## 1. 개요 및 목적 (Overview & Purpose)

본 기준표는 위상최적화(Topology Optimization) 프로젝트에서 수행되는 모든 벤치마크 예제(GE 브라켓, 휴머노이드 로봇 암, 쿼드콥터 프레임, 인버터 하우징 등)에 공통으로 적용되는 **표준 연구·설계 세트(6대 핵심 기둥, 6-Pillar Framework)**의 설계 조건, 평가 지표 및 산출물 규격을 정의합니다.

단순한 정적 컴플라이언스(Compliance) 수치 비교에 머무르지 않고, **정적-동적 성능 트레이드오프 분석**, **체적 제약 완화를 통한 동적 올-도미넌트(All-Dominant) 설계**, **제조성(압출·스킨-리브 일체형) 필터 적용**, 그리고 **100% 무결성 다물리 실측 검증**을 체계적인 세트로 완성하여 산업 현장의 실무 설계 요구를 완벽히 충족합니다.

---

## 2. 6대 표준 연구 세트 규격 (6-Pillar Benchmark Framework)

모든 표준 예제는 다음 6가지 항목을 하나의 완결된 연구 세트로 묶어서 수행합니다.

```
┌────────────────────────────────────────────────────────────────────────┐
│               위상최적화 표준 예제 6대 핵심 연구 세트 (6-Pillar)        │
├────────────────────────────────────────────────────────────────────────┤
│  [Pillar 1] 5대 알고리즘 벤치마크 (MMA, GCMMA, SEMDOT, BESO, Kratos-MMA) │
│  [Pillar 2] 정적 다중하중 vs 과도 동적 F-ESL (동일 체적률 V_target)     │
│  [Pillar 3] 체적 완화 고성능 모델 (All-Dominant: 정적·동적 동시 석권)   │
│  [Pillar 4] 제조공정 필터 (X/Y/Z 압출 및 한쪽 면 유지 스킨-리브 필터)   │
│  [Pillar 5] 100% 실측 다물리 교차 벤치마크 (K_x..K_θ, f1, 진동, 응력) │
│  [Pillar 6] 표준 산출물 & 3D HTML 리포트 (기본 커팅 OFF, 인터랙티브)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### [Pillar 1] 핵심 최적화 알고리즘 비교 (Algorithm Benchmark Suite)

동일한 설계 영역, 경계 조건, 필터 반경 및 목표 체적률 하에서 순수 내장 엔진(Built-in FEM) 및 Kratos 멀티피직스 엔진(Kratos Multiphysics MKL)과 연동된 핵심 최적화 알고리즘들을 1:1로 비교합니다.

| 분류 | 알고리즘 명칭 | 수치 역학 원리 | 권장 반복수 | 핵심 특징 및 평가 항목 |
| :--- | :--- | :--- | :---: | :--- |
| **순수 내장 엔진** | **SIMP + MMA** | 이동 점근선 근사법 (Svanberg 1987) | 40 ~ 60회 | 전역 안정성, 가장 널리 쓰이는 산업 표준 구배 기반 기법 |
| | **SIMP + GCMMA** | 보수적 볼록 이동 점근선 (Svanberg 2002) | 35 ~ 50회 | 내부 루프(Inner loop)를 통한 엄밀한 비선형 강건성 |
| | **SEMDOT** | 요소 내부 격자점 절단 (Fu et al. 2020) | 50 ~ 60회 | 요소 내부 경계, 연결성($r_{\min} \ge 2.2h$), 해석은 페널티 없는 선형 모듈러스 |
| | **BESO** | 양방향 진화적 구조최적화 (Huang & Xie 2007) | 40 ~ 60회 | 이산적 0/1 직접 제거 및 재생, 비체커보드 진화론적 기준선 |
| | **SE-AHCA** | 응력-에너지 적응형 HCA (Tovar / 2026) | 100회 (`docs/benchmarks/03`) | 비구배(Gradient-free) SED+FSD 응력 평탄화 및 초고속 수렴 |
| **Kratos 고성능 엔진** | **Kratos + MMA** | Kratos MKL PARDISO + MMA (Svanberg 1987) | 40 ~ 60회 | 대규모 산업용 FEM 연동 표준 벤치마크 |
| | **Kratos + GCMMA** | Kratos MKL PARDISO + GCMMA (Svanberg 2002) | 35 ~ 50회 | 대규모 자유도 하 비선형 보수적 수렴 안정성 |
| | **Kratos + SEMDOT**| Kratos MKL PARDISO + SEMDOT (Smooth-EDO) | 50 ~ 60회 | 산업용 대규모 격자 기반 위상 미분 골격 최적화 |
| | **Kratos + BESO** | Kratos MKL PARDISO + BESO (Huang & Xie) | 40 ~ 60회 | 대규모 유한요소 기반 양방향 진화적 요소 교체 |

- **필터 반경 기준**: 수치적 체커보드(Checkerboard) 방지 및 제조 최소 두께 보장을 위해 $r_{\min} \ge 2.0h \sim 2.5h$ 준수.
- **반복수(Iteration) 가이드**:
  - 단일 정밀도(MC40): 40 ~ 50회
  - 고해상도(MC80 이상): 50 ~ 60회 (SEMDOT의 경우 비연결 파편 방지를 위해 최소 50회 이상 보장)

---

### [Pillar 2] 정적 다중하중 vs 과도 동적 F-ESL 비교 (Static vs Dynamic F-ESL)

동일한 체적 제약 조건($V = V_{\text{target}}$) 하에서 정적 최적화 결과와 동적 F-ESL 최적화 결과를 정밀하게 대조합니다.

1. **정적 다중 하중 최적화 (Model 1: Static Multi-Load)**:
   - 기구의 대표 정적 하중 케이스(인장, 압축, 굽힘, 비틀림 등)에 가중치($w_k$)를 부여하여 다중 하중 가중 컴플라이언스(Weighted Compliance) 최소화.
   - 특징: 주 하중 방향의 굽힘 강성은 극대화되나, 과도 진동 및 편심 비틀림 저항력은 상대적으로 취약할 수 있음.
2. **과도 동적 등가정하중 최적화 (Model 2: Dynamic F-ESL)**:
   - 시간 영역 적분기(Generalized-$\alpha$ 기법, $\rho_\infty = 0.85$)를 통해 가속/감속 과도 응답 $u(t)$ 해석.
   - 0차 등가정하중: $F_{\text{esl}}(t) = \mathbf{K}(\rho) \mathbf{u}_{\text{dyn}}(t)$. 현재 구현은 과도 수반 해석이 없어 KKT 보장이 없다 (F-ESL 1차 보정 미구현).
   - 최대 변위 및 과도 천이 시점의 4~6개 핵심 시간 스텝을 등가 정적 하중으로 변환하여 다중 컴플라이언스 최적화.
   - 특징: 주 굽힘 강성을 약간 양보하는 대신, **사선 크로스 브레이싱(X-Rib)을 스스로 구축하여 비틀림 처짐 및 잔류 진동을 60% 이상 격감**.

---

### [Pillar 3] 체적 제약 완화 고성능 모델 (Volume-Relaxed All-Dominant High-Performance)

> **핵심 공학 질문**: "동일 체적에서는 동적 모델이 정적 굽힘 강성을 일부 양보했다. 그렇다면 체적을 약간($\Delta V = +5\% \sim +7\%$)만 완화하면, **정적 굽힘 강성(Ky)과 동적 비틀림/진동 억제 성능을 모두 석권(All-Dominant)**하는 이상적인 설계를 만들 수 있는가?"

- **체적 완화 모델 (Model 3: Dynamic Rib-Plus)**:
  - 체적률 설정: $V_{\text{target, plus}} = V_{\text{target}} + 0.05 \sim 0.07$ (예: 로봇 암 $28\% \rightarrow 35\%$, GE 브라켓 $15\% \rightarrow 20\%$).
  - 성능 목표:
    1. 정적 굽힘 강도($K_y$): Model 1(정적 모델) 대비 **+20% ~ +35% 이상 능가**.
    2. 동적 비틀림 처짐($U_z$): Model 1 대비 **-70% 이상 격감** (Model 2보다도 더욱 우수).
    3. 1차 고유진동수($f_1$): 기존 대비 **+20Hz ~ +40Hz 이상 상향**.
    4. 내부 리브 형상: 단순 벽체 형태에서 **고강성 조밀 와플 격자망(Isogrid)**으로 자발적 진화.

---

### [Pillar 4] 제조공정 필터 기준 (Manufacturing Constraints)

실제 주조(Casting), 절삭(Milling), 3D 프린팅(AM) 공정을 반영하기 위한 필터링 기법을 적용합니다.

1. **단방향 1D/2D 압출 필터 (X, Y, Z Extrusion / Draw Filter)**:
   - 특정 축(Z축 등) 방향으로 요소 밀도를 균일화하여 압출 성형 가능한 2.5D 형상 도출.
2. **외벽 쉘 유지 및 내부 리브 압출 필터 (`PlanarRibExtrusionFilter`)**:
   - **한쪽 면(Bottom/Skin Shell)** 및 **외벽(Side Walls)**: $2.0\text{ mm}$ 두께의 연속 외형 스킨을 비설계 영역으로 100% 보존.
   - **내부 코어 영역**: 바닥에서 상부로 수직 압출되는 2D 평면 원추형 필터($r_{\min, xy} = 2.0 \sim 2.6\text{ mm}$) + 점진적 Heaviside Continuation ($\beta = 1 \rightarrow 2 \rightarrow 4 \rightarrow 8 \rightarrow 16$) 적용.
   - 수반 감도 역전파(Adjoint Backpropagation) 체인룰 엄격 검증.
3. **체결 부위 보존 (Non-Design Bosses & Holes)**:
   - 볼트 구멍, 베어링 하우징, 로드 셀 허브는 $\rho = 1.0$ 패시브 솔리드로 고정.
   - 볼트 삽입 통로 및 회전축 클리어런스는 $\rho = 10^{-3}$ 패시브 보이드로 고정.

---

### [Pillar 5] 100% 실측 다물리 교차 벤치마크 테스트 (Rigorous Multi-Physics Audit)

최적화 완료 후, 모든 모델을 동일한 가상 시험 벤치(Virtual Test Rig)에 올려 공인 수치해석 엔진으로 전수 시험을 수행합니다. (🚫 Mocking, Random 값 절대 금지)
**보고서에는 (1) 동적 하중이 어떤 임의 프로파일로 생성되었는지에 대한 수학적 정의/시간-하중 그래프와, (2) 정하중과 동하중에 대해 각각 독립적으로 측정한 정량 수치 및 응답 곡선이 반드시 포함되어야 합니다.**

#### 1. 인가 동적 하중 프로파일 명시 규격 (Dynamic Load Profile Specification)
- **물리적 기원 및 시나리오 명시**: 슬루잉 급가속/급정지 관성력, 항공기 돌풍(Gust) 과도 충격, 바닥 노면 충격 진동 등.
- **수학적 하중 함수 정의**: S-Curve 7차 다항식 프로파일, 하프-사인(Half-sine) 펄스 $F(t) = F_0 \sin(\pi t / T_d)$, 삼각 충격파 등.
- **다축 복합 하중 벡터**: 시간 영역 3축 힘 $F_x(t), F_y(t), F_z(t)$ 및 비틀림 모멘트 $M_x(t), M_y(t), M_z(t)$ 시간 이력 그래프 제시.

#### 2. 정하중 vs 동하중 개별/교차 성능 시험 항목 (Dual Performance Breakdown)

| 하중 시험 구분 | 평가 항목 | 기호 및 단위 | 측정 및 평가 방법 |
| :--- | :--- | :--- | :--- |
| **정하중 성능 시험**<br>(Static Load Performance) | 축방향 인장/압축 강성 | $K_x$ ($\text{N/mm}$) | $F_x = 1000\text{ N}$ 인가 시 팁 처짐 역수 |
| | 슬루잉/주 굽힘 강성 | $K_y$ ($\text{N/mm}$) | $F_y = 1000\text{ N}$ 인가 시 팁 처짐 역수 |
| | 수직 전단 굽힘 강성 | $K_z$ ($\text{N/mm}$) | $F_z = 1000\text{ N}$ 인가 시 팁 처짐 역수 |
| | 비틀림 회전 강성 | $K_{\theta x}$ ($\text{N}\cdot\text{m/rad}$) | $M_x = 50\text{ N}\cdot\text{m}$ 인가 시 비틀림 각도 역수 |
| | 정적 팁 처짐 벡터 | $U_{x,\text{stat}}, U_{y,\text{stat}}, U_{z,\text{stat}}$ ($\text{mm}$) | 공칭 운용 정하중 하 팁 최대 변위 |
| | 정적 가중 컴플라이언스 | $C_{\text{stat}}$ ($\text{mJ}$) | 정격 복합 정하중 하 총 변형 에너지 |
| | 정적 피크 von Mises 응력 | $\sigma_{\text{stat, max}}$ ($\text{MPa}$) | 정하중 하 적분점 응력 복원 |
| **동하중 성능 시험**<br>(Dynamic Load Performance) | 과도 진동 피크 처짐 | $\max \|U_y(t)\|, \max \|U_z(t)\|$ ($\text{mm}$) | 과도 구간 중 최대 동적 변위 (시간응답 곡선) |
| | 잔류 진동 RMS 및 감쇠 | $\text{RMS}(U_{\text{res}})$, $t_{\text{settle}}$ ($\text{mm}$, $\text{ms}$) | 외력 종료 후 잔류 진동 구간 RMS 및 안정화 시간 |
| | 1~3차 고유진동수 | $f_1, f_2, f_3$ ($\text{Hz}$) | PARDISO Shift-Invert Lanczos 모달 해석 |
| | 동적 피크 von Mises 응력 | $\sigma_{\text{dyn, max}}$ ($\text{MPa}$) | 최대 변위 시점의 Hex8 적분점 응력 복원 |
| | 동적 항복 안전율 | $\text{SF}_{\text{dyn}} = S_y / \sigma_{\text{dyn, max}}$ | 재료 항복강도 대비 동적 안전율 |

---

### [Pillar 6] 표준 산출물 규격 및 대화형 3D 리포트 (Standard Deliverables)

모든 예제 산출물은 통일된 폴더 규칙과 파일 형식을 유지합니다.

#### 1. 폴더 저장 경로 표준
- 산출물 저장소: `wht_topopt/examples/outputs_v1/<ex_name>/`, 그 아래 `pillar1_...`~`pillar6_...` 폴더와 `manifest.json`
- `manifest.json`에는 Pillar별 실행 여부와, 실행하지 않은 Pillar의 이유를 적는다.

#### 2. 파일 명명 규칙 (File Naming Conventions)
- **예제 스크립트**: `wht_topopt/examples/src_v1/exNN_<name>/exNN[a-z]_<name>.py` (문제 정의만, 실행은 `src_v1/common/`의 Pillar 모듈)
- **3D 형상 모델**:
  - `robot_arm_rib_static.stl` / `robot_arm_rib_dynamic.stl` / `robot_arm_rib_dynamic_plus.stl`
  - (100% Watertight Marching Cubes 바이너리 STL 포맷)
- **정량 메트릭 데이터**: `<ex_name>_cross_stiffness_metrics.json`
- **시각화 그래프 및 애니메이션**:
  - `<ex_name>_transient_vibration.png` (`koreanize-matplotlib` 9pt, 2-패널)
  - `<ex_name>_faceted_mesh.png` (PyVista 3D 렌더링)
  - `<ex_name>_progression.gif` (★ **PyVista 3D Discrete Hex8 Voxel 기반 고화질 수렴 애니메이션**)
    - **표준 생성 규격**: 2D 단면 슬라이스(imshow) 사용을 엄단하고, **PyVista 3D discrete Hex8 voxel (`grid.threshold`, `show_edges=True, edge_color="#202020"`)** 렌더링 프레임으로 생성.
    - **시각화 테마**: Black 배경 (Paraview 다크 스타일), Viridis 컬러맵 (`clim=[threshold, 1.0]`), 3축 좌표계 (`show_axes`), Courier/Sans 12pt 메트릭 오버레이 (`Iter`, `Compliance`, `Volume Fraction`).
- **단독 실행 인터랙티브 HTML 보고서**: `<ex_name>_comprehensive_master_report.html`

#### 3. Three.js 3D 뷰어 필수 인터페이스 및 듀얼 평가표 규격
- **기본 상태(Default State)**: **절단(Clipping) 없는 100% 온전한 3D 형상 기본 표시** (사용자 피드백 엄수).
- **모델 전환 버튼**:
  - Model 1 단독 뷰 (호박색 골드)
  - Model 2 단독 뷰 (일렉트릭 블루)
  - Model 3 단독 뷰 (에메랄드 그린)
  - 3개 모델 1:1:1 반투명 오버레이 뷰 (퍼플)
- **단면 절단 옵션**: 사용자가 명시적으로 '내부 단면 절단' 체크박스를 켤 때만 Z축 슬라이더가 활성화되어 $0 \sim Z_{\max}$ 절단면 탐색 제공.
- **[필수 탑재] 듀얼 평가 비교표 (정적 vs 동적 독립 분리 테이블)**:
  1. **정적 강성 평가표 (Static Performance Table)**: 인장 $K_x$, 슬루잉 굽힘 $K_y$, 전단 $K_z$, 비틀림 $K_\theta$, 정적 처짐, 정적 컴플라이언스, 정적 von Mises 응력.
  2. **동적 결과 기준 평가 비교표 (Dynamic Performance Matrix)**:
     - 인가 동적 하중 사양 (하중 프로파일 수식, 피크 외력, 주기 $T_d$)
     - 1~3차 모달 고유진동수 ($f_1, f_2, f_3$)
     - 피크 과도 처짐 ($\max \|U_y(t)\|, \max \|U_z(t)\|$)
     - 잔류 진동 RMS 및 진동 감쇠 안정화 시간 ($t_{\text{settle}}$)
     - 동적 피크 von Mises 응력 및 동적 안전율 ($\text{SF}_{\text{dyn}}$)
     - 모델별 동적 진동 억제 개선율(%) 및 우승 항목 하이라이트.

---

## 3. 예제별 적용 현황 및 확장 로드맵 (Roadmap)

> ex00~ex06은 해당하는 Pillar만 적용하고, 제외한 Pillar의 이유를 `manifest.json`에 적는다. ex07, ex09는 6개 Pillar 전부를 적용한다.

| 예제 번호 | 대상 부품 | 현재 적용 단계 | 완료 항목 및 다음 진행 과제 |
| :---: | :--- | :---: | :--- |
| **ex00** | **2D 기본 (a 캔틸레버, b L-브라켓, c MBB)** | **Pillar 1, 5, 6** | • `common` 모듈로 재작성, 전체 해상도 결과 저장 |
| **ex01~ex06** | **응력 L-브라켓, 좌굴, 3D 캔틸레버·제조, 메타물질, MuJoCo 링크, 컴플라이언트 메커니즘** | **이전** | • 기존 코드를 옮기고 출력 경로만 변경, `common` 재작성은 다음 과제 |
| **ex07** | **GE 항공기 엔진 브라켓 (GE Bracket)** | **Pillar 1~6 코드 작성** | • 구 ex07, ex08 Kratos, ex11 GE 통합. 스모크와 전체 실행은 남음. Kratos 경로는 미실행으로 기록 |
| **ex08** | **eVTOL 쿼드콥터 프레임 (Quadcopter)** | **대기 중** | • 소스 없음 |
| **ex09** | **휴머노이드 로봇 암 (Robot Arm)** | **Pillar 1~6 코드 작성** | • 구 ex09, ex11 로봇 암 통합. 스모크와 전체 실행은 남음 |
| **ex10** | **전기차 파워 인버터 마운트 (Inverter)** | **대기 중** | • 소스 없음 |

---

## 4. 품질 보증 및 엔지니어링 원칙 (Quality Assurance)

1. **Anti-Mocking 100% 원칙**:
   - 강성 계수, 진동 응답 곡선, von Mises 응력에 난수 대입이나 임의 상수 스케일링을 절대 금지합니다.
   - 모든 숫자는 FEMEngine 및 Generalized-$\alpha$ 솔버의 연립방정식 해($\mathbf{K} \mathbf{u} = \mathbf{F}$, $\mathbf{M} \ddot{\mathbf{u}} + \mathbf{C} \dot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{F}(t)$)에서 직접 추출합니다.
2. **독립 검증 (Independent Verification)**:
   - 산출물 생성 후 메인 에이전트가 직접 파일 존재 여부, 크기, 메트릭의 물리적 타당성(예: 체적 증가 시 강성 증가 $K_{y,\text{plus}} > K_{y,\text{base}}$)을 검증합니다.
3. **Windows PowerShell 7 호환성 및 UTF-8 인코딩**:
   - 모든 파일은 BOM 없는 UTF-8로 저장하며, 터미널 출력 및 HTML 보고서는 한글 깨짐 없이 동작하도록 보증합니다.
