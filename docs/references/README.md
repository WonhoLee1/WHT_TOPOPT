# wht_topopt 연구 논문 및 이론 문헌 마스터 아카이브 (Master Reference Hub)

본 디렉토리(`wht_topopt/docs/references/`)는 `wht_topopt` 패키지 개발 및 알고리즘 구현의 학술적 기반이 되는 **53편의 최신 연구 논문 원문 PDF, 7대 한국어 심층 해설서, 4대 종합 서베이 리포트, 그리고 10대 동역학/역학 기술 조사 보고서**를 하나로 집대성한 공식 연구 문헌 포털입니다.

---

## 1. 아카이브 구조 (Directory Layout)

```
wht_topopt/docs/references/
├── README.md                      # [현재 파일] 마스터 문헌 포털 및 학술 분류 가이드
├── INDEX.md                       # 영문 마스터 논문 목록 및 다운로드 링크 색인
├── DYNAMICS_SURVEY.md             # 동적 과도응답 및 F-ESL/DiESL 종합 서베이 보고서
├── METHODS_SURVEY_2026.md         # 2019-2026 최신 위상최적화 기법 종합 서베이 보고서
├── FRAMEWORK_PLAN.md              # SOTA 위상최적화 엔진 아키텍처 설계 계획서
│
├── summaries_ko/                  # 7대 핵심 분야별 한국어 심층 해설서
│   ├── 01_foundations_and_filters_ko.md         # SIMP 기초, 필터(밀도/민감도), 투영법
│   ├── 02_optimizers_and_non_simp_ko.md         # MMA, GCMMA, BESO, 레벨셋, SEMDOT
│   ├── 03_stress_buckling_dynamics_ko.md       # 응력제약(P-norm/QP), 좌굴(KG), 과도동역학(F-ESL)
│   ├── 04_manufacturing_constraints_ko.md      # 압출(Extrusion), 주조(Draft), 3D프린팅 오버행
│   ├── 05_multiscale_and_neural_ko.md          # 신경망 기반(Neural TopOpt), 균질화 멀티스케일
│   ├── 06_kratos_topopt_and_lessons_ko.md      # Kratos C++ 인메모리 연동 및 구조 최적화 교훈
│   └── 07_frontistr_elements_and_topopt_ko.md  # FrontISTR 대규모 병렬 FEM 및 요소 역학
│
├── investigations/                # 10대 실증 해석 및 역학 심층 조사 보고서
│   ├── 3d_dynamic_load_profile_mechanics.md    # 로봇 암 3차원 과도 동적 하중 프로파일 기구학
│   ├── claude_peer_review_dynamic_topopt.md    # 동적 위상최적화 피어 리뷰 종합 의견서
│   ├── dynamic_topopt_before_vs_after_rev1_comparison.md # F-ESL 개선 전후 정량 비교
│   ├── dynamic_vs_static_comprehensive_evaluation_report.md # 정적 vs 동적 교차 검증 보고서
│   ├── dynamic_vs_static_topopt_theory_and_comparison.md  # 정적 vs 동적 최적화 이론 분석
│   ├── ge_bracket_advanced_algorithms_benchmark.md # GE 브라켓 고해상도 4대 알고리즘 벤치마크
│   ├── ge_bracket_challenge_real_stl_and_multi_load_benchmark.md # GE 브라켓 STL 복셀화 및 다중하중
│   ├── peer_review_prompt_dynamic_topopt.md    # 다분야 전문가 피어리뷰 프롬프트 사양
│   ├── reliability_inspection_and_argon_refinement_plan.md # 신뢰성 검증 및 6대 오류 수정 보고
│   └── topology_optimization_theories_and_formulations.md # 6대 위상최적화 정식화 집대성
│
└── papers/                        # 53편의 최신 연구 논문 원문 PDF 아카이브
    ├── 2007_huang-xie_beso-new-algorithm_JSME-49-4-1091.pdf
    ├── 2020_fu_semdot-smooth-edged-material-distribution_arXiv-2005.09233.pdf
    ├── 2022_triller_DiESL-topology-SIMP_SMO-65-2022.pdf
    ├── 2025_buzaglo_first-order-ESL-dynamic_arXiv-2508.07222.pdf
    ├── 2025_tian_beso-adaptive-evolutionary-ratio_CJME-38-122.pdf
    ├── 2026_park_topoptagents-llm-multiagent_arXiv-2605.23273.pdf
    └── ... (총 53편 완비)
```

---

## 2. 핵심 연구 분야별 논문 분류 및 wht_topopt 구현 매핑

### Track A: 동적 과도응답 및 등가정하중법 (Dynamic Transient TopOpt & True F-ESL)
* **Buzaglo & Pollini (2025/2026)**, *First-order equivalent static loads for dynamic response structural optimization*, arXiv:2508.07222:
  - **핵심 기여**: 기존 Kang & Park(2006) ESLM이 1차 KKT 정상성을 만족하지 못한다는 Stolpe(2018)의 반례를 극복하고, 동적 수반 민감도 보정항($\boldsymbol{\delta}_{dynamic} = -\mathbf{u}_S^T \frac{\partial \mathbf{M}}{\partial \rho} \ddot{\mathbf{u}}_D$)을 추가하여 엄밀한 1차 정상성(F-ESL)을 확립함.
  - **wht_topopt 구현**: `wht_topopt/examples/ex09_esl_dynamic_robot_arm.py`의 `RobotArmFESLOptimizer3D` 및 `compute_rigorous_fesl_sensitivity`에 100% 반영.
* **Triller, Immel & Harzheim (2022)**, *Topology optimization using difference-based equivalent static loads*, SMO:
  - **핵심 기여**: 기하학적/재료 비선형 구조에서 변위 증분 기반 DiESL 정식화 수립.
* **Behrou & Guest (2017)**, *Topology optimization for transient response of structures subjected to dynamic loads*, arXiv:1705.01538:
  - **핵심 기여**: 시간 영역 직접 수반 방정식(Direct time-domain adjoint)의 정석.

### Track B: 경계 평활화 및 비SIMP 기법 (SEMDOT & Level-Set)
* **Fu et al. (2020)**, *Smooth-edged material distribution for optimizing topology (SEMDOT)*, arXiv:2005.09233:
  - **핵심 기여**: 페널티를 주지 않는($p=1$) 재료 보간과 형상함수 기반 서브복셀 분할로 톱니 형상(Zig-zag)을 완벽히 제거하는 매끄러운 3D 경계 도출.
  - **wht_topopt 구현**: `wht_topopt/algorithms/semdot.py`에 이식, GE 브라켓 고해상도 벤치마크 완주.
* **Wegert, Roberts & Challis (2023/2024)**, *A Hilbertian projection method for constrained level set-based topology optimisation*, arXiv:2305.03913 / *GridapTopOpt.jl*, arXiv:2405.10478:
  - **핵심 기여**: $H^1$ 정칙화 연장 및 그람-슈미트 제약 직교화 투영을 통한 무재초기화 레벨셋.
* **Ježek et al. (2026)**, *Sequential topology optimization: SIMP initialization for level-set boundary refinement*, arXiv:2605.04735:
  - **핵심 기여**: SIMP 최적화 결과 밀도를 레벨셋 부호거리함수(SDF)의 초기값으로 인계하는 하이브리드 파이프라인.

### Track C: 적응형 진화 및 셀룰러 오토마타 (BESO & SE-AHCA)
* **Huang & Xie (2007)**, *A New Algorithm for Bi-Directional Evolutionary Structural Optimization*, JSME:
  - **핵심 기여**: 민감도 수(Sensitivity number) 기반 양방향 요소 추가/제거 알고리즘.
* **Tian & Zhang (2025)**, *Bi-Directional Evolutionary Topology Optimization with Adaptive Evolutionary Ratio*, CJME:
  - **핵심 기여**: 수렴 상태에 따라 하이퍼볼릭 탄젠트로 진화율(ER)을 자동 조절하는 적응형 BESO.
* **Stress-Energy Adaptive HCA (2025/2026)** (SOTA SE-AHCA):
  - **핵심 기여**: 수반 편미분방정식 풀이 없이 변형에너지밀도(SED)와 $qp$-완화 von Mises 응력을 가중 결합하여 노치 응력 집중을 완화하고 완전응력설계(FSD)를 달성하는 적응형 PID 게인 스케줄링 HCA.
  - **wht_topopt 구현**: `wht_topopt/algorithms/hca.py`에 이식되어 체적 보존 오차 $< 10^{-5}$ 및 응력 변동계수(COV) -16.8% 평탄화 실증.

### Track D: AI 에이전트 및 대규모 언어 모델 연계 (LLM-Agents in TopOpt)
* **Park & Chung (2026)**, *Self-refining topology optimization via an LLM-based multi-agent framework (TopOptAgents)*, arXiv:2605.23273.
* **Lin et al. (2026)**, *TO-Master: an LLM-agent framework for automated topology optimization*, arXiv:2607.01812.
* **Yang, Wang & Wang (2026)**, *LLMs as optimization controllers: adaptive continuation for SIMP*, arXiv:2603.25099.
  - **wht_topopt 구현**: Antigravity/Claude 자율 오케스트레이션 및 MCP 도구 인터페이스 규격의 사상적 배경.

---

## 3. 한국어 심층 해설서 안내 (`summaries_ko/`)

본 아카이브의 `summaries_ko/` 폴더에는 수식 유도부터 코드 라인 단위 매핑까지 기술한 7편의 한국어 마스터 가이드가 수록되어 있습니다:

1. [`01_foundations_and_filters_ko.md`](summaries_ko/01_foundations_and_filters_ko.md): SIMP 이론, 밀도 필터, 민감도 필터, Heaviside 투영법 수식 전개.
2. [`02_optimizers_and_non_simp_ko.md`](summaries_ko/02_optimizers_and_non_simp_ko.md): Svanberg MMA/GCMMA 듀얼 문제 전개, BESO, Level-Set, SEMDOT.
3. [`03_stress_buckling_dynamics_ko.md`](summaries_ko/03_stress_buckling_dynamics_ko.md): 응력 특이점 회피 $qp$-보간, 기하학적 강성 $K_G$ 좌굴 고윳값, F-ESL 과도 동역학.
4. [`04_manufacturing_constraints_ko.md`](summaries_ko/04_manufacturing_constraints_ko.md): 밀링 압출(Extrusion), 주조 탈형구배(Draft Angle), 적층제조 오버행 제한.
5. [`05_multiscale_and_neural_ko.md`](summaries_ko/05_multiscale_and_neural_ko.md): 점근 균질화(Asymptotic Homogenization), 신경망 재파라미터화.
6. [`06_kratos_topopt_and_lessons_ko.md`](summaries_ko/06_kratos_topopt_and_lessons_ko.md): Kratos C++ 인메모리 포인터 교환 기법 및 PARDISO 가속.
7. [`07_frontistr_elements_and_topopt_ko.md`](summaries_ko/07_frontistr_elements_and_topopt_ko.md): FrontISTR 대규모 SMP/MPI 분산 병렬 연동 메커니즘.

---

## 4. 논문 원문 PDF 열람 안내

PDF 원본 파일은 본 디렉토리 내 `papers/` 서브폴더에 저장되어 있으며, 각 논문의 세부 사양 및 DOI는 [`INDEX.md`](INDEX.md)에서 확인하실 수 있습니다.
- 저장 용량: 총 53편 (약 365 MB)
- 연구 목적의 로컬 참조 및 인스펙션 전용으로 활용됩니다.
