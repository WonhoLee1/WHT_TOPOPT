# Benchmark Study 1: High-Speed Industrial Robot Arm Dynamic Rib & Multi-Physics Study

> **주의 (2026-11-03).** 이 문서의 수치는 코드 리뷰(`docs/references/investigations/code_review_wht_topopt_20261103.md`) 수정 전 결과이며 재생성이 필요하다. 해당 결함: Generalized-alpha 우변(R1), 시간 스텝과 선형 질량(R8), 피크 응력과 가중치(R5), 신호 필터(R6). F-ESL 표기는 휴리스틱 보정이다(R7).

> 본 문서는 고속 산업용 픽앤플레이스 로봇 암(Robot Arm)에 대해 수행된 정적·동적 위상최적화, 평면 압출 리브 필터링(PlanarRibExtrusionFilter), 체적 완화(Volume Relaxation, $V=35\%$)를 통한 All-Dominant 고성능 리브 달성 및 SE-AHCA 벤치마크 결과를 집대성한 공식 기술 보고서입니다.

---

## 1. 벤치마크 개요 및 물리적 조건

### 1.1 지오메트리 및 재료 물성
- **설계 영역 (Design Domain)**: $400\text{ mm} \times 60\text{ mm} \times 40\text{ mm}$ 직육면체 박스 블록
- **비설계 영역 (Non-Design Domain)**:
  - Base Hub 1 (원점 $X=0$, 구속 허브): 외경 $\phi 50\text{ mm}$, 내경 $\phi 24\text{ mm}$
  - Tip Hub 2 ($X=370\text{ mm}$, 엔드이펙터 체결 허브): 외경 $\phi 44\text{ mm}$, 내경 $\phi 18\text{ mm}$
- **재료 물성 (Al6061-T6)**:
  - 탄성계수: $E = 71.0\text{ GPa} = 71,000\text{ MPa}$
  - 포아송 비: $\nu = 0.33$
  - 밀도: $\rho = 2.70\text{ g/cm}^3 = 2.70 \times 10^{-3}\text{ g/mm}^3$
  - 항복강도: $\sigma_{\text{yield}} = 276.0\text{ MPa}$

### 1.2 하중 조건
1. **정적 다중 하중 (4 Load Cases)**:
   - Case 1: 슬루잉 굽힘 주하중 $F_y = -750\text{ N}$ (가중치 $0.35$)
   - Case 2: 축방향 급가감속 하중 $F_x = +620\text{ N}$ (가중치 $0.25$)
   - Case 3: 수직 이송 하중 $F_z = +350\text{ N}$ (가중치 $0.15$)
   - Case 4: 엔드이펙터 편심 비틀림 모멘트 $M_x = 28.0\text{ N}\cdot\text{m}$ (가중치 $0.25$)
2. **동적 과도 충격 하중 (Transient Slewing & Torsion)**:
   - $120\text{ ms}$ 슬루잉 가감속 과도 토크 파형 인가 후 자유 감쇠 진동 해석 (Generalized-$\alpha$ Solver, 감쇠비 $\zeta = 0.02$).

---

## 2. 평면 압출 리브 필터 (PlanarRibExtrusionFilter)와 제조성

### 2.1 1-Rib vs 2-Rib의 위상학적 원인 분석
- **질문**: *"1-rib, 2-rib는 의도적으로 1, 2 rib를 설계한 것인가, 아니면 만들고 보니 그렇게 된 것인가?"*
- **해석학적 원인**:
  - `PlanarRibExtrusionFilter`는 Z축(두께 $40\text{ mm}$) 방향으로 밀도를 동일하게 유지하는 압출 투영 필터입니다.
  - 최적화기가 특정 rib의 개수를 사전에 정해둔 것이 아니라, **인가된 하중의 역학적 요구조건에 의해 자발적으로 분기(Bifurcation)**된 결과입니다.
  - **정적 하중 우선 시 (1-Rib/단일 아치 형성)**:
    - 슬루잉 굽힘 하중($F_y$)에 저항하기 위해 중립축에서 가장 먼 수직 외곽 플랜지와 중앙을 관통하는 굵은 단일 아치 리브(1-Rib)로 재료가 집중됩니다.
  - **동적 비틀림 및 충격 우선 시 (2-Rib/대각 X-브레이싱 형성)**:
    - 비틀림 모멘트($M_x$)에 의해 발생하는 전단 변형을 억제하기 위해 단일 아치가 상하 2갈래의 대각선 트러스(2-Rib / X-bracing)로 분할되어 비틀림 강성을 극대화합니다.

---

## 3. 핵심 모델별 정량 비교 평가 (Benchmark Matrix)

| 기계공학 평가 항목 | Model 1-Static (정적) | Model 2-Dynamic (동적) | Model 3-Rib-Plus (체적완화) | Model 4-HCA (SE-AHCA) |
|---|:---:|:---:|:---:|:---:|
| **목표 체적률 ($V$)** | $28.00\%$ | $28.00\%$ | **$35.00\%$** (완화) | $28.00\%$ |
| **최종 부품 질량 (Mass)** | $241.6\text{ g}$ | $241.6\text{ g}$ | $301.9\text{ g}$ ($+60\text{ g}$) | **$244.4\text{ g}$** |
| **슬루잉 굽힘 강성 ($K_y$)** | $4,627.1\text{ N/mm}$ | $3,984.5\text{ N/mm}$ ($-13.9\%$) | **$8,540.8\text{ N/mm}$ ($+84.6\%$)** | **$4,850.2\text{ N/mm}$ ($+4.8\%$)** |
| **비틀림 회전 강성 ($K_{\theta x}$)** | $1.42\times 10^7\text{ N}\cdot\text{mm/rad}$ | $3.15\times 10^7$ ($+121.8\%$) | **$4.82\times 10^7$ ($+239.4\%$)** | $2.10\times 10^7$ ($+47.9\%$) |
| **1차 고유진동수 ($f_1$)** | $265.7\text{ Hz}$ | $312.4\text{ Hz}$ ($+46.7\text{ Hz}$) | **$353.0\text{ Hz}$ ($+87.3\text{ Hz}$)** | $298.5\text{ Hz}$ ($+32.8\text{ Hz}$) |
| **피크 비틀림 변위 ($U_z$)** | $3.54\text{ mm}$ (극심한 진동) | $1.20\text{ mm}$ ($-66.1\%$) | **$0.95\text{ mm}$ ($-73.1\%$, 1mm 미만)** | $1.85\text{ mm}$ ($-47.7\%$) |
| **피크 폰미세스 응력 ($\sigma_{\max}$)** | $64.9\text{ MPa}$ | $52.3\text{ MPa}$ | $38.7\text{ MPa}$ | **$30.4\text{ MPa}$ ($-53.2\%$, FSD)** |
| **수반 감도 연산 횟수** | $30\text{ 회}$ | $30\text{ 회}$ | $25\text{ 회}$ | **$0\text{ 회}$ (Adjoint-Free)** |
| **STL 메쉬 수밀성** | $100\%$ Watertight | $100\%$ Watertight | **$100\%$ Watertight** ($138,935\text{ mm}^3$) | **$100\%$ Watertight** |

---

## 4. 핵심 공학적 발견 및 결론

1. **체적 완화(Volume Relaxation)를 통한 All-Dominant 입증**:
   - 동일 체적($V=28\%$)에서는 굽힘 강성과 비틀림 강성 간에 트레이드오프가 존재하여 어느 한쪽이 우세하면 다른 쪽이 희생됨.
   - 체적 제약을 $V=35\%$로 단 $60.3\text{ g}$ 완화한 `Model 3-Rib-Plus`는 굽힘 강성($+84.6\%$), 고유진동수($+87.3\text{ Hz}$), 비틀림 진동 억제($-73.1\%$)의 **모든 지표에서 정적·동적 기본 모델을 압도하는 All-Dominant 성능을 완벽히 달성**.
2. **SE-AHCA의 실용적 우수성 실증**:
   - 수반 편미분 방정식 풀이를 전면 배제하여 단 $158.5\text{ 초}$ 만에 최적화 완주.
   - $qp$-완화 von Mises 응력 평탄화 메커니즘을 통해 노치 응력 집중을 $30.4\text{ MPa}$로 대폭 완화하여 피로 수명 및 신뢰성을 획기적으로 개선.
