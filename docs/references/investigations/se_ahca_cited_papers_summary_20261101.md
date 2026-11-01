# SE-AHCA 인용 논문 7건 수집·분석 요약 (2026-11-01)

- 근거 문서: `wht_topopt/docs/references/investigations/advanced_hca_implementation_and_plan.md` §1 (인용 목록 1~4번) + `se_ahca_stress_leveling_and_reference_hub_20261101.md` §3 (집대성 보고)
- 수집 결과: **원문 PDF 2건 저장** (`papers/`), **미확보 5건** (아래 표, 사유 명시)
- 정독 라벨 규율: [정독]=원문 전수 독해 / [초록]=초록·서지정보 수준 / [미확보]=원문 없음 (summaries_ko 관례 준용). [초록]/[미확보] 절의 수식은 교차 문헌의 표준 형태이며 원전 직접 인용이 아님.

## 0. 수집 현황 일람

| # | 논문 | 상태 | 저장 파일 / 근거 |
|---|---|---|---|
| 1 | Tovar et al. 2006, HCA 원전, JMD 128 | [미확보] (ASME·ResearchGate 유료/차단) | 서지 + Penninger·Tong 내 인용으로 교차 정리 |
| 2 | Penninger et al. 2010, HCA 수렴 해석, SMO 40 | [정독] ✅ 신규 수집 | `papers/2010_penninger_hca-convergence-analysis_SMO-40.pdf` (227,980 B, `%PDF-` 확인) |
| 3 | Raeisi et al. 2026, Adaptive HCA, Eng Opt | [미확보] (T&F 403) | DOI·서지 확인, 내용은 계획서 기술로만 참조 |
| 4 | Battery-pack multi-case HCA + von Mises 2026 | [미확보] (ResearchGate 전용) | 서지 확인 불가 — 인용 보류 (아래 §4) |
| 5 | Tong et al. 2025, HCA 과도 충격 강성, Processes 13(11):3726 | [정독] ✅ 신규 수집 | `papers/2025_hca_transient-impact-stiffness-stress_Processes-13-3726.pdf` (6,231,950 B, `%PDF-` 확인, CC BY) |
| 6 | Wang·Song·Kan, Harmonic HCA, JMD 147(5):051702 | [초록] (ASME·RG 403) | 초록 스니펫 + 서지 확인. ※계획서는 "2024"로 기재하나 실제 게재는 **2025-05** (JMD 147권 5호) — 인용 정정 필요 |
| 7 | Le et al. 2010, 응력 qp-완화, SMO 41 | [초록] (Springer 유료; summaries_ko도 미확보 기록) | Illinois Experts 초록 + DOI `10.1007/s00158-009-0440-y` 확인 |

---

## 1. [정독] Penninger–Watson–Tovar–Renaud (2010), SMO 40:271–282

**서지.** C. L. Penninger, L. T. Watson, A. Tovar, J. E. Renaud, *Convergence analysis of hybrid cellular automata for topology optimization*, Struct. Multidisc. Optim. 40, 271–282 (2010). 원문 12 pp 전수 독해.

**핵심 결과.**
- HCA 갱신식은 불동점 반복 `x(t+1) = G(x(t))`이다 (proportional 제어만 고려, §3 eq. 36–44). 수축사상 조건 `‖DG‖ ≤ L < 1`, 실용 판정은 **spectral radius `ρ(DG(x̄)) < 1`** (eq. 47–50).
- Tovar et al. (2007)의 KKT 조건을 **수정**: multiobjective `min f(U)+g(M)` (eq. 7) 기준 자극 `S_i = U_i/x_i`, 목표 `S*_i = (ω2/ω1)(ρ0i/p)` (eq. 29–30). 포화 셀(`x=0/1`) 조건 별도 유도 (eq. 31–35).
- **이웃 평균(`S̄_i`) 사용 시 KKT 불일치**: `e_i = S̄_i − S*_i = 0` 이어도 `S_i ≠ S*_i`이면 최적성 위반. 수렴 설계는 반드시 **empty neighborhood (N̂=0)** 기준으로 받아야 한다 (§5).
- 2×2 외팔보 수치실험: `ρ(DG) < 1` 영역이 **서로 단절**되어 있고, 초기점·게인 `c_p`에 따라 서로 다른 불동점(국소 최소, FMINCON 대조 확인)에 수렴하거나 순환 발산(`c_p=1.0` 균일 초기점). 즉 **HCA 수렴 ≠ 최적** (§4 Fig. 4–6, Table 1).

**SE-AHCA 연관성 (직접 적용).**
- SE-AHCA의 PID 게인(THD `c_p, c_i, c_d`)·이웃 반경 선택은 곧 다른 불동점 함수 `G`의 선택이다. 고정 게인 HCA가 초기점 의존적 국소 수렴을 보인다는 본 논문이 **적응형 게인 스케줄링(기술 5)의 이론적 정당화**가 된다.
- 이웃 평균 PDE 필터 + Heaviside 투영을 쓰는 SE-AHCA는 Penninger §5 지적과 정면으로 맞물린다: 최종 수렴 판정은 필터 반경을 축소한 상태(≈empty neighborhood)에서 KKT 잔차로 재확인해야 한다. `wht_topopt`의 `kkt_residual` 유틸(`aieng/topopt/esl.py`)을 SE-AHCA 종료 조건에 연결할 것 (구현 이슈로 등록).
- COV −16.8% 같은 균일화 지표는 "자극 평탄화"이지 최적성 증명이 아님을 본 논문이 명시 (§6). 보고서 서술 시 "FSD 달성"과 "KKT 최적"을 분리 표기할 것.

## 2. [정독] Tong–Lin–Li–Chen (2025), Processes 13(11):3726 (CC BY)

**서지.** Z. Tong et al., *Application of Hybrid Cellular Automata Method for High-Precision Transient Stiffness Design of a Press Machine Frame*, Processes 2025, 13, 3726. DOI `10.3390/pr13113726`. 원문 27 pp 전수 독해.

**핵심 내용.**
- JH31-250 프레스 프레임: Abaqus explicit 과도 해석(90,633 요소, ~40분) → **3D→2D 쉘 변환**(12,128 요소) → HCA 개념설계(SED 균일화, 35회 반복) → 2D→3D 복원(보강 리브) → 응답면(CCD)+PSO 치수 최적화. 상대변위 0.7098 → **0.4896 mm (−31.02%)**, 최대 등가응력 117.80 → 97.52 MPa (§3–§5).
- HCA 정식화: 상태 `α_i(t)`, SED 오차 `e_i = U_i − U*` (eq. 10), 두께 갱신 `Δt_i = α·e_i` (eq. 12), 체적 수렴 판정 (eq. 13–15). 이웃 8셀 모델.
- 방법론적 정직성: "2D 쉘 응력 절대값은 3D와 직접 비교 불가, 고SED 영역 식별이 목적" 명시 (§5.1); "HCA는 단일 목적 최소화가 아니라 제약 하 SED 균질화, 최종값은 균형 상태" 명시 (§5.2); 기본주파수 저하(22.2→20.2 Hz)·60 Hz대 고차 공진 잔류라는 **부작용도 보고** (§5.4).
- Tovar 2006 (ref 27–28), Da et al. 2017 HCA 재료설계 (ref 31) 인용 — Tovar 원전 교차 근거로 활용 (§0-1번 보강).

**SE-AHCA 연관성 (직접 적용).**
- 본 논문이 SE-AHCA Mode 1 (외부 솔버 무구배 최적화)의 **실증 선례**: Abaqus 결과 파일의 요소 SED만 읽어 C++ 자체 코드로 밀도 갱신 — `wht_topopt`의 CalculiX/FrontISTR 파서 + `hca.py` 구조와 동형. 2D↔3D 왕복(리브 복원) 절차는 ex09 로봇 암 허브 보강 설계에 직접 이식 가능.
- 한계도 그대로 계승됨을 주지: 과도 최대상대변위 단일 목적, 체적 고정, 수렴이 KKT가 아님. ex09 연동 시 동일 한계 고지 필요.

## 3. [미확보] Tovar et al. (2006), JMD 128:1205–1216 — HCA 원전

- 서지 확인 (ASME 221회 인용 기록, ResearchGate 2006-01 등재). 원문은 ASME 유료 + RG 차단으로 미확보.
- 내용은 Penninger §2 (PID 갱신식 eq. 5–6, 이웃 평균 eq. 4, Wolff's law 동기) + Tong §4.1 (SED 균일화, eq. 7–12) + DYNAMICS_SURVEY §2·§7 ([search] 정리)의 **삼중 교차**로 재구성됨. 독자 정리는 summaries_ko/03 §4.3 ESL 대응표 참조.
- 추가 확보 경로: Tovar (2004) Notre Dame 박사학위논문 (Penninger ref 인용) — 학위논문은 통상 공개 저장소 경유 확보 가능. 후속 작업으로 지정.

## 4. [미확보] Adaptive HCA (Raeisi et al. 2026), Eng Opt — 적응형 게인 근거

- 서지 확정: S. Raeisi et al., *Control-based structural optimization for crashworthiness using adaptive hybrid cellular automata*, Engineering Optimization, 1–28 (2026), DOI `10.1080/0305215X.2025.2608738` (Tovar CV·Scilit Open Access 표기). T&F는 403 차단.
- SE-AHCA 기술 5 (요소별 적응형 PID 게인)의 직접 근거로 계획서에 인용되어 있으나, 원문 미확인 상태에서는 게인 스케줄 수식을 계획서에서 **독자 제안([proposal])으로 격하 표기**해야 한다. 확보 후 `hca.py` L319 주석과 대조 검증 예정.

## 5. [미확보→보류] Battery-pack multi-case HCA + von Mises (2026)

- 계획서 인용: ResearchGate 398115713, CTB 배터리팩 다중 하중 von Mises HCA. 검색으로 서지 확인 불가 (유사 주제 무관 논문만 반환).
- **인용 보류 판정**: 저자·게재지·연도 미확인 인용은 AGENTS.md §10-8a (silent fill 금지) 위반 소지. SE-AHCA 다중 하중 포락선 기능의 근거로 쓰려면 서지 확정 후 재인용. 그 전까지 다중 하중 정당화는 Tong §5 (단일 목적 + 체적 제약) 범위로 축소 서술.

## 6. [초록] Wang–Song–Kan, Harmonic HCA, JMD 147(5):051702 (2025-05)

- 서지: Z. Wang, X. Song, Z. Kan, *A Harmonic Response Topology Optimization Method Based on Hybrid Cellular Automaton*, J. Mech. Des. 147(5):051702. 초록: "MDM(Mode Displacement Method)…" 기반 조화응답 HCA, **20–30회 반복 수렴** 주장.
- SE-AHCA 연관: 과도(50 ms S-curve)가 아닌 **주파수 영역** HCA 선례. ex09 비교 파이프라인의 모달 해석(`compute_modal_frequencies`)과 주파수 응답 평가 확장의 근거로 활용 가능. 원문 확보 후 MDM-HCA ↔ Gen-α 과도 HCA 정식화 비교를 후속 작업으로 지정.
- ※ 계획서 "2024" 표기는 **2025로 정정** (147권 = 2025년호).

## 7. [초록] Le et al. (2010), SMO 41:605–620 — 응력 qp-완화 원전

- 서지 확정: C. Le et al., DOI `10.1007/s00158-009-0440-y`, 16 pp (Illinois Experts 초록 확보). 핵심: 밀도 필터 + SIMP + **SIMP-motivated 응력 정의로 특이점 해소** + 전역/영역 응력 측도 + 적응 정규화로 국소 응력 제어.
- SE-AHCA 기술 1 (qp-완화 von Mises `σ̂_vm`)의 직접 근거. summaries_ko/03 §4 (미확보 고지 + 후속 문헌 교차 정리)와 정합. 원문 확보 시 `hca.py` 완화식과 대조 검증 예정. 확보 경로 후보: DTU orbit / 저자 홈페이지 — 후속 작업으로 지정.

---

## 8. 코드·문서 반영 사항 (본 작업 수행분)

1. `papers/` 신규 2건 저장 (헤더 검증済). 총 55편 (53→55).
2. `INDEX.md` HCA·응력 섹션에 신규 행 추가 + "Not downloaded" 목록 갱신 (미확보 5건 + 사유).
3. 계획서 정정 요구 2건 발행: harmonic 연도 2024→2025 (§6), battery-pack 인용 보류 (§5). adaptive 게인 [proposal] 격하 (§4).

## 9. 후속 작업 (미확보 해소 우선순위)

1. Tovar (2004) 학위논문 공개본 확보 → #1 교차 정리 검증.
2. Raeisi 2026 OA 경로(Scilit 표기) 재시도 → 기술 5 수식 대조.
3. Le 2010 DTU orbit/저자본 확보 → 기술 1 대조.
4. Wang 2025 원문 확보 → MDM-HCA 비교 절 추가.
