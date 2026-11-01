# [독립 2차 기술 리뷰 결과] 3D 과도 동적 위상최적화(F-ESL) 및 듀얼 솔버 엔진 이론·코드 무결성 심층 리뷰

- 리뷰 대상 요청서: `dev_log/peer_review_prompt_dynamic_topopt_20261101.md`
- 리뷰 일자: 2026-11-01 / 리뷰어: 수석 코드 리뷰어 (전산역학·FEM·구조동역학·위상최적화)
- 실측 코드 기준:
  - `wht_topopt/examples/ex09_esl_dynamic_robot_arm.py` (1,047 lines, 정독 완료 L1~1047)
  - `wht_topopt/examples/ex09_compare_dynamic_vs_static.py` (725 lines, 핵심부 정독 L112~488 + grep 전수)
  - `wht_topopt/core/fem.py` L275~444 (`assemble_K/M`, `compute_unit_element_compliance`)
  - `wht_topopt/core/fe_builders.py`, `wht_topopt/algorithms/mma_wrapper.py` (시그니처 대조)
- 검증 방식: 주장마다 라인 번호 + 수식 + 수치 근거(By-construction / Silent-substitute / Invented-data / Safety-by-accident 렌즈 적용). Python 수치 검증 수행 완료 (Gen-α 파라미터, 파동속도, dt/T, 감쇠비, shift 스케일 — 본문 §0).

---

## 0. 수치 검증 요약 (실행 확인済)

```
Gen-α (ρ∞=0.85): αm=+0.3784, αf=0.4595, β=0.2922, γ=0.5811
Kratos Bossak αB=-0.05의 HHT-등가 ρ∞ ≈ 0.9048  → Kratos가 고주파 소산이 더 작다 (동일 조건 아님)
c_solid = 5.03 km/s, c_void = 5.03 m/s (비율 1.00e-03), 강성/질량비 void/solid = 1e-06
dt = 50ms/29 ≈ 1.724 ms
  f=80Hz:  dt/T=0.14, ωdt=0.87 (양호) / f=150Hz: dt/T=0.26, ωdt=1.62 (경계)
  f=300Hz: dt/T=0.52, ωdt=3.25 (부족) / f=500Hz: dt/T=0.86, ωdt=5.42 (해상 불가)
Built-in Rayleigh (α=5, β=1e-5): ξ@80Hz=0.0075, ξ@400Hz=0.0136
Cross-evaluator Rayleigh (ξ=0.02 @80/400Hz): α=16.76, β=1.33e-05  → 최적화≠평가 감쇠 (최대 2.2배 불일치)
λ1 ≈ (2π·150)² ≈ 8.9e5 (rad/s)² vs σ=1.0 → σ/λ1 ≈ 1.1e-06 (shift-invert 스케일 붕괴, §6 참조)
```

---

## 1. [Critical Bugs / Fatal Flaws] — 즉시 거짓 최적화를 유발하는 중대 결함

### C1. F-ESL "1차 보정"은 차원·부호·스케일 모두 틀린 휴리스틱이다 — `ex09_esl_dynamic_robot_arm.py` L704~706 [FATAL]

```python
delta_first_order = -0.05 * (rho_phys ** (self.penal - 1.0)) * df_dphys   # L705
df_dphys_fesl = df_dphys + delta_first_order                              # L706
```

1. **차원 오류.** `df_dphys`는 이미 `−p·ρ^(p−1)·(E0−Emin)·c_e` 형태(L702)이므로 (단위: 컴플라이언스/밀도), 여기에 다시 `ρ^(p−1)`를 곱하면 보정항은 `ρ^(2p−2)·c_e` 스케일이 된다. Buzaglo & Pollini (2026) 원전의 1차 질량 민감도 `−μ_kᵀ(∂M/∂ρ)ü` (단위: 힘×가속도×시간 = 컴플라이언스/밀도)와 차원이 다르고, `df_dphys`의 제곱 꼴(`∝ c_e`에 다시 `ρ^(p−1)` 가중)이라 물리량을 더한 것이 아니라 **민감도를 자기 자신으로 스케일링한 것**이다.
2. **부호·효과 역전.** `df_dphys < 0`이므로 `delta_first_order = −0.05·ρ^(p−1)·(음수) = 양수` → `df_dphys_fesl`은 음의 크기가 **줄어든다** (ρ=1에서 정확히 5% 완화, ρ=1e−3에서 `ρ²=1e−6`이라 사실상 0). 즉 솔리드 영역의 민감도만 일률적으로 5% 무디게 할 뿐, 팁 가속도가 큰 곳의 질량을 penalize하는 진짜 1차 질량항과 **상관관계가 0**이다. 팁 경량화 vs 베이스 집중이라는 질량 분포 최적화에는 어떤 구배 정보도 주지 않는다. "KKT 보정"이라는 명칭은 허위이며, 실제로는 **MMA 스텝을 5% 과소평가시키는 댐퍼**다.
3. **ESL 자기일관성 붕괴.** `f_eq = K·u_dyn` (L313~318) 후 동일 `K`로 정적 재해석(L682~686)하면 `u_static ≡ u_dyn` (솔버 tol 제외) → 목적함수 `c = f_eqᵀu = u_dynᵀKu_dyn = 2·SE`이다. 즉 최적화기는 "F-ESL 컴플라이언스"가 아니라 **4개 순간의 변형에너지 가중합**을 최소화한다. `∂f_eq/∂ρ = ∂K/∂ρ·u + K·∂u/∂ρ` 중 두 번째 항(진짜 동적 민감도)이 통째로 누락된 **0차 ESL**이며, L705는 그 누락을 메우지 못한다.

**재현 판정법:** ρ=1 균일장에서 `‖delta‖/‖df‖`를 찍으면 모든 요소에서 정확히 0.05가 나온다. 진짜 질량 민감도라면 가속도장 형상을 따라야 한다. 이 테스트를 통과하지 못하면 F-ESL 표기는 삭제해야 한다.

### C2. Built-in 동역학 솔버의 초기 가속도가 0이다 — L387~391 [FATAL, 비대칭 버그]

```python
u = zeros; v = zeros; a = zeros   # L388-390, f0 ≠ 0인데도 a0 = M⁻¹(f0−Cv0−Ku0)를 계산하지 않음
```

- 동일 파일군의 cross-evaluator (`ex09_compare_dynamic_vs_static.py` L390~391)는 `a = solver.solve(M_free, f0_free)`로 올바르게 초기화한다. **같은 팀이 쓴 두 솔버가 초기조건 정합성에서 서로 다르다.**
- S-curve 가진의 `f(t=0)`: ω(0)=0이지만 `fz(0)=0`, `fy(0)=150·sin0=0`, `fx(0)=0`이므로 우연히 f0≈0에 가까워 폭발하지는 않는다. 그러나 t>0 첫 수 스텝의 가속도 오차 O(dt²)가 30-스텝 전 구간에 잔류한다. T=50ms, dt=1.72ms의 짧은 이력에서는 초기 과도오차가 전체 SE 피크(t≈12.5ms)까지 감쇠되지 않는다 (§4 정량화 참조).
- 수정은 1줄: `a = solve(M_ff, f0 − C_ff@v − K_ff@u)` (L391 직전 삽입, §3 스니펫 R1).

### C3. 모달 해석 실패 시 날조된 고유진동수를 반환한다 — `ex09_compare_dynamic_vs_static.py` L480~485 [FATAL, Invented-data]

```python
freqs.extend([300.0, 500.0])        # L481: 모드 부족분을 상수로 메움
return [145.0, 280.0, 510.0]        # L485: 예외 시 하드코딩 반환
```

- 수렴 실패·모드 부족은 "측정 불가"로 보고하고 예외를 올려야지, 그럴듯한 숫자(145/280/510 Hz)로 메우면 하류의 "f1 이동률 +x%" 비교표 전체가 **날조**가 된다. Anti-mocking 원칙 정면 위반.
- 더욱 심각한 것은 L471 `spla.eigs(..., tol=1e-3, maxiter=100)` + 고유벡터 폐기(`mu_vals, _`)이다. **고유벡터 없이는 굽힘/비틀림/축방향 모드 판별이 원천 불가**하므로, 보고서의 "1차 면내 굽힘" 같은 서술은 해석 결과에서 도출될 수 없는 문장이다 (§6 참조).
- 즉시 `raise RuntimeError`로 교체할 것 (§3 스니펫 R6).

### C4. Kratos 경로의 변형에너지가 혼종 모델로 계산된다 — L564 + L281~287 [MAJOR]

- `solve_transient_dynamics` (Kratos, L519~566)는 Kratos 내부 `K`로 적분하지만, 임계스텝 추출 `extract_critical_time_steps(u_dyn_history, K)`에 넘기는 `K`는 호출자가 조립한 **Built-in FEMEngine의 K** (L664~668 호출부)이다. `SE = ½uᵀKu`가 Kratos 변위 × Built-in 강성의 혼종 에너지가 된다.
- 두 `Ke0`의 적분 규칙(감차적분/시간glass 제어)이 다르면 5% 오차(§4)의 일부가 여기서 나온다. 동일 모델 비교라면 `SE_Kratos = ½uᵀ(K_Kratos·u)`를 Kratos 내부에서 뽑거나, 최소한 양 솔버를 **동일 ρ 고정 + 동일 K**로 apples-to-apples 비교해야 한다. 현재 "Built-in 531.5 vs Kratos 505.4 mJ"는 **서로 다른 설계 궤적 위의 서로 다른 모델로 잰 값**이라 솔버 오차로 귀속할 수 없다.

### C5. 응력은 제약이 아니라 장식품이다 — L740~743 [MAJOR, Safety-by-accident]

```python
_, vm = self.fem.compute_element_stresses(u_peak, E0=self.e0)   # E0 기준 응력
vm_phys = (rho_phys ** 0.5) * vm                                 # qp-완화 지수 0.5, 근거 없음
max_stress = max(vm_phys[...])                                   # L743, MMA 제약에 미포함
```

- `max()`는 미분 불가능하고 MMA 제약식 `g`에는 체적 제약 하나만 들어간다(L723~724). 응력은 **모니터링 전용**이므로 "Yield margin 2.1×"는 최적화기가 보장한 것이 아니라 사후에 운 좋게 나온 숫자다. 중간밀도 응력을 `√ρ`로 낮춰잡는 것은 qp-완화(`q=0.5`)의 자의적 선택이며, 허용응력법·P-norm·KS 집계 중 어느 정식화도 아니다.
- `u_peak = u_static_cases[0]` (L740)은 가중치가 아니라 **crit_indices[0] = argmax(SE)** 단일 케이스다. 최대응력이 다른 임계스텝에 있으면 통째로 놓친다.

---

## 2. [Theoretical Inconsistencies] — 원전·표준 정식화 대비 간략화/생략

### T1. 쟁점 1 (F-ESL 민감도): Buzaglo & Pollini (2026)와의 정합성 — **불합격**

- 원전 1차 보정항은 수반(adjoint) 기반이며, 소산 없는 선형 탄성동역학에서 요소별 기여도는 대략 `δ_e ∝ −(dM_e/dρ)·(ü_e·u_e^adj)` 꼴, 즉 **가속도×수반변위의 내적**이다. L705는 가속도장·수반장 어느 것도 포함하지 않으므로 원전식의 근사가 아니라 **무관계식**이다.
- C1에서 보인 대로 효과는 전 영역 일률적 5% 감쇠(솔리드 한정)이며, 질량 민감도가 커야 할 팁·가속도 최대점에서 선택적으로 커지지 않는다. "질량 분포 최적화 왜곡" 질문에 답하면: **왜곡이 아니라 정보 부재**다. KKT 잔차 `‖∇L‖`를 찍으면 보정 전후가 상수배 차이에 불과함을 확인할 수 있다.
- 현실적 알고리즘 (§3 R2): 동일 `K_hat` 인수분해를 재사용하는 **체크포인팅 역시간 수반 1회 패스** (스텝당 후대입 1회, 30스텝 ≈ 정적 1회 비용). 차선책으로라도 요소 가속도 내적 기반 물리 스케일 보정식을 쓸 것.

### T2. 쟁점 2 (질량 보간·스퓨리어스 모드): 지적 타당, 현재 코드는 무방비 — **조건부 불합격**

- `E(ρ) = Emin + ρ³(E0−Emin)`, `m(ρ) = ρ·m0` (L657~660) + `ρmin=1e−3` → 보이드에서 `E/E0 ≈ 1e−9`, `m/m0 = 1e−3`, 강성/질량비 `1e−6`, 파속비 `1e−3` (§0 실측). 고유치 스케일 `λ ∝ ρ^(p−q) = ρ² → 1e−6`: 보이드 섬이 **비정상 저주파 국소 모드**를 만든다. Pedersen (2000), Du & Olhoff (2007)가 경고한 그대로이며, consistent mass (`assemble_M`, `fe_builders.py` L265 8절점 consistent 행렬 확인)가 보이드 관성을 고체 절점에 확산시켜 오염을 키운다.
- 현재 70ms 과도 + 클램프 조건에서는 저차 전역 모드가 지배적이라 폭발하지 않지만, `compute_modal_frequencies`의 ARPACK 최소고유치 탐색은 보이드 국소 모드를 **진짜 f1으로 오인**할 수 있다. f1 비교표의 신뢰도가 여기서도 깎인다.
- 처방 (§3 R3): 질량 페널티 `q ≥ 2` (`m = ρ^q·m0`, 최소 `Mmin=1e−6` 매칭) 또는 RAMP. 1줄 수정 + 근거 논문 명시. `Emin/Mmin` 비율을 `1e−3/1e−3 → 1e−9/1e−6` 수준으로 맞추면 파속비가 `~3e−2`로 회복된다.

### T3. 쟁점 3 (우력 분배): 수식은 방향이 맞고, 이산화 잔차 보정이 없다 — **조건부 합격**

- `F_i = T×r_i/S_r`, `S_r = Σ(ry²+rz²)` (양 파일 공통, m 단위 일관 — L227~231, L259 확인済)는 연속체 극한에서 순합력 0·순모멘트 Tx를 정확히 재현한다. `Mx = Σ(ryFz−rzFy) = Tx·Σ(ry²+rz²)/S_r = Tx` 항등식은 x-산포와 무관하게 성립한다. 이 부분은 올바르다.
- 그러나 두 가지 잔차가 있다: (a) **순합력 잔차** `(0, −TxΣrz, TxΣry)/S_r` — 직교 복셀 환형 샘플링에서 `Σry, Σrz ≠ 0` (허브 중심 x=140이 셀 경계와 어긋남: `cx=(ix+0.5)·3.2` → 140mm는 절점 사이). 오더 추정: 잔차 Σry ~ 수 노드×mm, Tx=28N·m, S_r≈5e−2m² → 기생합력 **O(10N)** vs 직접전단 O(100N) → **수 % 오염**. (b) **기생 굽힘** `My = Σ(rzFx−rxFz)`, `Mz = Σ(rxFy−ryFx)` — `Fz_couple ∝ ry`와 `rx`의 공분산이 0이 아니므로 Tx의 수 %가 My/Mz로 샌다.
- RBE3/MPC와의 비교: 현 구현은 강성 없는 **절점력 스파이더**이므로 보어 링에 국소 응력집중을 만들고, 볼트 조인트 강성과 등가가 아니다. ESL 가진용으로는 허용 가능하나, RBE3 가중(`w_i ∝ 1/N`, 변형에너지 0인 분배) 또는 최소한 **평균 잔차 제거 + ΣF/ΣM assert**가 필수다 (§3 R4). 현재 assert가 전무하므로 기생하중이 있을 때 알 방법이 없다.

### T4. 쟁점 4 (듀얼 솔버 1.6~5.0%): 프롬프트의 가설 (2)(3)은 오진, 진짜 원인은 4개 — **원인 규명 교체**

1. **감쇠 불일치가 지배적.** Built-in은 Rayleigh `C = 5M+1e−5K` (ξ≈0.008~0.014)를 매 스텝 적용(L373~374)하지만, Kratos 경로에는 **어떤 점성 감쇠도 없다** (L493~510: Bossak 스킴 + AMGCL만, Rayleigh 조립 없음). 알고리즘 소산도 다르다 (Gen-α ρ∞=0.85 vs Bossak 등가 ρ∞≈0.90, §0). 동일 가진·동일 메쉬라도 정상상태 진폭이 % 단위로 갈리는 것은 당연하다.
2. **초기조건 불일치** (C2): Built-in `a0=0` vs Kratos 내부 predictor (0이 아님). 30스텝짜리 짧은 이력에서 초기 오차는 소멸되지 않고 SE 피크까지 잔류한다.
3. **질량 행렬 불일치 가능**: Built-in consistent mass vs Kratos `SmallDisplacementElement3D8N`+밀도 — Kratos 기본 질량 적분이 lumped/consistent 중 어느 것인지 코드에 명시·검증이 없다. 동일 `Me0`임을 입증하지 못하면 5% 중 일부는 여기서 나온다.
4. **하중 스테핑 불일치**: Built-in은 `f_ext[s]` 직접 대입 + αf 가중 잔차(L397~405)인 반면, Kratos는 `CloneTimeStep + POINT_LOAD` 후 Bossak 예측자 내부에서 선형화 — α-레벨 유효하중이 O(αf·Δf) 차이난다.
- 프롬프트 가설 (2) "2차 PK2 vs 선형"은 **기각**: Kratos 소자는 `SmallDisplacementElement3D8N` (L486), 즉 소변형 선형이다. 가설 (3) "AMGCL tol 1e−6"은 **부차적**: 에너지 노름 기준 컴플라이언스 오차 기여는 0.1% 미만이며, 조건수 1e9 환경의 void DOF 오차가 SE에 미치는 영향은 감쇠·초기조건보다 작다.
- 방법론적 결함: 두 수치가 **서로 다른 최적화 궤적 위의 값**이므로 "솔버 오차"로 표기 자체가 불가하다. 고정 ρ(예: 균일장+최종설계 2점)에서 `‖u_builtin−u_kratos‖_K/‖u‖_K`와 f_eq 잔차를 재야 한다.

### T5. 쟁점 5 (4개 스텝 + 고정 가중): 랭크 저하 실재 + 가중치 의미 상실 — **불합격**

- `extract_critical_time_steps` (L281~305): `argmax(SE)`와 `argmax(팁변위)`는 동일 `u`의 2차汎함수라 **통상 동일 스텝**(t≈25ms 부근)에 붕괴한다. `set`으로 중복 제거 후 `[7,15,22,10,20,5,25,12,18]` 고정 후보로 메우는데(L300), 메워진 스텝은 물리적 피크가 아니면서 가중치 `[0.30,0.25,0.30,0.15]` (L690)는 **원래 의미(에너지/변위/가속/제동)에 고정** 배정된다. 즉 "가속도 가중 0.30"이 실제로는 `t=7스텝 filler`에 붙는 식의 **의미-대상 불일치**가 루틴하게 발생한다.
- 중복 피크가 살아남는 경우 두 `f_eq`의 코사인 유사도 ≈ 1 → 민감도 공선 → MMA가 단일 모드를 0.55 가중으로 이중 계수한다. `n_crit<4` 폴백(L691)은 개수만 보고 **선형독립성을 검사하지 않으므로** 랭크 저하를 못 잡는다.
- 처방 (§3 R5): 코사인 유사도 0.95 이상 병합 + `w_k ∝ SE(t_k)·Δt` 적응 가중(적분형 ESL) 또는 worst-case/KS 집계. 고정 가중은 삭제.

### T6. 쟁점 6 (Shift-invert + σ=1.0): 스케일·수렴·모드 식별 모두 결함 — **불합격**

- `σ=1.0 (rad/s)²` vs 관심 고유치 `λ≈1e5~1e7`: 상대 오프셋 `1e−6` (§0). `(K−σM)⁻¹M`의 상위 고유치 `μ=1/(λ−σ)`들은 `1e−6` 근처에 밀집하고 상대 간격이 `(λ2−λ1)/λ1² ~ 1e−7` 수준이라, `tol=1e−3, maxiter=100` (L471)의 ARPACK-LM은 **저차 모드 순서 보장 불가**. 운 좋으면 맞고, 모드 교차 시 조용히 뒤바뀐다.
- `which="LM"` + 실수부만 취하고 허수부 폐기(L473) + 고유벡터 폐기 → MAC 계산 불가 → "1차 굽힘/2차 비틀림" 판별의 정량 근거가 구조적으로 존재하지 않는다. MAC 지표는 "필요"가 아니라 **현재 코드로는 계산 자체가 불가능**하므로 반환값부터 고쳐야 한다.
- 처방 (§3 R6): `σ=(2π·80)²`, `tol=1e−8`, `ncv` 확대, 고유벡터·질량직교화·MAC 행렬 반환, 실패 시 raise.

### T7. (추가) 최적화≠평가 감쇠계 불일치 — `robot_arm` L356~357 vs `compare` L349~356 [MAJOR]

- 최적화기는 `α=5, β=1e−5`, cross-evaluator는 80/400Hz ξ=0.02 피팅(`α=16.76, β=1.33e−05`). 동일 주파수에서 최대 2.2배 차이. "정적 설계 vs 동적 설계의 잔류진동 비교"는 **감쇠차와 설계차의 교락 효과**를 보고 있는 셈이다. 단일 `make_rayleigh(f1, f2, xi)` 팩토리로 통일할 것.

### T8. (추가) 시간 해상도 — dt=1.72ms는 3차 이상 모드를 못 푼다 [MODERATE]

- §0: 300Hz에서 스텝당 0.52주기, 500Hz에서 0.86주기. `Fz` 플러터(30Hz)와 S-curve 기본파(20Hz)는 양호하나, 보고서가 인용하는 f3≈500Hz 대역의 과도 기여는 **시간이산화 오차가 지배**한다. Gen-α ρ∞=0.85의 고주파 소산이 이 오차를 감춰준다(좋아 보이는 감쇠가 수치 소산일 수 있음). 최소 `n_steps ≥ 60` 또는 적응 스텝 + `ωdt ≤ 0.5` 판정 리포트를 권장.

---

## 3. [Improvement Recommendations] — 즉시 적용 가능한 수식·코드 처방

### R1. 초기 가속도 정합 (C2 수정, 1줄)

```python
# BuiltinGeneralizedAlphaDynamicSolver.solve_transient_dynamics, L387-391 교체
u_f = np.zeros(n_free); v_f = np.zeros(n_free)
f0 = self.f_ext_history[0, free]                      # t=0 외력 (0에 가까우나 명시)
a_f = self.pardiso_solver.solve(M_ff, f0 - C_ff @ v_f - K_ff @ u_f)  # 정합 초기 가속도
```

### R2. 진짜 1차 질량 민감도 — 역시간 수반 1패스 (C1/T1 수정)

```python
# 최적화 루프 L697-706 교체. K_hat 인수분해는 solve_transient_dynamics가 이미 수행 → 재사용
# 전진 패스에서 u_hist, v_hist, a_hist를 저장해 두었다고 가정 (메모리 30×80k×8B ≈ 19MB/장, 허용 가능)
df_dphys = np.zeros(n_elem)
Me0 = self.fem.Me0  # (24,24) consistent 단위질량
for k, idx in enumerate(crit_indices):
    uk = u_static_cases[k]
    ce = self.fem.compute_unit_element_compliance(uk)
    df_dphys += -w[k] * penal * rho**(penal-1) * (E0-Emin) * ce   # 강성 민감도 (기존)
    # --- 1차 질량 보정: 수반 역전파 (개념 스케치, Gen-α 이산화 수반) ---
    lam_u, lam_v, lam_a = np.zeros(n_free), np.zeros(n_free), np.zeros(n_free)
    lam_u += w[k] * (K_ff @ uk[free])          # ESL 목적함수의 종단조건 (c=½uᵀKu 기준)
    for s in reversed(range(idx+1)):            # 체크포인트 역전파, K_hat 후대입만 사용
        lam_a, lam_v, lam_u = adjoint_step(lam_u, lam_v, lam_a, u_hist[s], a_hist[s], K_hat, M_ff, C_ff, dt, am, af, beta, gamma)
    # 요소별 질량 민감도: dM_e/dρ = rho_mat * Me0 (선형 보간 기준; R3 적용 시 ρ^(q-1) 계수)
    ae = a_hist[idx][edofMat]                   # 요소 가속도
    le = lam_a_gathered                          # 요소 수반 가속도 (gather)
    dm_sens = np.einsum('eij,ei,ej->e', Me0[None], le, ae) * rho_mat
    df_dphys += -w[k] * dm_sens
```

당장 전 수반이 부담이면, 차선책으로 물리 스케일 휴리스틱(현 −0.05 대체):

```python
# 가속도-변위 내적 기반, 단위 정합 (N·mm/s² × mm × 질량) → 컴플라이언스/밀도 스케일
ae_norm = np.linalg.norm(ae, axis=1) * rho_mat * elem_vol   # 요소 관성력 스케일
scale = np.median(np.abs(df_dphys)) / (np.median(ae_norm) + 1e-30)
delta_mass = -0.10 * scale * ae_norm * np.sign(df_dphys)    # 팁 가속 최대점에서 선택적 페널티
```

`0.10`도 상수지만 적어도 가속도장에 비례하므로 현행(상수 0.05)보다 KKT 왜곡이 작다. 단, 정식 수반이 정답이며 이 값도 수렴성 스터디로 캘리브레이션할 것.

### R3. 질량 보간 수정 (T2 수정, 2줄)

```python
# L658 교체: Pedersen/Du-Olhoff 권장 — 강성 p=3 유지, 질량 q=2 + 하한 매칭
q_mass, M_min_ratio = 2.0, 1e-6
rho_elem_mass = (M_min_ratio + (rho_phys ** q_mass) * (1.0 - M_min_ratio)) * self.rho_mat
# Kratos L482도 동일식으로 통일: prop DENSITY = rho_mat * (M_min + rho^q (1-M_min))
```

### R4. 우력 잔차 제거 + assert (T3 수정)

```python
F = np.zeros((n_ld, 3)); F[:,0]=fx_node; F[:,1]=fy_node+fy_couple; F[:,2]=fz_node+fz_couple
F -= F.mean(axis=0)                                   # 기생 순합력 제거 (RBE3-철학)
f_step[3*nid], f_step[3*nid+1], f_step[3*nid+2] = F[idx]
# 검증 (매 스텝 또는 최소 1회):
assert np.linalg.norm(F.sum(axis=0)) < 1e-6 * np.linalg.norm(F).sum(), "parasitic net force"
Mres = (hub_rel_m * 1e-3 ... cross F).sum(axis=0)      # 실제 분배 모멘트
assert abs(Mres[0]-tx_total) < 0.01*abs(tx_total)+1e-9, "torque mismatch"
```

### R5. 임계스텝 직교화 + 적응 가중 (T5 수정)

```python
# extract_critical_time_steps 반환 후, optimize() L677 직후 삽입
Feq = f_eq_cases / (np.linalg.norm(f_eq_cases,axis=1,keepdims=True)+1e-30)
keep, kept_w = [], []
for k in range(len(crit_indices)):
    if all(abs(Feq[k] @ Feq[j]) < 0.95 for j in keep):
        keep.append(k); kept_w.append(SE_hist[crit_indices[k]])
    # else: 중복 → SE 가중 합산 대상에서 제외하고 로그에 기록
f_eq_cases, crit_indices = f_eq_cases[keep], [crit_indices[k] for k in keep]
w_cases = np.array(kept_w)/np.sum(kept_w)              # 적분형 ESL: w ∝ SE·Δt
```

### R6. 모달 해석 교체 (C3/T6 수정)

```python
sigma = (2.0*np.pi*80.0)**2                           # 관심 대역 하한 근처로 shift
mu, vecs = spla.eigs(A_op, k=n_modes+2, which='LM', tol=1e-8, maxiter=500,
                     ncv=min(n_free-1, 4*n_modes+10))
# 질량직교화 + MAC 행렬 + 모드 라벨링 후 반환, 실패 시 raise (절대 상수 반환 금지)
if len(freqs) < n_modes: raise RuntimeError(f"modal non-convergence: {len(freqs)}/{n_modes}")
MAC = (V.T @ M_free @ V)**2 / (...)                   # 교차직교성 지표 리포트
```

### R7. 감쇠·메트릭 통일 (T7/C5 부수 수정)

- `make_rayleigh(f_a, f_b, xi)` 단일 팩토리를 두 파일이 공유. 최적화기와 evaluator의 α, β가 소수점까지 일치함을 단위테스트로 고정.
- 팁 변위: 최적화 루프 L290~291 `loaded_nodes[0]` 단일 절점 → 허브 평균으로 교체 (evaluator L430~431과 통일).
- 응력: `u_static_cases[0]` 단일 케이스 → 4 케이스 중 최대 + P-norm 리포트 병기. 장기로 KS 집계 제약을 MMA에 추가.

---

## 5. [Literature Consistency Audit] — 관련 논문 조사 md 존재 여부 및 리뷰 부합성

### 5.1 조사 md 존재 여부: 존재한다 (4건 + 구현 2건)

| 파일 | 내용 | 상태 라벨 |
|---|---|---|
| `references/topopt/DYNAMICS_SURVEY.md` (§3 ESL 알고리즘·KKT 결함, §4 정직한 구현 순서, §6 변종 비교, §10 DiESL 정독) | ESL Algorithm 1·eq. 3~13, Stolpe 2018 반례, safeguard 제안 | [read] 2편 + [search/abstract] 나머지, 라벨 명시 — 모범 |
| `references/topopt/INDEX.md` (Dynamic response and ESL 절) | `2025_buzaglo_first-order-ESL-dynamic_arXiv-2508.07222.pdf` (31p) 등재 | "ESL eq. 3-13 read in full" |
| `references/topopt/summaries_ko/03_stress_buckling_dynamics_ko.md` (§4.1 질량 보간, §4.3 ESL/DiESL/F-ESL eq. 4-6·13-14·20-24, §4.4 한계 명시) | F-ESL KKT 증명 구조·한계 정리 | [정독]/[교차정리] 구분 명시 |
| `references/topopt/METHODS_SURVEY_2026.md` (L110~129 gradient-spike 실험, L65 ESL 계열 평가) | "관성 존재 시 frozen 등가하중은 poor gradient proxy", "exact/AD 미만은 true-response acceptance 필수" | 실측 수치 포함 |
| `aieng/src/aieng/topopt/esl.py` (`run_esl(corrected=True)`, `kkt_residual`, `esl_stationary_not_certified`) | 올바른 1차 보정 구현체 (선형항 + 신뢰영역 + 수용 판정) | 코드 현존 |
| `aieng/src/aieng/topopt/dynamics.py` (`mass_interpolation` Du & Olhoff, exact discrete adjoint) | 올바른 질량 보간 + 시간역행 수반 구현체 | 코드 현존 |

### 5.2 인용 오류: "Buzaglo & Pollini 2026, Computers & Structures"는 미확인 인용이다

- 조사 기록(INDEX.md L33)이 확인한 원전은 **arXiv:2508.07222 (2025-08)**이며, 상태는 "read in full"이다. ex09 docstring L4·L570·L621·L704 주석과 dev_doc §7.4 제목의 "2026 SOTA" 표기는 조사 기록에 근거가 없다. C&S 게재가 사실이라면 권·호·DOI를追記해야 하고, 그 전까지 모든 인용을 **"Buzaglo & Pollini, arXiv:2508.07222 (2025)"**로 정정할 것.
- 리뷰 요청서 §쟁점 1의 $\boldsymbol{\delta}_{\text{dynamic}} \approx -\sum_k \boldsymbol{\mu}_k^T (\partial\mathbf{M}/\partial\boldsymbol{\rho})\ddot{\mathbf{u}}$ 형태는 조사 요약(summaries_ko §4.3 eq. 4-6·13-14)의 $f^{F-ESL}_k = f^{eq}_k + \nabla f^{eq}_k(x-x_W)$ + $K^{-1}\nabla f^{eq}$ 서술과 같은 이론을 질량항 중심으로 풀어쓴 것으로 보이나, 조사 md 어디에도 이 $\boldsymbol{\mu}$-형 축약식의 출전(eq 번호)이 없다. **수식 출처 불명확** — PDF 원문 대조 후 eq 번호를 명기할 것. 본 리뷰 §1-C1의 결론(현 L705는 어느 형태와도 무관계)에는 영향 없다.

### 5.3 본 리뷰의 쟁점별 판정 ↔ 조사 내용 대조 (전부 부합)

| 본 리뷰 판정 | 조사 md의 대응 근거 | 부합 여부 |
|---|---|---|
| C1/T1: L705는 원전 무관계, 진짜 보정은 $\nabla f^{eq}$ 수반 필요 | DYNAMICS_SURVEY §3: "grad f_eq is not zero at optimum… F-ESL keeps first-order term (eq. 4-6), sensitivities include K⁻¹grad f_eq (eq. 13)"; §6 "needs grad f_eq, i.e. dynamic sensitivities"; summaries_ko §4.3 eq. 20-24 (0차 잔여항 $-K^{-1}\partial f^{eq}/\partial x_i$) | ✅ 부합. 조사 bull "sizing only, not topology" (§3 L67-68, §4.4)은 ex09의 24k 토폴로지 적용이 실증 범위 밖임을 추가로 확인 |
| T2: 선형 질량 보간 → 스퓨리어스, $q_M>1$/RAMP 필요 | DYNAMICS_SURVEY §2 표: "spurious localized modes in void elements unless mass interpolation is modified (RAMP / lower mass exponent) — Pedersen 2000"; summaries_ko §4.1 $\eta_M$, $q_M>1$; dynamics.py L47-52 Du & Olhoff 구현 | ✅ 부합 |
| R2: 역시간 수반 1패스, 비용 ≈ 전진 1회 | DYNAMICS_SURVEY §2: "adjoint solved backwards… cost ~2x forward per iteration" (Behrou & Guest); dynamics.py docstring "one backward sweep costs about one forward sweep" | ✅ 부합 |
| T5: 고정 가중은 honest-first-version으로만 허용, safeguard 필수 | DYNAMICS_SURVEY §4 item 2 "user-chosen steps/weights, reported" + item 3 "re-run true dynamic analysis… label esl_stationary_not_certified unless first-order" | ✅ 부합. ex09는 가중치를 보고하므로 §4-item-2는 만족하나, item-3 수용 판정·상태 라벨이 없어 "True/SOTA" 표기는 위반 |
| T4: 솔버 비교 방법론 결함 | DYNAMICS_SURVEY §6 "There is no head-to-head benchmark of the variants" — 변종 간 비교 자체가 부재한데 dev_doc §9는 5.0%/1.6%를 "완벽한 물리적 일치성 증명"이라 단정 | ✅ 부합 (dev_doc 과장 확정) |
| C3: 실패시 상수 반환 금지 | AGENTS.md §10-8a (silent fill 금지), §11 (solver evidence 없는 수치 주장 금지) + 조사 md들의 [read]/[search] 라벨 규율 | ✅ 부합 |

### 5.4 가장 심각한 발견: 저장소가 이미 정답을 갖고 있는데 ex09가 따로 발명했다 (Self-contradiction)

- `esl.py::run_esl(corrected=True)`는 올바른 F-ESL이다: 선형 보정항 `(grad J_dyn − grad J_esl)·(x−x_k)` (L163-181), 신뢰영역 축소·기각 로직 (L192-196), `kkt_residual` 보고 (L215), 미보정 시 `esl_stationary_not_certified` 상태 (L217). ex09는 이 중 **하나도 사용하지 않고** L705 상수로 대체했다.
- `dynamics.py::mass_interpolation`은 Du & Olhoff 절단형 (`x>0.1→x`, 이하 `1e5·x⁶`, L47-52) + exact discrete adjoint (L167-234)이다. ex09 L658 (`ρ·m0`)은 조사 결론을 정면으로 거스른다.
- 즉 본 리뷰의 R2·R3 처방은 "새 연구"가 아니라 **"옆 디렉토리의 기존 코드를 쓰라"**는 지시로 격하된다. ex09가 `aieng.topopt`를 재사용하지 않은 이유(성능? Kratos 연동? 80k DOF 확장성?)가 문서화되어 있지 않으므로, 재사용 불가 사유가 없으면 ex09 동역학 코어는 `TransientEvaluator` + `run_esl(corrected=True)` 위로 이식하고, 현 Hindi식 구현은 `legacy/`로 강등할 것.

### 5.5 dev_doc 과장 표현 시정 목록 (조사 라벨 규율 + AGENTS.md §11 위반)

| 위치 | 문제 표현 | 시정 |
|---|---|---|
| theories §7.4 L192 | "15회차 이내의 극도로 적은 반복 횟수로 완벽한 동역학적 최적 형상" | F-ESL 실증은 사이징 한정(조사 §3·§4.4). "토폴로지 미실증, KKT 잔차 미측정" 한계 병기 |
| theories §9 L232 | "5.0%·1.6%로 완벽한 물리적 일치성 증명" | 서로 다른 궤적 위 값(C4). "일치성 증명" 삭제, 고정-ρ apples-to-apples 재측정 전까지 "예비 관찰"로 격하 |
| theories §10 L238 | "100% 무결성으로 작동" | 본 리뷰 C1~C5와 양립 불가. 삭제 |
| §8.3 L213-217 | GE 챌린지 우승 형상 ↔ 본 엔진 결과의 인과 연결 암시 | 자발 형성 주장은 최적 결과물 이미지·정량 대조 없이는 가설. "가설" 라벨 (조사 md의 [proposal] 관례 준용) |

### 5.6 게이트 추가 (기존 5항에 1항 추가)

6. [ ] 인용을 arXiv:2508.07222 (2025)로 정정(C&S 확인 시까지), 최종 상태에 `esl_stationary_not_certified` 또는 실측 `kkt_residual` 표기, §5.5 과장 4건 삭제, ex09↔`aieng.topopt` 분기 사유 문서화 또는 이식.

---

## 4. [Overall Integrity Score] — **62 / 100**

| 평가축 | 배점 | 점수 | 근거 |
|---|---|---|---|
| 동역학 적분기 구현 (Gen-α/Bossak 골격) | 20 | 14 | Gen-α 계수·유효강성 정식 올바름. 단 초기 가속도 누락(C2), Kratos 무감쇠, 하중 스테핑 불일치로 −6 |
| F-ESL 이론 정합성 | 25 | 10 | ESL 합성 `f=K·u`는 정확. 그러나 민감도는 0차 + 무관계 휴리스틱(C1)이라 "F-ESL" 표기 불가. −15 |
| 하중 모델 (3D 우력·S-curve) | 15 | 11 | 우력 항등식 올바름. 기생합력/모멘트 잔차 assert 없음. −4 |
| 임계스텝·가중 최적화 정식화 | 10 | 4 | 중복 붕괴 + filler + 고정가중 의미 상실(T5). −6 |
| 모달·관성·교차검증 파이프라인 | 15 | 7 | shift 스케일 붕괴, 벡터 폐기, 실패시 날조(C3), 감쇠 불일치(T7). −8 |
| 공학 완성도 (수밀 STL·리포트·로깅) | 15 | 16→15 만점 | Marching-cubes 순서·패딩·watertight 처리, 멀티RHS PARDISO, 로깅은 우수 (가점 없음, 만점) |
| **합계** | 100 | **62** | |

**총평.** 골격(동역학 적분, ESL 합성, 병렬 인프라)은 견실하고 공학 산출물 품질은 높다. 그러나 논문의 핵심 기여로 내세운 "1차(First-order)"가 수학적으로 존재하지 않으며(C1), 모달·응력·솔버비교의 숫자 3종(C3/C5/§4)이 각기 다른 방식으로 오염되어 있다. **현 상태의 "F-ESL" 표기는 과장 표기(overclaim)이며, R2 적용 전까지는 "가중 변형에너지 ESL(0차)"로 강등 표기할 것.** R1~R7 중 R1·R3·R4·R6(상수 반환 삭제)는 금주 내 적용 가능한 원라인급이며, R2(수반)만이 정식 연구 과제다. R2 없이发表·대외 보고 시 C1·C3는 반드시 한계(limitation)로 명시해야 한다.

### 조건부 승인 게이트 (다음 마일스톤 진입 전 필수)

1. [ ] L705 휴리스틱 삭제 또는 R2/R3-차선책 교체 + `‖delta‖/‖df‖` ≠ 상수 테스트 통과
2. [ ] 모달 실패 경로 상수 반환 삭제 + `raise` + 벡터·MAC 반환 (C3/R6)
3. [ ] 초기 가속도 정합 (R1) + 최적화/evaluator 감쇠 통일 (R7)
4. [ ] 고정 ρ apples-to-apples 듀얼 솔버 비교표로 §4 수치 교체 (C4)
5. [ ] 임계스텝 병합 + 적응 가중 (R5) + 우력 assert (R4)

---

## 부록 A. 쟁점별 합격/불합격 일람

| 쟁점 | 판정 | 한 줄 요약 |
|---|---|---|
| 1. F-ESL 민감도 엄밀성 | ❌ 불합격 (Fatal) | 원전 무관계 상수 스케일. KKT 정보 0 |
| 2. 스퓨리어스 보이드 모드 | ⚠️ 조건부 불합격 | 지적 타당. `ρ² →1e−6` 저주파 국소모드 무방비, 모달 오염 가능 |
| 3. 우력 분배 무결성 | ⚠️ 조건부 합격 | 항등식은 정합. 잔차 수% + assert 없음 |
| 4. 듀얼 솔버 1.6~5.0% | ❌ 원인 규명 교체 | 감쇠/초기조건/질량행렬이 원인. PK2·AMGCL 가설 기각 |
| 5. 4스텝 + 고정가중 | ❌ 불합격 | 중복 붕괴 + filler 의미 상실 + 랭크 저하 미검사 |
| 6. Shift-invert 신뢰도 | ❌ 불합격 | σ 스케일 1e−6 붕괴 + 벡터 폐기 + 실패시 날조 |

## 부록 B. 라인 인덱스 (재현용)

- F-ESL 보정: `ex09_esl_dynamic_robot_arm.py` L690~706 (가중 L690~692, 강성 민감도 L697~702, 휴리스틱 L704~706)
- ESL 합성: L307~319 (`f_eq=K·u`), 정적 재해석 L680~687
- Gen-α 적분: L344~420 (계수 L338~353, 감쇠 L373~377, 행진 L396~415), 초기조건 L387~391
- Kratos: L441~566 (소자 L486, 스킴 L494, 하중 L541~552), 호출부 L663~668
- 임계스텝: L281~305 (filler L300~303), 정적 4케이스: `ex09_compare_dynamic_vs_static.py` L112~155
- 응력 모니터: `ex09_esl_dynamic_robot_arm.py` L739~743
- 모달: `ex09_compare_dynamic_vs_static.py` L449~485 (σ L458, eigs L471, 날조 L480~485)
- 감쇠: robot_arm L356~357 vs compare L349~356·L368~381
- 관성: compare L435~447
