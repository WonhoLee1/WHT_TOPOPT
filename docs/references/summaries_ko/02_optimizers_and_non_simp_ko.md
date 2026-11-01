# 옵티마이저와 비(非)-SIMP 위상최적화 — OC/MMA·BESO·Hilbertian 레벨셋·SEMDOT (한국어 요약 02)

> 원문 근거와 미확보 문헌 고지 (필독)
>
> - 본 요약이 직접 읽은 원문(PDF 전문 발췌)은 다음 5건이다.
>   1. `2023_wegert_hilbertian-projection-levelset_arXiv-2305.03913.pdf` — Wegert·Roberts·Challis, "A Hilbertian projection method for constrained level set-based topology optimisation", arXiv:2305.03913v4 (2023).
>   2. `2026_jezek_sequential-simp-to-levelset_arXiv-2605.04735.pdf` — Jezek 외, "Sequential topology optimization: SIMP initialization for level-set boundary refinement", arXiv:2605.04735v1 (2026).
>   3. `2025_tian_beso-adaptive-evolutionary-ratio_CJME-38-122.pdf` — Tian·Zhang, "Bi-Directional Evolutionary Topology Optimization with Adaptive Evolutionary Ratio for Nonlinear Structures", CJME 38:122 (2025, CC-BY 4.0).
>   4. `2007_huang-xie_beso-new-algorithm_JSME-49-4-1091.pdf` — Huang·Xie·Burry, "A New Algorithm for Bi-Directional Evolutionary Structural Optimization", JSME 49(4):1091 (2007).
>   5. `2020_fu_semdot-smooth-edged-material-distribution_arXiv-2005.09233.pdf` — Fu 외, "SEMDOT: Smooth-Edged Material Distribution for Optimizing Topology", arXiv:2005.09233 (2020).
> - 아래 모든 수식 번호·파라미터 기본값은 각 원문에서 가져왔다. 원문 발췌 텍스트는 `references/topopt/_new/*.pdf.txt`에 저장되어 있다.
> - **`03_Svanberg_1987_MMA` 원문(PDF)은 본 세션에서 미확보다.** 따라서 OC/MMA/GCMMA 서술은 (a) 공개 문서(`mmapy` = GCMMA-MMA-Python, `mma-dual` PyPI 패키지 문서), (b) DTU 코드 내 MMA 호출부(`mmasub`/`subsolv`/`gcmmasub`, SEMDOT 부록의 `mmasub(m,n,loop,...)` 호출), (c) 워크스페이스 래퍼 `aieng/src/aieng/topopt/optimizers/mma.py`의 교차 정리이며, **Svanberg 1987 원문 직접 인용이 아님**을 명시한다. `INDEX.md`가 지시한 두 백엔드(`mma-dual`, `mmapy`, 모두 GPLv3) 서술과 일치한다.
> - 프레임워크 정합성: `FRAMEWORK_PLAN.md` §2의 플러그인 지점 **Optimizer**(OC·MMA·GCMMA·BESO·레벨셋 드라이버) + **GradientSource** 등급(§3) + **Constraint registry**이며, 본 파일은 서베이 `METHODS_SURVEY_2026.md` §1(한 번에 한 가지만 바꾼다)·§7(구배 스파이크 ESL 신뢰도) 대응 정독 산출물이다.

## 0. 범위와 공통 표기 (Nomenclature)

다루는 옵티마이저·알고리즘 5계열: (A) OC, (B) MMA/GCMMA, (C) BESO 고전형+적응 ER 비선형 확장, (D) Hilbertian 투영 레벨셋 + 순차 SIMP→레벨셋, (E) SEMDOT.

공통 표기:

| 기호 | 의미 |
|---|---|
| `x_e (rho_e)` | 요소 설계변수, `0 <= x_e <= 1` (BESO는 `{x_min, 1}`, `x_min=0.001`) |
| `N (M, nele)` | 전체 요소 수 (원문별 표기 상이, 본문에서 통일 표기 병기) |
| `K(x), u, F` | 전체 강성행렬·변위·외력, `K(x) u = F` |
| `K_e, u_e, V_e` | 요소 강성·변위·체적 |
| `C` | 컴플라이언스 `C = F^T u = u^T K u = sum_e u_e^T K_e u_e` |
| `V(x), V*` | 현 체적·목표 체적(또는 체적분율 `Vf`) |
| `p` | SIMP 패널티 지수 |
| `r_min (R)` | 민감도/밀도 필터 반경, 가중 `w(e,l) = max(0, r_min - d(e,l))` |
| `phi` | 레벨셋 함수 (`phi<0` 고체, `phi=0` 경계, `phi>0` 공극) |
| `g` | 형상민감도(shape derivative) 체적분해완화(right-hand side) 장 |
| `v (v_Omega)` | 레벨셋 법선속도장 (Hamilton–Jacobi 이송용) |
| `l (beta)` | Hilbertian 정규화 길이척도 (원문별 기호 상이) |
| `lambda` | OC 라그랑주 승수 / 레벨셋 구속보정 비율 / AL 승수(문맥별 구분) |
| `ER` | BESO 진화율(evolutionary ratio), `RRV`는 체적제거율 |

원문 간 기호 충돌(`beta` = Heaviside 급경도(SEMDOT) vs 정규화길이(Wegert) vs 구속보정률(Jezek))은 각 절에서 원문 기호를 그대로 쓰고 대괄호로 통일 기호를 병기한다.

---

## 1. OC — Optimality Criteria (이동제한 + 람다 이분법)

### 1.1 정식화와 업데이트식 (수식 완전전개)

컴플라이언스 최소화 문제:

$$
\min_{x} C(x) = F^T u(x) \quad \mathrm{s.t.} \quad K(x)u=F, \quad V(x)=\sum_e V_e x_e \le V^*, \quad 0 < x_{\min} \le x_e \le 1
$$

$$
E_e(x_e) = E_{\min} + x_e^p\,(E_0 - E_{\min}), \qquad K_e(x_e) = E(x_e)\,K_0^e
$$

KKT 정류조건에서 요소별 비율을 정의한다:

$$
B_e = \frac{-\partial C / \partial x_e}{\lambda\,\partial V / \partial x_e} = \frac{-\partial C / \partial x_e}{\lambda\,V_e}, \qquad \frac{\partial C}{\partial x_e} = -p\,x_e^{p-1}(E_0-E_{\min})\,{u_e^T K_0^e u_e}
$$

감쇠지수(damping) `eta=0.5`와 이동제한 `move m`(기본 `m=0.2`)을 둔 OC 고정점 반복:

$$
x_e^{\mathrm{new}} = \max\Big(x_{\min},\, \max\big(x_e - m,\, \min\big(1,\, \min(x_e + m,\, x_e\,B_e^{\eta})\big)\big)\Big)
$$

$$
\eta = 0.5,\quad m = 0.2\ (\text{기본}),\quad V(x^{\mathrm{new}}(\lambda)) = V^*
$$

`lambda`는 체적등식 `V(x(lambda)) = V*`를 만족하도록 **이분법(bisection)** 으로 찾는다: `lambda`가 크면 `B_e`가 작아져 체적이 줄고, 작으면 체적이 는다는 단조성을 이용한다. 수동(passive) 요소(하중·지지·비설계 영역)는 업데이트에서 제외하고 고정한다 (`x_e = 0` 또는 `1` 고정). Jezek 순차 파이프라인의 SIMP 단계가 정확히 이 방식이다: 균일초기장 `rho_e = Vf = 0.4`, `p = 3`, 민감도필터 반경 `R = 2h`, 수렴판정 `max_e |x_e^{k+1}-x_e^{k}| < Delta_rho_max` (0.5%~8% 구간 실험).

### 1.2 민감도 유도와 FD 검증

자기수반(self-adjoint)이므로 별도 수반해 없이 직접 미분:

$$
\frac{\partial J}{\partial \rho_e} = -p\,\rho_e^{p-1}(E_0 - E_{\min})\,u_e^T k_0^e u_e
$$

일반 메시용 민감도 필터(Jezek Eq.(4)):

$$
\frac{\widetilde{\partial J}}{\partial \rho_e} = \frac{1}{\rho_e V_e \sum_{i \in \mathcal{N}_e} H(x_i)} \sum_{i \in \mathcal{N}_e} \rho_i V_i\,H(x_i)\,\frac{\partial J}{\partial \rho_i}, \quad H(x_i) = \max(0, R - \|x_e - x_i\|)
$$

FD 검증(필수): 임의의 `e`에 대해 중심차분과 대조한다.

$$
\frac{\partial C}{\partial x_e} \approx \frac{C(x + h\,e_e) - C(x - h\,e_e)}{2h}, \quad h = 10^{-6}\sim 10^{-8}
$$

상대오차 `< 10^{-4}` (선형탄성, 필터 끈 상태 원시감도 기준)을 요구한다. `FRAMEWORK_PLAN.md` §3의 `GradientSource` 등급상 OC+완전수반은 `exact_adjoint`이며 FD 검사가 CI에서 통과해야 한다. 비선형·ESL 동결하중은 `surrogate`로 강등하고 참응답 개선 시에만 수용한다(§9).

### 1.3 의사코드와 하이퍼파라미터

```text
# OC-SIMP 한 스텝 (Jezek §2.1 + 표준 OC)
입력: x, u(Ku=F 해), p=3, m=0.2, eta=0.5, xmin, V*, passive 집합
1. dC[e] = -p*x[e]^(p-1)*(E0-Emin)*u_e^T K0 u_e   # 전 요소
2. dC_tilde = sensitivity_filter(dC, R)            # Eq.(4)
3. lambda = bisect(lambda -> V(x_new(lambda)) - V* = 0)
     x_new[e] = clamp(x[e]*B_e^eta, x[e]-m, x[e]+m, xmin, 1)
4. passive 요소는 고정값으로 덮어쓰기
5. max|x_new - x| < tol 이면 종료
```

| 파라미터 | 기본값 | 비고 |
|---|---|---|
| `p` | 3 | SIMP 패널티 |
| `move m` | 0.2 | 1스텝 최대 변화량 |
| `eta` | 0.5 | 감쇠 |
| `R` | `2h` | 민감도필터 반경 |
| `Delta_rho_max` | 0.5%~8% | Jezek SIMP 종료판정 (느슨할수록 후단 LS가 메움, §6) |

### 1.4 wht_topopt 매핑과 pitfalls

- 매핑: `aieng/src/aieng/topopt/optimizers/oc.py`의 `oc_update` + `evaluator.py`(수반감도) + `filters.py`(민감도필터). Jezek 파이프라인 전체는 `problem.py`+`driver.py`의 SIMP→SDF→LS 순차 구동에 대응한다.
- pitfalls: (i) 이동제한을 끄면 체터링·발산한다. (ii) 필터 켠 감도로 FD 검증을 하면 안 된다 — 원시감도로 검증 후 필터를 켠다. (iii) 수동요소를 감도필터 분모에서 빼면 안 된다(0/0 또는 누설). (iv) `lambda` 이분법 구간을 체적단조가 깨지는 극단값에서 시작하면 실패한다 — 구간을 넓게 잡고 단조를 먼저 확인한다.

---

## 2. MMA / GCMMA (Svanberg) — 이동점근선·볼록분리가능근사·쌍대·내점법

> **출처 한계 명시:** Svanberg 1987 원문 미확보. 아래는 `mmapy`(GCMMA-MMA-Python, GPLv3)·`mma-dual`(GPLv3) 공개 문서, SEMDOT 부록의 `mmasub` 호출부, DTU `top444/top88`류 코드의 MMA 호출 관례, 워크스페이스 래퍼(`optimizers/mma.py`: `MMA(n,m,...)` 상태형 `update`, 백엔드 `auto→mmadual→mmapy`, 없으면 `MMABackendUnavailable`)의 교차 정리이다.

### 2.1 볼록 분리가능 근사 (수식 완전전개)

반복점 `x^{(k)}`에서 목적·제약 `f_i(x)` (`i=0` 목적, `i=1..m` 제약 `f_i <= 0`)를 변수분리형 볼록근사로 둔다:

$$
\tilde{f}_i(x) = \sum_{e=1}^{n} \left( \frac{p_{ie}^{(k)}}{U_e^{(k)} - x_e} + \frac{q_{ie}^{(k)}}{x_e - L_e^{(k)}} \right) + r_i^{(k)}
$$

$$
p_{ie}^{(k)} = \begin{cases} (U_e^{(k)}-x_e^{(k)})^2\,\partial f_i/\partial x_e, & \partial f_i/\partial x_e > 0 \\ 0, & \text{그 외} \end{cases}, \qquad q_{ie}^{(k)} = \begin{cases} 0, & \partial f_i/\partial x_e > 0 \\ -(x_e^{(k)}-L_e^{(k)})^2\,\partial f_i/\partial x_e, & \text{그 외} \end{cases}
$$

`r_i^{(k)}`는 `tilde{f}_i(x^{(k)}) = f_i(x^{(k)})`, 기울기 일치 조건으로 정해지는 상수항이다. 이동점근선 `L_e^{(k)} < x_e^{(k)} < U_e^{(k)}`가 곡률을 조절한다: 점근선이 가까우면 보수적(강한 볼록), 멀면 공격적이다.

근사 부분문제 (박스 `alpha_e <= x_e <= beta_e`, `alpha/beta`는 `xmin/xmax`와 `x^{(k)} ± move`의 교집합):

$$
\min_x \tilde{f}_0(x) + a_0 z + \sum_i (c_i y_i + \tfrac{1}{2} d_i y_i^2) \quad \mathrm{s.t.} \quad \tilde{f}_i(x) - a_i z - y_i \le 0,\ z,y_i \ge 0
$$

SEMDOT 부록 호출의 `a0=1, ai=0, ci=1000, di=0`이 바로 이 인공변수 페널티이다.

### 2.2 이동점근선 업데이트 (수식 완전전개)

표준 Svanberg 규칙(공개 구현 `asymp`/`raaupdate` 기준 정리):

$$
L_e^{(k)} = x_e^{(k)} - s^{(k)}(x_{\max} - x_{\min}), \qquad U_e^{(k)} = x_e^{(k)} + s^{(k)}(x_{\max} - x_{\min})
$$

진동 감지(부호가 바뀌면 점근선을 당기고, 단조면 밀어낸다):

$$
s^{(k+1)} = \begin{cases} s^{(k)} / s_{\mathrm{dec}}, & (x_e^{(k)}-x_e^{(k-1)})(x_e^{(k-1)}-x_e^{(k-2)}) < 0 \ (\text{진동}) \\ s^{(k)} \cdot s_{\mathrm{inc}}, & \text{단조} \end{cases}
$$

SEMDOT·DTU 관례 기본값: `s_init = 0.5`(초기 간격계수), `s_dec = 0.7`, `s_inc = 1.2` (워크스페이스 `my_plan.md` 표기 `s_init/s_dec/s_inc`와 동일). `low/upp`가 바로 `L/U`이며 `mmasub` 입출력으로 매 반복 갱신된다(`vxold1/vxold2`가 진동판정용 2스텝履歴).

### 2.3 쌍대문제와 primal-dual 내점법 (수식 완전전개)

근사 부분문제의 라그랑주 쌍대함수:

$$
W(\lambda) = \min_{x \in [\alpha,\beta],\,z,y \ge 0} \left\{ \tilde{f}_0(x) + a_0 z + \sum_i \big(c_i y_i + \tfrac{1}{2}d_i y_i^2\big) + \sum_i \lambda_i\big(\tilde{f}_i(x) - a_i z - y_i\big) \right\}
$$

$$
\max_{\lambda \ge 0} W(\lambda)
$$

`x_e(lambda)`는 분리형이라 스칼라 최소화의 닫힌형(클램프된 유리식)으로 구해지고, 쌍대문제는 저차원(`m << n`)이라 뉴턴법으로 푼다. `mmapy` 원형은 Svanberg MATLAB `subsolv`의 **primal-dual interior-point(주-쌍대 내점법)** 을 그대로 포팅한 것이고, `mma-dual`은 쌍대 뉴턴 직접해로 같은 해를 더 싸게 구한다(제약이 적을 때). GCMMA(Svanberg 2002)는 근사가 비보수적이면 내부 반복으로 점근선을 조이며(`raaupdate`/`concheck`) 단조 하강을 강제하는 **전역수렴 변형**이다 —py측 `gcmmasub` 서명이 이를 노출한다.

### 2.4 민감도/FD 검증

MMA 자체는 1차 옵티마이저이므로 감도는 외부(ResponseModel)가 준다. 검증 의무는 동일하다: `exact_adjoint` 등급이면 FD 상대오차 `< 10^{-4}`를 CI에서 확인하고, ESL·동결하중·신경망 대리면 `surrogate/learned`로 표기하고 참응답 개선시에만 수용한다. GCMMA 내부반복은 함수값만 추가로 쓰므로(워크스페이스 `evaluator.py` 주석) 구배 카운트와 함수 카운트를 분리 기록한다.

### 2.5 의사코드와 하이퍼파라미터

```text
# MMA 한 스텝 (mmasub 관점, SEMDOT 부록 호출 대응)
입력: x(k), xmin/xmax, xold1/xold2, L/U(low/upp), f0/df0, fval/dfdx, a0/ai/ci/di
1. [L,U] = asymp_update(x, xold1, xold2, L, U, s_init/s_dec/s_inc)  # §2.2
2. [p0,q0,pi,qi] = separable_approx(df0, dfdx, L, U, x)             # §2.1
3. lambda* = solve_dual(W)   # mmapy: primal-dual 내점법(subsolv), mma-dual: 쌍대 뉴턴
4. x(k+1) = x_boxclip(lambda*)   # 유리식 닫힌형 + move/box 클램프
5. GCMMA라면: 보수성 검사(concheck) 실패 시 점근선 조이기(raaupdate) 후 3으로
```

| 파라미터 | 기본값 | 비고 |
|---|---|---|
| `move` | 0.2 | 박스 `alpha/beta` 폭 제한 |
| `s_init/s_dec/s_inc` | 0.5/0.7/1.2 | 점근선 간격·진동시 축소·단조시 확대 |
| `a0/ai/ci/di` | 1/0/1000/0 | SEMDOT 부록값, 인공변수 페널티 |
| `m < n` | 필수(GCMMA) | 워크스페이스 `GCMMA` 생성자에서 강제 |

### 2.6 wht_topopt 매핑과 pitfalls

- 매핑: `optimizers/mma.py` `MMA(n,m,xmin,xmax,move,c)` + `GCMMA`(내부 `gcmmasub`) + `evaluator.py`(감도 공급) + `constraints.py`(부피·변위 제약을 `fval/dfdx`로). SEMDOT 활성화 코드는 `semdot.py`가 이 인터페이스를 호출한다.
- pitfalls: (i) 백엔드 미설치 시 조용히 다른 옵티마이저로 대체 금지 — `MMABackendUnavailable`을 그대로 올린다(래퍼 명세). (ii) `m < n` 위반 시 GCMMA 생성 거부. (iii) 쌍대-원시 혼동: `W(lambda)` 최대화가 본문제 최소화가 아니다 — 부호를 뒤집으면 발산한다. (iv) 점근선 `low/upp`를 매 반복 외부에서 덮어쓰면 진동판정履歴이 깨진다 — `mmasub` 반환값을 그대로 다음 입력에 쓴다. (v) GCMMA 내부반복의 추가 함수평가는 `n_solves`와 분리 집계한다.

---

## 3. BESO 고전형 — Huang·Xie 2007 (단일 제거율·성능지수)

### 3.1 민감도수와 스무딩 (수식 완전전개)

선형탄성 컴플라이언스 문제(원문 Eq.(1)):

$$
\min C = \tfrac{1}{2}F^T u \quad \mathrm{s.t.} \quad g = V^* - \sum_i V_i x_i = 0,\ x_i \in \{0,1\}
$$

요소 제거 시 컴플라이언스 변화(원문 Eq.(2))와 민감도수(원문 Eq.(3)):

$$
\Delta C_i = \tfrac{1}{2}{u_i^T[K_i]u_i}, \qquad \alpha_i = \frac{\tfrac{1}{2}{u_i^T[K_i]u_i}}{V_i}
$$

절점 스무딩(원문 Eq.(4)) — 절점에 연결된 `m`개 요소의 체적가중 평균:

$$
\alpha_k = \frac{\sum_{i=1}^{m} V_i\,\alpha_i}{\sum_{i=1}^{m} V_i}
$$

구조 바깥 후보요소를 위한 선형외삽(원문 Eq.(5), Fig.1):

$$
\alpha_k^{x-} = 2\alpha_{k1} - \alpha_{k2}
$$

가능한 모든 ±x·±y(±z) 방향으로 반복 후 평균한다. 후보요소 평활민감도수(원문 Eq.(6)):

$$
\alpha_e^* = \frac{1}{N}\sum_{n=1}^{N} \alpha_n
$$

`N`은 후보요소의 절점 수이다.

### 3.2 단일 파라미터 제거·추가와 임계값 이분법 (수식 완전전개)

모든 요소(고체+공극)를 함께 순위매김한다(원문 Eq.(7)):

$$
\text{제거: } \alpha_e^* \le \alpha_{\mathrm{th}}, \qquad \text{추가: } \alpha_e^* > \alpha_{\mathrm{th}}
$$

`alpha_th`는 현 제거율 `RRV_i`가 지시하는 체적에 맞추는 **순위 절단값**이다: 예컨대 설계영역 1000요소 중 `RRV_i`가 800요소 유지를 지시하면 정렬된 `alpha_800`이 `alpha_th`다. 즉 `alpha_th`를 직접 정하는 게 아니라 목표체적→정렬→절단값의 **이분탐색(binary search)** 으로 역산한다. 같은 `RRV_i`로 FEA→제거/추가 사이클을 수렴까지 반복한 뒤 진화율 `ER`만큼 올린다(원문 Eq.(8)):

$$
RRV_{i+1} = RRV_i + ER, \quad i = 0,1,2,\dots
$$

초기 `RRV_0 = 0.8`(최소설계에서 시작, 원문 Fig.8·13 예제), `ER = 0.0`(형상조정 단계) 관례.

### 3.3 성능지수 수렴판정 (수식 완전전개)

성능지수(원문 Eq.(9))와 수렴오차(원문 Eq.(10)):

$$
PI = \frac{1}{C\,V}, \qquad \mathrm{error}_i = \frac{|PI_i - PI_{i-1}|}{PI_i} \le \mathrm{error}
$$

허용오차 예 `error = 0.001`(엄격) 또는 `0.001~0.0001`(원문 예제). 진동상태(같은 요소군이 제거↔추가 반복)도 최적도달로 인정하고 다음 `RRV`로 넘어간다(Step 11→12). 최종 체적 `V*`만 필요하면 중간 `RRV`의 수렴판정을 생략하고 시간 절약이 가능하다.

### 3.4 의사코드와 하이퍼파라미터

```text
# Huang-Xie BESO (원문 §4 Steps 1-12)
1. 설계영역 조밀메시, 하중·경계·재료 정의
2. FEA 수행 → u
3. 요소민감도 Eq.(3) → 절점평균 Eq.(4) → 외삽 Eq.(5) → 후보평활 Eq.(6)
4. 목표체적(RRV_i) → 정렬 → alpha_th 이분탐색 → Eq.(7a/7b) 제거/추가
5. 같은 RRV로 PI Eq.(9) 수렴 Eq.(10) 또는 진동까지 2-4 반복
6. RRV += ER (Eq.8) 후 2-5 반복, V* 도달·수렴 시 종료
```

| 파라미터 | 기본값/예제값 | 비고 |
|---|---|---|
| `RRV_0` | 0.8 | 최소설계 시작 |
| `ER` | 0.0~0.02 | 형상조정시는 0, 체적감축시는 소량 |
| `error` | 0.001 (엄격 0.0001) | PI 상대변화 |
| 체커보드 | 절점평균 Eq.(4) | 필터와 별개 고전 스무딩 |

### 3.5 FD 검증·매핑·pitfalls

- FD 검증: `alpha_i`는 변형에너지 밀도이므로 요소강성 `K_i` 제거차분 `Delta C ≈ 0.5 u_i^T K_i u_i`와 직접 대조한다. 공극요소는 `x_min` 강성 기준이므로 FD 스텝이 `x_min`을 건드리지 않게 주의한다.
- 매핑: `aieng/src/aieng/topopt/beso.py`(민감도·순위·제거/추가 드라이버) + `filters.py`(BESO식 절점 스무딩) + `evaluator.py`(선형정적 `K u = F`).
- pitfalls: (i) 제거·추가를 분리순위하면 논리모순 — 반드시 전요소 통합순위를 쓴다(원문 강조). (ii) `error`를 너무 작게 잡으면 진동무한루프 — 진동인정 탈출을 둔다. (iii) `ER` 과대 시 수렴불안·과소 시 저속 — §4 적응ER로 해소한다.

---

## 4. BESO 적응 진화율(Adaptive ER) 비선형 확장 — Tian·Zhang 2025

### 4.1 비선형 목적·수반민감도 (수식 완전전개)

외력이 한 일(음영면적, 사다리꼴 수치적분) 최소화(원문 Eq.(6)):

$$
\min_x f(x) = W = \lim_{n\to\infty} \frac{1}{2}\sum_{i=1}^{n} (U_i^T - U_{i-1}^T)(F_i + F_{i-1})
$$

$$
\mathrm{s.t.}\ R = F - F_{\mathrm{int}} = 0,\quad V(x) = \sum_e V_e x_e = V^*,\quad x_e \in \{x_{\min}, 1\}
$$

`F_int = sum_e x_e^p C_e^T F_{e,0}^{int}` (원문 Eq.(13), `F_{e,0}^{int}`는 고체요소 내력). 수반벡터 `lambda_i = -(U_i - U_{i-1})`(원문 Eq.(12))를 두면 설계민감도는(원문 Eq.(13)→(15)→(16)):

$$
\frac{\partial f}{\partial x_e} = \lim_{n\to\infty}\frac{1}{2}p\,x_e^{p-1}\sum_{i=1}^{n}(U_i^T-U_{i-1}^T)C_e^T\big(F_{e,i}^{\mathrm{int}}+F_{e,i-1}^{\mathrm{int}}\big)
$$

이를 단위체적 최종 탄·소성 변형에너지 `E_n^e`(Abaqus `SENER`+`PENER`)로 축약한 요소민감도수(원문 Eq.(16)→(17)):

$$
\alpha_e = \frac{1}{p}\frac{\partial f}{\partial x_e} = x_e^{p-1} E_n^e, \qquad \alpha_e = \begin{cases} E_n^e, & x_e = 1 \\ x_{\min}^{p-1}\,E_n^e, & x_e = x_{\min} \end{cases}
$$

기하비선형은 전체라그랑지안 정식화(TL, 원문 §2.1.2) + Newton–Raphson 증분반복 `K_t = -dR/dU`, `R = F - F_int = 0`(원문 Eq.(1)→(5))으로 푼다. 재료비선형은 다중선형 등방경화 곡선(Fig.1), `SENER`(탄성)·`PENER`(소성) 필드출력 합산(Fig.2)이다.

### 4.2 필터·이력평균 (수식 완전전개)

요소민감도 필터(원문 Eq.(18)):

$$
\hat{\alpha}_e = \frac{\sum_j w(r_{ej})\,\alpha_j}{\sum_j w(r_{ej})}, \qquad w(r_{ej}) = \max(0,\,r_{\min} - r_{ej})
$$

이력평균(원문 Eq.(19)):

$$
\tilde{\alpha}_e^{(k)} = \frac{\hat{\alpha}_e^{(k)} + \hat{\alpha}_e^{(k-1)}}{2}
$$

### 4.3 적응 ER 4함수와 체적업데이트·종료판정 (수식 완전전개)

체적업데이트(원문 Eq.(21))와 종료판정(원문 Eq.(22), `tau = 0.001`, `N = 5` → 과거 10스텝 평균변화):

$$
V_{k+1} = V_k\,(1 \pm ER), \qquad \mathrm{change} = \frac{\sum_{i=1}^{N}|W_{k-i+1} - W_{k-N-i+1}|}{\sum_{i=1}^{N} W_{k-i+1}} \le \tau
$$

`+`는 최소설계 시작 시 증량, `−`는 충만 시작 시 감량이다. 적응 ER 4함수(원문 Eq.(23), `ER_max = 0.05`, `ER_min = 0.02`, `k_1 = 25`, `k_2 = 30`, `k_3 = 6`, `v_h`는 현 체적분율):

$$
ER_1 = \frac{ER_{\max}-ER_{\min}}{2}\tanh\!\big(k_1(v_h-0.5)\big) + \frac{ER_{\max}+ER_{\min}}{2}
$$

$$
ER_2 = (ER_{\max}-ER_{\min})\frac{1}{1+\exp(-k_2(v_h-0.5))} + ER_{\min}
$$

$$
ER_3 = (ER_{\max}-ER_{\min})\frac{\log\!\big(1+\exp(2k_3(v_h-0.5))\big)}{\log(1+\exp(k_3))} + ER_{\min}
$$

$$
ER_4 = ER_{\min} + (ER_{\max}-ER_{\min})\,v_h
$$

초기(고체과잉, `v_h` 큼)에는 큰 `ER`로 저민감도 요소를 빨리 걷어내고, 후기(`v_h` 작음)에는 작은 `ER`로 안정수렴한다. 양단클램프보(80×10×4mm, C3D8R 1mm, 3200요소, `F=300N`, `Vf=0.3`, `r=2`)에서 고정ER 75회 대비 `ER_1` 47회(**−37.3%**), `ER_2` 55회(−26.7%), `ER_3` 66회(−12%), `ER_4` 61회(−18.7%)이다. 최종 위상은 4함수 간 거의 동일하며 SIMP·MMC 결과와 컴플라이언스 오차가 극소하다. 비선형 3D 2케이스에서 하중 증가 시 재료·기하·결합 비선형이 서로 다른 최적위상을 만들며(500N 기하 V자화, 750N 재료 단부반전), 저하중(300N)에서는 선형과 유사하다.

### 4.4 의사코드와 하이퍼파라미터

```text
# Tian 적응ER BESO (Abaqus FEA + Python 옵티마이저, 원문 §3)
1. Abaqus CAE: 설계영역·재료(탄소성)·하중·경계·메시, 비선형스텝(NLGEOM)
2. V*, ER함수(ER_1~ER_4 중 택1), rmin, M(최대반복) 설정
3. loop:
   a. Newton-Raphson 비선형 FEA (K_t 매 증분 갱신)
   b. SENER+PENER → E_n^e → Eq.(17) alpha_e
   c. Eq.(18) 필터 → Eq.(19) 이력평균
   d. 목표체적 V_{k+1}=V_k(1±ER) → alpha_th 이분탐색 → x 업데이트
   e. 현 v_h → Eq.(23) ER 갱신
   f. V* 만족 + Eq.(22) change<=0.001 (N=5) 시 종료
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| `ER_max/ER_min` | 0.05/0.02 | 적응구간 |
| `k_1/k_2/k_3` | 25/30/6 | 곡선 급경도 |
| `tau/N` | 0.001/5 | 종료판정 |
| `x_min` | 0.001 | 공극 강성 하한 |

### 4.5 FD 검증·매핑·pitfalls

- FD 검증: 선형영역(`F` 미소)에서 Eq.(17)이 §3 고전 `alpha_i`로 환원됨을 먼저 확인한 뒤, 비선형에서는 하중증분을 조밀히 한 수반감도와 조악증분 수반감도를 비교한다. `SENER/PENER` 단위체적·현배치 기준을 혼동하면 체적가중이 이중적용된다.
- 매핑: `beso.py`(순위·이분탐색·제거/추가) + `evaluator.py`(비선형은 외부 Abaqus 어댑터 — `adapters/`에 추가 시) + `constraints.py`(체적) + `FRAMEWORK_PLAN` LoadProvider(비선형 하중이력).
- pitfalls: (i) 저강성요소 과대변형 → 접선강성 부정부호·발산: 에너지보간·요소제거·자유도제거 중 택1이 필수다(원문 §1). (ii) `ER_max` 과대 시 후기진동 — `ER_1`이 가장 빠르나 곡선이 가팔라 `k_1`을 함부로 키우지 않는다. (iii) 결합비선형 결과를 선형직관으로 해석 금지 — 하중별 위상이 다르다(Table 2).

---

## 5. Hilbertian 투영 레벨셋 — Wegert 2023 + Jezek 2026 순차 파이프라인

### 5.1 레벨셋·형상미분·Hilbertian 연장-정규화 (수식 완전전개)

레벨셋 정의(양 논문 공통, Wegert Eq.(1)-(2), Jezek Eq.(5)-(6)):

$$
\phi(x) < 0\ (x \in \Omega),\quad \phi(x) = 0\ (x \in \partial\Omega),\quad \phi(x) > 0\ (x \in D\setminus\bar{\Omega})
$$

Hamilton–Jacobi 이송과 CFL 시간스텝(Wegert Eq.(3), Jezek Eq.(21)-(22), Wegert Eq.(38)):

$$
\frac{\partial\phi}{\partial t} + v\,|\nabla\phi| = 0, \qquad \Delta t = \frac{\gamma\,\Delta x}{\|v_\Omega\|_\infty} \le \frac{h}{\max|\bar{v}|}
$$

$$
\gamma \in (0,1),\quad \gamma_{\min}=0.001,\ \gamma_{\max}=0.1,\ \delta_{\mathrm{inc}}=1.1,\ \delta_{\mathrm{dec}}=0.7
$$

재초기화(부호거리 복원, Wegert Eq.(5), Jezek Eq.(23)-(24)):

$$
\frac{\partial\phi}{\partial t} + S(\phi_0)\,(|\nabla\phi|-1) = 0, \qquad S(\phi) = \frac{\phi}{\sqrt{\phi^2 + |\nabla\phi|^2\,\Delta x^2}}
$$

$$
\|\phi_q - \phi_{q-1}\|_\infty < 5\times10^{-5}\ \text{까지 반복}
$$

형상미분 정의(Wegert Def.1·Lemma 1, Eq.(6)-(8))와 ersatz 재료(Wegert Eq.(18)-(20), Jezek Eq.(7)-(8)):

$$
J(\Omega_\theta) = J(\Omega) + J'(\Omega)(\theta) + o(\theta), \qquad J'(\Omega)(\theta) = \int_{\partial\Omega} f\,\theta\!\cdot\! n\,d\Gamma
$$

$$
C_{ijkl}(x) = C_{ijkl}\,(x\in\Omega) + \varepsilon_{\mathrm{void}}C_{ijkl}\,(x\in D\setminus\Omega),\ \varepsilon_{\mathrm{void}}=0.001
$$

$$
H_\eta(\phi) = \begin{cases} 0 & \phi<-\eta \\ \tfrac{1}{2}+\tfrac{\phi}{2\eta}+\tfrac{1}{2\pi}\sin(\pi\phi/\eta) & |\phi|\le\eta \\ 1 & \phi>\eta \end{cases},\quad \eta = 1.5\Delta x
$$

균질화 강성·체적과 형상미분(Wegert Eq.(17)(21)(24)-(27)):

$$
\bar{C}_{ijkl}(\Omega) = \int_\Omega C_{pqrs}(\varepsilon_{pq}(\tilde{u}_{(ij)})+\bar{\varepsilon}^{(ij)}_{pq})\,\bar{\varepsilon}^{(kl)}_{rs}\,d\Omega
$$

$$
\bar{C}'_{ijkl}(\Omega)(\theta) = \int_{\partial\Omega} C_{pqrs}(\varepsilon_{pq}(\tilde{u}_{(ij)})+\bar{\varepsilon}^{(ij)}_{pq})(\varepsilon_{rs}(\tilde{u}_{(kl)})+\bar{\varepsilon}^{(kl)}_{rs})\,\theta\!\cdot\! n\,d\Gamma
$$

컴플라이언스 형상미분(Jezek Eq.(14)):

$$
J'(\Omega)(v_n) = -\int_{\partial\Omega} \sigma(u)\!:\!\varepsilon(u)\,v_n\,ds
$$

체적분해완화(Jezek Eq.(15)-(18), `theta = -w\,n` 매개화):

$$
J'(\Omega)(-wn) = \int_D \sigma\!:\!\varepsilon\ w\,H'_\eta(\phi)\,|\nabla\phi|\,dx
$$

$$
C'(\Omega)(-wn) = -\int_D \frac{1}{V_D}\,w\,H'_\eta(\phi)\,|\nabla\phi|\,dx
$$

$$
g_{\mathcal{L}}(x) = \sigma(u)\!:\!\varepsilon(u) + \frac{\lambda - \Lambda\,C(\phi)}{V_D}
$$

Hilbertian 연장-정규화(Wegert Eq.(9)-(10), Jezek Eq.(19)-(20)) — 요구된 연산자 형태 `-\beta^2\nabla^2 v + v = -g`의 약형식이다:

$$
\langle g_\Omega, w \rangle_H = -J'(\Omega)(wn)\ \forall w \in H, \qquad \langle u,v \rangle_H = \beta^2\!\!\int_D \nabla u\!\cdot\!\nabla v\,d\Omega + \int_D uv\,d\Omega
$$

$$
\int_D \big(\ell^2\,\nabla\bar{v}\!\cdot\!\nabla w + \bar{v}w\big)\,dx = \mathcal{L}'(\Omega)(-wn)\ \forall w \in H^1(D),\quad A_H\,\bar{\mathbf{v}} = \mathbf{g}_{\mathcal{L}}
$$

기본값 `beta = 4Δx`(Wegert Table 1), `ell`은 요소크기·스텝 비례(Jezek). 하중작용 트랙션 경계 `Gamma_N`에는 동차 Dirichlet `bar{v} = 0`을 건다. `A_H`는 메시·`ell`에만 의존하므로 1회 인수분해 후 목적·제약 우변에 재사용한다.

### 5.2 Hilbertian 투영 — Gram–Schmidt + 속도결합 (수식 완전전개)

목적·제약 각각 연장문제(Wegert Eq.(29)-(30)):

$$
\langle g_\Omega, v \rangle_H = -J'(\Omega)(vn),\qquad \langle \mu_{\Omega i}, v \rangle_H = -C'_i(\Omega)(vn),\quad i=1..N
$$

제약감도 집합 `{mu}`에 Gram–Schmidt 직교화를 걸어 종속을 제거한 정규직교기저 `{bar{mu}_p}`, `p=1..bar{N} ≤ N`을 얻는다. 직교투영자(Wegert Eq.(31)):

$$
P_{C^\perp}\,g_\Omega = g_\Omega - \sum_{p=1}^{\bar{N}} \frac{\langle \bar{\mu}_{\Omega p}, g_\Omega \rangle_H}{\|\bar{\mu}_{\Omega p}\|_H^2}\,\bar{\mu}_{\Omega p}
$$

결합속도(Wegert Eq.(34), 단일체적구속 시 Jezek Eq.(13)과 동형):

$$
v_\Omega = \sqrt{1-\sum_{p=1}^{\bar{N}}\alpha_p^2}\,\frac{P_{C^\perp}g_\Omega}{\|P_{C^\perp}g_\Omega\|_H} + \sum_{p=1}^{\bar{N}} \alpha_p\,\frac{\bar{\mu}_{\Omega p}}{\|\bar{\mu}_{\Omega p}\|_H}
$$

$$
v_\Omega = \sqrt{1-\alpha^2}\,\frac{P_{\mathcal{C}^\perp}g_\Omega}{\|P_{\mathcal{C}^\perp}g_\Omega\|_H} + \alpha\,\frac{\mu_\Omega}{\|\mu_\Omega\|_H}\quad (\text{단일구속})
$$

계수 `alpha_p`는 구속 1차개선 요구(Wegert Eq.(32)-(33))에서 하삼각 연립(Wegert Eq.(35))으로 푼다:

$$
J'(\Omega)(v_\Omega n) = \min,\quad C'_i(\Omega)(v_\Omega n) = -\lambda\,C_i \iff \langle g_\Omega,v_\Omega\rangle_H = \max,\ \langle \mu_{\Omega i},v_\Omega\rangle_H = \lambda\,C_i
$$

$$
\lambda\,C_p = \sum_{l=1}^{p-1}\alpha_l\,\frac{\langle \bar{\mu}_{\Omega l},\mu_{\Omega p}\rangle_H}{\|\bar{\mu}_{\Omega l}\|_H} + \alpha_p\,\|\bar{\mu}_{\Omega p}\|_H
$$

전진대입으로 푼다. 파라미터는 `alpha_min^2 = 0.1`(기본), `1 - sum alpha_p^2 ≥ 0`, `sum alpha_p^2 ≥ alpha_min^2`를 만족하도록 `lambda`를 정한다(Wegert §3.3). `alpha_min = 1`이면 구속만족 문제로 귀착한다. 선형종속 구속민감도(등방성 6개 중 nullity 2 등)도 특수처리 없이 통과한다 — 역행렬이 없고 LICQ를 요구하지 않는 것이 Feppon/Barbarosie null-space법 대비 장점이다(Wegert §3.4). 등식구속만 지원하며 부등식은 slack 변수 확장이 후속과제이다.

제약이 하나뿐인 Jezek형에서는 `alpha` 하한이 준가능 초기장에서 진동을 유발하므로 적응하한(Jezek Eq.(31))을 둔다:

$$
\alpha_{\min}^{\mathrm{eff}} = \alpha_{\min}\,\min\!\left(1,\,\frac{|C|}{\tau}\right),\quad \tau = 0.01,\ \alpha_{\min}^2 = 0.1
$$

### 5.3 Godunov upwind HJ + 재초기화 + SLERP (수식 완전전개)

HJ·재초기화는 1차 Godunov upwind 유한차분(Peng·Osher–Fedkiw, Wegert §4.1, Jezek §2.2.7-8)으로 푼다. HJ는 고정 스텝수(`⌊n/10⌋` 2D, `⌊n/3⌋` 3D) 전진, 재초기화는 정상상태까지 전진한다. CFL 위반·비단조 `L` 시 `gamma`를 25% 삭감한다.

다중 레벨셋(color-LS, Wegert Eq.(22)-(23), 최대 4상/2함수)이나 방향장 보간이 필요할 때 단위벡터 보간은 **SLERP**로 구면 위를 등각속으로 잇는다(구현측 결합식, 원문 미기재 — 수치관례로 포함):

$$
\mathrm{slerp}(q_0,q_1;t) = \frac{\sin((1-t)\Theta)}{\sin\Theta}\,q_0 + \frac{\sin(t\Theta)}{\sin\Theta}\,q_1,\quad \cos\Theta = q_0\!\cdot\! q_1,\ t \in [0,1]
$$

`Theta → 0`이면 선형보간으로 폴백한다. 색상-레벨셋 상 보간 `C_ijkl(d_D1,d_D2)`(Wegert Eq.(23))의 계수 가중치를 이 형태로 주면 상경계에서 법선 뒤집힘을 막는다.

### 5.4 순차 SIMP→레벨셋 기하전달 (Jezek §3.2-3.4, 수식 완전전개)

1. 절점 재구성(국소 선형 최소제곱, Jezek Eq.(25)-(27)):

$$
\rho(x) = [1\ x^T]\,a,\quad X^TX\,a = X^T\rho,\quad \rho_n = [1\ x_n^T]\,a
$$

`m ≥ 4`·비공면이면 유일해, `1 < m < 4`면 거리역가중 폴백 `w_i ∝ 1/‖x_n - x_i‖`, `m = 1`이면 그대로 할당한다.
2. 등면추출: `rho_t = 0.5` Marching Cubes/Tetrahedra (ParaView `pvpython`/VTK).
3. SDF 부호배정(Jezek Eq.(28)-(30)):

$$
d(x_n) = \min_{k=1..n_T}\|x_n - x(\lambda_k^*)\|,\quad \phi(x_n) = \begin{cases} -d(x_n) & \rho_n \ge \rho_t \\ +d(x_n) & \rho_n < \rho_t \end{cases}
$$

`k`d-트리+ AABB 선별 시공, 노드당 `O(log n_T + n_c)`이다.
4. LS 정제: GridapTopOpt.jl, 체적등식 + Hilbertian 투영(§5.2) 또는 augmented Lagrangian (Jezek Eq.(11)-(12)):

$$
\mathcal{L}(\phi,\lambda,\Lambda) = J(\phi) - \lambda\,C(\phi) + \tfrac{\Lambda}{2}C(\phi)^2,\quad \lambda_{k+1} = \lambda_k - \Lambda_k C(\phi_k)
$$

$$
C(\phi) = (V(\phi)-V_f V_D)/V_D
$$

검증 수치(40×20×20 육면체, `E_0=1MPa`, `nu=0.3`, `Vf=0.4`, Apple M3 단일코어): 캔틸레버 모서리지지(`f=(0,0,-1)N`)에서 SIMP→LS가 다공초기 대비 **최대 4.6×**(SIMP `Delta=8%`+LS, 342.0s vs 1561.2s), 실용구간 `Delta=1~4%`에서 2.4~2.9×, 최종 컴플라이언스 스프레드 2% 이내. MBB 대칭하중에서는 4% 이내 스프레드이며 SIMP 초기 전부(66.31J)가 다공초기(68.44J)보다 우수 — 국소최적 회피다. LS 반복당 비용은 투영법이 AL 대비 ~1.4×(14s vs 10s)이나 벽시계 기준 speed-up에 이미 반영됨. 추출·SDF 비용은 3.8s로 무시 가능하다.

### 5.5 의사코드와 하이퍼파라미터

```text
# Hilbertian 투영 LS (Wegert Alg.1 + Jezek 순차)
SIMP: OC(p=3,R=2h,Vf) → Delta_rho_max 만족까지
전달: 절점LSQ Eq.(26) → rho_t=0.5 등면 → SDF Eq.(29)(30)
LS loop q=1..qmax(1000):
  1. 상태식(균질화/탄성) → J, Ci, 형상미분 §5.1
  2. Hilbertian 연장: 목적·제약 각각 A_H v = g (beta=4Δx, ell)
  3. Gram-Schmidt → P_C⊥ → Eq.(34) v_Omega (alpha_min^2=0.1, Jezek 적응식(31) 선택)
  4. HJ Godunov 전진(고정스텝, CFL gamma) → 재초기화(5e-5)
  5. line-search: J_new < J + xi|J| (xi=0.005) 이면 gamma*=1.1 수용, 아니면 gamma*=0.7 기각
  6. |J_q - J_{q-j}|<=eps1|J_q| (eps1=0.01, j=1..5, jmax=5) and |Ci|<eps2=1e-4 이면 종료
```

| 파라미터 | 값 | 출처 |
|---|---|---|
| `qmax/kmax` | 1000/10 | Wegert Table 1 |
| `eta/eps_void` | 1.5Δx/0.001 | Heaviside·ersatz |
| `beta(Hilbertian)` | 4Δx | 정규화길이 |
| `alpha_min^2/lambda` | 0.1/0.5 | 투영법 (Jezek는 (31) 적응) |
| `gamma_min/max/reinit` | 0.001/0.1/0.1 | HJ CFL |
| `xi/delta_inc,dec` | 0.005/1.1,0.7 | line search |
| `eps1/eps2/jmax` | 0.01/1e-4/5 | 정지판정 |
| `rho_t` | 0.5 | SIMP→SDF 임계 |
| `J_tol` | 기본/5 (Jezek) | LS 정제 수렴깊이 |

### 5.6 FD 검증·매핑·pitfalls

- FD 검증: 경계적분 민감도 `sigma:eps`를 경계노드 섭동 FD와 대조하되 ersatz 대역(`eta`)을 고정한다. `|grad phi|=1` 재초기화 없이 완화식(15)를 쓰면 오차가 커진다 — 반드시 재초기화 후 검증한다.
- 매핑: `levelset.py`(HJ·재초기화·Hilbertian 연장) + `evaluator.py`(균질화/탄성 상태식) + `problem.py`(체적등식) + 순차전달(SDF)은 `driver.py` 신규 스텝. 등방성 6제약(2D, Wegert Eq.(44)-(50), 정규화 `sqrt(4k^2+8mu^2)`는 미분 시 상수 취급) 같은 다중구속은 투영법이 담당한다.
- pitfalls: (i) 2D에서 HJ만으로 구멍생성 불가 — 초기 구멍분포가 빈약하면 SIMP 초기 없이는 실패한다. (ii) `alpha_min` 하한 과대 시 준가능점에서 발진 — Jezek (31) 적응하한을 쓴다. (iii) SLP 신뢰영역(Wegert Eq.(39) `|lambda_i| ≤ Δx/(2‖theta_i‖_H)`)은 문제의존적 — 투영법이 파라미터 튜닝이 적다. (iv) ersatz 제거 strip 최적화 시 경계 스트립을 남겨야 한다. (v) `bar{v}=0` 트랙션 고정을 빼먹으면 하중점이 사라진다.

---

## 6. SEMDOT — Smooth-Edged Material Distribution (Fu 2020)

### 6.1 서브그리드 표현과 비처벌 결합강성 (수식 완전전개)

격자점 밀도 설계변수 `rho_{e,g} ∈ {rho_min, 1}`, 요소체적분율(Fu Eq.(2)):

$$
X_e = \frac{1}{N}\sum_{g=1}^{N} \rho_{e,g},\qquad X_e \in [\rho_{\min}, 1]
$$

격자점 재료보간(Fu Eq.(1)): `E_e(rho_{e,g}) = rho_{e,g}^p E_1`. 고체·공극 요소는 SIMP 표준식(Fu Eq.(3)):

$$
E_e(X_e) = X_e^p\,E_1,\quad X_e \in \{\rho_{\min}, 1\}
$$

경계(중간) 요소는 이상(bi-material) 혼합이라 정확한 함수가 없으므로 **비처벌 선형결합**으로 근사한다(Fu Eq.(4) — 요구된 "비처벌결합"의 정의):

$$
K_e(X_e) = (1-X_e)\,K_e^0 + X_e\,K_e^1 = (1-X_e)\,\rho_{\min}^p\,K_e^1 + X_e\,K_e^1,\quad X_e \in [\rho_{\min},1]
$$

`□ = (1-X_e)□_void + X_e□_solid` 꼴이며 단순성을 위해 선형을 택했다고 명시한다.

### 6.2 민감도 근사 (수식 완전전개)

고체·공극 단의 정확한 감도에 가중합을 건 근사(Fu Eq.(5)):

$$
\frac{\partial C}{\partial X_e} \approx (1-X_e)\left.\frac{\partial C}{\partial X_e}\right|_{X_e=\rho_{\min}} + X_e\left.\frac{\partial C}{\partial X_e}\right|_{X_e=1} = -p\big[(1-X_e)\rho_{\min}^{p-1} + X_e\big]\,u_e^T K_e^1 u_e
$$

컴플라이언스 문제(Fu Eq.(6)):

$$
\min C(X_e) = f^T u \ \mathrm{s.t.}\ K(X_e)u = f,\ \frac{\sum_e X_e V_e}{\sum_e V_e} - V^* \le 0,\ 0 < \rho_{\min} \le X_e \le 1
$$

컴플라이언트 메커니즘(Fu Eq.(7)-(8), 더미하중 `K tilde{u} = -L`):

$$
\min C = -u_{\mathrm{out}} = -L^T u = \tilde{u}^T K u,\quad \frac{\partial C}{\partial X_e} \approx p\big[(1-X_e)\rho_{\min}^{p-1}+X_e\big]\,\tilde{u}_e^T K_e^1 u_e
$$

필터 2단(Fu Eq.(9)-(13)): 요소필터 `tilde{X} = H X`(Eq.(9)(10)) → 연쇄미분 Eq.(11) → BESO식 발견적 격자필터로 절점밀도(Fu Eq.(12)(13)):

$$
\tilde{X}_e = \frac{\sum_l \omega_{el}X_l}{\sum_l \omega_{el}},\ \omega_{el}=\max(0,r_{\min}-\Delta(e,l))
$$

$$
\frac{\partial C(\tilde{X}_e)}{\partial X_l} = \sum_e \frac{\omega_{el}}{\sum_\varsigma \omega_{e\varsigma}}\frac{\partial C}{\partial\tilde{X}_e},\quad \rho_n = \frac{\sum_e \omega_{ne}\tilde{X}_e}{\sum_e \omega_{ne}},\ \omega_{ne}=\max(0,\Upsilon_{\min}-\Delta(n,e))
$$

`Upsilon_min ≥ 1` 요소폭이 필수이며, Heaviside 투영·모폴로지·체적보존 투영 같은 흑백 강제 필터는 경계 중간요소를 죽이므로 **사용금지**(Fu §2.3 명시)이다.

### 6.3 Heaviside와 수렴판정 (수식 완전전개)

스텝(Fu Eq.(15)) vs 스무드(Fu Eq.(16)-(17)):

$$
\rho_{e,g} = \begin{cases} 1 & \rho_{e,g} > \Psi \\ \rho_{\min} & \rho_{e,g} \le \Psi \end{cases},\qquad \rho_{e,g} = \frac{\tanh(\beta\Psi)+\tanh[\beta(\rho(x,y)-\Psi)]}{\tanh(\beta\Psi)+\tanh[\beta(1.0-\Psi)]}
$$

$$
\beta_k = \beta_{k-1} + \Lambda,\quad \beta_0 = 0.5,\ \Lambda = 0.5
$$

스무드 쪽이 초기 20회 파동·산재재료·얇은 바·비대칭(수동영역 교란)을 피하고 제조성이 좋다(단순지지 깊은보 180×90, `r_min=2`: 스텝 21.3875J/167회 vs 스무드 21.1049J/142회; 고정홀 케이스 차 0.2% 이내이나 스텝안은 얇은 바·미세홀). 요소 갱신(Fu Eq.(18)(19))과 레벨셋 표현(Fu Eq.(20)):

$$
\tilde{X}_e^{\mathrm{new}} = \frac{1}{N}\sum_g \rho_{e,g}^{\mathrm{new}},\quad \delta_e = \tilde{X}_e^{\mathrm{new}} - \tilde{X}_e,\quad \Phi(x,y) = \rho(x,y) - \Psi
$$

수렴판정: 스텝형은 전체 위상변화율(Fu Eq.(21)):

$$
\frac{\sum_e |X_e^k - X_e^{k-1}|}{\sum_e X_e^k} \le \tau,\quad \tau = 0.001
$$

스무드형은 위상경계오차(Fu Eq.(22))를 병행한다:

$$
\frac{N_v}{M} \le \epsilon,\quad \epsilon = 0.001
$$

`N_v`는 경계에 없는 중간요소 수이다. `delta_e` 히스토그램은 ±0.2에 집중, 다수 요소는 0이다.

### 6.4 의사코드와 하이퍼파라미터

```text
# SEMDOT (Fu Fig.2 + 부록 MATLAB)
초기화: X, beta=0.5, MMA(low/upp, vxold1/2, a0=1, ai=0, ci=1000, di=0)
loop:
  1. FEA(K(X)u=f) → Eq.(5)/(8) 감도 → Eq.(9)-(11) 필터
  2. MMA(mmasub) → X_new (설계변수 업데이트)
  3. X_tilde 필터 → 절점밀도 Eq.(12) → 격자보간 Eq.(14)
  4. Heaviside Eq.(16) (beta+=0.5) → rho_new → Eq.(18) X_new_tilde
  5. Phi Eq.(20) → Eq.(21) 전체변화 + Eq.(22) 경계오차 판정
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| `p` | 1.5 | SEMDOT 패널티 (SIMP 3과 다름) |
| `beta_0/Lambda` | 0.5/0.5 | 스무드 급경도·증분 |
| `tau/epsilon` | 0.001/0.001 | 종료판정 |
| `10×10` | 요소당 격자점 | Fu 권장 |
| `r_min/Upsilon_min` | 2(예제)/≥1 | 요소·발견적 필터 반경 |
| `V*` | 0.3 | 벤치 공통 |
| 옵티마이저 | MMA 기본파라미터 | Svanberg `mmasub/subsolv` 외부파일 필요 |

### 6.5 FD 검증·매핑·pitfalls

- FD 검증: 고체·공극 단 감도는 표준 SIMP FD와 일치해야 하며, 경계요소 근사 Eq.(5)는 컴플라이언트 메커니즘(force inverter) 케이스로 타당성을 입증한다(Fu §3.4). FD는 `tilde{X}`가 아닌 `X` 기준으로 한다(연쇄미분 이전).
- 매핑: `aieng/src/aieng/topopt/semdot.py`(서브그리드·Heaviside·경계오차) + `filters.py`(2단 필터) + `optimizers/mma.py`(설계갱신) + `interpolation.py`(격자보간).
- pitfalls: (i) `Upsilon_min < 1`이면 위상 자체가 안 나온다. (ii) 흑백 강제 필터를 물리면 경계 중간요소가 사라져 매끄러운 경계가 깨진다. (iii) 요소당 격자점 축소는 계산비 절감이나 경계오차 증가 — 10×10이 기준이다. (iv) MMA `mmasub.m/subsolv.m`은 외부 제공(Svanberg, KTH) — 라이선스 확인 없이 재배포 금지(부록 명시).

---

## 7. SIMP↔BESO↔SEMDOT 비교 (Fu §3.5 수치, 동일조건)

깊은 캔틸레버·L-브라켓( Fu Table 3-10 ): SEMDOT가 SIMP-D(밀도필터, 최악 60.0841J)보다 우수하고 SIMP-H·BESO·ETO와 대등하거나 우세하다. BESO·ETO는 수렴이 빠르나(68~119회) 얇은 부재를 만들기 쉽고, SIMP-H는 200회 이상 걸린다. SEMDOT-S(단일 발견적 필터)는 얇은 부재·단절에 취약하므로 정식 2단 필터를 쓴다. 메쉬·종횡비·수동영역에 대한 결론은 동일하다.

---

## 8. GradientSource 신뢰계층과 Trust-Region 수용 (FRAMEWORK_PLAN §3 + Jezek/Wegert 선탐색)

| Tier | 의미 | 드라이버 규칙 |
|---|---|---|
| `exact_adjoint` | 이산화와 일치하는 이산수반 | FD 통과 시 `converged` 허용 |
| `autodiff` | 체크포인팅 포함 AD | FD 통과 시 허용 |
| `surrogate` | ESL/DiESL·축소모델·동결하중 구배 | 참응답 개선 시에만 수용, `converged` 보고 금지 |
| `learned` | 신경망 대리·생성제안 | `surrogate` + OOD 플래그 |
| `finite_difference` | 직접차분 | 미소문제만, 비용경고 |
| `none` | 무구배 | 구배불요 옵티마이저(CMA-ES)만 |

동결하중 ESL 구배의 참구배 대비 코사인 0.5~0.77(준선형 동적) 사례가 강등 근거이다. Trust-Region 수용비(구배 스파이크·GCMMA 내부반복·HJ line-search 공통형):

$$
\rho_{TR} = \frac{J(x) - J(x^+)}{\tilde{J}(x) - \tilde{J}(x^+)},\quad \rho_{TR} > c_1\ \text{수용},\ \rho_{TR} > c_2\ \text{영역확대},\ \rho_{TR} < c_0\ \text{영역축소·기각}
$$

Wegert HJ line-search(`J_new < J + xi|J|`, `xi=0.005`, 아니면 `gamma*=0.7`)와 Jezek CFL 25% 삭감은 이 판정의 HJ 특수형이다. 모든 `Evaluation`은 `{objective, constraints[], gradients, gradient_source tier, provenance}`를 달고, `Optimizer.step`은 tier가 허용할 때만 `converged`를 선언한다.

---

## 9. wht_topopt 전체 매핑표와 공통 pitfalls

| 원문 방법 | `aieng/src/aieng/topopt/` 매핑 | 비고 |
|---|---|---|
| OC-SIMP (Jezek §2.1) | `optimizers/oc.py`, `filters.py`, `evaluator.py`, `interpolation.py` | `move/eta/passive` 그대로 |
| MMA/GCMMA | `optimizers/mma.py` (`MMA`/`GCMMA`, `mmapy`/`mmadual`) | GPLv3 백엔드, `m<n` 강제 |
| BESO 고전 (Huang-Xie) | `beso.py`, `filters.py`(절점스무딩), `evaluator.py` | 순위·`RRV/ER`·`PI` |
| BESO 적응ER 비선형 (Tian) | `beso.py`(ER함수 4종 추가), 외부 FEA는 `adapters/` | `SENER/PENER`, TL·N-R |
| Hilbertian LS (Wegert) | `levelset.py`, `evaluator.py`(균질화), `constraints.py`(다중등식) | Gram–Schmidt·투영 |
| 순차 SIMP→LS (Jezek) | `driver.py`(순차), `levelset.py`, `problem.py` | SDF 전달·(31) 적응하한 |
| SEMDOT (Fu) | `semdot.py`, `filters.py`, `optimizers/mma.py` | 서브그리드·`N_v/M` |

공통 pitfalls 체크리스트:

1. **수식생략 금지**: `K_e` 보간·감도·필터·투영 중 하나라도 빼먹으면 재현이 깨진다 — 본 파일의 완전전개식을 그대로 옮긴다.
2. **이분탐색 혼동**: OC `lambda`, BESO `alpha_th`, SEMDOT `Psi`는 모두 "목표체적→정렬→절단값" 역산이다. 정방향(값 지정)으로 쓰면 체적이 어긋난다.
3. **신뢰계층 위반**: `surrogate/learned` 구배로 `converged`를 찍지 않는다. ESL·생성제안은 재해석(Re-evaluation) 후 보고한다.
4. **라이선스**: `mmapy`·`mma-dual`·`mmasub.m`은 GPLv3/연락제공 — `wht_topopt` 코어에 벤더코드를 복사하지 말고 선택적 백엔드로 둔다(현 래퍼 방식 유지).
5. **단위·부호**: 레벨셋 부호(`phi<0` 고체, Wegert·Jezek 공통), `SENER+PENER` 합산, `alpha_min` 하한, `bar{v}=0` 트랙션 고정을 매 구현 전 assert로 확인한다 (`require` 가드 권장).
