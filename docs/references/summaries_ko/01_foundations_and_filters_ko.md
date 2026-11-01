# 01 — Foundations & Filters: SIMP 기초와 밀도·민감도·Helmholtz·Heaviside 필터 (요약, 한국어)

> 원문 근거: 본 문서는 아래 PDF의 직접 텍스트 추출에 근거한다.
> - `references/topopt/papers/04_Andreassen_2011_88line_SMO.pdf` (88-line, SMO 43, pp.1–16, 전 페이지 텍스트 추출로 확인)
> - `references/topopt/papers/25_Ferrari_Sigmund_2020_top99neo_top3d125_SMO.pdf` (top99neo/top3d125, arXiv:2005.05436v2, 전 페이지 텍스트 추출로 확인)
> - 제외: `references/topopt/2007_huang-xie_beso-new-algorithm_JSME-49-4-1091.pdf`는 본 문서 범위에서 제외한다 (BESO는 후속 문서 담당).
> - 미확보 문헌 `01_Bendsoe_Sigmund_1999` (SIMP 보간 원전), `06_Lazarov_Sigmund_2011_PDE_Filter` (Helmholtz PDE 필터 원전)는 DTU 88-line / top99neo 코드 내 동일 수식 + 후속 논문 `05_Wang_Lazarov_Sigmund_2011`(투영·강건 정식화) / `17_Guest_2004 계열`(Heaviside 투영·최소 길이 스케일) 계열의 교차 추출로 기술하며, 각 수식마다 출처를 명시한다. 즉 본 문서의 SIMP/필터/투영 수식은 04·25번 코드·본문에 그대로 존재하는 것을 전사한 것이고, 01/06 고유 서술(예: HS 경계 p조건의 증명, PDE 필터 수렴 이론)은 원전 미확보이므로 "교차 추출"임을 밝힌다.
> - spec 매핑: `topology_optimization_framework_spec.md` §1(연산자 합성·체인룰), §5.1(물성 보간·필터 레퍼런스 표), §6(참조 코드), `references/topopt/FRAMEWORK_PLAN.md`(Representation/필터), `references/topopt/INDEX.md`(Phase 0 다운로드 표, 미확보 목록).

---

## 1. 서지 정보 + Nomenclature 표

### 1.1 서지

| # | 서지 | 본 문서에서 쓰는 것 | 상태 |
|---|---|---|---|
| 04 | E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, O. Sigmund, "Efficient topology optimization in MATLAB using 88 lines of code", *Struct. Multidisc. Optim.* 43(1):1–16 (2011). DOI 10.1007/s00158-010-0594-7 | Modified SIMP, 민감도/밀도 필터 W행렬, 체인룰, OC 업데이트, Helmholtz PDE 이산화(§4.2), Heaviside 투영(§5.4), 하이퍼파라미터 | PDF 전문 추출·확인 |
| 25 | F. Ferrari, O. Sigmund, "A new generation 99 line MATLAB code for compliance topology optimization and its extension to 3D", *SMO* (2020), arXiv:2005.05436v2 | 완화 Heaviside 투영(tanh형), 체적보존 $\eta^*$ Newton 탐색, $\lambda^\#$ 추정·이분법 단축, 조립 대칭 절반화·imfilter, continuation 구조 | PDF 전문 추출·확인 |
| 01 (미확보) | M. P. Bendsøe, O. Sigmund, "Material interpolation schemes in topology optimization", *Arch. Appl. Mech.* 69:635–654 (1999) | SIMP/RAMP/HS 경계 p조건의 원전 귀속만 명시. 수식 자체는 04 eq.(1)·25 eq.(6)과 동일하므로 코드 구현은 04·25로 완결됨 | 교차 추출 (원전 미확보) |
| 06 (미확보) | B. S. Lazarov, O. Sigmund, "Filters in topology optimization based on Helmholtz type differential equations" (2010/2011) | Helmholtz PDE 필터식·$l=r_{\min}/2\sqrt{3}$ 관계. 04 §4.2에 축약형으로 포함되어 있으므로 구현은 04 §4.2로 완결됨 | 교차 추출 (원전 미확보) |
| 05 (후속) | F. Wang, B. S. Lazarov, O. Sigmund, "On projection methods, convergence and robust formulations in topology optimization", *SMO* 43:767–784 (2011) | tanh 투영식(25 eq.(1)의 출처), 강건 erode/intermediate/dilate 개념의 출처 | 25번이 직접 인용하므로 2차 출처로 명시 |
| 17 (후속) | J. K. Guest, J. H. Prévost, T. Belytschko, "Achieving minimum length scale … using nodal design variables and projection functions", *IJNME* 61:238–254 (2004) | Heaviside 투영 아이디어·최소 길이 스케일 개념의 출처 (04 §5.4가 인용) | 04번이 직접 인용하므로 2차 출처로 명시 |

### 1.2 Nomenclature (기호 통일표 — 코드 구현자가 그대로 사용)

| 기호 | 의미 | 차원/범위 | 등장 |
|---|---|---|---|
| $x_e$ | 설계변수 (요소 밀도) | $[0,1]$, $e=1,\dots,N$ (또는 $m$) | 04 §2.1, 25 §2 |
| $\tilde{x}_e$ ($\tilde{\rho}_e$) | 필터된(중간) 밀도 = $W x$ | $[0,1]$ | 04 eq.(9), 25 eq.(2) |
| $\bar{x}_e$ ($\hat{x}_e$) | 투영된 물리 밀도 (Heaviside 적용 후) | $[0,1]$ | 04 eq.(21), 25 eq.(1) |
| $E_e$, $E_0$, $E_{\min}$ | 요소/고체/void 영률 | $E_{\min}=10^{-9}E_0$ (코드 기본값) | 04 eq.(1), 25 eq.(6) |
| $p$ | SIMP 페널티 지수 | 보통 $3$ | 04 eq.(1) |
| $q$ | RAMP 페널티 파라미터 | $q \ge 0$ | 01 귀속 (아래 §2.3) |
| $\mathbf{K}, \mathbf{u}, \mathbf{F}$ | 전역 강성행렬·변위·하중 | $\mathbf{Ku}=\mathbf{F}$ | 04 eq.(2), 25 eq.(5) |
| $\mathbf{k}_0$ ($\mathbf{K}_e^0$, `KE`/`Ke0`) | 단위 영률 요소 강성행렬 | Q4: $8\times 8$, H8: $24\times 24$ | 04 §3.1, 25 §3/PRE.2 |
| $c$ | 컴플라이언스 $c=\mathbf{U}^T\mathbf{KU}=\mathbf{f}^T\mathbf{u}$ | 스칼라 | 04 eq.(2), 25 eq.(3) |
| $V, V_0, f$ | 재료 체적·설계영역 체적·체적분율 | $V/V_0 = f$ | 04 eq.(2), 25 eq.(4) |
| $r_{\min}$ | 고전 필터 반경 (요소 크기 단위) | 예: MBB $2.4$, $6$, $16$ | 04 eq.(8) |
| $R_{\min}$ ($l$) | Helmholtz PDE 길이 스케일 | $R_{\min}=r_{\min}/2\sqrt{3}$ | 04 eq.(17) |
| $H_{ei}$, $w_{ei}$ | 필터 가중 $=\max(0,r_{\min}-\Delta(e,i))$ | 스칼라 | 04 eq.(8) |
| $\beta, \eta$ | Heaviside 선예도·임계값 | $\beta: 1\to 512$ (2배씩), $\eta \approx 0.5$ | 04 §5.4, 25 §3.2 |
| $m$ ($\mu$) | OC 이동제한 | $0.2$ (04), `move` (25) | 04 eq.(3), 25 eq.(10) |
| $\lambda$ ($\tilde{\lambda}$) | 체적제약 Lagrange 승수 | $>0$, 이분법 탐색 | 04 eq.(4), 25 eq.(11) |
| $d_c, d_v$ (`dc`, `dv`) | 목적·체적 민감도 벡터 | 요소별 | 04 §3.3, 25 RL.3 |
| $M_{nd}$ ($Mnd$) | grayness (비이산도) 지표 | $M_{nd}=100\cdot 4\sum_e \bar{x}_e(1-\bar{x}_e)/N$ | 25 §3.3 정의 전사 |

---

## 2. 지배·이산화 수식 완전전개 (LaTeX)

### 2.1 연속 지배방정식 (선형 탄성 정적 문제)

설계영역 $\Omega$에서 변위장 $\mathbf{u}:\Omega\to\mathbb{R}^d$:

$$
-\nabla\cdot\boldsymbol{\sigma}(\mathbf{u}) = \mathbf{0} \quad \text{in } \Omega,
\qquad
\boldsymbol{\sigma} = \mathbf{D}(E(\rho)) : \boldsymbol{\varepsilon}(\mathbf{u}),
\qquad
\boldsymbol{\varepsilon}(\mathbf{u}) = \tfrac{1}{2}(\nabla\mathbf{u}+\nabla\mathbf{u}^T).
$$

경계조건 $\mathbf{u}=\mathbf{0}$ on $\Gamma_D$, $\boldsymbol{\sigma}\cdot\mathbf{n}=\mathbf{t}$ on $\Gamma_N$.
약형식: 모든 시험함수 $\mathbf{v}\in V_0$에 대해

$$
a(\mathbf{u},\mathbf{v};\rho) = l(\mathbf{v}),
\quad
a(\mathbf{u},\mathbf{v};\rho) := \int_\Omega \boldsymbol{\varepsilon}(\mathbf{v})^T \mathbf{D}(E(\rho))\,\boldsymbol{\varepsilon}(\mathbf{u})\,d\Omega,
\quad
l(\mathbf{v}) := \int_{\Gamma_N}\mathbf{t}\cdot\mathbf{v}\,d\Gamma.
$$

### 2.2 이산화 (Q4/H8 등매개 요소 — 04 §3.1, 25 PRE.2/§3.1)

요소 $e$의 절점변위 $\mathbf{u}_e$, 형상함수 행렬 $\mathbf{N}$, 변형률-변위 행렬 $\mathbf{B}$:

$$
\mathbf{K}_e^0 = \int_{\Omega_e} \mathbf{B}^T \mathbf{D}_0 \mathbf{B}\,d\Omega_e,
\qquad
\mathbf{K}(\bar{\mathbf{x}}) = \mathop{\mathrm{A}}_{e=1}^{N} E(\bar{x}_e)\,\mathbf{K}_e^0,
\qquad
\mathbf{K}\mathbf{U} = \mathbf{F}.
$$

04 코드는 정규 직사각형 메쉬이므로 모든 요소가同一 $\mathbf{K}_e^0$ (`KE`)를 공유한다.
등방성 평면응력 $\mathbf{D}_0$ (04 부록 line 8–12):

$$
\mathbf{K}_e^0 = \frac{1}{1-\nu^2}\frac{1}{24}\left(
\begin{bmatrix}\mathbf{A}_{11}&\mathbf{A}_{12}\\\mathbf{A}_{12}^T&\mathbf{A}_{11}\end{bmatrix}
+\nu
\begin{bmatrix}\mathbf{B}_{11}&\mathbf{B}_{12}\\\mathbf{B}_{12}^T&\mathbf{B}_{11}\end{bmatrix}
\right),
$$

$\mathbf{A}_{11},\mathbf{A}_{12},\mathbf{B}_{11},\mathbf{B}_{12}$는 부록의 $4\times 4$ 수치행렬 그대로 사용한다
(전사 생략이 아닌 코드 내 리터럴 복사 — 구현자는 04 부록 line 8–12를 그대로 붙여넣을 것).

### 2.3 재료 보간: SIMP / Modified SIMP / RAMP (01 귀속 + 04·25 확인)

**(a) Classical SIMP** (01 원전, 99-line 계열):

$$
E(x_e) = x_e^p\,E_0, \qquad 0 < x_{\min} \le x_e \le 1.
$$

**(b) Modified SIMP** (04 eq.(1), 25 eq.(6) — 본 프레임워크 기본):

$$
\boxed{E(\bar{x}_e) = E_{\min} + \bar{x}_e^p\,(E_0 - E_{\min}), \qquad \bar{x}_e \in [0,1],}
$$

$$
\boxed{\frac{\partial E}{\partial \bar{x}_e} = p\,\bar{x}_e^{p-1}\,(E_0 - E_{\min}).}
$$

$E_{\min}=10^{-9}E_0$이므로 하한 구속 $x_{\min}$이 불필요하고, 추가 필터 구현이 단순해진다 (04 §2.1).

**(c) RAMP** (Stolpe–Svanberg 2001; spec §5.1이 `algorithms/simp.py`에 요구. 01 계열 보간이므로 본 문서에 포함):

$$
\boxed{E(\bar{x}_e) = E_{\min} + \frac{\bar{x}_e}{1+q(1-\bar{x}_e)}\,(E_0-E_{\min}), \qquad q\ge 0,}
$$

$$
\boxed{\frac{\partial E}{\partial \bar{x}_e} = \frac{1+q}{\left[1+q(1-\bar{x}_e)\right]^2}\,(E_0-E_{\min}).}
$$

RAMP는 $\bar{x}_e\to 0$에서 기울기가 $0$이 아니므로 동적/모드 문제의 저밀도 국부 허위모드를 억제한다 (spec §4.1).

**(d) Hashin–Shtrikman(HS) 경계와 $p$ 조건** (01 귀속, 교차 추출임을 명시):

- 2상(강성 $E_0$ + void) 체적분율 $f$ 복합재의 HS 상한은 벌크·전단계수에 대해 닫힌형 부등식을 준다.
- SIMP가 물리적으로 실현 가능한 미세구조로 달성 가능하려면 벌크계수 기준 $p \ge 3$ (2D, $\nu=1/3$ 대표 조건), 전단계수 기준 $p \ge$ 해당 임계값을 만족해야 한다는 것이 01의 결론으로 인용된다.
- 실무帰結: 정적 컴플라이언스에서는 $p=3$이 표준이며, $p$를 $1\to 3$으로 continuation하면 국부최적 회피에 유리하다 (25 §3 `penalCnt`).
- 원전 미확보이므로 HS 상한식의 완전한 텐서 전개는 본 문서에 싣지 않는다 — 코드 구현에는 $(p,q)$ 수치 조건만 필요하고 이는 04·25에 확정되어 있다.

### 2.4 최적화 문제 정식화 (04 eq.(2) ≡ 25 eq.(3)–(4))

$$
\boxed{
\begin{aligned}
\min_{\mathbf{x}\in[0,1]^N} \quad & c(\bar{\mathbf{x}}) = \mathbf{U}^T\mathbf{K}(\bar{\mathbf{x}})\mathbf{U}
= \sum_{e=1}^{N} E(\bar{x}_e)\,\mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e, \\
\text{s.t.} \quad & V(\bar{\mathbf{x}})/V_0 = \frac{1}{N}\sum_{e=1}^{N}\bar{x}_e = f, \\
& \mathbf{K}(\bar{\mathbf{x}})\mathbf{U} = \mathbf{F}.
\end{aligned}
}
$$

25는 수동(passive) 고체 $P_1$/void $P_0$를 분리한다 (25 eq.(4)):

$$
V(\bar{\mathbf{x}}) = \frac{1}{m}\left(m_{P1} + \sum_{e\in\mathcal{A}}\bar{x}_e\right) \le f,
$$

능동집합 $\mathcal{A}$에 대해서만 설계변수를 업데이트한다.

---

## 3. 필터 연산자 완전전개

### 3.1 고전 콘(cone) 가중행렬 $W$ (04 §3.2, eq.(7)–(9))

이웃집합 $\mathcal{N}_e = \{i \mid \Delta(e,i) < r_{\min}\}$ ($\Delta$는 요소 중심간 거리):

$$
\boxed{w_{ei} = H_{ei} = \max\big(0,\, r_{\min} - \Delta(e,i)\big).}
$$

희소행렬 $\mathbf{H}\in\mathbb{R}^{N\times N}$, $(\mathbf{H})_{ei}=w_{ei}$, 정규화 벡터 $H_{s,e}=\sum_{i\in\mathcal{N}_e} w_{ei}$.

**(a) 민감도 필터** (04 eq.(7) — Sigmund 1994/1997 계열):

$$
\boxed{
\widehat{\frac{\partial c}{\partial x_e}}
= \frac{1}{\max(\gamma,x_e)\sum_{i\in\mathcal{N}_e} w_{ei}}
\sum_{i\in\mathcal{N}_e} w_{ei}\,x_i\,\frac{\partial c}{\partial x_i},
\qquad \gamma = 10^{-3}.
}
$$

**(b) 밀도 필터** (04 eq.(9) — Bruns–Tortorelli/Bourdin 계열):

$$
\boxed{
\tilde{x}_e = \frac{\sum_{i\in\mathcal{N}_e} w_{ei}\,x_i}{\sum_{i\in\mathcal{N}_e} w_{ei}}
\quad\Longleftrightarrow\quad
\tilde{\mathbf{x}} = \mathbf{W}\mathbf{x},
\;\;
W_{ei} = \frac{w_{ei}}{\sum_{j\in\mathcal{N}_e} w_{ej}}.
}
$$

밀도 필터는 체적보존($\sum_e \tilde{x}_e = \sum_e x_e$에 비례, 균일 내장에서 정확)을 만족하므로 OC 이분법의 체적검사를 설계변수 $\mathbf{x}$에 대해 수행해도 된다 (04 §3.3, 25 §3.2).

**(c) $v_e$ 가중 일반형** (구현 일반화 — 불균일 요소 체적 대응):

$$
\tilde{x}_e = \frac{\sum_{i\in\mathcal{N}_e} w_{ei}\,v_i\,x_i}{\sum_{i\in\mathcal{N}_e} w_{ei}\,v_i},
$$

균일 메쉬에서는 $v_i\equiv 1$로 (b)에 귀착한다. `wht_topopt`의 `spatial.py`는 CSR 희소행렬로 $\mathbf{W}$를 캐싱한다.

### 3.2 Helmholtz PDE 필터 (04 §4.2, 06 교차 추출)

연속형 (04 eq.(15)–(16), 동차 Neumann 경계):

$$
\boxed{
-R_{\min}^2\,\nabla^2 \tilde{\rho} + \tilde{\rho} = \rho \quad \text{in } \Omega,
\qquad
\frac{\partial\tilde{\rho}}{\partial n} = 0 \quad \text{on } \partial\Omega.
}
$$

길이 스케일 관계 (04 eq.(17)):

$$
\boxed{R_{\min} = \frac{r_{\min}}{2\sqrt{3}}.}
$$

유한요소 이산화 (04 eq.(18)–(19)):

$$
\boxed{\mathbf{K}_F\,\tilde{\mathbf{x}}_N = \mathbf{T}_F\,\mathbf{x},}
\qquad
\boxed{\tilde{\mathbf{x}} = \mathbf{T}_F^T\,\tilde{\mathbf{x}}_N.}
$$

- $\mathbf{K}_F = R_{\min}^2\mathbf{A} + \mathbf{M}$: 스칼라 문제의 표준 FE 행렬.
  04 부록 코드는 Q4 요소행렬을 직접 준다:
  `KEF = Rmin^2*([4 -1 -2 -1; -1 4 -1 -2; -2 -1 4 -1; -1 -2 -1 4]/6) + ([4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4]/36)`.
- $\mathbf{T}_F$: 요소값을 절점으로 분배하는 $(4N\times N)$ 희소행렬, 성분 $1/4$.
- 04 구현은 직접분해 `LF = chol(KF,'lower')`를 사용한다:
  `xPhys(:) = (TF'*(LF'\(LF\(TF*xnew(:)))))`.

민감도 역필터 (체인룰, 04 §4.2 코드):

$$
\text{ft}=2:\quad
\mathbf{dc} \leftarrow \mathbf{T}_F^T(\mathbf{L}_F^{-T}(\mathbf{L}_F^{-1}(\mathbf{T}_F\,\mathbf{dc}))),
\quad
\mathbf{dv} \leftarrow \mathbf{T}_F^T(\mathbf{L}_F^{-T}(\mathbf{L}_F^{-1}(\mathbf{T}_F\,\mathbf{dv}))).
$$

PDE 필터도 체적보존이므로 이분법 내 재필터가 불필요하다 (04 §4.2).

### 3.3 conv2/imfilter 구현 동등형 (04 §4.1, 25 PRE.5)

직사각형 균일 메쉬에서 밀도 필터는 2D 이산 컨볼루션이다 (04 eq.(11)–(14)):

$$
\tilde{x}(k,l) = \frac{\sum_{m,n} h(m,n)\,x(k-m,l-n)}{\sum_{m,n} h(m,n)},
$$

$$
h(m,n) = \max\big(0,\,r_{\min}-\sqrt{m^2+n^2}\big).
$$

- 04: `Hs = conv2(ones(nely,nelx),h,'same')`, `xPhys = conv2(xnew,h,'same')./Hs`.
- 25: `imfilter` 사용 — `conv2`는 zero-Dirichlet만 가능하지만 `imfilter`는 `'symmetric'`(zero-Neumann) 선택이 가능하므로 경계처리 자유도가 높다 (25 §3).
  - Neumann(`ftBC='N'`→`'symmetric'`): 경계에서 필터가 잘리지 않음. 3D 캔틸레버에서 Neumann이 $c=3993.7$, Dirichlet이 $c=4074.3$으로 Neumann이 더 좋은 국부최적에 도달 (25 §4).
  - `wht_topopt` 매핑: `spatial.py`는 KD-Tree+CSR이 기본이며, 직사각형 균일 메쉬 fast-path로 컨볼루션을 허용한다.

---

## 4. Heaviside 투영 완전전개

### 4.1 Guest형 (지수형, 04 §5.4 eq.(21)–(23))

중간 밀도 $\tilde{x}_e$ → 물리 밀도 $\bar{x}_e$:

$$
\boxed{
\bar{x}_e = 1 - e^{-\beta\tilde{x}_e} + \tilde{x}_e\,e^{-\beta},
\qquad \beta \ge 0.
}
$$

- $\beta=0$이면 밀도 필터와 동일, $\beta\to\infty$이면 Heaviside 계단함수.
- 도함수 (04 eq.(23)):

$$
\boxed{
\frac{d\bar{x}_e}{d\tilde{x}_e} = \beta e^{-\beta\tilde{x}_e} + e^{-\beta}.
}
$$

- 체인룰 (04 eq.(22)):

$$
\frac{\partial f}{\partial \tilde{x}_e} = \frac{\partial f}{\partial \bar{x}_e}\cdot\frac{d\bar{x}_e}{d\tilde{x}_e}.
$$

- 코드 (04 §5.4): `dx = beta*exp(-beta*xTilde)+exp(-beta); dc(:) = H*(dc(:).*dx(:)./Hs);` 동일하게 `dv`.

### 4.2 Wang–Lazarov–Sigmund tanh형 (25 eq.(1), eq.(9) — 05 교차 추출)

$$
\boxed{
\bar{x}_e = H(\tilde{x}_e;\eta,\beta)
= \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde{x}_e-\eta))}{\tanh(\beta\eta) + \tanh(\beta(1-\eta))}.
}
$$

도함수 (25 eq.(9)):

$$
\boxed{
\frac{\partial H}{\partial \tilde{x}_e}
= \beta\,\frac{1-\tanh^2(\beta(\tilde{x}_e-\eta))}{\tanh(\beta\eta)+\tanh(\beta(1-\eta))}.
}
$$

$\eta$ 도함수 (25 `deta`, Newton 탐색용):

$$
\frac{\partial H}{\partial \eta}
= -\beta\,\mathrm{csch}(\beta)\,\mathrm{sech}^2(\beta(\tilde{x}_e-\eta))\,
\sinh(\beta\tilde{x}_e)\,\sinh(\beta(1-\tilde{x}_e)).
$$

### 4.3 체적보존 $\eta^*$ Newton 탐색 (25 §3.2 eq.(17)–(18))

투영은 일반적으로 체적을 보존하지 않으므로, 각 이터레이션에서

$$
\boxed{
\eta_k^* = \arg\min_{\eta\in[0,1]} \big|V(\hat{\mathbf{x}}_k(\eta)) - V(\tilde{\mathbf{x}}_k)\big|,
\quad
g(\eta) = V(\hat{\mathbf{x}}(\eta)) - V(\tilde{\mathbf{x}}),
}
$$

$g(0)g(1)\le 0$이고 $dg/d\eta < 0$이므로 해가 존재·유일하다.
Newton 반복:

$$
\eta \leftarrow \eta - \frac{g(\eta)}{g'(\eta)},
\qquad
g'(\eta) = \frac{1}{N}\sum_{e}\frac{\partial H(\tilde{x}_e;\eta,\beta)}{\partial \eta}.
$$

25 코드 line 61–66: 허용오차 $|f|<10^{-6}$, 보통 1–2회 수렴, $\eta_k^*\in[0.4,0.52]$ 관측.
`ft==3`(체적보존 투영)에서만 탐색하고, `ft==2`(고정 $\eta$)에서는 체적제약을 $\bar{\mathbf{x}}$에 걸어야 한다
($\eta<0.5$면 $V(\bar{\mathbf{x}})>f$, $\eta>0.5$면 반대 — 25 §3.2 경고).

### 4.4 전체 체인룰 (필터+투영, 04 eq.(10) + 25 eq.(8))

$$
\boxed{
\frac{\partial f}{\partial x_j}
= \sum_{e\in\mathcal{N}_j} W_{ej}\,\frac{d\bar{x}_e}{d\tilde{x}_e}\,\frac{\partial f}{\partial \bar{x}_e},
}
$$

행렬형 (25 eq.(8), 요소별 곱 $\odot$, $\mathbf{F}=\mathrm{diag}(d\bar{\mathbf{x}}/d\tilde{\mathbf{x}})$):

$$
\boxed{
\nabla_{\mathbf{x}} f = \mathbf{W}^T \mathbf{F}\, \nabla_{\bar{\mathbf{x}}} f,
\qquad
\nabla_{\mathbf{x}} V = \mathbf{W}^T \mathbf{F}\, \nabla_{\bar{\mathbf{x}}} V.
}
$$

코드 대응:
- 04 (line 66–67): `dc(:) = H*(dc(:)./Hs); dv(:) = H*(dv(:)./Hs);`
- 25 (RL.3): `dc = imfilter(reshape(dc,nely,nelx)./dHs,h,bcF)` (여기서 `dHs = Hs./dprj`이므로 나눗셈이 $\mathbf{F}$ 곱과 동일).

### 4.5 Mnd grayness + continuation 스케줄

비이산도 (25 §3.3):

$$
\boxed{
M_{nd} = 100\cdot\frac{4}{N}\sum_{e=1}^{N}\bar{x}_e(1-\bar{x}_e).
}
$$

완전 0/1이면 $0$, 전면 $0.5$이면 $100$.

Continuation (25 line 6–7, 92):

```text
parCont = {istart, maxPar, isteps, deltaPar}
penalCnt = {1, 3, 25, 0.25}   # p: loop>=1부터 25마다 +0.25, 최대 3
betaCnt  = {1, 2, 25, 2}      # beta: loop>=1부터 25마다 +2 (ft>1일 때; 통상 beta 2→16)
par = par + (loop>=istart).*(par<maxPar).*(mod(loop,isteps)==0).*deltaPar
```

04 Heaviside 스케줄 (§5.4): $\beta=1$ 시작, 50회마다 또는 $\text{change}<0.01$이면 $\beta\leftarrow 2\beta$, 최대 $512$.

---

## 5. 수반/체인룰 민감도 + FD 검증식

### 5.1 컴플라이언스 자기수반 민감도 (04 eq.(5), 25 eq.(7))

$\mathbf{KU}=\mathbf{F}$를 미분하면 $\frac{\partial\mathbf{K}}{\partial\bar{x}_e}\mathbf{U}+\mathbf{K}\frac{\partial\mathbf{U}}{\partial\bar{x}_e}=0$.
$c=\mathbf{F}^T\mathbf{U}$이므로:

$$
\frac{\partial c}{\partial \bar{x}_e}
= \mathbf{F}^T\frac{\partial\mathbf{U}}{\partial\bar{x}_e}
= -\mathbf{U}^T\frac{\partial\mathbf{K}}{\partial\bar{x}_e}\mathbf{U}
= -\mathbf{u}_e^T\frac{\partial\mathbf{K}_e}{\partial\bar{x}_e}\mathbf{u}_e.
$$

Modified SIMP를 대입하면 (04 eq.(5)):

$$
\boxed{
\frac{\partial c}{\partial \bar{x}_e}
= -p\,\bar{x}_e^{p-1}\,(E_0-E_{\min})\,\mathbf{u}_e^T\mathbf{k}_0\mathbf{u}_e,
\qquad
\frac{\partial V}{\partial \bar{x}_e} = 1\;(\text{또는 }1/N).
}
$$

RAMP면 $\partial E/\partial\bar{x}_e$만 §2.3(c)로 교체한다.

### 5.2 일반 응답의 이산 수반식 (spec §4.1 인용 — 정적 범위로 한정)

스칼라 응답 $g(\mathbf{U}(\bar{\mathbf{x}}),\bar{\mathbf{x}})$에 대해 수반벡터 $\boldsymbol{\lambda}$:

$$
\boxed{
\mathbf{K}\,\boldsymbol{\lambda} = -\left(\frac{\partial g}{\partial\mathbf{U}}\right)^T,
\qquad
\frac{dg}{d\bar{x}_e}
= \left.\frac{\partial g}{\partial\bar{x}_e}\right|_{\text{explicit}}
+ \boldsymbol{\lambda}_e^T\frac{\partial\mathbf{K}_e}{\partial\bar{x}_e}\mathbf{u}_e.
}
$$

컴플라이언스는 $\partial g/\partial\mathbf{U}=\mathbf{F}^T$이므로 $\boldsymbol{\lambda}=-\mathbf{U}$가 되어 자기수반으로 귀착한다.

### 5.3 FD 검증식 (반드시 구현 — `test_filters_gradcheck`, `test_solvers_cantilever` 대응)

중심차분 (2차 정확도):

$$
\boxed{
\left(\frac{\partial f}{\partial x_j}\right)_{\!FD}
= \frac{f(\mathbf{x}+h\mathbf{e}_j) - f(\mathbf{x}-h\mathbf{e}_j)}{2h} + \mathcal{O}(h^2).
}
$$

- $h = 10^{-6}$ (밀도 스케일 $10^{-3}$이면 $h=10^{-7}$도 병행).
- 판정: 상대오차 $\le 10^{-4}$ (투영 $\beta\le 2$인 영역), $\beta$가 큰 영역은 절댓값이 작아지므로 절댓값+상대값 혼합 판정.
- 검사 대상 3종 (각각 독립):
  1. $\partial c/\partial\bar{\mathbf{x}}$ (필터 없음, 소규모 메쉬 예: $8\times 4$).
  2. $\partial c/\partial\mathbf{x} = \mathbf{W}^T\nabla_{\bar{\mathbf{x}}}c$ (밀도 필터만, $\beta$ 없음).
  3. $\partial c/\partial\mathbf{x} = \mathbf{W}^T\mathbf{F}\nabla_{\bar{\mathbf{x}}}c$ (필터+투영, $\beta=2$, $\eta=0.5$).
- 주의: OC 이동제한·이분법은 끄고($\mathbf{x}$ 고정) 순수 함수+기울기만 비교한다. 투영 임계점($\tilde{x}_e=\eta$)에서는 미분 불연속이 아니라 급경사이므로 FD 스텝이 임계점을 가로지르면 오차가 커진다 — 검증용 $\mathbf{x}$는 균일장+$10^{-3}$摂動으로 선택한다.

---

## 6. 의사코드 + 하이퍼파라미터표

### 6.1 의사코드 (04 부록 + 25 RL.1–RL.4 통합, Python 표기)

```python
# 입력: nelx, nely, volfrac f, penal p, rmin, ft(1=밀도필터, 2/3=투영), eta, beta, move, maxit
E0, Emin, nu = 1.0, 1e-9, 0.3
Ke0 = build_Q4_unit_stiffness(nu)          # 04 부록 line 8-12 리터럴
edofMat, iK, jK = build_assembly_index(nelx, nely)  # 25 §3.1: 하삼각 절반만 + int32
W, Hs = build_cone_filter(nelx, nely, rmin)          # w=max(0,rmin-dist), Hs=sum(W,axis=1)
# 또는 Helmholtz: KF, TF 조립 후 LF=cholesky(KF); W 대신 (TF',LF) 연산자 사용
# 또는 imfilter fast-path: h 커널 + Hs=imfilter(ones,h,bcF)

x = full(f, (nely, nelx))                  # 균일 초기화; passive 있으면 능동집합만 조정
xPhys, xOld, loop, change = x.copy(), 1.0, 0, 1.0
while change > 0.01 and loop < maxit:      # 25는 ch=||xPhys-xOld||/sqrt(N) > 1e-6
    loop += 1
    # RL.1 물리장
    xTilde = (H @ x.ravel()) / Hs          # 밀도 필터; PDE면 TF'*(LF'\(LF\(TF*x)))
    xPhys = xTilde.copy()
    if ft in (2, 3):
        if ft == 3:
            eta = newton_volume_preserving_eta(xTilde, eta, beta, f)  # §4.3, tol 1e-6
        dproj = dH_dx(xTilde, eta, beta)   # §4.2 도함수
        xPhys = H_proj(xTilde, eta, beta)
    # RL.2 조립+해석
    sK = Emin + xPhys**p * (E0 - Emin)
    K = sparse(iK, jK, (Ke0.ravel()[:,None]*sK).ravel())
    K = (K + K.T)/2                        # 25는 하삼각+cholesky lower로 대체 가능
    U[free] = solve(K[free][:,free], F[free])
    # RL.3 민감도
    ce = sum((U[edofMat] @ Ke0) * U[edofMat], axis=1)  # 요소 변형에너지
    dc = -p*(E0-Emin)*xPhys**(p-1) * ce     # ∂c/∂xbar
    dv = ones_like(xPhys)                   # ∂V/∂xbar
    dc = backfilter(dc * dproj, W, Hs)      # W^T F dc; 민감도필터면 x 가중 포함 (§3.1a)
    dv = backfilter(dv * dproj, W, Hs)
    # RL.4 OC 업데이트 (이분법은 xPhys 체적 기준; 체적보존 필터면 x 기준과 동일)
    lam = bisect_lambda(x, dc, dv, move, f) # 초기구간 [0, mean(ocP)/f] (25 §3.2 단축)
    x = oc_update(x, dc, dv, lam, move)
    # continuation
    p = cnt(p, penalCnt, loop); beta = cnt(beta, betaCnt, loop)
    change = max(abs(x - x_prev))
```

OC 업데이트식 (04 eq.(3), 25 eq.(10)–(11)):

$$
x_e^{\text{new}} =
\begin{cases}
\max(0, x_e-m), & x_e B_e^{\eta} \le \max(0,x_e-m),\\
\min(1, x_e+m), & x_e B_e^{\eta} \ge \min(1,x_e+m),\\
x_e B_e^{\eta}, & \text{그 외},
\end{cases}
\qquad
B_e = \frac{-\partial c/\partial x_e}{\lambda\,\partial V/\partial x_e},
$$

$\eta_{\text{damp}} = 1/2$ (04), 25는 $F_{k,e}=x_{k,e}(-\partial_e c_k/\lambda_k\partial_e V_k)^{1/2}$ 표기.
$\lambda$ 초기구간 단축 (25 eq.(19)):

$$
\boxed{
\lambda^{\#}_k = \left[\frac{1}{mf}\sum_{e=1}^{m} x_{k,e}
\left(\frac{-\partial_e c_k}{\partial_e V_k}\right)^{1/2}\right]^2,
\qquad \Lambda^{(0)}_k = [0,\, \lambda^{\#}_k].
}
$$

이분법 횟수가 약 50% 감소한다 (25 §3.2).

### 6.2 하이퍼파라미터표 (코드 기본값 그대로)

| 파라미터 | 기호 | 04 기본값 | 25 기본값 | 허용 범위/지침 |
|---|---|---|---|---|
| 체적분율 | $f$ | $0.5$ (MBB) | $0.5$ (MBB), $0.12$ (3D 캔틸레버), $0.2$ (프레임) | $(0,1)$ |
| 페널티 | $p$ | $3$ | $3$ (continuation $1\to 3$, 25마다 $+0.25$) | $[1,5]$, $p\ge 3$ 권장 |
| 필터 반경 | $r_{\min}$ | $2.4/6/16$ (메쉬별, 설계영역 폭의 $0.04$) | $4/8/16$ (MBB 3종), $8.75$ (Sigmund 2007 mesh-independent) | $>1$ 요소; 클수록 회색 감소·해상도 저하 |
| PDE 길이 | $R_{\min}$ | $r_{\min}/2\sqrt{3}$ | — | 위 식 고정 |
| 투영 선예도 | $\beta$ | $1\to 512$ (2배씩) | $2\to 16$ (25마다 $+2$) | $[0,512]$; 큰 값은 continuation 필수 |
| 투영 임계값 | $\eta$ | — (지수형은 불필요) | $0.5$ 또는 $\eta^*$ Newton 탐색 | $[0,1]$ |
| 이동제한 | $m$ | $0.2$ | `move` (통상 $0.2$) | $(0,1]$ |
| 감쇠 | $\eta_{\text{damp}}$ | $1/2$ | OC 지수 $1/2$에 내장 | 고정 |
| void 강성 | $E_{\min}$ | $10^{-9}$ | $10^{-9}$ | $[10^{-12},10^{-6}]E_0$ |
| 수렴 판정 | change | $\lVert\mathbf{x}^{\text{new}}-\mathbf{x}\rVert_\infty<0.01$ | $\lVert\bar{\mathbf{x}}-\bar{\mathbf{x}}_{\text{old}}\rVert_2/\sqrt{N}<10^{-6}$ | 둘 중 하나 선택·기록 |
| 최대 반복 | maxit | — (change만) | 명시 (예: 100 성능측정, 2500 수렴측정) | 필수 지정 권장 |

---

## 7. `wht_topopt` 매핑표

| 본 문서 요소 | `wht_topopt` 모듈 | 함수/클래스 | 비고 |
|---|---|---|---|
| Modified SIMP/RAMP 보간 | `algorithms/simp.py` | `material_model(rho, p, q, E0, Emin)`, `dEdrho` | SIMP/RAMP 전환자, $p$/$q$ continuation 지원 |
| 콘 필터 $\mathbf{W}$ | `filters/spatial.py` | `ConeFilter(rmin)` (KD-Tree+CSR 캐싱) | 균일직사각형 fast-path: 컨볼루션 허용 |
| Helmholtz PDE 필터 | `filters/helmholtz.py` | `HelmholtzFilter(Rmin)` (`KF`, `TF`, Cholesky/반복해) | 비정렬 3D 메쉬용 기본값 (spec §1) |
| Heaviside 투영 (지수/tanh) | `filters/projection.py` | `HeavisideProjection(beta, eta, mode)` | `dHs` 수정 포함, $\eta^*$ Newton 내장 |
| 필터 연산자 합성·역방향 체인룰 | `filters/pipeline.py` | `FilterPipeline`, `FilterOperator.forward/backward_sensitivity` | $\nabla_{\mathbf{x}}=\mathbf{W}^T\mathbf{F}\nabla_{\bar{\mathbf{x}}}$ 자동 연쇄 |
| 컴플라이언스 수반 민감도 | `sensitivity/compliance.py` | `compliance_sensitivity(U, Ke0, dEdrho)` (einsum 벡터화) | $\mathbf{u}_e^T\mathbf{k}_0\mathbf{u}_e$ 일괄 계산 |
| OC 옵티마이저 | `optimizers/oc.py` | `OCUpdate(move, eta_damp)`, `bisect_lambda` | $\lambda^{\#}$ 초기구간 단축 포함 |
| FD 검증 | `tests/test_filters_gradcheck.py` | 중심차분 대조 (§5.3 3종) | 필터·투영 야코비안 검증 |
| 솔버 일치성 | `tests/test_solvers_cantilever.py` | 1차/2차 요소 컴플라이언스·민감도 일치 | Kratos/FrontISTR/CalculiX 어댑터 공통 |

---

## 8. 수치 pitfalls (구현자가 반드시 피할 것 — 원문 근거 포함)

1. **민감도 필터 분모 $0$**: $x_e=0$에서 나눗셈 발산 → $\max(\gamma,x_e)$, $\gamma=10^{-3}$ (04 eq.(7)). Modified SIMP에서 도입된 항이며 classical SIMP 코드에는 없다.
2. **이분법의 체적검사 대상**: 비체적보존 필터(고정 $\eta$ 투영, Heaviside 지수형 초기장)에서는 반드시 $\bar{\mathbf{x}}$(또는 $\hat{\mathbf{x}}$) 기준으로 검사한다. $\mathbf{x}$ 기준으로 검사하면 $\eta<0.5$에서 체적 초과, $\eta>0.5$에서 체적 미달이 발생한다 (25 §3.2). 체적보존(밀도·PDE·$\eta^*$) 필터에서만 $\mathbf{x}$ 기준이 허용된다.
3. **이분법 내 재필터 금지**: 체적보존 필터를 쓰면 이분법 루프 안에서 필터를 다시 적용하지 않는다. 04 Table 2의 300×100 밀도필터 $5.67$s/iter 중 상당 부분이 이 때문이다. 25는 $\lambda^{\#}$ 추정+체적보존으로 $t_U$ 비중을 $4$–$5\%$로 낮췄다.
4. **대칭 조립 후 평균화**: `K=(K+K')/2`를 생략하면 MATLAB이 Cholesky 대신 LU를 선택해 해석이 느려진다 (04 §3.1). 25는 하삼각만 조립+`decomposition(…,'chol','lower')`로 대체한다.
5. **PDE 필터의 Heaviside 직접 결합 금지**: PDE 커널은 지수 감쇠형이라 compact support가 없어 Heaviside의 길이 스케일 보장이 성립하지 않는다 (04 §5.4). 길이 스케일이 필요하면 고전 콘 필터+투영을 쓴다.
6. **지수형 Heaviside의 체적 위반 초기장**: `xPhys=1-exp(-beta*xTilde)+xTilde*exp(-beta)` 초기화는 체적제약을 어긴 채 시작하므로 OC가 교정하도록 둔다 (04 §5.4). 수렴 판정을 $\mathbf{x}$가 아닌 $\bar{\mathbf{x}}$ 기준으로 하면 이 과도기를 정확히 포착한다.
7. **$\beta$ continuation 없는 고선예도 시작**: $\beta$를 큰 값에서 시작하면 국부최적에 갇힌다. 반드시 $\beta=1$–$2$에서 시작해 점진 증가 (04: 2배씩 최대 $512$ / 25: $+2$씩 최대 $16$) (04 §5.4, 25 §3).
8. **s-size 메모리**: `iK/jK/sK`는 강성행렬 자체보다 크다. $300\times 100$이 04 코드 한계, conv2/imfilter는 $700\times 233$까지 가능했다 (04 §6.2). 25의 절반조립+int32+fsparse는 메모리 $1/2$–$1/8$로 감축한다 (25 Table 1).
9. **FD 검증 시 투영 임계점 회피**: §5.3의 주의사항. 검증용 설계는 균일장+미소摂動으로 잡고, $\beta\le 2$에서 먼저 통과시킨 뒤 $\beta$를 올린다.
10. **3D 필터 경계조건**: Dirichlet(0) 필터는 경계 요소를 깎아내려 열등한 국부최적($c=4074.3$ vs $3993.7$)을 만든다. Neumann(`'symmetric'`)을 기본값으로 한다 (25 §4).

---

## 9. 재현 체크리스트 (코드 구현자가 원문 없이 확인할 것)

- [ ] MBB $60\times 20$, $f=0.5$, $p=3$, $r_{\min}=2.4$: 민감도 필터 $c\approx 216.8$, 밀도 필터 $c\approx 233.7$ (04 Fig. 3).
- [ ] 동일 문제 PDE 필터: 민감도 $c\approx 218.8$, 밀도 $c\approx 237.6$ (04 Fig. 4, 고전 필터와 유사하나 불일치 — 커널 감쇠형 차이).
- [ ] Heaviside(지수형, $r_{\min}=0.03\times$폭): $60\times 20$에서 $c\approx 189.1$ (04 Fig. 8).
- [ ] 속도: $150\times 50$에서 88-line이 99-line 대비 약 $100$배 (04 Table 1); top99neo가 top88 대비 $2.66$–$5.5$배 (25 Table 3).
- [ ] 체적보존 Newton: $|\bar{V}-f|<10^{-6}$, 1–2회 수렴, $\eta^*\in[0.4,0.52]$ (25 §3.2 Fig. 4a).
- [ ] FD 3종 통과 (허용오차 §5.3).

*문서 끝. 다음 문서 `02_optimizers_oc_mma_robust_ko.md`에서 OC/MMA/GCMMA·강건 정식화를 다룬다.*
