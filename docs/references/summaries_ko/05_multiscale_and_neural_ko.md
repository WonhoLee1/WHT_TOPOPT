# 05 멀티스케일·신경망 위상최적화 (Multiscale & Neural TO)

## 0. 원문 상태와 증거 등급 (먼저 읽을 것)

| 번호 | 요청 라벨 | 실제 확보 내용 | 상태 |
|---|---|---|---|
| 19, 20 | Groen & Sigmund 2018 후속 2건 | 미확보 (다운로드 목록에 없음) | **[미확보]** 본 요약에서는 Groen & Sigmund 2018 계열 정식 경로를 직접 인용하지 않고, 다운로드된 21번 내 인용 수식(균질화·dehomogenization 인용链: Pantz–Trabelsi 2008, Rumpf–Pazos 2012, Groen & Sigmund 2018, Groen et al. 2020) + 표준 수치균질화 정식으로 **교차 정리**했음을 명시한다 |
| 21 | Spinodal Dehomogenization 2025 | 실제 파일은 Altassan–Wang–Sanders, PolyAnisoMat, SMO 2026, 50p (다방성·가변 상대밀도/방향성 교육 코드) | **[파일명–내용 불일치]** 스피노달 전용 논문이 아님. 단, Spinodal70 재료·Andreassen & Andreasen 2014 전산균질화·상대밀도/방향 피팅식·dehomogenization 인용을 포함하므로 균질화→매크로 최적화→후처리链 근거로 사용 |
| 22a | GINOT 2025 | 실제 파일은 GA-VINO (Mindlin–Reissner 판 기하인식 변분 물리정보 신경연산자, 32p) | **[파일명–내용 불일치]** GINOT 본편이 아니나, p.5–6 Fig.2 + 식 (14)에 GINOT 구조(경계 포인트클라우드 기하 인코더 + 연산자 디코더 + 보조장 인코더, 어텐션 정의)를 직접 기술하므로 GINOT 인터페이스 근거로 사용 |
| 22b | TONIC 2026 | 실제 파일은 광도파관 전벡터 모드해석 PINO (Eng. with Computers 2026, 22p) | **[파일명–내용 불일치]** TONIC 조건·정제 루프 기술을 포함하지 않음. TONIC 항목은 METHODS_SURVEY_2026 §5(E) + Hoyer/NTopo/GM-TOuNN의 신경망→결정론 정제 패턴을 근거로 **[교차정리]** 표기하고, 인터페이스 계약 형태로만 기술 |
| GM | Chandrasekhar et al., GM-TOuNN, arXiv:2204.06682 | 16p 전문 확보 | **[읽음]** §2–3 전부 |
| Hoyer | Hoyer et al., Neural reparameterization, arXiv:1909.04240 | 6p 전문 확보 | **[읽음]** 전부 |
| NTopo | Zehnder et al., NTopo, arXiv:2102.10782 | 19p 전문 확보 | **[읽음]** §4–5 중심 |
| Survey | METHODS_SURVEY_2026.md | §2(A), §5(E), §12 | **[읽음]** 신경망·생성사전·TOuNN/GM-TO 언급부 |

본 파일의 모든 수식은 생략 없이 전개한다. 표준 균질화식·스피노달 가우시안 랜덤장·임계값·GINOT 어텐션·GM-TOuNN 소프트맥스·Hoyer 체적보존 sigmoid·NTopo SIREN 정의는 아래에 완전한 형태로 적는다.

---

## 1. 수치균질화: 단위셀, 주기 경계, 유효 텐서와 민감도

### 1.1 단위셀과 미시 변위장

단위셀 $Y \subset \mathbb{R}^d$ ($|Y|$는 셀 체적/면적), 미시 좌표 $y \in Y$,
거시 변형률 $\varepsilon^0$이 주어질 때 미시 변위 $\chi^{kl}(y)$는 셀 문제(cell problem)의 해이다.
21번(PolyAnisoMat)이 전산균질화 구현으로 명시 인용한 Andreassen & Andreasen 2014 방식이 바로 이 셀 문제의
유한요소 풀이이며, 21번에 동봉된 Spinodal70·SC/BCC/Oct/Kelvin 격자 특성치가 이 경로로 계산된 예이다.

### 1.2 주기 경계조건: 마스터–슬레이브와 $P_{\mathrm{pbc}}$

주기성은 변위의 점프가 거시 변형률에 의한 것만 허용되는 조건이다:

$$u^+ - u^- = \varepsilon^0 \cdot (y^+ - y^-) \quad \text{on } \partial Y.$$

이산화 후 종속(슬레이브) 자유도를 독립(마스터) 자유도로 소거하는 투영 행렬을 $P_{\mathrm{pbc}}$라 적으면,
주기 구속이 걸린 강성행렬·하중은

$$K_{\mathrm{per}} = P_{\mathrm{pbc}}^T K P_{\mathrm{pbc}}, \quad f_{\mathrm{per}} = P_{\mathrm{pbc}}^T f$$

로 축소된다. 구현상으로는 반대면 노드 쌍을 마스터–슬레이브로 묶고 한쪽 모서리(2D)·한쪽 모서리선(3D)의
강체 모드를 고정한다. 21번 본문은 이 행렬을 직접 적지 않으나, 인용한 Andreassen & Andreasen 2014 코드가
정확히 이 마스터–슬레이브 소거이며, PolyAnisoMat의 `MatProperties`가 그 출력(밀도·방향별 텐서 표)을 입력으로 받는다.

### 1.3 하중 경우: 2D 3개, 3D 6개의 $\varepsilon^0_{(kl)}$

유효 텐서의 독립 성분 수만큼 단위 거시 변형을 가한다.

2D (평면응력/평면변형, Voigt 순서 $[11, 22, 12]$):

$$\varepsilon^0_{(11)} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad \varepsilon^0_{(22)} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}, \quad \varepsilon^0_{(12)} = \begin{pmatrix} 0 & 1/2 \\ 1/2 & 0 \end{pmatrix}.$$

3D (Voigt 순서 $[11, 22, 33, 23, 13, 12]$) 6개:

$$\varepsilon^0_{(11)},\ \varepsilon^0_{(22)},\ \varepsilon^0_{(33)},\ \varepsilon^0_{(23)},\ \varepsilon^0_{(13)},\ \varepsilon^0_{(12)}$$

이며 각 텐서는 해당 Voigt 슬롯에만 1(전단 슬롯은 공학전단 정의에 따라 $1/2$ 대칭 배치)을 갖는다.
각 경우의 등가節点 하중이 다음 항의 $f_e^{kl}$이다.

### 1.4 요소 하중과 유효 텐서

셀을 $C^e$ (미시 요소 탄성행렬), $B$ (변형률–변위 행렬)로 이산화하면 $kl$ 하중 경우의 요소 하중 벡터는

$$f_e^{kl} = \int_{\Omega_e} B^T C^e \, \varepsilon^0_{(kl)} \, d\Omega.$$

셀 문제를 풀어 특성 변위 $\chi_e^{kl}$를 얻으면 균질 유효 탄성텐서는 체적 평균 에너지 등가로

$$E^H_{ijkl} = \frac{1}{|Y|} \sum_e \int_{\Omega_e} \left(\varepsilon^0_{(ij)} - \varepsilon(\chi_e^{ij})\right)^T C^e \left(\varepsilon^0_{(kl)} - \varepsilon(\chi_e^{kl})\right) d\Omega$$

$$= \frac{1}{|Y|} \sum_e \left( (u_e^{0,ij} - \chi_e^{ij})^T k_e \, (u_e^{0,kl} - \chi_e^{kl}) \right)$$

이다 ($u_e^{0,ij}$는 $\varepsilon^0_{(ij)}$를 요소 절점에 전개한 벡터, $k_e = \int B^T C^e B\,d\Omega$).
등방성 베이스 + SIMP류 보간 $C^e(\hat{\rho}) = E(\hat{\rho})\,C_0$ 하에서 설계변수(물리 밀도 $\hat{\rho}$)에 대한 민감도는

$$\frac{\partial E^H_{ijkl}}{\partial \hat{\rho}_e} = \frac{1}{|Y|} \int_{\Omega_e} \left(\varepsilon^0_{(ij)} - \varepsilon(\chi_e^{ij})\right)^T \frac{\partial C^e}{\partial \hat{\rho}_e} \left(\varepsilon^0_{(kl)} - \varepsilon(\chi_e^{kl})\right) d\Omega$$

이며, 수반법상 셀 문제의 자기수반(self-adjoint) 구조 덕분에 추가 수반해 없이 위 6(3)개 해만으로 모든
$(ij,kl)$ 조합의 민감도가 닫힌다. 21번은 최적화 루프 안의 반복 균질화를 피하기 위해 이 $E^H$ 표를
상대밀도 $r$·방향 $\theta$의 해석 피팅함수 $F_i(r,\theta)$로 먼저 근사한다(§1.5).

### 1.5 21번의 피팅 경로 (균질화표를 루프 밖으로)

21번 식 (21)–(25). 각 재료 $i$의 텐서 성분을 기저–계수 분리형으로 둔다:

$$F_i(r, \theta) = \sum_{s=0}^{n} [A_i]_s \, [b_i(r,\theta)]_s, \tag{21}$$

밀도 방향 3차 다항식

$$F_i(r) = [A_i]_0 + [A_i]_1 r + [A_i]_2 r^2 + [A_i]_3 r^3, \tag{23}$$

방향 2항 푸리에 급수 ($h = 2\pi/T$, $T$는 재료 주기)

$$F_i(\theta) = [A_i]_0 + [A_i]_1\cos(h\theta) + [A_i]_2\sin(h\theta) + [A_i]_3\cos(2h\theta) + [A_i]_4\sin(2h\theta), \tag{24}$$

결합형

$$\begin{aligned} F_i(r,\theta) &= [A_i]_0 + [A_i]_1 r + [A_i]_2 r^2 + [A_i]_3 r^3 \\ &\quad + [A_i]_4\cos(h\theta) + [A_i]_5\sin(h\theta) + [A_i]_6\cos(2h\theta) + [A_i]_7\sin(2h\theta) \\ &\quad + [A_i]_8 r\cos(h\theta) + [A_i]_9 r\sin(h\theta) + [A_i]_{10} r^2\cos(h\theta) + [A_i]_{11} r^2\sin(h\theta). \end{aligned} \tag{25}$$

계수 행렬 $[A_i]_s$와 $B^T(\cdot)B$ 적분을 최적화 전에 한 번만 수행하면 요소 강성은 내적(dot product)으로
환원되므로 매 반복 수치적분이 사라진다. 즉 §1.4가 "정확한 표를 만드는 법"이라면 §1.5는 "그 표를 루프 밖으로
빼는 법"이다. 방향 설계변수는 주기 덫을 피하기 위해 최급강하 + 모듈러 래핑으로 갱신된다(21번 식 (47)–(48)):

$$s^{(k+1)} = \max(0.99\,s^{(k)},\, 0.01), \tag{47}$$

$$\theta^{(k+1)}_{\ell i} = \mathrm{mod}\!\left({\theta'}^{(k+1)}_{\ell i} - \underline{\theta}_i,\; \overline{\theta}_i - \underline{\theta}_i\right) + \underline{\theta}_i. \tag{48}$$

---

## 2. 디호모지나이제이션과 스피노달: 거시 장 → 단일 스케일 실형상

### 2.1 디호모지나이제이션 파이프라인 (19/20 교차정리 선언)

19/20번(Groen & Sigmund 2018 후속)이 미확보이므로, 정식 디호모지나이제이션 파이프라인은 21번이 인용한
문헌链(Pantz–Trabelsi 2008; Rumpf–Pazos 2012; Groen & Sigmund 2018; Groen et al. 2020)과
표준 절차의 교차 정리로 기술한다 **[교차정리]**:

1. 거시 최적화에서 상대밀도장 $\rho_M(x)$와 방향장 $\theta(x)$를 얻는다(§1의 균질 재료로 최적화).
2. 방향장 $\theta(x)$에서 적분선(스트라이프·격자 궤적)을 추적하거나, 투영 연산자로 미세 패치를 생성한다.
3. 밀도장 $\rho_M(x)$를 국소 파장·두께로 변환하여 실형상을 잘라낸다.
4. 경계·연결성을 검사하고 체적·컴플라이언스 오차를 거시 예측과 비교한다.

21번이 "후처리(post-processing) 기법과 결합될 것"이라 적시한 대상이 바로 이 단계이다.

### 2.2 스피노달 가우시안 랜덤장: 생성식과 임계값 (수식 생략 금지)

스피노달형 미시구조는 $N_w$개 평면파의 중첩으로 만든 정상 가우시안장 $G(x)$를 목표 체적분율 $\rho_M$에 맞는
레벨 $\tau$로 자른 것으로 등가 생성한다. 파수 벡터 $k_m$, 위상 $\phi_m \sim \mathcal{U}[0,2\pi)$,
$|k_m| \approx k_0$ (단일 특성 파수 대역)라 두면

$$G(x) = \sqrt{\frac{2}{N_w}} \sum_{m=1}^{N_w} \cos\!\left(k_m \cdot x + \phi_m\right).$$

각 $\cos$ 항의 분산이 $1/2$이므로 규격화 인자 $\sqrt{2/N_w}$에 의해 $G(x) \sim \mathcal{N}(0,1)$ (점별 표준정규),
서로 다른 점 사이 공분산은 파수 스펙트럼의 푸리에 변환(특성상관함수)이 된다.
고체 영역을 $\{x : G(x) > \tau\}$ (또는 부호 반전 규약 $\{G < \tau\}$; 한 규약으로 고정하고 체적식을 맞춘다)로
정의할 때, 표준정규 CDF $\Phi$에 의해 고체분율= $\rho_M$ 조건은

$$P(G > \tau) = 1 - \Phi(\tau) = \rho_M \quad \Longrightarrow \quad \tau(\rho_M) = \Phi^{-1}(1 - \rho_M) = \sqrt{2}\,\mathrm{erf}^{-1}(1 - 2\rho_M)$$

이다. 확인: $\rho_M = 0.5 \Rightarrow \tau = 0$; $\rho_M = 0.7 \Rightarrow \tau = \sqrt{2}\,\mathrm{erf}^{-1}(-0.4) \approx -0.370$ (21번의 Spinodal70 예와 부호 규약 일치 필요 — $\{G > \tau\}$ 규약에서 양의 고체분율이 $\tau < 0$ 쪽으로 열리는 것은 정상이다).

거시 밀도장 $\rho_M(x)$가 공간 가변일 때 임계값도 공간 가변 $\tau(x) = \tau(\rho_M(x))$이 되며,
국소 파수 $k_0(x)$를 굵기·이방성 요구에 연동하면 graded spinodal이 된다.
이 $G(x)$–$\tau(\rho_M)$ 쌍이 본 요약이 요구한 필수 수식이며, 어떤 경우에도 $\sqrt{2/N_w}$ 규격화와
$\sqrt{2}\,\mathrm{erf}^{-1}(1-2\rho_M)$ 임계식을 빼고 적지 않는다.

---

## 3. 등급 멀티스케일과 신경망 재매개화: GM-TOuNN, Hoyer, NTopo

### 3.1 GM-TOuNN (Chandrasekhar et al. 2022) — 기저 전환을 분류기로

미리 뽑은 $M$개 미세구조 후보를 매 점에서 "어느 것을 얼마나 쓸지"로 푼다.
신경망 입력은 공간 좌표 $x \in \mathbb{R}^d$, 출력은 조성 $\rho = \{\rho_1,\dots,\rho_M\}$ + 체적분율 $v$:

- 조성 출력 활성화 = 소프트맥스 → 분할통일(partition of unity) 자동 만족

  $$\rho_i(x;w) = \frac{\exp(z_i(x;w))}{\sum_{j=1}^{M}\exp(z_j(x;w))}, \quad \sum_{i=1}^{M}\rho_i = 1,\;\; 0 \le \rho_i \le 1.$$

- 체적분율 출력 활성화 = 시그모이드 → $0 \le v(x;w) \le 1$, 추가 제약 불필요.
- 설계변수는 격자 밀도가 아니라 네트워크 가중치·편향 $w$ 전부: $\rho(x;w)$, $v(x;w)$.
- 최적화 문제는 $\min_w J(w) = f^T u(w)$ s.t. $K(w)u = f$ + 로그-배리어 체적벌칙

  $$L(w) = J + \psi(g_v), \quad \psi_t(g) = \begin{cases} -\frac{1}{t}\log(-g), & g \le -\frac{1}{t^2}, \\ t\,g - \frac{1}{t}\log\!\left(\frac{1}{t^2}\right) + \frac{1}{t}, & \text{otherwise}, \end{cases} \tag{9--10}$$

  ($t$는 반복마다 키워 제약을 조인다.)
- 민감도는 JAX 위 end-to-end 자동미분(FEA 포함 전체 계산을 미분 가능하게 작성)으로 수해석을 제거한다.
  옵티마이저는 Adam. 은닉층은 Swish ($x\cdot\mathrm{sigmoid}(x)$) 밀집층.

### 3.2 Hoyer et al. 2019 — 재매개화가 곧 정칙화

격자 밀도 $x$ 직접 최적화 대신 CNN 가중치 $w$ 최적화: $x = \mathrm{CNN}(w)$.
핵심은 무제약 로짓 $\hat{x}$ → 체적 정확 만족 밀도로 가는 **제약 sigmoid 변환**을 순방향에 박은 점이다:

$$x_{ij} = \frac{1}{1 + \exp\!\left[\hat{x}_{ij} - b(\hat{x}, V_0)\right]}, \quad \text{s.t. } V(x) = V_0, \tag{2}$$

여기서 $b(\hat{x},V_0)$는 체적조건을 맞추는 스칼라 오프셋으로 이분탐색으로 풀고, 역방향은 최적점에서
음함수 미분으로 통과시킨다. MMA/OC는 신경망 학습에 부적합하므로 L-BFGS(신경망 측)·MMA(픽셀 측) 대조로
116개 과제에서 신경망 재매개화가 최고설계 빈도 약 50% 향상을 보고한다. 단, Sanu et al. 2024 주의점
(부드러운 문제에서 수렴 지연 가능, 아키텍처 의존성)을 METHODS_SURVEY_2026 §2가 병기한다.

### 3.3 NTopo (Zehnder et al. 2021) — SIREN 두 개 + 몬테카를로

변위장 $\Phi_u(\omega;\gamma)$와 밀도장 $\Phi_\rho$를 각각 SIREN-MLP(사인 활성화 다층퍼셉트론)로 둔다:

$$\Phi(x) = W_l(\phi_{l-1} \circ \phi_{l-2} \circ \cdots \circ \phi_0)(\omega_0 x) + b_l, \quad \phi_i(x) = \sin(W_i x + b_i). \tag{5}$$

교대 학습: (a) $\Phi_u$는 선형탄성 전퍼텐셜 최소화(정평형), (b) 밀도 공간 기울기→민감도 필터링→목표장
$\hat{\rho}$ 작성, (c) $\Phi_\rho$는 볼록 토폴로지 목적 $L_{\mathrm{topo}} = \|\rho - \hat{\rho}\|^2$ 최소화.
적분은 몬테카를로 샘플링(메시 없음). 표준 메시 기반 솔버와 대등한 컴플라이언스를 보고하며,
연속 설계공간(파라미터 범위 전체)의 자기지도 학습으로 확장된다.

---

## 4. 연산자 학습 인터페이스: GINOT와 TONIC

### 4.1 GINOT (22a 파일 내 Fig.2·식(14) 근거, [읽음])

22a는 GINOT(Geometry-Informed Neural Operator Transformer)을 선행 연산자로 소개하면서 구조를
그대로 적시한다: **경계 포인트클라우드 기하 인코더 + 질의점 연산자 디코더 + 보조 물리장 인코더**.
계약:

- 입력 1 (기하): 계산영역 경계 포인트클라우드 → 샘플링/그룹화(S/G) → 지오메트리 인코더.
- 입력 2 (물리): 재료장·하중장·스칼라 파라미터용 추가 인코더(22a의 GA-VINO는 재료/하중/스칼라 3분기를 둠).
- 디코더: 질의점(query point) + 위치인코딩 + MLP → 해당 점의 장 예측.
- 융합 연산자: 크로스어텐션. 어텐션 정의(22a 식 (14)):

  $$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_e}}\right)V, \tag{14}$$

  $Q \in \mathbb{R}^{N_q \times d_e}$ (질의), $K,V$ (인코딩된 기하·물리 토큰).
- 출력: 질의점에서의 상태장(22a 구현은 Mindlin–Reissner 판의 처짐·회전; GINOT 일반형은 임의 장).
- 학습: 변분汎함수(이산화 전퍼텐셜 에너지) 기반 물리 구동 비지도 학습 가능 — 라벨 불필요 경로가 열려 있다.
  서로 다른 이산화 집합(입력 클라우드 ≠ 질의점) 허용, 공통 격자 보간 강제 없음.

구현 계약(호스트 쪽): `encode_geometry(boundary_cloud) → tokens_G`, `encode_physics(fields, scalars) → tokens_P`,
`decode(query_xyz, tokens_G, tokens_P) → field(query)` 세 함수 + 어텐션식 (14) 고정.
이 셋이 GINOT 인터페이스의 전부이며, 위상최적화 반복 안에서는 "재메시·재해석 없는 대리 상태장"으로 쓴 뒤
반드시 결정론 해석으로 재평가한다(§5).

### 4.2 TONIC (파일 22b에 없음 — [교차정리] 계약 정의)

22b 파일은 광도파관 PINO이며 TONIC 기술을 포함하지 않는다. 따라서 TONIC은 본 요약에서
**신경망 초기치 + 결정론 정제(MMA/OC) 하이브리드**의 인터페이스 계약으로 정의하고, 근거는
Hoyer(신경망이 좋은 basins 제공) + NTopo(신경망↔결정론 교대) + GM-TOuNN(Adam 탐색 후 물리 정합) +
METHODS_SURVEY_2026 §5(E)("생성·신경망 제안은 재해석 대상이지 결과가 아님")로 명시한다:

```
neural_propose(loads, BCs, vf) -> rho_init            # 신경망: 수 초 내 초기장
deterministic_refine(rho_init, mode={MMA|OC}) -> rho*  # 결정론: 체적·민감도 정합 정제
accept(rho*) iff reanalysis(rho*) improves J and satisfies constraints
```

규칙: (i) 신경망 출력은 항상 `rho_init` 취급, 단독 결과 주장 금지. (ii) 정제는 MMA 또는 OC 중 하나로
고정하고 민감도는 §1.4 또는 AD 기반 참기울기만 사용(ESL류 대리 기울기 사용 시 등급 표기).
(iii) 최종 수치는 정제 후 결정론 재해석값만 인용. 파일 22b의 PINO 전진속도(1–2% 상대오차, GPU 최대 29배)는
연산자 대리의 속도 근거로만 인용하고 TONIC 자체 성능으로 오인하지 않는다.

---

## 5. 통합 검증·한계·구현 수칙

1. **균질화→최적화→실형상** 한 방향 검증: §1.4 표 → 21번 §1.5 피팅 → 거시 최적화 → §2 디호모지나이제이션/
   스피노달 절단 → 체적분율·컴플라이언스를 거시 예측과 대조. 오차는 표 단계·피팅 단계·절단 단계로 귀속한다.
2. **신경망→결정론** 한 방향 검증: §3–§4의 모든 신경망 출력은 제안으로 취급하고 MMA/OC 정제 + 재해석 통과분만
   결과로 기록한다 (Survey §8의 기울기 등급: exact/AD vs surrogate/learned/FD/none 표기 준수).
3. **정직성**: 19/20 미확보와 21·22a·22b 파일명–내용 불일치는 §0 표 그대로 유지한다. TONIC 수치를 22b에서
   가져왔다고 적지 않는다. 스피노달 임계식은 $\{G > \tau\}$ 규약과 함께 적고, 부호 규약을 바꾸면 체적식을
   함께 바꾼다.
4. **구현 인터페이스 최소집합**: `homogenize(cell) → E^H + dE^H/dρ̂` (§1.4), `fit_table(E^H 표) → F_i(r,θ)` (§1.5),
   `spinodal_field(k0, Nw, φ) → G` + `threshold(ρM) → τ` (§2.2), `gm_tounn_step / reparam_step / ntopo_step` (§3),
   `ginot_encode_decode` (§4.1), `tonic_propose_refine` (§4.2). 각 함수는 입출력 물리단위와 근거 등급을 함께 반환한다.

*작성: summaries_ko/05. 원문 대조 완료일 2026-10-31. 인코딩 UTF-8(BOM 없음).*
