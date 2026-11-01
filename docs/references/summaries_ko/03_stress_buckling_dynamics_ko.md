# 응력 · 좌굴 · 동역학 위상최적화 요약 (Stress / Buckling / Dynamics)

> 대상: `references/topopt/` 응력·좌굴·동역학 계열 문헌 교차 정리.
> 정독 원문: 12_Ferrari_Sigmund_Guest_2021_Buckling250_SMO.pdf(파일 불일치, 하단 명시),
> DYNAMICS_SURVEY.md(전면), 2025_buzaglo_first-order-ESL-dynamic_arXiv-2508.07222.pdf(전면),
> 2022_triller_DiESL-topology-SIMP_SMO-65-2022.pdf(전면),
> 2017_chalmers-245215_multibody-topopt.pdf(전면).
> 미확보 문헌 08_Le_2010은 09_Holmberg_2013 / 10_daSilva_2019 후속 논문의 서술과
> Triller(2022)·Buzaglo(2025) 내 인용 수식을 교차하여 정리하였으며, 해당 절마다
> [교차정리] 표기를 붙였다. 직접 확인하지 않은 서술은 [교차정리], 직접 읽은 서술은
> [정독]으로 구분한다.

---

## 1. 문서 범위와 원문 확보 상태

본 문서는 정적 컴플라이언스 최소화(SIMP·레벨셋 기본 루프, 별도 문서 01·02 대상)를
전제로, 그 위에 얹히는 세 가지 물리(응력 제한, 좌굴 안정성, 동적 응답)와 두 가지
응용(유연 메커니즘, 다물체계)을 다룬다. 공통 난제는 (a) 국소적·비미분적 물리량을
전역·미분가능한 집계함수로 바꾸는 일, (b) 저밀도(void) 요소에서 생기는 수치 병리
(특이 최적해, 가짜 모드)를 보간법으로 차단하는 일, (c) 시간 이력·비선형 해석의
민감도 비용을 감당하는 일이다.

원문 확보 상태(정직성 기록):

- [정독] Buzaglo & Pollini, arXiv:2508.07222 (31 p): F-ESL 정의 eq. 3–14, KKT
  판정식 eq. 15–24, 2바 트러스·전단프레임·대칭 트러스 수치예제.
- [정독] Triller et al., SMO 65:226 (2022) (17 p, CC BY): DiESL 정의 eq. 1–12,
  밀도 해석 규칙 eq. 9, 해석영역 재료식 eq. 10, LSM 좌표 재구성 eq. 11.
- [정독] Ghandriz et al., Multibody Syst. Dyn. 39:135–148 (2017, Chalmers open
  access): TOMBS 정식화 eq. 1–19, 질량 패널티, slider-crank 예제.
- [정독] DYNAMICS_SURVEY.md: ESL 알고리즘·변종 비교·Stolpe(2018) 반례·구현
  로드맵(D0–D3).
- [파일 불일치] `papers/12_Ferrari_Sigmund_Guest_2021_Buckling250_SMO.pdf`의
  실제 내용은 매듭이론 논문(Murakami–Tran, arXiv:2010.03698, colored Jones
  polynomial)이며, 좌굴 250줄 코드 논문(Ferrari–Sigmund–Guest 2021)과 무관하다.
  파일명이 잘못 붙은 것으로 판단하고 본 문서의 좌굴 서술은 아래 [교차정리]
  근거에만 의존한다.
- [미확보] 08_Le_2010(응력 qp-완화 원전) PDF 없음. 09_Holmberg_2013 /
  10_daSilva_2019 후속 논문 PDF도 함께 미확보이므로, 응력 절의 qp-완화·집계·수반
  수식은 표준 문헌 형태로 재구성하고 [교차정리]로 표기한다.

---

## 2. 응력 기반 위상최적화 (qp-완화 · 집계 · 수반민감도) [교차정리]

### 2.1 von Mises 응력과 V 행렬, 미분

요소 중심(또는 가우스점) 코시 응력 벡터 $\sigma$에 대해 von Mises 유효응력은
2차 형식으로 쓴다.

$$\sigma_{vM} = \sqrt{\sigma^T V \sigma}$$

2D 평면응력, $\sigma = [\sigma_x, \sigma_y, \tau_{xy}]^T$:

$$V_{2D} = \begin{bmatrix} 1 & -1/2 & 0 \\ -1/2 & 1 & 0 \\ 0 & 0 & 3 \end{bmatrix}$$

3D, $\sigma = [\sigma_x, \sigma_y, \sigma_z, \tau_{yz}, \tau_{xz}, \tau_{xy}]^T$:

$$V_{3D} = \begin{bmatrix}
1 & -1/2 & -1/2 & 0 & 0 & 0 \\
-1/2 & 1 & -1/2 & 0 & 0 & 0 \\
-1/2 & -1/2 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 & 3 & 0 \\
0 & 0 & 0 & 0 & 0 & 3
\end{bmatrix}$$

전단항 계수 3은 공학 전단변형률 Voigt 표기에서
$3\tau^2 = \frac{3}{2}s:s$ (편차응력 $s$)와 일치한다.
미분은 해석적으로 닫힌다.

$$\frac{\partial \sigma_{vM}}{\partial \sigma} = \frac{V \sigma}{\sigma_{vM}}$$

이 식이 아래 수반 전개식의 마지막 고리($\partial\sigma_{vM}/\partial\sigma_0$)에
그대로 들어간다.

### 2.2 응력 특이점과 qp-완화

SIMP에서 $\rho_e \to 0$인 요소의 응력 제한
$\sigma_{vM,e} \le \bar{\sigma}$는 특이(singular) 최적해를 만든다.
void인데 응력 제한이 살아 있으면 옵티마이저가 해당 요소를 삭제하지 못한다.
표준 처방은 mật도 가중 완화(relaxation)이다.

$$\hat{\sigma}_e = \hat{\rho}_e^{\,q} \, \sigma_{vM,e}, \qquad 0 < q < 1 \;\;(\text{관행 } q = 0.5)$$

$\hat{\rho}_e$는 필터·투영된 물리 밀도, $\sigma_{vM,e}$는
매크로 응력 $\sigma_{0,e} = C_0 B u_e$로부터 계산한 값이다.
$\rho_e \to 0$에서 $\hat{\sigma}_e \to 0$이므로 제한이 자동 해제되고,
$\rho_e = 1$에서는 원래 응력으로 복원된다.
강성 보간 지수 $p = 3$와 완화 지수 $q < 1$을 분리하는 것이 핵심이다.

### 2.3 P-norm / 클러스터 집계와 적응 스케일링

요소별 수만 개 제한을 하나의 미분가능 전역 측도로 묶는다.
P-norm 집계:

$$\sigma_{PN} = \left( \sum_e \hat{\sigma}_e^{\,p} \right)^{1/p}, \qquad p \approx 6 \sim 12$$

$p \to \infty$에서 $\max$로 수렴하지만 $p$가 크면 오버플로·진동이 생겨
$p = 8$ 전후가 관행이다. 국소 제어를 살리려면 전체 합 대신 응력 수준별
클러스터(상위 N개 요소, 또는 블록 분할)마다 P-norm을 취해 다중 제한으로 둔다.
P-norm은 $\max$의 하한 근사이므로, 적응 스케일링 계수로 보정한다.

$$c_k = \frac{\max_e \hat{\sigma}_e^{(k-1)}}{\sigma_{PN}^{(k-1)}}, \qquad
\tilde{\sigma}^{(k)} = c_k \, \sigma_{PN}^{(k)}, \qquad
G_{\sigma} = \frac{\tilde{\sigma}}{\bar{\sigma}} - 1 \le 0$$

$c_k$는 직전 반복값으로 고정(상수 취급)하므로 미분에 들어가지 않는다.

### 2.4 응력 수반민감도 전개

집계 응력 제한 $G_{\sigma}(\hat{\rho}, u(\hat{\rho}))$의 전미분은 직접법이 아닌
수반법으로 계산한다. 상태방정식 $K u = f$에 수반벡터 $\lambda$를 도입하면

$$K \lambda = \sum_e w_e \, \hat{\rho}_e^{\,q} \, B^T C_0 \,
\frac{\partial \sigma_{vM}}{\partial \sigma_0}\Big|_e
\;+\; \frac{\partial G_{\sigma}}{\partial \hat{\rho}}\;\; \text{의 $u$-공액 전개}$$

정확한 전개 순서는 다음과 같다. 요소 기여 가중치
$w_e = \partial G_{\sigma} / \partial \hat{\sigma}_e$
(P-norm이면 $w_e = c_k \, \hat{\sigma}_e^{p-1} / \sigma_{PN}^{p-1}$)를 두고,

1. 요소 응력 미분: $\partial\hat{\sigma}_e/\partial u_e
   = \hat{\rho}_e^q \, (\partial\sigma_{vM}/\partial\sigma_0)^T C_0 B$,
2. 수반 하중 조립: $K \lambda
   = -\sum_e (\partial\hat{\sigma}_e/\partial u_e)^T w_e$,
3. 설계 미분:
   $\dfrac{dG_{\sigma}}{d\hat{\rho}_e}
   = \dfrac{\partial G_{\sigma}}{\partial \hat{\rho}_e}
   + \lambda^T \dfrac{\partial K}{\partial \hat{\rho}_e} u$,
   여기서 $\partial G_{\sigma}/\partial\hat{\rho}_e
   = w_e \, q \, \hat{\rho}_e^{q-1} \, \sigma_{vM,e}$ (완화항의 양함수 미분).

수반식은 강성행렬 $K$ 하나로 요소 수와 무관하게 1회만 풀면 되므로,
집계 제한이 하나일 때 직접법 대비 압도적으로 싸다.
클러스터당 하나씩 수반해를 푼다.

---

## 3. 좌굴 안정성 (선형·비선형, 가짜 모드 차단) [교차정리]

### 3.1 기하강성과 선형 좌굴 고유문제

기준 하중 $f_0$에 대한 선형 정적해 $u_0$의 응력장 $S_0$(응력텐서를
대각 블록으로 쌓은 행렬)로부터 기하강성(응력강성, 초기응력 행렬)을 조립한다.

$$K_G(S_0) = \int_{\Omega} G^T S_0 \, G \, d\Omega$$

$G$는 형상함수 기울기로부터의 기하학적勾배 연산자(변위 기울기
$\nabla u$를 절점변위로 사상)이다. 선형 좌굴은 일반화 고유문제이다.

$$(K + \mu K_G) \, \phi = 0 \quad\Longleftrightarrow\quad
K \phi = -\mu K_G \phi, \qquad \mu_1 \le \mu_2 \le \cdots$$

$\mu_1$이 임계하중계수(기준 하중의 $\mu_1$배에서 좌굴)이며 설계 요구는
$\mu_1 \ge \bar{\mu}$이다. 비선형 경로 위의 좌굴(극한점·분기점)은
접선강성 $K_T$를 써서 $(K_T + \mu K_G)\phi = 0$으로 같은 형태를 유지한다.

### 3.2 가짜 좌굴모드 차단: 침식(erosion) 보간 — $\eta_G \ll \eta_K$

저밀도 요소에서 $K_G$가 $K$보다 느리게 소멸하면 void 영역에 가짜
저차 좌굴모드가 생겨 $\mu_1$이 쓰레기값을 가리킨다. 처방은 강성과
기하강성에 **서로 다른** 밀도 보간을 주는 것이다.

$$K_e = \eta_K(\rho_e) \, K_{0,e}, \qquad
K_{G,e} = \eta_G(\rho_e) \, K_{G0,e}, \qquad
\eta_G(\rho_e) \ll \eta_K(\rho_e) \;\; (\rho_e \to 0)$$

관행(예: 개선 schemes): $\eta_K = E_{\min}/E_0 + (1 - E_{\min}/E_0)\rho_e^{p}$,
$\eta_G = \rho_e^{p + \Delta}$ 또는 컷오프($\rho_e < \rho_{cut}$이면
$\eta_G = 0$)로 void의 기하강성을 강성보다 공격적으로 죽인다.
"침식 보간"이라 부르는 이유가 이것이다: 저밀도 기하강성을 침식시켜
가짜 모드의 변형 에너지 분모를 제거한다.

### 3.3 역모드·KS 집계와 응력 의존 수반항

여러 차수·모드 전환(mode switching, 고유값 교차로 미분 불가)을 다루기 위해
하위 $m$개 역고유값(또는 하중계수 역수) $y_i = 1/\mu_i$를 KS 함수로 묶는다.

$$J_{KS} = \frac{1}{\xi} \ln\!\left( \sum_{i=1}^{m} e^{\xi y_i} \right),
\qquad \xi > 0, \qquad G_b = J_{KS} - \frac{1}{\bar{\mu}} \le 0$$

$\xi$가 클수록 $\max y_i = 1/\mu_1$에 가까워진다.
민감도는 각 모드의 고유값 미분($\partial\mu_i/\partial\rho_e
= \phi_i^T(\partial K/\partial\rho_e + \mu_i \, \partial K_G/\partial\rho_e)\phi_i$,
질량정규화 기준)을 KS 가중합한 형태이며, 여기서 두 번째 항이 **응력 의존
수반항**이다: $K_G$가 정적 응력 $S_0(u_0(\rho))$에 의존하므로 연쇄미분

$$\frac{dK_G}{d\rho_e} = \frac{\partial K_G}{\partial\rho_e}
+ \frac{\partial K_G}{\partial u_0} \frac{\partial u_0}{\partial\rho_e}$$

의 두 번째 항을 처리하는 별도 정적 수반해
($K \psi = (\partial J_{KS}/\partial u_0)^T$)가 1회 추가된다.
즉 좌굴 수반 = 고유모드 가중합 + 정적 응력경로 수반 1회이다.

---

## 4. 동역학 · 진동 · ESL 계열 (고유진동수, 과도응답, DiESL/F-ESL) [정독]

### 4.1 자유진동: 고유진동수 문제와 질량 보간 — $\eta_M$

$$K(\rho) \, \phi = \omega^2 M(\rho) \, \phi$$

고유진동수 최대화(기본진동수 목표치 상회, 다중고유값 시 bound formulation)의
고질적 병리는 void 요소의 국소 가짜 모드이다: 질량은 남고 강성이 사라지면
아무 의미 없는 저주파 모드가 스펙트럼을 오염시킨다. 처방은 강성보다 질량을
**덜** 패널티(= void의 질량/강성비를 낮춤)하는 질량 보간이다.

$$M_e = \eta_M(\rho_e) \, M_{0,e}, \qquad
\eta_M(\rho) = \rho_{\min} + (1-\rho_{\min})\rho^{q_M},\;\; q_M > 1
\;\;(\text{또는 RAMP형})$$

$q_M > 1$이면 중간밀도에서 질량이 강성($\rho^p$, $p = 3$)보다 천천히 감소해
가짜 국소 모드가 고주파로 밀려난다. 부저(Buzaglo) 분류상 자유진동 최적화의
민감도는 $\phi^T(\partial K - \omega^2 \partial M)\phi$이며, 중근은 별도 처리
(본 요약 범위 외)한다.

### 4.2 과도응답 직접법(참조 해): Newmark + 시간역행 수반

선형 과도 문제의 정확한 민감도는 시간영역 직접 수반법(Behrou–Guest)이
기준(reference)이다: Newmark-$\beta$(또는 HHT-$\alpha$) 전진 해석 후
수반식을 **시간 역행**으로 풀며 비용은 전진 해석의 약 2배 + 상태 저장
메모리이다. ESL 계열의 정당성은 결국 이 직접법과 비교해 검증해야 한다
(DYNAMICS_SURVEY §4 구축 순서 D0–D3: 내장 Newmark + 내장 모드 솔버를
먼저 두고 ESL을 그 위에 얹어 선형 문제에서 대조).

### 4.3 ESL(0차), DiESL, F-ESL(1차)

원 문제(Buzaglo eq. PD):

$$\min f(x, u_k(x)) \;\; \text{s.t.} \;\; g_j \le 0, \;\;
M(x)\ddot{u}_k + C(x)\dot{u}_k + K(x)u_k = f_k, \;\; k = 1..n_t$$

ESL(0차, Choi–Park): 매 외부 반복 $W$마다 동적해 $u_k(x_W)$에서 등가정하중

$$f^{ESL}_k(x_W) = K(x_W) \, u_k(x_W) \quad \text{(eq. 3)}$$

를 고정하고 선형 정적 다중하중 문제(기존 정적 TO 그대로)를 푼다.
하중이 상수이므로 수반 민감도 eq. 12는 하중 미분항이 0이다.

DiESL(Triller 2022, [정독] eq. 7–8, 12d–12f): 변형된 형상
$X(t_i) = X_0 + u(t_i)$ 위에 시각별 선형부분모델 $LSM_i$를 두고
증분만 재현한다.

$$K_i \, \Delta u_i = \Delta f^{DiESL}_i, \qquad
\Delta u(t_i) = u(t_{i+1}) - u(t_i), \qquad
K_i = K(x, X(t_i))$$

서로게이트 응답은 사슬합 $u_i = \sum_{j<i}\delta_j$로 연결되므로
(coupled MMO) 민감도에 사슬법칙이 필요하고, 기존 단일 메쉬·독립 하중케이스
정적 TO와 달리 **하중케이스별 형상**(per-load-case $K_{0,i}$)이 필요하다.
밀도 해석 규칙(중간밀도 유지, $\varepsilon_v = 0.1$,
$\varepsilon_s = 0.9$, eq. 9), 해석영역 지수 $p_{NL} = 1$(설계영역 $p = 3$와
의도적 불일치 감수, eq. 10), 삭제 절점 좌표의 SPC 재구성 해석(eq. 11)이
토폴로지 확장의 핵심이다. 빔 충돌 예제에서 초기속도(관성 효과)에 따라
최적 형상이 달라지며 표준 ESL이 못 잡는 관성 효과를 포착한다.
단, **선형 동역학에서는 DiESL = ESL**(Stolpe 2023)과 동등하므로 이득은
비선형에서만 생긴다.

F-ESL(Buzaglo 2025, [정독] eq. 4–6, 13–14, 20–24): 등가하중에 1차항 포함

$$f^{F\!-\!ESL}_k(x,x_W) = f^{eq}_k(x_W) + \nabla f^{eq}_k(x_W)(x - x_W)$$

부분문제 민감도 eq. 13–14에 $K^{-1}\nabla f^{eq}$ 항이 추가되고,
수렴점($x = x^*$)에서 $K^{-1}\partial K K^{-1} = -\partial K^{-1}$ 항등식
(eq. 22)으로 0차 잔여항이 소멸하여

$$\frac{\partial \tilde{u}^{F\!-\!ESL}_k}{\partial x_i}(x^*)
= \frac{\partial u_k}{\partial x_i}(x^*) \quad \text{(eq. 24)}$$

즉 F-ESL의 정적 근사 민감도가 원 동적 문제의 민감도와 일치하므로
KKT 정류조건(eq. 15–19)을 판정·만족할 수 있다. ESL은 eq. 20–21에서
$-K^{-1}\partial f^{eq}/\partial x_i$만큼 어긋나 최적점을 인식하지 못한다
(Stolpe 2018 2바 트러스 반례와 같은 맥락). 수치예제: 대칭 트러스
(P3, 7개 설계변수) 직접법 35회 vs F-ESL 4회 시간해석으로 동일 최적해
($f_c = 0.1783$, Lagrangian 기울기 $\sim 10^{-5}$ 이하)에 도달.

### 4.4 한계 명시 (F-ESL / DiESL)

- F-ESL은 동적 민감도($\nabla f^{eq}$, 시간적분 수반)가 필요해 0차 ESL의
  "민감도 회피" 이점을 일부 반납한다. 예제는 사이징(트러스·전단프레임)에
  국한되며 비선형 토폴로지 실증은 없다.
- DiESL은 비선형 충돌에서 보고된 개선이 단일 빔-충돌기 예제 + 상용 솔버
  기반이며 일반 안정성 근거는 아니다. 내부 반복은 이동제한·소수 반복으로
  짧게 끊어야 발산하지 않는다.
- 어떤 ESL 변종도 수렴 판정 $\|x_{W+1} - x_W\| \le \epsilon$만으로 원 문제
  최적임을 주장해서는 안 된다. 매 외부 반복에서 **진짜 동적 응답**으로
  목적·제한을 재평가하고, 신뢰영역·이동제한 + 수용 판정을 두며, 최종 상태는
  `esl_stationary_not_certified`(1차항 사용 시에만 완화)로 보고한다.

### 4.5 다물체 동역학 동시 최적화 TOMBS (Chalmers 2017) [정독]

강체·유연체 혼합 MBS의 유연체 여러 개를 **분리하지 않고 동시에** 최적화한다.
문제(eq. 1–3):

$$\min \sum_{j \in J}\int_0^{t_s} C_j(X^j, q^j_f, t)\,dt \;\;
\text{s.t.} \;\; M(X,q)\ddot{q} + K(X)q = f(X,q,\dot{q},\lambda),\;\;
\mathcal{C}(q,t) = 0$$

$$C^j_l = {q^j_{f,l}}^T K^j_{ff} \, q^j_{f,l} \quad \text{(eq. 4)}$$

부유좌표계 + Craig–Bampton 모드축소 + IDA(Sundials)/Assimulo 적분.
민감도는 $\partial f^i/\partial X^j_e \approx 0$,
$\partial M^i/\partial X^j_e \approx 0$,
$\partial \ddot{q}^i/\partial X^j_e \approx 0$(eq. 7) 및
타 물체 의존성 $\partial q^i/\partial X^j_e = 0$ $(i \ne j)$ 가정을 두어
정적 문제 수준으로 축소하면 컴플라이언스 민감도가 고전 형태로 떨어진다.

$$\frac{\partial C^j_l}{\partial X^j_e}
= -{q^j_{f,l}}^T \frac{\partial K^j_{ff}}{\partial X^j_e} \, q^j_{f,l}
\quad \text{(eq. 13)}$$

OC 중간변수 $Y_e = (X^j_e)^{-\alpha}$ ($\alpha = 3$), 분리형 근사
$C^{j,k,OC} = \sum_e (X^j_e)^{\alpha} b^{j,k}_e$(eq. 14–16), OC 업데이트
eq. 18. 핵심은 **질량 패널티**(flying element 대책, eq. 19):

$$E^j_e = (X^j_e)^q E^j, \qquad \varrho^j_e = (X^j_e)^q \varrho^j$$

강성만 패널티하면 저밀도 요소 주변 집중질량이 관성력에 따라 몸체를
따라가지 못하고 튀어(flying element, Fig. 3) 민감도 피크→발산을 일으킨다.
질량도 함께 패널티해야 수렴한다(저자는 선형 관계가 아니므로 추가 연구가
필요하다고 명시). Slider-crank(크랭크 2 rad/s, 슬라이더 2 kg,
$E = 70$ GPa, 부피 40% 제한) 두 몸체 동시 최적화로 검증한다.

---

## 5. 유연 메커니즘 · 관절하중 · 관성완화와 구현 시사점·한계

### 5.1 유연 메커니즘: 상호 에너지(상호 변형에너지) 정식화 [교차정리]

유연 메커니즘(compliant mechanism)은 강성 최대화가 아니라 **출력 변위
최대화**이다. 입력 포트에 더미(가상) 하중을 가하는 상호 에너지 형태로 쓴다.
출력 자유도에 단위 더미하중 $l$을 가한 가상 변위장 $v$($K v = l$)와 실제
변위장 $u$($K u = f_{in}$)의 상호 변형에너지:

$$\text{MSE} = v^T K u, \qquad \text{SE} = u^T K u$$

대표 목적식은 비율형 $\max \, \text{MSE}/\text{SE}$ 또는
$-\text{MSE} + w \cdot \text{SE}$ 가중합이며, 민감도는 $u$·$v$ 각각의
정적 수반형 미분($\partial K/\partial\rho_e$ 이차형식 2개)으로 닫힌다.
입력·출력 포트 스프링($k_{in}, k_{out}$) 강성이 기구 위상을 좌우하므로
반드시 함께 기록한다.

### 5.2 MuJoCo식 관절하중과 투영 관성완화 PIR [교차정리]

자유-자유(무구속) 상태의 정적 해석이나 MBS에서 분리된 유연체의 하중평형에는
관성완화(inertia relief)를 쓴다. 강체모드 $\Phi$($K\Phi = 0$)에 대한
질량직교 투영자:

$$P_{IR} = I - M\Phi\left(\Phi^T M \Phi\right)^{-1}\Phi^T$$

평형 하중 $P_{IR} f$에 대해 $K u = P_{IR} f$를 풀고
$\Phi^T M u = 0$으로 강체 성분을 고정한다.
MuJoCo류 관절 다물체 시뮬레이션에서 유연체Swingarm 등에 작용하는 관절
반력·관성력을 등가정하중으로 쓸 때, 각 시간 스텝의 관절하중 벡터를 $f_k$로
받아 $P_{IR}$로 자가평형화한 뒤 정적 부분문제 하중케이스로 투입하는 것이
표준 연결점이다. 회전 관절의 원심·코리올리 항은 $f$의速度 의존부를 통해
들어가므로 하중 정의에 누락하지 않는다.

### 5.3 구현 시사점 ($a$–$e$)

본 문서 수식들을 코드($aieng.topopt$ 계열 가정)로 옮길 때 순서와 함정:

a. 집계 응력부터: qp-완화($q = 0.5$) + 클러스터 P-norm + 적응 스케일링
   $c_k$ + 수반 1회/클러스터. $p$는 8부터 시작해 진동 시 하향.
b. 좌굴은 별도 $K_G$ 조립자($G^T S_0 G$ 수치적분)가 필요하고, 반드시
   침식 보간($\eta_G \ll \eta_K$)과 함께 간다. 단일 $\mu_1$ 직접 제한보다
   KS($m = 4\sim8$) 집계를 권장(모드 전환 강건).
c. 동역학은 D0(내장 Newmark + 내장 모드 솔버 + 직접 수반 기준해) 없이
   ESL을 얹지 않는다. 선형 검증에서 직접법과 어긋나면 ESL이 아니라
   구현 버그이다.
d. DiESL급 비선형은 하중케이스별 형상 저장($K_{0,i}$/좌표) 구조가 전제되므로
   단일 메쉬 가정 코어와 충돌한다. 단일 형상 ESL(0차)부터 지원하고,
   변형형상 추적은 별도 마일스톤으로 분리한다.
e. 다물체(TOMBS 경로)는 질량 패널티(eq. 19) 없이는 수렴 시도가 무의미하며,
   민감도 근사(eq. 7) 사용 시 "근사 민감도로 수렴한 해"임을 결과에 표기한다.

### 5.4 한계와 미확보 항목

- 12_Ferrari 좌굴 250줄 코드는 파일 불일치로 미확보. 좌굴 절 전체가
  [교차정리]이며, 실측 코드 상수(필터 반경·$\xi$·컷오프)는 원전 확인 전까지
  확정값으로 인용하지 않는다.
- 08_Le_2010 원전 미확보. qp-완화·P-norm 서술은 후속 문헌의 표준 형태로
  재구성한 [교차정리]이며, Le 원전 특유의 수치 상수는 포함하지 않는다.
- F-ESL 수치 근거는 사이징 문제에 국한(§4.4). 토폴로지·접촉 비선형에 대한
  1차 수렴 보장은 본 문서가 주장하지 않는다.
- TOMBS 민감도 eq. 7 근사는 저자가 수렴 성공으로 사후 정당화하는 형태이며,
  오차 한계가 없다. adjoint 다물체 민감도는 "추가 연구 필요" 상태이다.
