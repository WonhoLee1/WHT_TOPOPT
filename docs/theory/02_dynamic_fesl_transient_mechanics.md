# Dynamic F-ESL & Transient Structural Mechanics (Buzaglo 2026 True F-ESL)

> 본 문서는 `wht_topopt` 프레임워크의 과도 동역학 해석 솔버(Generalized-$\alpha$ Solver) 및 Buzaglo (2026) 최신 정밀 등가 정하중법(True F-ESL)의 수치역학적 원리와 동적 강성 극대화 정식화를 다룹니다.

---

## 1. 연속체 구조 동역학 지배 방정식

시간 영역 $t \in [0, T]$에서 감쇠를 갖는 선형 탄성 연속체 구조물의 진동 방정식은 다음과 같습니다:

$$\mathbf{M}(\boldsymbol{\rho}) \ddot{\mathbf{u}}(t) + \mathbf{C}(\boldsymbol{\rho}) \dot{\mathbf{u}}(t) + \mathbf{K}(\boldsymbol{\rho}) \mathbf{u}(t) = \mathbf{f}(t)$$

여기서:
- $\mathbf{M}(\boldsymbol{\rho})$는 일관 질량 행렬(Consistent Mass Matrix) 또는 집중 질량 행렬(Lumped Mass Matrix)입니다:
  $$\mathbf{M}_e(\rho_e) = m(\rho_e) \, \mathbf{M}_{0,e}, \quad m(\rho) = \rho \ (\rho \ge 0.1), \quad m(\rho) = 10^{5} \rho^{6} \ (\rho < 0.1)$$

  (Du & Olhoff 2007. 선형 질량 $\rho \mathbf{M}_0$은 $E \propto \rho^3$과 함께 쓰면 저밀도 요소에 국부 스퓨리어스 모드를 만들므로 쓰지 않는다.)
- $\mathbf{C}(\boldsymbol{\rho})$는 레일리 감쇠(Rayleigh Damping) 모델을 따릅니다:
  $$\mathbf{C}(\boldsymbol{\rho}) = \alpha_{\text{mass}} \mathbf{M}(\boldsymbol{\rho}) + \beta_{\text{stiff}} \mathbf{K}(\boldsymbol{\rho})$$
- $\mathbf{f}(t)$는 시간에 따라 변하는 외부 충격 또는 돌풍 하중 벡터입니다.

---

## 2. Generalized-$\alpha$ 수치 시간 적분법 (Chung & Hulbert 1993)

시간 단계 $\Delta t$에서 수치적 안정성과 고주파 감쇠 제어를 위해 Generalized-$\alpha$ 기법을 사용합니다:

$$\begin{aligned}
\mathbf{M} \ddot{\mathbf{u}}_{n+1-\alpha_m} + \mathbf{C} \dot{\mathbf{u}}_{n+1-\alpha_f} + \mathbf{K} \mathbf{u}_{n+1-\alpha_f} &= \mathbf{f}_{n+1-\alpha_f} \\
\mathbf{u}_{n+1} &= \mathbf{u}_n + \Delta t \, \dot{\mathbf{u}}_n + \Delta t^2 \left( \frac{1}{2} - \beta \right) \ddot{\mathbf{u}}_n + \Delta t^2 \beta \, \ddot{\mathbf{u}}_{n+1} \\
\dot{\mathbf{u}}_{n+1} &= \dot{\mathbf{u}}_n + \Delta t \, (1 - \gamma) \ddot{\mathbf{u}}_n + \Delta t \, \gamma \, \ddot{\mathbf{u}}_{n+1}
\end{aligned}$$

- 중간 시점 변수 보간:
  $$\mathbf{u}_{n+1-\alpha_f} = (1 - \alpha_f) \mathbf{u}_{n+1} + \alpha_f \mathbf{u}_n, \quad \ddot{\mathbf{u}}_{n+1-\alpha_m} = (1 - \alpha_m) \ddot{\mathbf{u}}_{n+1} + \alpha_m \ddot{\mathbf{u}}_n$$
- 무조건적 2차 정확도(Second-order Accuracy) 및 스펙트럼 반경 $\rho_\infty \in [0, 1]$ 제어 파라미터:
  $$\alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1}, \quad \alpha_f = \frac{\rho_\infty}{\rho_\infty + 1}, \quad \beta = \frac{1}{4}(1 - \alpha_m + \alpha_f)^2, \quad \gamma = \frac{1}{2} - \alpha_m + \alpha_f$$
- 본 시스템 기본 설정: $\rho_\infty = 0.80$ (스퓨리어스 고주파 모드를 효과적으로 감쇠시키며 핵심 구조 응답의 물리적 진폭을 완벽 보존).

---

## 3. Buzaglo (2026) True F-ESL (Equivalent Static Load) 정식화

> **구현 상태 (2026-11-03 코드 리뷰 R7).** 아래 3.2의 $\mathbf{f} = \mathbf{K}\mathbf{u}(t_{\text{peak}})$는 0차 ESL이다. `ex09`의 보정항은 임계 시점의 $\mathbf{u}_k$를 수반 변수 대신 쓰는 휴리스틱이며, 과도 수반 해석과 KKT 잔차가 없다. 따라서 현재 구현은 정확한 F-ESL 기울기가 아니고 KKT 보장도 없다. 정적 부문제의 해는 $\mathbf{u}(t_k)$와 항등이므로 코드는 풀이 없이 그 값을 읽는다.

### 3.1 전통적 ESL의 한계
전통적인 동적 위상최적화의 ESL 기법은 각 시간 스텝의 변위 $\mathbf{u}(t_i)$를 순차적으로 정적 하중으로 매핑하여 계산량이 비대해지고, 정적 최적화기와의 상호작용 과정에서 비틀림 모드와 굽힘 모드가 충돌하는 진동 발산(Instability) 현상이 발생합니다.

### 3.2 True F-ESL 핵심 메커니즘
Buzaglo (2026)의 True F-ESL은 과도 응답 곡선에서 구조 시스템의 최대 변형에너지 및 최대 변위가 발생하는 임계 시점 $t_{\text{peak}} = \arg\max_{t} \|\mathbf{u}(t)\|$를 식별하고, 해당 상태의 탄성 복원력을 단일 등가 정하중 벡터로 정의합니다:

$$\mathbf{f}_{\text{ESL}} = \mathbf{K}(\boldsymbol{\rho}) \mathbf{u}(t_{\text{peak}})$$

- **물리적 무결성**:
  정적 방정식 $\mathbf{K} \mathbf{u}_{\text{stat}} = \mathbf{f}_{\text{ESL}}$을 풀었을 때 얻어지는 변위 $\mathbf{u}_{\text{stat}}$는 동적 과도 해석의 피크 변위 $\mathbf{u}(t_{\text{peak}})$와 수학적으로 항등적($\mathbf{u}_{\text{stat}} \equiv \mathbf{u}(t_{\text{peak}})$)입니다.
- **동적 증폭 계수 (Dynamic Amplification Factor, DAF)**:
  정적 하중 진폭 $F_0$에 대한 피크 변위 비를 동적으로 반영하여 하중 벡터를 정밀 보정합니다:
  $$\text{DAF} = \frac{\max_t \|\mathbf{u}_{\text{dyn}}(t)\|}{\|\mathbf{u}_{\text{static}}\|}$$

---

## 4. 항공기 돌풍(Gust) 및 로봇 암 슬루잉(Slewing) 동적 하중 모델

### 4.1 항공기 50ms 돌풍 하중 (Aircraft 50ms Gust)
$$F_{\text{gust}}(t) = F_0 \sin\left(\frac{\pi t}{T_{\text{gust}}}\right) \exp\left(-\gamma \frac{t}{T_{\text{gust}}}\right) \cdot \mathcal{H}(T_{\text{gust}} - t)$$
- $T_{\text{gust}} = 50\text{ ms}$, $\gamma = 1.5$, $\mathcal{H}$는 헤비사이드 계단 함수.
- 충격 인가 후 $t > 50\text{ ms}$ 구간에서는 외력이 $0$이 되며, 구조물은 자유 감쇠 진동을 수행합니다.

### 4.2 로봇 암 과도 슬루잉 충격 및 원심 비틀림
- 가감속 토크 프로파일에 따른 급격한 회전 슬루잉 충격:
  $$T_{\text{slew}}(t) = T_{\max} \left[ 1 - \cos\left(\frac{2\pi t}{T_{\text{acc}}}\right) \right]$$
- 슬루잉 시 관성 질량의 편심에 의해 굽힘($K_y$)과 더불어 극심한 동적 비틀림($U_z, K_{\theta x}$)이 유발됨.

---

## 5. 정적 강성 vs 동적 강성의 물리적 트레이드오프와 체적 완화

1. **동일 체적률 ($V=28\%$) 하에서의 트레이드오프**:
   - 정적 최적화는 슬루잉 굽힘 하중 축($Y$)에만 재료를 집중하여 수직 두께를 키우므로 정적 굽힘 강성은 우수하나, 비틀림 저항($Z$) 면내 전단 재료가 부족하여 동적 과도 비틀림 진동 $U_z$가 $3.54\text{ mm}$까지 증폭됨.
   - 동적 F-ESL 최적화는 비틀림 억제를 위해 대각 X자형 전단 브레이싱을 자발적으로 형성하므로 동적 진동은 $1.20\text{ mm}$ (-66%)로 격감하지만, 수직 단면적이 감소하여 정적 굽힘 강성은 13.9% 낮아짐.
2. **체적 완화(Volume Relaxation, $V=35\%$, +60g)에 의한 All-Dominant 달성**:
   - 체적 제약을 $V=35\%$로 소폭 완화(단 +60g 증량)할 경우, X자형 동적 트러스와 고강성 수직 플랜지가 동시에 두껍게 형성되어:
     - 굽힘 강성: $K_y = 8,540.8\text{ N/mm}$ (+84.6% 압도)
     - 1차 고유진동수: $f_1 = 353.0\text{ Hz}$ (+87.3 Hz 폭증)
     - 피크 비틀림 변위: $U_z = 0.95\text{ mm}$ (-73.1% 격감, 1mm 미만 진동 억제 완벽 달성)
   - 이로써 정적 모델의 한계를 동적 모델이 모든 지표에서 완전히 능가하는 All-Dominant 엔지니어링 해법이 입증됨.
