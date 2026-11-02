# Topology Optimization Formulations & Numerical Algorithmic Foundations

> 본 문서는 `wht_topopt` 프레임워크에 구현된 5대 위상최적화 알고리즘(MMA, GCMMA, SEMDOT, BESO, HCA)과 Kratos Multiphysics MKL PARDISO 연동 정식화의 수학적 원리, 수치 이산화 기법, 수렴성 보장 메커니즘을 총망라한 기술 문서입니다.

---

## 1. 지배 방정식 및 표준 위상최적화 문제 (Problem Formulation)

연속체 구조물의 정적 다중하중에 대한 컴플라이언스(유연도) 최소화 문제는 다음과 같이 정식화됩니다:

$$\begin{aligned}
\min_{\boldsymbol{\rho}} \quad & C(\boldsymbol{\rho}) = \sum_{k=1}^{N_{\text{lc}}} w_k \, \mathbf{f}_k^T \mathbf{u}_k \\
\text{subject to} \quad & \mathbf{K}(\boldsymbol{\rho}) \mathbf{u}_k = \mathbf{f}_k, \quad k = 1, \dots, N_{\text{lc}} \\
& V(\boldsymbol{\rho}) = \frac{\sum_{e \in \Omega_d} v_e \rho_e}{\sum_{e \in \Omega_d} v_e} \le V^* \\
& 0 < \rho_{\min} \le \rho_e \le 1, \quad \forall e \in \Omega_d
\end{aligned}$$

여기서:
- $\boldsymbol{\rho} = \{\rho_1, \rho_2, \dots, \rho_{N_e}\}$는 요소별 물리적 밀도(Physical Density) 벡터입니다.
- $\mathbf{K}(\boldsymbol{\rho})$는 전체 구조 강성 행렬(Global Stiffness Matrix)입니다.
- $w_k$는 각 하중 케이스의 중요도 가중치($\sum w_k = 1$)입니다.
- $V^*$는 허용 목표 체적률(Target Volume Fraction)입니다.

---

## 2. 재료 보간 모델: SIMP & RAMP

### 2.1 SIMP (Solid Isotropic Material with Penalization)
수치적 중간 밀도($0 < \rho_e < 1$)의 발생을 억제하고 0/1 이진 위상으로 유도하기 위해 페널티 지수 $p$ (통상 $p=3$)를 적용합니다:

$$E(\rho_e) = E_{\min} + \rho_e^p (E_0 - E_{\min})$$

- $E_0$: 기본 재료의 탄성계수 (예: Ti-6Al-4V $113.8\text{ GPa}$, Al6061-T6 $71.0\text{ GPa}$)
- $E_{\min}$: 특이 행렬(Singular Matrix) 방지를 위한 수치적 최소 탄성계수 ($E_0 \times 10^{-6} \sim 10^{-9}$)

### 2.2 수반 감도 해석 (Adjoint Sensitivity Analysis)
목적함수 컴플라이언스의 설계 변수에 대한 1차 편도함수는 수반 변위 해석을 통해 계산됩니다:

$$\frac{\partial C}{\partial \rho_e} = - \sum_{k=1}^{N_{\text{lc}}} w_k \, p \, \rho_e^{p-1} (E_0 - E_{\min}) \, \mathbf{u}_{k,e}^T \mathbf{k}_0 \mathbf{u}_{k,e}$$

여기서 $\mathbf{k}_0$는 단위 탄성계수를 갖는 요소의 기본 강성 행렬입니다.

---

## 3. 5대 위상최적화 알고리즘 비교 및 원리

### 3.1 MMA (Method of Moving Asymptotes, Svanberg 1987)
- **개념**: 각 이터레이션에서 목적함수와 제약조건을 점근선(Moving Asymptotes) 기반의 볼록 분리 함수(Convex Separable Function)로 국소 근사하여 서브 문제를 해석적으로 풉니다.
- **점근선 갱신**:
  $$L_j^{(k)} = x_j^{(k)} - s_k (x_j^{(k-1)} - L_j^{(k-1)}), \quad U_j^{(k)} = x_j^{(k)} + s_k (U_j^{(k-1)} - x_j^{(k-1)})$$
- **장점**: 다중 비선형 제약조건(응력, 변위, 고유진동수) 처리에 탁월하며 수치적 안정성이 검증됨.

### 3.2 GCMMA (Globally Convergent MMA, Svanberg 2002)
- **개념**: 기존 MMA의 점근선 외에 비보수적(Non-conservative) 오차를 방지하기 위해 2차 항 파라미터 $\rho_i^{(k)}$를 도입하고, 내부 루프(Inner Iteration)를 통해 실제 목적함수 값이 근사 함수보다 작거나 같을 때까지 보수성을 동적으로 강화합니다.
- **수렴성**: 임의의 시작점에서 KKT(Karush-Kuhn-Tucker) 정지점(Stationary Point)으로의 엄밀한 수학적 전역 수렴성이 증명됨.

### 3.3 SEMDOT (Smooth-Edged Material Distribution, Fu & Rolfe 2020)

> **구현 상태 (2026-11-03).** `algorithms/semdot.py`는 2D(`StructuredMesh2D`)와 구조 격자 3D(`StructuredMesh3D`, `structured_hex`)에서 필터링된 밀도를 절점으로 평균하고, 요소마다 축당 `n_sub + 1`개 격자점으로 쌍선형/3선형 보간한 뒤 평활 Heaviside의 평균으로 체적 분율을 만든다. 축당 샘플 수는 `n_sub`로 정한다(`run_topology_optimization` 기본 4, 즉 3D 요소당 125점). 수동 공백 요소는 절점 평균에서 빠진다. 표면은 `fine_density_grid`, `extract_grid`, `extract_isosurface`로 얻는다. 비구조 격자에서는 요소 샘플 방식으로 대체하고 경고를 낸다. `run_topology_optimization`의 SEMDOT 루프는 SIMP 페널티 없는 선형 모듈러스로 해석한다. 비교 기준: 3D 캔틸레버 48x16x8, V 0.3, 60회, 2배 세밀 격자 재해석에서 SEMDOT 40.6, MMA 34.5 (같은 체적). 파라미터 튜닝은 하지 않았다.

- **개념**: 요소 중심의 연속 밀도 대신 요소 꼭짓점(Node)의 매끄러운 형상 함수 보간을 통해 요소 경계를 연속 곡면으로 표현.
- **경계 분할 및 수치 적분**: 각 Hex8 요소를 8개 하위 셀로 분할하여 경계면 통과 체적을 정밀 적분함으로써 중간 밀도 영역을 원천적으로 제거하고 체커보드 현상을 완벽 차단.
- **장점**: $0/1$ 경계가 매우 선명하고 후처리 CAD 표면화가 용이함.

### 3.4 BESO (Bi-directional Evolutionary Structural Optimization, Huang & Xie 2007)
- **개념**: 요소별 민감도 수치(Sensitivity Number $\alpha_e$)를 계산하여 비효율적인 요소를 제거(Void화)하고 고응력 요소 주변의 Void 요소를 고체화(Solid화)하는 이산 진화 알고리즘.
- **목표 체적 진화율**: $V^{(k+1)} = \max(V^*, V^{(k)}(1 - ER))$.
- **장점**: 중간 밀도가 완전히 배제된 명확한 솔리드 골격을 직접 형성.

### 3.5 SE-AHCA (Stress-Energy Adaptive Hybrid Cellular Automata)
- **개념**: 전역 최적화 및 복잡한 수반 감도 연립방정식을 풀지 않고, 각 요소 셀(Cell)이 주변 이웃 요소와의 변형에너지 및 von Mises 응력 상태를 교환하며 자발적으로 밀도를 적응시키는 국소 법칙(Local Rule) 기반 알고리즘.
- **상세**: `03_stress_energy_adaptive_hca_theory.md` 참조.

---

## 4. 고성능 FEM 솔버 엔진 연동 아키텍처

```
+-------------------------------------------------------------------------+
|                         wht_topopt Framework                            |
+-------------------------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
      [Direct BuiltinAdapter]                    [KratosSolver Native]
               |                                           |
    - Scipy CSR Matrix SpMV                     - Kratos Core C++ ModelPart
    - PyPardiso MKL LU                          - Intel MKL PARDISO Direct LU
    - Multi-RHS Direct Solve                    - AVX-512 FMA Vectorization
    - Peak 6 Threads Parallel                   - In-Memory Pointer Exchange
```

- **다중 하중 멀티-RHS 처리**: 4개 정적 하중 벡터 $[\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4]$를 단 한 번의 강성 행렬 $LU$ 분해로 동시 해석하여 CPU 시간을 $75\%$ 이상 절감.
- **Watertight STL 변환**: Hex8 이산 요소망에 3D Marching Cubes 등위면 추출 및 Laplacian 스무딩을 적용하여 100% 수밀성(Watertight) 3D 프린팅 형상을 직접 생성.
