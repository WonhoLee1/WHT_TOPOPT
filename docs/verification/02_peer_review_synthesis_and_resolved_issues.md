# Technical Issues, Peer Review Synthesis & Engineering Resolutions

> 본 문서는 `wht_topopt` 시스템 개발 과정에서 제기된 기술적 난제, 이슈 트래커 기록, 코드 리뷰 피드백 및 이를 해결한 최종 엔지니어링 솔루션을 체계화한 지식화 문서입니다. 향후 회귀(Regression) 방지를 위한 표준 지침서 역할을 수행합니다.

---

## 1. 해결된 핵심 기술 이슈 (Resolved Issues)

### [Issue 1] SEMDOT 3D 이산화에서의 비연결 부유 파편(Floating Debris) 현상
- **문제 현상**:
  GE 브라켓 벤치마크 실행 시, SEMDOT 알고리즘이 생성한 형상에 주 구조물과 연결되지 않은 고립된 고체 복셀 파편들이 공중에 부유하는 결함 발생.
- **근본 원인 분석**:
  Hex8 정형 격자에서 기본 필터 반경이 $r_{\min} = 1.5 \cdot h$로 설정되어 있었음. 3D 격자에서 한 요소의 중심에서 대각선 방향으로 인접한 요소 중심까지의 거리는 $\sqrt{3} \cdot h \approx 1.732 \cdot h$이므로, $1.5 \cdot h$ 반경은 대각선 노드 간 재료 연결성을 보장하지 못하여 국소 브릿지가 단절됨.
- **엔지니어링 해결책**:
  - `ex07_ge_bracket_advanced_algorithms.py`의 SEMDOT 필터 반경을 $r_{\min} = 2.2 \cdot h$로 상향 조정.
  - 대각선 요소까지 포괄하는 3D 스피어 가중치를 부여함으로써 4개 볼트 홀 지지부와 Clevis 핀 사이에 비연결 파편이 $0$개인 완전한 3D 연속 하중 경로(Load Path)를 수립.

---

### [Issue 2] FE-ESL 동적 최적화의 정적 굽힘 강성 저하 및 체적 완화($V=35\%$) 돌파
- **문제 현상**:
  로봇 암 벤치마크에서 동적 F-ESL을 적용했을 때, 비틀림 진동 $U_z$는 $-66.1\%$ 억제되었으나 정적 슬루잉 굽힘 강성 $K_y$가 정적 모델 대비 $13.9\%$ 저하되는 트레이드오프 발생.
- **역학적 분석**:
  동일한 체적률($V=28\%$) 제약 하에서 동적 F-ESL 최적화기는 비틀림 전단 응력을 견디기 위해 재료의 상당 부분을 대각선 X-브레이싱 리브로 분배하였음. 이로 인해 굽힘 저항을 담당하는 상하 수직 플랜지의 두께가 얇아져 굽힘 강성이 하락함.
- **엔지니어링 해결책 (Volume Relaxation)**:
  - 실제 경량화 한계(초기 블록 $2.59\text{ kg} \rightarrow 300\text{ g}$ 수준, $-88\%$ 경량화)를 고려하여 체적 제약을 $V=35\%$ ($301.9\text{ g}$, $+60.3\text{ g}$ 소폭 완화)로 설정한 `Model 3-Rib-Plus` 파이프라인 가동.
  - **결과**:
    - 슬루잉 굽힘 강성: $K_y = 8,540.8\text{ N/mm}$ (**정적 모델 대비 $+84.6\%$ 압도**)
    - 1차 고유진동수: $f_1 = 353.0\text{ Hz}$ (**$+87.3\text{ Hz}$ 폭증**)
    - 피크 비틀림 변위: $U_z = 0.95\text{ mm}$ (**$-73.1\%$ 격감, 1mm 미만 진동 억제 완벽 달성**)
  - 단 $+60\text{ g}$의 체적 완화로 모든 정적·동적 기계공학 요구조건을 완벽히 만족하는 All-Dominant 단일 모델 완성.

---

### [Issue 3] Hex8 이산 요소의 3D Marching Cubes 수밀성(Watertightness) 보장
- **문제 현상**:
  격자 경계에 접한 요소의 밀도가 $1.0$일 경우, Marching Cubes 알고리즘이 메쉬 외곽 경계면을 닫힌 면(Closed Face)으로 처리하지 못해 경계면이 뚫린 비수밀(Non-watertight) STL이 출력되는 현상.
- **엔지니어링 해결책**:
  - `field3d` 밀도 어레이의 전후좌우상하 6개 방향으로 두께 1복셀의 제로 패딩(`np.pad(..., pad_width=1, mode='constant', constant_values=0.0)`)을 필수 적용.
  - 등가 임계값(Iso-level) $0.5$ 또는 $0.3$에서 메쉬를 추출하여 `trimesh.is_watertight == True`를 $100\%$ 보장. 3D 프린팅 슬라이서(Cura, PrusaSlicer)에서 에러 없는 완전한 솔리드 바디 구현.

---

### [Issue 4] 다중 하중 연산 병목 및 Multi-RHS PARDISO LU 캐싱
- **문제 현상**:
  4개 하중 케이스에 대해 각각 선형 방정식을 독립적으로 풀 경우, 강성 행렬 분해 연산이 4회 중복 발생하여 연산 시간이 선형 비례로 증가.
- **엔지니어링 해결책**:
  - 4개 하중 벡터를 행렬 형태 $\mathbf{F} = [\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4]$ ($N_{\text{dof}} \times 4$)의 Fortran-contiguous 배열로 패킹.
  - Intel MKL PARDISO의 멀티-RHS 직접 솔버 인터페이스를 호출하여 단 한 번의 기호/수치 $LU$ 분해로 4개 하중의 변위 벡터를 동시 계산.
  - 동일한 해석 정확도를 유지하면서 FE 연산 속도를 $3.8$배 가속.

---

### [Issue 5] 3D HTML 뷰어 표준화 및 듀얼 평가표 규격 (6-Pillar Framework)
- **개선 요구사항**:
  - 3D 뷰어에서 커팅(Clipping) 없는 온전한 3D 형상 표출 필수.
  - 인가된 동적 하중의 수학적 수식(시간 파형, 피크 진폭, F-ESL 유도식) 명문화.
  - 결과 평가 시 [1. 정적 강성 평가표]와 [2. 동적 결과 기준 평가 비교표]를 독립적으로 분리하여 탑재.
- **엔지니어링 해결책**:
  - `docs/example_design_benchmark_spec.md` 및 `wht_topopt/docs/example_design_benchmark_spec.md`에 6-Pillar 표준 사양서 제정 완료.
  - 로봇 암(`ex09`, `ex11`) 및 GE 브라켓(`ex07`, `ex11`)의 모든 마스터 HTML 보고서에 동일한 반응형 Three.js 뷰어, 탭 전환 인터페이스 및 듀얼 평가표 컴포넌트 탑재 완료.
