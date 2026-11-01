# [Implementation Plan] GE 제트엔진 브래킷 대회 실제 형상(STL) 및 4대 다중하중 벤치마크 구축

**문서 코드**: `PLAN-GE-BRACKET-REAL-STL-20261101`  
**작성 일자**: 2026-11-01  
**목표**: `wht_topopt`의 단순 합성 박스형 모델(`ex07`)을 탈피하고, `external/freeto_stls`의 실제 대회 STL(`GE_domain.STL`, `GE_fixed.STL`, `GE_force.STL`) 및 4대 다중 하중(수직 8,000 lbf, 수평 8,500 lbf, 42도 9,500 lbf, 토크 5,000 lb-in)과 Ti-6Al-4V 공학 물성, RBE형 강체 핀 결합을 완전히 반영한 마스터 벤치마크 파이프라인 구현 및 검증.

---

## 1. 당면 과제 및 배경 분석

### 1.1 기존 ex07 구현의 한계 및 문제점 진단
- **기존 ex07 모델**: $120 \times 60 \times 50\text{ mm}$ 합성 박스에 인공 원통을 부착한 단순 형상.
- **컴플라이언스 동일 현상 (88.6417 ~ 88.6644, 0.03% 오차)**:
  - 4개 볼트 보스와 1개 러그 보스의 `passive_solid` 및 외곽 `passive_void`가 도메인의 약 75%를 고정.
  - 최적화 가능한 자유 설계 영역(`design_mask`)이 전체의 25% 미만으로 극도로 협소하여, 어떤 알고리즘(OC, MMA, BESO, SEMDOT)을 사용하더라도 해가 사실상 사전에 결정되는(Over-constrained) 구조였음.
  - 또한 단일 수직 하중($-1000\text{ N}$)만 가해져 복합 다축 하중 간의 상충(Trade-off) 최적화가 발생하지 않음.

### 1.2 실제 대회 규격 (GrabCAD GE Jet Engine Bracket Challenge)
- **설계 영역**: $176 \times 62.5 \times 107\text{ mm}$ 복합 유선형 아치 및 러그 구조 (`GE_domain.STL`).
- **재료**: Ti-6Al-4V (E = 113.8 GPa, nu = 0.342, 항복강도 903 MPa, 밀도 4.43 g/cm³).
- **경계 조건**:
  - 볼트 4개 고정면 (`GE_fixed.STL`): 모든 자유도($u_x, u_y, u_z$) 완전 구속.
  - 러그 핀 구멍 2개 (`GE_force.STL`): 0.75인치 핀을 강체(RBE형)로 모델링하여 핀 중심축 기준 절점에 연결.
- **4대 독립 하중 케이스**:
  - Case 1 (수직 상향): $8,000\text{ lbf} = 35,585.8\text{ N}$ (+y)
  - Case 2 (수평 외향): $8,500\text{ lbf} = 37,809.9\text{ N}$ (+z)
  - Case 3 (42도 외향): $9,500\text{ lbf} = 42,258.1\text{ N}$ ($9,500\cos 42^\circ\mathbf{e}_y + 9,500\sin 42^\circ\mathbf{e}_z$)
  - Case 4 (수직축 비틀림): $5,000\text{ lb}\cdot\text{in} = 564,924\text{ N}\cdot\text{mm}$ (y축 비틀림 토크)
- **공식 평가 기준**:
  - 원본 부품 질량: $2,033\text{ g}$
  - 대회 우승작 질량: $327\text{ g}$
  - 우리 목표: $V=0.15$ 기준 질량 $245\text{ g}$ 및 최대 von Mises 응력 $< 903\text{ MPa}$ 통과 검증.

---

## 2. 통합 대상 핵심 전산역학 기법 (`aieng/src/aieng/topopt`)

1. **격자 열 단위 Ray-Casting 복셀화 (`stl_domain.py`)**:
   - `inside_grid`: 비대칭 홀수 교차 판정으로 STL 도메인 완전 수밀 복셀화.
   - `RigidBody`: CalculiX/Nastran 호환 RBE 강체 커플링 $\mathbf{u}_s = \mathbf{u}_r + \boldsymbol{\theta} \times (\mathbf{x}_s - \mathbf{x}_r)$ 및 축소 자유도 행렬 $\mathbf{T}^T \mathbf{K} \mathbf{T}$.
2. **보이드 제외 축소 시스템 풀이 (`_void_free_system`)**:
   - 도메인 외곽 보이드 요소(전체 격자의 70% 이상)를 전역 강성 행렬에서 배제하여 5~10배 고속화 및 메모리 절감.
3. **동일 체적 0/1 공통 재해석 평가 (`binary` + `check`)**:
   - 밀도 필터/투영 결과에 의한 중간 밀도 착시를 제거하고, 체적 분율 기준 상위 요소를 1.0으로 이진화한 후 공통 FEM 해석으로 4개 하중 케이스별 요소 중심 von Mises 응력(최대치 및 99.5 백분위수) 정밀 산출.
4. **적층제조(AM) 오버행 필터 (`am_filter.py`)**:
   - Langelaar (2016/2017) 수반 기반 45도 자립(Self-supporting) 오버행 필터 (+y 빌드 방향).
5. **국부 체적 제약 (`local_volume`) 및 Robust formulation**:
   - 최대 부재 크기 제약 (MMA p-norm 국부 체적) 및 최소 부재 크기 제약 (삼중 필터 침식/팽창).

---

## 3. 단계별 구현 및 실행 계획 (Step-by-Step Roadmap)

```
1. [Step 1: 환경 및 데이터셋 검증]
   - external/freeto_stls STL 3종 형상 및 Bounding Box 정합성 확인
   - mesh_control=40 격자 생성 및 절점/요소수 검증
   verify: python 스크립트로 STL 로드 및 ray-casting 정상 동작 확인

2. [Step 2: 마스터 벤치마크 러너 구현 (ex07_ge_bracket_challenge.py)]
   - aieng의 검증된 STL 복셀화 및 BuiltinAdapter (MKL Pardiso) 연계
   - 4개 하중 케이스 및 RBE 핀 결합
   - OC, BESO, MMA(local_volume), Robust, AM(+y) 알고리즘 스위트 구현
   - 라이브 진행 GIF (Front | Side | Top) 및 0/1 형상 렌더링
   verify: 1회차 시험 실행 및 4개 하중 케이스별 컴플라이언스/응력 정상 계산 확인

3. [Step 3: 벤치마크 실행 및 데이터 추출 (V=0.15 기준)]
   - BESO, OC, AM(+y), Local Volume 제약 실행
   - 4개 하중 케이스별 max von Mises, 99.5% von Mises, 컴플라이언스, 질량, 반복수, 소요시간 측정
   verify: summary_mc40.md 데이터와 1:1 대조 및 항복 응력(903 MPa) 통과 검증

4. [Step 4: ex07 컴플라이언스 0.03% 동일 현상 원인 규명 및 비교 분석]
   - 기존 ex07 합성 모델 vs 실제 STL 모델의 설계 영역 비율 및 하중 모드 팩트체크
   - 정량적 원인 분석 리포트 작성
   verify: dev_doc 및 dev_log 문서 생성

5. [Step 5: 모던 반응형 HTML 종합 대시보드 리포트 생성 및 배포]
   - outputs/ge_bracket/ge_bracket_challenge_report.html
   - Antigravity 아티팩트 등록 및 사용자 브리핑
   verify: 브라우저 열람 및 렌더링 무결성 확인
```

---

## 4. 반증 가능한 성공 기준 (Verifiable Success Criteria)
1. **무결성**: 🚫 Anti-Mocking 준수 — 난수, 더미 상수 전면 배제.
2. **형상 정합성**: `GE_domain.STL`, `GE_fixed.STL`, `GE_force.STL`을 격자 열 단위 ray casting으로 정확히 복셀화.
3. **하중 및 물성**: Ti-6Al-4V ($E=113.8\text{ GPa}$, $\nu=0.342$, $\sigma_y=903\text{ MPa}$, $\rho=4.43\text{ g/cm}^3$) 및 4대 하중 공식 규격 적용.
4. **성능 벤치마크**: $V=0.15$에서 질량 $\approx 245\text{ g}$, 4개 하중 케이스 최대 응력 $\le 903\text{ MPa}$ 통과 입증.
5. **산출물**: 4대 하중 케이스 응력 표, 실시간 진행 GIF, 3D 렌더링 이미지, HTML 종합 리포트 및 기술 문서.
