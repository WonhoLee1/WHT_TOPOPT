# 로봇 프레임 구조 위상최적화 모듈 개발 계획서 (슬라이드 초안, md)

> 최종 HTML: `robot_frame_topopt_plan_20261102.html` (동일 폴더)
> 성격: 내부 개발용 · 기술 교류 목적 · 수식 포함
> 핵심 메시지: 로봇 프레임 동적 거동을 고려한 경량·고신뢰성 강성/강도 설계 기술로서의 위상최적화 기법 확보

## S01 표지
- 제목: 로봇 프레임 구조 위상최적화 모듈 개발 계획서
- 부제: 동적 거동을 고려한 경량·고신뢰성 강성/강도 설계 기술 확보
- wht_topopt · 2026-11-01 · 내부 개발용 (기술 교류)

## S02 배경 — 왜 동적 고려 위상최적화인가
- 고속 슬루잉(수십 ms) 하 관성 하중 지배, 정적 설계는 과설계 아니면 파손
- 목표像: 체적분율 25%급 + 피크응력/COV 관리 + 제조 가능 형상

## S03 최신 연구 조사 ① — 동적·HCA 계열
- **F-ESL (Buzaglo·Pollini 2025)**: 독창성=등가하중 1차항으로 KKT 보장 / 차별성=0차 ESL(최적점 인식 불가)과 대비 / 활용도=외부 반복당 과도 수반 1회 / 연관성=ex09 핵심엔진 / 흡수=보정항+`kkt_residual` 보고+신뢰영역. 한계: 실증은 사이징 한정
- **HCA (Tovar 2006 / Penninger 2010 / Tong 2025)**: 독창성=무구배 SED 균질화 / 차별성=수렴≠최적(불동점 다수) / 활용도=외부솔버 결과만으로 갱신 / 연관성=SE-AHCA 모태 / 흡수=워밍업+PID 적응게인, 종료조건 KKT 연계

## S04 최신 연구 조사 ② — 응력·옵티마이저·수반해석
- **응력 (Le 2010 qp-완화)**: 독창성=SIMP-motivated 응력 정의로 특이점 해소 / 흡수=SE-AHCA 기술1 완화식+P-norm
- **MMA/GCMMA, BESO, Level-Set**: 흡수=`algorithms` 래퍼+이동제한
- **과도 수반 (Behrou·Guest, Gen-α/Newmark)**: 흡수=`dynamics` 시간역행 exact adjoint, 비용≈전진 1회
- 자산: `wht_topopt/docs/references/` 55편, [정독]/[초록]/[미확보] 근거 라벨

## S05 고려사항 총괄
- 6대축: 동적 하중 프로파일 / 회전 관성 / 강성-강도 / 제조 가능성 / 수치 무결성 / 허브 인터페이스

## S06 Loadcase 정의 기술
- S-curve 시간 이력 → 대표 스텝 (피크 SE·팁변위·가감속 충격), 유사도 병합, 에너지 적분 비례 가중
- $f_{eq,k} = K(x)\,u_k$, 다중 정하중 부분문제 + 1차 보정
- 등급: $w_k \propto \mathrm{SE}(t_k)\,\Delta t$

## S07 회전 관성 고려 기술
- $F_x = m\omega^2L$, $F_y = -m\alpha L$, $F_i = T \times r_i / \Sigma r^2$ (순합력 제거 assert)
- $I_{zz} = \Sigma\rho\,m\,r^2$ 교차평가 지표
- Du·Olhoff 질량 보간 ($q>1$) — 보이드 스퓨리어스 억제
- $m(\rho)=\rho m_0$ 선형 보간 금지 근거: 강성/질량비 $10^{-6}$ 붕괴

## S08 제조 필터링 기술
- 콘/KD-Tree 필터, Helmholtz PDE, Heaviside 투영, SIMP 연속화
- 탈형 구배·압출·AM 오버행·대칭 링크 연산자

## S09 기타 고려사항
- ωΔt ≤ 0.5, 정합 초기 가속도 $a_0 = M^{-1}f_0$, 최적화↔평가 감쇠 통일
- shift-invert 스케일·MAC, 허브 passive 영역·RBE3 분배
- Anti-mocking: 실측 로그, 난수 금지, 인용 라벨

## S10 wht_topopt 설계 매핑
- 표: `core`→필터·질량보간 / `dynamics`→loadcase·관성·SE-AHCA / `algorithms`→MMA·신뢰영역 / `ex09`→통합 실증

## S11 솔버 전략 (부전략)
- 자체 Hex8+PARDISO+Gen-α 1차 경로, CalculiX/FrontISTR/Kratos/OpenRadioss 어댑터 교차검증

## S12 향후 계획(안) + 기대효과
- M1 코어·필터 → M2 동역학·SE-AHCA → M3 ex09 실증 → M4 어댑터·자동 리포트
- 기대: 상대변위 −30%급, 피크응력 −15~40%, 체적 25%급, 문헌-코드 직결 자산
- 핵심 참고문헌 5종
