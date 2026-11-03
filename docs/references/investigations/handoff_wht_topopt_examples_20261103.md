# Handoff: wht_topopt 예제 재구성 (2026-11-03)

대상은 wht_topopt만이다. aieng는 지금 작업하지 않는다. 승인된 계획: `C:\Users\GOODMAN\.claude\plans\glimmering-baking-kite.md`.
커밋 대상은 소스와 문서다. `examples/outputs_v1/`의 결과물은 커밋하지 않았다.

## 지금 상태

| 항목 | 상태 |
|---|---|
| 코드 리뷰 수정 (R1~R10, 4절) | 완료. `docs/references/investigations/code_review_wht_topopt_20261103.md` 8~10절. R7은 일부 (수반 해석 미구현, 보고서 HTML의 F-ESL 표기 다수 남음) |
| SEMDOT 3D | 구현. MMA보다 약 18 % 나쁨 (48x16x8, 60회, 튜닝 안 함) |
| `examples/src_bak/` | 기존 50개 파일 복사 완료 |
| `src_v1/common/` | 2D와 3D 통합 완료: `structure`, `study`, `pillar1_algorithms` ~ `pillar6_deliverables` |
| ex00 (a 캔틸레버, b L-브라켓, c MBB) | 공용 모듈 위로 이전 완료 |
| ex07 GE, ex09 로봇 암 | 공용 모듈 위 작성, 스모크 통과. 전체 벤치마크 미실행 |
| ex01~ex06 | **옮기기만 함** (경로만 변경). 공용화 안 됨 |
| 기존 `examples/*.py` | 삭제 안 함 (계획 7단계) |

## 이어서 할 일 (순서대로)

1. **통합 후 ex00 대조 확인.** 커밋 시점에 재실행이 끝나지 않았다. 다시 돌리고 통합 전 값과 비교한다 (0/1 컴플라이언스, 전체 해상도).

   | 예제 | MMA | GCMMA | SEMDOT | BESO | SE-AHCA |
   |---|---:|---:|---:|---:|---:|
   | ex00a | 206.8 | 201.9 | 200.5 | 205.6 | 205.6 |
   | ex00b | 204.6 | 205.0 | 205.7 | 202.9 | 202.7 |
   | ex00c | 227.1 | 228.1 | 231.0 | 234.2 | 229.0 |

   주의: 통합 전과 통합 후는 해석 모델이 조금 다르므로(SE-AHCA와 BESO의 Emin, SEMDOT 등) 완전 일치가 아니라 작은 차이를 기대한다. 차이가 크면 원인을 찾는다. 3개를 동시에 돌리면 PARDISO 경합으로 많이 느려지므로 순서대로 돌린다.
2. **테스트 전체 실행.** 통합 후 `tests/test_examples_src_v1*.py` 중 `full_study` 2개와 전체 스위트는 아직 돌리지 못했다. 통합 전 기준: 112 통과 + 추가 테스트, 실패 2건은 Kratos, mujoco 항목으로 기존과 같음.
3. **ex07 f1 확인.** 스모크(MC24, 4회)에서 설계가 달라도 f1 = 179 Hz로 같았다. 떠 있는 조각의 국부 모드일 수 있다. `pillar5_audit`에 떠 있는 solid 조각 수를 보고하도록 추가하고 전체 해상도에서 확인한다.
4. **ex01~ex06 공용화** (사용자 승인 순서: 검증 후 공용화). ex03, ex01, ex02, ex06, ex04, ex05 순서. 다시 쓸 때마다 옛 결과와 대조한다. 목적함수(응력, 좌굴, 균질화, MuJoCo, 메커니즘)는 예제에 두고 끼워 넣는 인터페이스를 `pillar1_algorithms`에 만든다. `ex03c_beams_3d_suite`는 옛 ex06 전체라 ex00과 겹친다.
5. **PyVista GIF.** 임시 설치한 PyVista가 `pyvista_validation` 누락으로 불러와지지 않았다. 지금은 GIF 없이 이유만 기록한다.
6. **Kratos 경로.** 사양서 Pillar 1의 Kratos 행은 미실행으로 기록만 한다. 옛 코드는 `src_bak/ex08_kratos_*`.
7. **ex07, ex09 전체 벤치마크** (별도 지시 후). ex07은 omega_1 dt <= 0.5로 동적 해석이 가장 오래 걸린다. 시간은 재지 않았다.
8. **정리 (계획 7단계).** 스모크와 테스트 후 기존 `examples/*.py` 삭제, 테스트의 경로 갱신 (`test_review_fixes_20261103.py`가 `examples/ex07_*`, `ex09_*`를 직접 읽음), `src_bak` 파일 수 확인 후에만 삭제.

## 실행 방법

```
SITE=<scikit-image, gmsh, tifffile, imageio, lazy_loader, networkx, pyflakes를 --target으로 설치한 폴더>
PYTHONPATH="D:\PythonCodeStudy\CAD-CAE-COPILOT;$SITE" .venv/Scripts/python.exe -m pytest wht_topopt/tests -p no:cacheprovider --ignore=wht_topopt/tests/test_examples_showcase.py --ignore=wht_topopt/tests/test_homogenization.py -o addopts="" -q
.venv/Scripts/python.exe wht_topopt/examples/src_v1/ex00_2d_basic/ex00c_mbb.py [--smoke] [--algos mma se_ahca] [--iters 30] [-o 출력폴더]
```

임시 폴더는 세션 scratchpad라 사라질 수 있다. `.venv`에는 scikit-image가 없다 (`pyproject.toml`에는 필수로 적혀 있다). 제외한 두 테스트는 수정 전부터 수집되지 않는다.

## 알아 둘 것

- `ex09_esl_dynamic_robot_arm.py`(옛)는 Kratos를 최상위에서 import해 Kratos 없이 실행되지 않는다. 새 `ex09_robot_arm`은 그렇지 않다.
- 새 동적 모델은 0차 ESL이다. 옛 ex09의 보정항과 ex11의 `x 1.45` 증폭, 고정 f1 대체는 옮기지 않았다.
- `docs/benchmarks/01, 02, 04`와 `verification/01`에는 재생성 필요 표시가 있다. 03은 해당 없음.
- `docs/benchmarks/04_*`의 json, md 변경 일부는 다른 작업에서 온 것이다.
- `aieng/`의 topopt 변경(`hca.py`, 예제 등)은 이번 커밋에 넣지 않았다.
- Git Bash heredoc 안의 백슬래시가 그대로 전달되지 않은 사례가 있었다. 백슬래시가 든 패치는 파일로 쓴 뒤 실행한다.
