# FrontISTR 요소 정식화와 위상최적화 연동 규격 (한국어 요약 07)

> 소스코드 직접 분석 문서. 모든 정식화 서술은 `FrontISTR-master` 실측 라인,
> 모든 실행·키워드 규격은 `docs_ko/consolidated` 8대 통합 문서에서 가져왔다.
> 파일:행번호 인용을 원칙으로 하며, 핵심 수식은 생략 없이 전개한다.
>
> - 분석 대상 소스 루트: `D:\PythonCodeStudy\FrontISTR\FrontISTR-master\fistr1\src`
> - 문서 루트: `D:\PythonCodeStudy\FrontISTR\docs_ko\consolidated`
> - 워크스페이스 규칙: `D:\PythonCodeStudy\FrontISTR\AGENTS.md`
> - **파일명 정정 고지 (필독):** 작업 지시에 적힌
>   `static_LIB_C3D8_Bbar.f90`, `static_LIB_shell_mitc4.f90` 두 파일은
>   실재하지 않는다. `fistr1/src/lib/` 실측 목록에 존재하는 실제 파일은
>   `static_LIB_C3D8.f90` (B-bar 정식화) 와 `static_LIB_shell.f90`
>   (MITC 셸 통합 구현) 이며, 본 문서는 실재 파일 기준으로 인용한다.
> - **TopOpt 내장 여부 고지:** `fistr1/src/**` 전체에서
>   `TOPOPT|SIMP|BESO|level.?set` 패턴 검색 결과는 0건이다.
>   즉 FrontISTR v6.0에는 SIMP/BESO/Level-Set 옵티마이저가 내장되어 있지
>   않으며, §5의 연동 프로토콜은 외부 최적화 루프 + FrontISTR 재해석
>   구조임을 명시한다.

본 문서는 아래 **5대 구성**을 순서대로 다룬다:

1. SMP vs Hybrid 실행 규격 (`thread_openblas\fistr1.exe` 단독 실행 vs
   `hecmw_part1.exe` + `mpiexec -np N fistr1.exe -t T`).
2. `hecmw_ctrl.dat` / `.msh` / `.cnt` 3파일 규격.
3. `FrontISTR Completed !!` 성공 판정 규칙.
4. 요소별 정식화: 361(HEX8: IC 기본/FBAR vs FORM361=FI/BBAR, 민감도 해석 시
   `!SECTION FORM361` 명시 이유), 341(TET4 FI vs SELECTIVE_ESNS+DEPTH=2),
   342/362/731/741 번호·절점 순서.
5. 멀티머티리얼 `EB_0001`~`EB_0064` 버케팅과 SIMP/BESO/Level-Set 연동 프로토콜.

---

## 1. SMP vs Hybrid 실행 규격

### 1.1 바이너리 배치 (AGENTS.md:16–17)

| 모드 | 디렉토리 | 구성품 |
|------|---------|--------|
| SMP (Windows, OpenMP + OpenBLAS + MUMPS + METIS + ML) | `FrontISTR-latest.thread_openblas/` | `fistr1.exe`, `hecmw_part1.exe`, `hecmw_vis1.exe` |
| Hybrid MPI (MS-MPI + OpenMP + OpenBLAS + MUMPS) | `FrontISTR-latest.hybrid_msmpi_openblas/` | `fistr1.exe`, `mpiexec.exe` |

SMP = 단일 프로세스 + OpenMP 스레드 병렬. Hybrid = MS-MPI 다프로세스 ×
프로세스당 OpenMP 스레드의 2단계 병렬이다.

### 1.2 SMP 실행 명령

`01_execution_guide.md:488` 원문 그대로:

```text
fistr1 -t 4
```

4스레드 OpenMP 실행이다. 동일 파일 `:511` 로그 예시에서
`threads: 4`, `cores: 4`, `processes: 1`으로 기록된다.
입력 파일이 있는 디렉토리에서 실행하면 해석이 시작된다
(`01_execution_guide.md:89`).

### 1.3 Hybrid 실행 명령

`01_execution_guide.md:101` 원문 그대로:

```text
mpirun -np 4 fistr1
```

Windows 번들 바이너리에는 MPI 런타임이 포함되어 있어 별도 MPI 설치 없이
실행 가능하다 (`01_execution_guide.md:108`). 전체 Hybrid 절차는 2단계이다
(`01_execution_guide.md:284`, `01_execution_guide.md:313`):

```text
hecmw_part1
mpirun -np 4 fistr1
```

1. `hecmw_part1` — 전체 메시(`model.msh`)를 읽어 분할 메시
   (`model_8.0`~`model_8.n`)를 생성한다. 단일 CPU 해석에서는 이 단계가
   불필요하다 (`01_execution_guide.md:69`).
2. `mpirun -np 4 fistr1` — 분할 메시 + `.cnt`를 읽어 해석하고
   `model.res.0`~`model.res.n.1`을 출력한다.

스레드 수 병행 지정 형태는 `mpiexec -np N fistr1.exe -t T`
(N = MPI 프로세스 수, T = 프로세스당 OpenMP 스레드 수)이다.

### 1.4 분할 관련 키워드

- `!PARTITION, TYPE=NODE-BASED, METHOD=RCB, DOMAIN=8, DEPTH=1, UCD=mesh.inp`
  (`03_keyword_reference_dat_msh_part.md:1898`). `DEPTH` = 오버랩 깊이
  (중첩 층 수).
- SELECTIVE_ESNS 사용 시에는 `DEPTH=1`이 아니라 **`DEPTH=2`가 필수**이다
  (이유는 §4.3, `02_functions_reference.md:295`,
  `02_functions_reference.md:2175`).

---

## 2. hecmw_ctrl.dat / .msh / .cnt 3파일 규격

FrontISTR는 3종 입력 파일을 읽는다 (`01_execution_guide.md:50`):

| 파일 | 역할 (`01_execution_guide.md:68`) |
|------|----------------------------------|
| `hecmw_ctrl.dat` (전체 제어 데이터) | 메시 데이터·해석 제어 데이터의 입력 파일명과 가시화 데이터의 출력 파일명을 지정 |
| `.msh` (메시 데이터) | 절점·요소, 재료 물성, 그룹 데이터를 저장. 단일 CPU 해석은 전체 메시, 병렬 CPU 해석은 `hecmw_part1` 분할 메시 사용 |
| `.cnt` (해석 제어 데이터) | 해석 조건, 경계 조건, 하중, 시간 이력 제어를 정의 |

### 2.1 hecmw_ctrl.dat 예시

`01_execution_guide.md:740–748` (tutorial hinge 예제) 원문:

```text
!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE
 hinge.msh
!CONTROL, NAME=fstrCNT
 hinge.cnt
!RESULT, NAME=fstrRES, IO=OUT
 hinge.res
!RESULT, NAME=vis_out, IO=OUT
 hinge_vis
```

- `!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE` — 전체 메시 파일 지정
  (`03_keyword_reference_dat_msh_part.md:104`).
  병렬 분할 메시 사용 시 `TYPE=HECMW-DIST`이며, 이때 파일명 뒤에
  `.<rank>`가 붙는다 (`03_keyword_reference_dat_msh_part.md:139`).
- `!CONTROL, NAME=fstrCNT` — 해석 제어 파일 지정
  (`03_keyword_reference_dat_msh_part.md:47–51`).
- `!RESULT, ..., IO=OUT` — 결과·가시화 출력 지정
  (`03_keyword_reference_dat_msh_part.md:207`).

### 2.2 .msh 파일 규격 (요약)

`03_keyword_reference_dat_msh_part.md:900` (msh index). 핵심 블록:

- `!NODE` — 절점 좌표 (`03_keyword_reference_dat_msh_part.md:827`).
- `!ELEMENT, TYPE=<type> [, EGRP]` — 요소 연결성. 예시
  `!ELEMENT, TYPE=361, EGRP=A`
  (`03_keyword_reference_dat_msh_part.md:683`),
  `!ELEMENT, TYPE=341, EGRP=ESLD`
  (`03_keyword_reference_dat_msh_part.md:1612`).
- `!EGROUP, EGRP=<egrp>` — 요소 그룹명 정의. `GENERATE` 옵션으로 범위
  일괄 추가 가능 (`03_keyword_reference_dat_msh_part.md:513–583`).
- `!MATERIAL, NAME=<name> [, ITEM]` — 재료 정의
  (`03_keyword_reference_dat_msh_part.md:929`,
  `:1073`, `:1115`). `ITEM` 번호로 물성 입력 항목을 구분한다.
- `!SECTION, TYPE=<type>, EGRP=<egrp> [, MATERIAL]` — 그룹–재료–정식화
  연결. 예시
  `!SECTION, EGRP=SOLID1, TYPE=SOLID, MATERIAL=STEEL`
  (`03_keyword_reference_dat_msh_part.md:1603`).
  §5 버케팅의 `EB_0001`~`EB_0064`는 바로 이 `EGRP` 이름 자리에 들어간다.

### 2.3 .cnt 파일 규격 (요약)

`01_execution_guide.md:812` (`hinge.cnt`). 해석 조건·경계·하중·
제어 키워드 집합이며, 본 문서와 관련된 핵심은:

```text
!SECTION, SECNUM=<n>, FORM361=<FI|BBAR|IC|FBAR|UP>, FORM341=<FI|SELECTIVE_ESNS>
```

- `SECNUM` = 메시 데이터에서의 `!SECTION` 입력 행 번호
  (`04_keyword_reference_cnt.md:2754`).
- `FORM361 = FBAR`(비선형 해석 기본값) / `IC`(선형 해석 기본값) /
  `BBAR` / `FI` / `UP`
  (`04_keyword_reference_cnt.md:2754`).
- `FORM341 = FI`(일반 1차 사면체 기본값) / `SELECTIVE_ESNS`(활성화 옵션)
  (`04_keyword_reference_cnt.md:2755`).
- `FORM611 = EULER`(기본값) / `TIMOSHENKO`
  (`04_keyword_reference_cnt.md:2755`).

---

## 3. "FrontISTR Completed !!" 성공 판정

`AGENTS.md:26–27` 핵심 규칙: `fistr1.exe` 실행 후 종료 코드
(`returncode == 0`)만 확인하지 말고, 표준 출력 로그에
**`FrontISTR Completed !!`**가 출력되었는지 반드시 검증하라.

정상 종료 로그 꼬리 (`01_execution_guide.md:545–556`):

```text
 fstr_setup: OK
### FSTR_SOLVE_NLGEOM FINISHED!

 ====================================
    TOTAL TIME (sec) :     59.99
           pre (sec) :      0.71
         solve (sec) :     59.29
 ====================================
 FrontISTR Completed !!
```

판정 규칙:

1. `FrontISTR Completed !!` 문자열 존재 = 해석 완료
   (`01_execution_guide.md:556`: "표시되면 해석이 완료된 것입니다").
2. 부등호 주의: `FSTR_SOLVE_* FINISHED!`는 솔버 단계 종료 표시일 뿐,
   최종 성공 판정은 `FrontISTR Completed !!` 한 줄이다.
3. 자동화 판정 (헬퍼·스크립트 연동 시): `returncode == 0 AND
   ("FrontISTR Completed !!" in stdout)`를 모두 만족해야 성공.
   비선형 미수렴·중단 시나리오는 returncode만으로 가려지지 않으므로
   로그 문자열 검사가 필수이다.

---

## 4. 요소별 정식화

### 4.0 요소 번호 체계와 자유도

`02_functions_reference.md:208` 규칙: 요소 타입 번호는 3자리, 첫째 자리는
대략의 형상·차원, 뒤 2자리는 차수와 배열을 나타낸다
(예외: 2D 트러스 301, 9절점 셸 743).

솔리드·셸 번호–절점–자유도 일람
(`07_coding_guide_and_release_notes.md:48–63`,
`02_functions_reference.md:233–256`):

| 번호 | 명칭 | 절점 수 |節당 자유도| 비고 |
|------|------|--------|-----------|------|
| 341 | 1차 사면체 (TET4) | 4 | 3 | FI / SELECTIVE_ESNS |
| 342 | 2차 사면체 (TET10) | 10 | 3 | — |
| 351/352 | 1차/2차 삼각주 (6/15절점) | 6/15 | 3 | — |
| 361 | 1차 육면체 (HEX8) | 8 | 3 | FI/BBAR/IC/FBAR/UP |
| 362 | 2차 육면체 Serendipity (HEX20) | 20 | 3 | 변환 불필요 |
| 731 | MITC3 삼각형 셸 | 3 | 6 | INFINITESIMAL 재료에서만 |
| 741 | MITC4 사각형 셸 | 4 | 6 | 유한회전 가능 (유일) |
| 743 | MITC9 사각형 셸 | 9 | 6 | — |
| 761/781 | MITC3/4-shell361 혼합 (3×2/4×2) | — | 3 | 솔리드 결합용 |

소스 상수 정의 (`element/element.f90:76–100`):

```fortran
fe_tet4n = 341 | fe_tet10n = 342 | fe_hex8n = 361 | fe_hex20n = 362
fe_mitc3_shell = 731 | fe_mitc4_shell = 741 | fe_mitc3_shell361 = 761 | fe_mitc4_shell361 = 781
```

절점 순서는 HEC-MW 입력 순서와 FrontISTR 내부 순서가 다르며,
`hecmw2fstr_mesh_conv`가 변환 테이블로 흡수한다
(`07_coding_guide_and_release_notes.md:88–90`):

- 342 (2차 사면체): `Table342`, 10절점을
  `{1,2,3,4,7,5,6,8,9,10}` 순서로 재배열.
- 352 (2차 삼각주): `Table352`, 15절점을
  `{1,2,3,4,5,6,9,7,8,12,10,11,13,14,15}` 순서로 재배열.
- 362 (2차 육면체): **변환 불필요** (HECMW 입력과 내부 순서 일치).
- 출력 역변환 `fstr2hecmw_mesh_conv`가 HEC-MW 순서로 되돌리므로,
  사용자는 입·출력 모두 HEC-MW 순서만 알면 된다.

형상함수 (아래 수식은 `05_theory_reference.md:904–957` 원문, 생략 없음):

자연좌표: 육면체
\(\boldsymbol{r} = (\xi, \eta, \zeta), -1 \le \xi,\eta,\zeta \le 1\);
사면체 체적좌표 \((L_1,L_2,L_3,L_4), \sum L_i = 1\)
(`05_theory_reference.md:909–910`).

- 341 (1차 사면체):
  \[ N_1 = L_1, \quad N_2 = L_2, \quad N_3 = L_3, \quad N_4 = L_4. \]
- 361 (1차 육면체):
  \[ N_\alpha = \tfrac{1}{8}(1+\xi_\alpha\xi)(1+\eta_\alpha\eta)(1+\zeta_\alpha\zeta),
     \quad \alpha=1,\dots,8, \]
  \((\xi_\alpha,\eta_\alpha,\zeta_\alpha)\)는 절점 자연좌표 (각 성분 \(\pm 1\)).
- 342 (2차 사면체, 모서리 중점 5–10은 모서리 1-2, 2-3, 1-3, 1-4, 2-4, 3-4
  위에 대응):
  \[
  \begin{aligned}
  N_1 &= L_1(2L_1-1), & N_2 &= L_2(2L_2-1), \\
  N_3 &= L_3(2L_3-1), & N_4 &= L_4(2L_4-1), \\
  N_5 &= 4L_1L_2, & N_6 &= 4L_2L_3, \\
  N_7 &= 4L_1L_3, & N_8 &= 4L_1L_4, \\
  N_9 &= 4L_2L_4, & N_{10} &= 4L_3L_4.
  \end{aligned}
  \]
- 362 (2차 육면체 Serendipity, 20절점):
  모서리 절점 (\(\alpha=1,\dots,8\)):
  \[ N_\alpha = -\tfrac{1}{8}(1+\xi_\alpha\xi)(1+\eta_\alpha\eta)(1+\zeta_\alpha\zeta)
     (2-\xi_\alpha\xi-\eta_\alpha\eta-\zeta_\alpha\zeta), \]
  중변 절점 (\(\alpha=9,\dots,20\); \(\xi_\alpha=0\)인 경우):
  \[ N_\alpha = \tfrac{1}{4}(1-\xi^2)(1+\eta_\alpha\eta)(1+\zeta_\alpha\zeta), \]
  \(\eta\)·\(\zeta\) 방향 중변 절점도 대응 좌표를 치환하여 동일 형태.

### 4.1 361 (HEX8) — IC 기본 / FBAR vs FORM361=FI/BBAR

`!SECTION` 파라미터 허용 집합은 소스에 하드코딩되어 있다
(`common/fstr_ctrl_common.f90:425–439`):

```fortran
character(19) :: form341list = 'FI,SELECTIVE_ESNS '
character(19) :: form361list = 'FI,BBAR,IC,FBAR,UP '
...
fstr_ctrl_get_param_ex(ctrl, 'FORM341 ', form341list, ...)
fstr_ctrl_get_param_ex(ctrl, 'FORM361 ', form361list, ...)
```

5종 의미 (`02_functions_reference.md:271–279`):

- `FI` — 완전적분 표준 요소.
- `BBAR` — 체적 변형률 선택적 저감적분. 체적 잠김(volumetric locking)
  완화, 전단 잠김에는 무력하므로 F-Bar가 일반적 권장.
- `IC` (Incompatible modes) — 부가 내부 자유도로 굽힘 성능 개선.
  **선형 해석 기본값.**
- `FBAR` — 변형 구배의 체적부를 중심값으로 치환.
  **비선형 대변형 기본값.**
- `UP` (u-p 혼합) — 비압축성 전용, 압력 자유도 추가.

해석 종류별 기본값 분기 (`common/fstr_setup.f90:394–409`):

```fortran
if (solution_type == STATIC .or. solution_type == DYNAMIC) then
  if (nlgeom) then
    elemopt361 = kel361FBAR   ! 대변형 → F-bar
  else
    elemopt361 = kel361IC     ! 미소변형 → IC
  end if
else if (solution_type == EIGEN) then
  elemopt361 = kel361IC
else if (solution_type == STATICEIGEN) then
  elemopt361 = kel361FBAR
else
  elemopt361 = kel361FI
end if
elemopt341 = kel341FI
```

내부 정수 매핑 (`lib/m_fstr.F90:84–88`):
`kel361FI=1, kel361BBAR=2, kel361IC=3, kel361FBAR=4, kel361UP=5`.

서브루틴 대응 (`lib/static_LIB_3dIC.f90`,
`lib/static_LIB_C3D8.f90`, `lib/static_LIB_Fbar.f90`,
`lib/static_LIB_3d_up.f90`):

- IC: `STF_C3D8IC` (`static_LIB_3dIC.f90:21`), `UPDATE_C3D8IC` (`:222`),
  `TLOAD_C3D8IC` (`:684`).
- B-bar: `STF_C3D8Bbar` / `Update_C3D8Bbar` / `TLOAD_C3D8Bbar`
  (`static_LIB_C3D8.f90:23,205,457`).
- F-bar: `STF_C3D8Fbar` / `Update_C3D8Fbar` / `TLOAD_C3D8Fbar`
  (`static_LIB_Fbar.f90:22,340,663`).

B-bar 핵심식 (`05_theory_reference.md:987`):

\[ \Delta B_{i\alpha} = \tfrac{1}{3}(\bar{B}_{i\alpha}(\boldsymbol{0}) - B_{i\alpha}(\boldsymbol{r})), \]

\(\bar{\boldsymbol{B}}\)는 요소 중심 \(\boldsymbol{r}=\boldsymbol{0}\)에서
평가한 B 행렬, 체적 변형률 \((\varepsilon_{11},\varepsilon_{22},
\varepsilon_{33})\) 성분에만 적용하고 전단 성분
\((\varepsilon_{12},\varepsilon_{23},\varepsilon_{31})\)은 일반
\(\boldsymbol{B}\)를 쓴다. 소스 구현
(`static_LIB_C3D8.f90:102–104`):

```fortran
B4 = ( Bbar(j, 1)-gderiv(j, 1) )/3.0D0
B6 = ( Bbar(j, 2)-gderiv(j, 2) )/3.0D0
B8 = ( Bbar(j, 3)-gderiv(j, 3) )/3.0D0
```

F-bar 핵심식 (`05_theory_reference.md:1000`):

\[ \bar{\boldsymbol{F}} = \left(\frac{J_0}{J}\right)^{1/3}\boldsymbol{F},
   \quad J_0 = \det\boldsymbol{F}(\boldsymbol{0}),\;
   J = \det\boldsymbol{F}(\boldsymbol{r}), \]

\(\det\bar{\boldsymbol{F}} = J_0\)이 되어 요소 전체 체적이 중심값으로
통일된다. 소스 구현 (`static_LIB_Fbar.f90:176`):

```fortran
Fbar(1:ndof, 1:ndof) = Jratio(LX)*(I33(1:ndof,1:ndof) + gdispderiv(1:ndof, 1:ndof))
```

IC 핵심식 (`05_theory_reference.md:1017` 이후):

\[ \boldsymbol{u}(\boldsymbol{r}) = \sum_{\alpha=1}^{8} N_\alpha^e(\boldsymbol{r})\,
   \boldsymbol{u}^e_\alpha + \sum_{k=1}^{3} M_k(\boldsymbol{r})\,\boldsymbol{\alpha}_k, \]

\(M_1 = 1-\xi^2,\ M_2 = 1-\eta^2,\ M_3 = 1-\zeta^2\).
응축 후 요소 강성:

\[
\begin{bmatrix} \boldsymbol{K}_{dd} & \boldsymbol{K}_{d\alpha} \\
\boldsymbol{K}_{\alpha d} & \boldsymbol{K}_{\alpha\alpha} \end{bmatrix}
\begin{bmatrix} d\boldsymbol{u}^e \\ d\boldsymbol{\alpha} \end{bmatrix} =
\begin{bmatrix} \boldsymbol{F}^e_{\text{ext}} \\ \boldsymbol{0} \end{bmatrix},
\qquad
\boldsymbol{K}^e = \boldsymbol{K}_{dd} - \boldsymbol{K}_{d\alpha}\,
\boldsymbol{K}_{\alpha\alpha}^{-1}\,\boldsymbol{K}_{\alpha d}.
\]

**위상최적화 민감도 해석 시 `!SECTION FORM361` 명시 이유:**

1. 기본값이 해석 종류에 따라 자동 전환되므로
   (`fstr_setup.f90:394–409`), 민감도·随伴 해석 체인에서 정식화가
   바뀌면 강성행렬·随伴 하중이 불일치한다. `FORM361=FI` (또는 의도한
   `BBAR`/`IC`)를 `.cnt`에 명시해야 정방향–随伴 해석이 동일 연산자를 쓴다
   (`04_keyword_reference_cnt.md:2754`, `AGENTS.md:32`:
   `!SECTION, SECNUM=<n>, FORM361=FI` 권장).
2. `B-bar` 근거: 체적 변형률을 중심값으로 치환하므로
   Hughes 선택적 저감적분 일반화
   (`static_LIB_C3D8.f90:17–19` 주석: "Generalization of Selective
   Integration Procedures")에 해당한다. 1차 육면체 완전적분이
   비압축 극한에서 보이는 체적 잠김을 제거하는 정식 근거이다.
3. 대변형 추적 시 일관 접선 강성이 엄밀 구현된 것은 `FI`/`BBAR`
   경로이므로, Total Lagrangian 2차 뉴턴 수렴을 위해서는
   `FORM361=FI` (또는 `BBAR`) + `!SOLVER, METHOD=MUMPS` 조합을 권장한다
   (`AGENTS.md:32`).

### 4.2 341 (TET4) — FI vs SELECTIVE_ESNS

허용 집합은 `FORM341 = FI / SELECTIVE_ESNS`, 기본값은 항상 `FI`
(`common/fstr_ctrl_common.f90:425`, `common/fstr_setup.f90:409`,
`02_functions_reference.md:287`).

- `FI` — 일반 1차 사면체 강성 (`lib/static_LIB_3d.f90:STF_C3` 계열,
  `:50`).
- `SELECTIVE_ESNS` — edge-based smoothing을 요소 선택적으로 적용하는
  평활화 FEM. 강성·열전도 행렬에 평활 변형률을 사용하고
  (`lib/static_LIB_C3D4_selectiveESNS.f90:193 STF_C3D4_SESNS`,
  `:374 UPDATE_C3_SESNS`), 평활 활성 시
  `fstrSOLID%is_smoothing_active = .true.`가 켜진다
  (`common/fstr_setup.f90:1500`).

내부 정수 매핑 (`lib/m_fstr.F90:81–82`):
`kel341FI=1, kel341SESNS=2`.

판정 지침 (`02_functions_reference.md:293`): 341 영역에 평활화가 필요하고
표준 `FI`로 품질이 부족하다고 판단될 때 `SELECTIVE_ESNS`를 쓴다.
361 사용이 가능하면 361의 `IC`/`BBAR`/`FBAR`가 일반적으로 더 좋은
선택이다.

### 4.3 SELECTIVE_ESNS + DEPTH=2 근거

`02_functions_reference.md:295` 원문 취지: `SELECTIVE_ESNS`는 MPI 병렬 시
반드시 이웃 층까지 필요하므로 `!PARTITION`에 `DEPTH=2`를 지정한다.
기본값 `DEPTH=1`로는 경계 층의 평활화에 필요한 이웃 요소 정보가
모자라 결과가 틀어진다. 동일 취지가
`02_functions_reference.md:2175`에 재확인되어 있다:

> 1차 사면체 341의 평활화 정식화(`FORM341=SELECTIVE_ESNS`)를 MPI 병렬과
> 함께 쓰는 경우 오버랩 2 이상이 필요 … 기본값 `DEPTH=1`로는 경계 층
> 평활화가 수행되지 않는다.

### 4.4 342/362 — 2차 요소 절점 순서

- 342 (TET10): 모서리 절점 1–4 + 중변 절점 5–10
  (모서리 1-2, 2-3, 1-3, 1-4, 2-4, 3-4 대응,
  `05_theory_reference.md:942–948`).
- 362 (HEX20 Serendipity): 모서리 절점 1–8 + 중변 절점 9–20
  (`05_theory_reference.md:950–957`, `07_coding_guide:54`).
- HEC-MW 입력 순서와 내부 순서가 다른 것은 342/352뿐이며 변환 테이블이
  흡수한다 (§4.0). 362는 변환 불필요.

### 4.5 731/741 — MITC 셸 번호·절점 순서

- 731 = MITC3 삼각형 셸 (3절점, 절점당 6자유도),
  741 = MITC4 사각형 셸 (4절점, 절점당 6자유도)
  (`07_coding_guide_and_release_notes.md:60–61`,
  `02_functions_reference.md:251–253`).
- 구현: `lib/static_LIB_shell.f90` — `STF_Shell_MITC`,
  `STF_Shell_MITC33`, `UPDATE_Shell_MITC`, `UPDATE_Shell_MITC33`,
  `ElementStress_Shell_MITC` (`:28–34` 공개 선언),
  tying 평가는 `ShellMITC_TyingZeta` (`:105–111`)와
  `MITC_Tying` 모듈 연동 (`:9`).
- 이론 근거 (`05_theory_reference.md:1039–1043`):
  Reissner-Mindlin 가정 셸의 전단·막 잠김을 tying点 재보간으로 회피
  ([Dvorkin1984] [Bathe1986]). MITC9가 최대 케이스
  (tying 6점 × 5성분, `static_LIB_shell.f90:19–21`).
- 대변형 가용성 (`02_functions_reference.md:344–349`):
  **유한회전을 지원하는 셸은 741뿐**이다
  (Total Lagrange 기본, `!ELASTIC`에 `CAUCHY` 지정 시 Updated Lagrange).
  731/743은 `INFINITESIMAL` 재료에서만 사용 가능.
  611/641/761/781은 대변형 불가.
- 731 ↔ 761, 741 ↔ 781은 삼각형·사각형 셸의 솔리드 혼합 버전
  (절점당 3자유도, `02_functions_reference.md:322–323`).

---

## 5. 멀티머티리얼 EB_0001~EB_0064 버케팅과 SIMP/BESO/Level-Set 연동 프로토콜

### 5.1 전제: FrontISTR에 옵티마이저는 없다

§머리말 고지 반복: 소스 전체 검색에서 SIMP/BESO/Level-Set/TopOpt
구현은 0건이다. 따라서 최적화 반복은 **외부 드라이버**(Python/MATLAB
등)가 담당하고, FrontISTR는 각 반복의 정방향·随伴 해석기로 호출된다.

### 5.2 EB 버케팅 정의

설계 영역의 요소들을 밀도 구간별로 최대 64개 재료 그룹으로 나눈다:

| 항목 | 규격 |
|------|------|
| 그룹명 | `EB_0001` ~ `EB_0064` (64 버킷, 0-패딩 4자리) |
| `.msh` 귀속 | `!ELEMENT, TYPE=<341\|361\|...>, EGRP=EB_<nnnn>` — 요소→그룹 |
| `.msh` 재료 | `!MATERIAL, NAME=MAT_<nnnn>[, ITEM=...]` — 버킷별 재료 카드 |
| `.cnt`/`.msh` 연결 | `!SECTION, TYPE=SOLID, EGRP=EB_<nnnn>, MATERIAL=MAT_<nnnn>` + 필요 시 `FORM361`/`FORM341` |
| 버킷 수 산정 | SIMP 연속 밀도 → 64분위 양자화. BESO 이진 → `EB_0001`(공극, \(E_{\min}\)) + `EB_0064`(고체) 2버킷 사용이 표준이며, 중간 버킷은 이력·필터용으로 예약 |

`EGRP`는 임의 문자열이 허용되므로 (`03_keyword_reference_dat_msh_part.md:519–526`),
`EB_0001` 형태는 규격 위반이 아니다. `!SECTION`이 `EGRP`–`MATERIAL`을
1:1로 묶는 구조이므로 (`03_keyword_reference_dat_msh_part.md:1603–1618`),
버킷마다 다른 영률을 부여하는 멀티머티리얼 해석이 성립한다.

### 5.3 SIMP 연동

SIMP 보간 (외부 드라이버가 버킷 영률을 갱신):

\[ E_e(\bar{x}_e) = E_{\min} + \bar{x}_e^p\,(E_0 - E_{\min}), \quad p \ge 1, \]

\(\bar{x}_e\)는 필터·투영된 물리 밀도이다.

프로토콜 (반복 \(k\)):

1. 드라이버가 \(\bar{x}_e\)를 64분위로 양자화 → 각 요소의 `EGRP`를
   `EB_0001`~`EB_0064` 중 하나로 배정 (`.msh` 재생성).
2. 버킷 \(b\)의 영률 \(E_b = E_{\min} + \hat{x}_b^p\,(E_0-E_{\min})\)
   (\(\hat{x}_b\)는 버킷 대표 밀도)을 `!MATERIAL, NAME=MAT_<b>` 카드에 기록.
3. `.cnt`에는 `!SECTION, SECNUM=<n>, EGRP=EB_<b>, MATERIAL=MAT_<b>,
   FORM361=<고정값>` — **FORM361 명시 필수** (§4.1 이유 1).
4. `fistr1 -t T` (SMP) 또는 `hecmw_part1` + `mpiexec -np N fistr1 -t T`
   (Hybrid) 실행 → `FrontISTR Completed !!` 확인 (§3).
5. 변위·随伴 해석 결과에서 컴플라이언스 민감도
   \(\partial c/\partial \bar{x}_e = -p\,\bar{x}_e^{p-1}(E_0-E_{\min})\,
   \mathbf{u}_e^T \mathbf{K}_0 \mathbf{u}_e\)를 드라이버에서 조립
   (FrontISTR는 \(\mathbf{u}_e\), \(\mathbf{K}_0\) 근거장 제공).
6. OC/MMA 업데이트 → 1로 복귀. SIMP 연속화
   \(p = 1.0 \to +0.125/40\text{iter}\) 등은 드라이버 스케줄.

`E_{\min} > 0` (예: \(10^{-9}E_0\))을 유지해야 강성행렬이 정칙을 유지한다.

### 5.4 BESO 연동

- 이진 설계이므로 `EB_0001` (\(E_{\min}\), 공극) + `EB_0064` (\(E_0\),
  고체) 2버킷만 사용한다.
- 민감도 수(转能 밀도)는 FrontISTR 요소 변형률 에너지 출력에서
  드라이버가 계산하고, 진화율(ER)·필터 반경은 드라이버 파라미터이다.
- 공극 요소 삭제 대신 \(E_{\min}\) 버킷에 두어 메시 토폴로지를 고정하므로
  `hecmw_part1` 재분할 없이 `.msh` 재료 카드만 갱신하면 된다.

### 5.5 Level-Set 연동

- 레벨셋 함수 \(\phi\)의 부호로 고체/공극을 나누어 BESO와 동일한 2버킷
  (`EB_0001`/`EB_0064`) 매핑을 쓴다.
- 경계 이동(속도장 \(V_n\) = 형상 민감도)은 드라이버의 Hamilton-Jacobi
  업데이트가 담당하고, FrontISTR는 속도장 계산에 필요한 随伴장만 제공한다.
- 고정 메시 + ersatz material 방식이므로 재메싱 없이 버킷 재지정으로
  충분하다.

### 5.4–5.5 공통 주의 (정식화 고정)

随伴 해석 체인 전체에서 `FORM361`/`FORM341`을 고정하라. 특히 361 사용 시
`.cnt`의 모든 `!SECTION`에 동일 `FORM361`을 명시하고, 대변형 문제에서는
`FORM361=FI` (또는 `BBAR`) + `!SOLVER, METHOD=MUMPS`를 권장한다
(`AGENTS.md:32`, §4.1). 341 평활화 사용 시 Hybrid 실행은 `DEPTH=2`
(`02_functions_reference.md:2175`).

---

## 부록 A. 인용원 일람 (파일:행)

| # | 파일 | 행 | 내용 |
|---|------|----|------|
| 1 | `FrontISTR/AGENTS.md` | 16–17 | SMP/Hybrid 바이너리 배치 |
| 2 | `FrontISTR/AGENTS.md` | 26–27 | `FrontISTR Completed !!` 판정 규칙 |
| 3 | `FrontISTR/AGENTS.md` | 29–32 | 2D 대변형 361 돌출, `FORM361=FI`/`BBAR`+MUMPS 권장 |
| 4 | `fistr1/src/lib/element/element.f90` | 76–100 | 요소 번호 상수 (341/342/361/362/731/741/761/781) |
| 5 | `fistr1/src/lib/m_fstr.F90` | 81–88 | `kel341FI=1/SESNS=2`, `kel361FI=1/BBAR=2/IC=3/FBAR=4/UP=5` |
| 6 | `fistr1/src/common/fstr_ctrl_common.f90` | 425–439 | FORM341/FORM361 허용 문자열 |
| 7 | `fistr1/src/common/fstr_setup.f90` | 394–409 | 해석 종류별 361 기본값 분기 |
| 8 | `fistr1/src/common/fstr_setup.f90` | 1500–1539 | SESNS 평활 활성 플래그·요소 리스트 |
| 9 | `fistr1/src/lib/static_LIB_3dIC.f90` | 21, 222, 684 | `STF/UPDATE/TLOAD_C3D8IC` |
| 10 | `fistr1/src/lib/static_LIB_C3D8.f90` | 17–23, 102–104 | B-bar 근거 주석·\(\Delta B\) 구현 |
| 11 | `fistr1/src/lib/static_LIB_Fbar.f90` | 22, 176 | `STF_C3D8Fbar`·\(\bar F\) 구현 |
| 12 | `fistr1/src/lib/static_LIB_C3D4_selectiveESNS.f90` | 7–19, 193, 374 | SESNS 모듈·서브루틴 |
| 13 | `fistr1/src/lib/static_LIB_3d.f90` | 50 | `STF_C3` (FI 경로) |
| 14 | `fistr1/src/lib/static_LIB_shell.f90` | 9, 28–34, 105–111 | MITC tying·공개 API |
| 15 | `docs_ko/consolidated/01_execution_guide.md` | 50, 68–69, 89, 101, 108 | 3파일 체계·SMP/Hybrid 명령 |
| 16 | `docs_ko/consolidated/01_execution_guide.md` | 284, 313, 488, 511 | `hecmw_part1` 절차·`fistr1 -t 4` |
| 17 | `docs_ko/consolidated/01_execution_guide.md` | 545–556 | Completed 로그·판정문 |
| 18 | `docs_ko/consolidated/01_execution_guide.md` | 740–748 | `hecmw_ctrl.dat` 예제 |
| 19 | `docs_ko/consolidated/02_functions_reference.md` | 208, 233–256 | 번호 체계·지원 요소표 |
| 20 | `docs_ko/consolidated/02_functions_reference.md` | 271–295 | FORM361 5종·FORM341 2종 해설 |
| 21 | `docs_ko/consolidated/02_functions_reference.md` | 2175 | SESNS MPI `DEPTH=2` 요구 |
| 22 | `docs_ko/consolidated/03_keyword_reference_dat_msh_part.md` | 513–583, 683 | `!EGROUP`·`!ELEMENT` |
| 23 | `docs_ko/consolidated/03_keyword_reference_dat_msh_part.md` | 929–1115 | `!MATERIAL` |
| 24 | `docs_ko/consolidated/03_keyword_reference_dat_msh_part.md` | 1603–1618 | `!SECTION` EGRP–MATERIAL 연결 예제 |
| 25 | `docs_ko/consolidated/03_keyword_reference_dat_msh_part.md` | 1898 | `!PARTITION ... DEPTH=1` 예제 |
| 26 | `docs_ko/consolidated/04_keyword_reference_cnt.md` | 2754–2755 | `FORM361`/`FORM341` 문법 |
| 27 | `docs_ko/consolidated/05_theory_reference.md` | 904–957 | 형상함수 (341/351/361/342/352/362) |
| 28 | `docs_ko/consolidated/05_theory_reference.md` | 987–1023 | B-bar·F-bar·IC·u-p 이론식 |
| 29 | `docs_ko/consolidated/05_theory_reference.md` | 1039–1043 | MITC 셸 이론·지원 번호 |
| 30 | `docs_ko/consolidated/02_functions_reference.md` | 344–349 | 741 유한회전 유일·731/743 제한 |
| 31 | `docs_ko/consolidated/07_coding_guide_and_release_notes.md` | 48–90 | 번호–절점표·Table342/352 변환·362 불필요 |

## 부록 B. 미확인·외부 위임 사항

- SIMP/BESO/Level-Set 옵티마이저·민감도 조립식: FrontISTR 내 구현 없음.
  본 문서 §5의 수식 (\(E_e\) 보간, \(\partial c/\partial\bar{x}_e\),
  Hamilton-Jacobi)은 표준 TopOpt 이론의 외부 인용이며 소스 근거가 아니다.
- 342/362/731/741의 내부 적분점 수·적분차수는 본 세션에서 미실측
  (`quadrature.f90` 미판독). 필요 시 후속 판독 대상이다.
- `static_LIB_3d_up.f90` (u-p 혼합) 서브루틴 시그니처는 본 문서에서
  상세 인용하지 않았다 (`kel361UP=5` 존재만 확인).
