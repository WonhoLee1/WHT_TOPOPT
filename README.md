# `wht_topopt` — Next-Generation Topology Optimization Framework (v3.0)

`wht_topopt`는 정형(Structured 2D/3D) 및 비정형(Unstructured TRI3, Q4, TET4, TET10, HEX8, HEX20, MITC3, MITC4) 유한요소 격자 위에서 다중 솔버(**SciPy SuperLU/CG**, **PyAMG**, **Kratos Multiphysics 10.4.3 In-Memory/Subprocess**, **FrontISTR v6.0 SMP/MPI**)와 다중 최적화 알고리즘(**OC**, **MMA**, **BESO**, **Hilbertian Level-Set**, **SEMDOT**)을 결합한 통합 위상최적화 패키지입니다.

## 핵심 구성 모듈

- `wht_topopt.core`: 정형/비정형 메시(`mesh.py`), 해석적/가우스/B-bar 요소 강성 및 질량 행렬(`fem.py`, `fe_builders.py`), KDTree 밀도 필터 및 Helmholtz PDE 필터(`filter.py`), SIMP/RAMP/다중재료 보간(`interpolation.py`), 매끄러운 Heaviside 투영(`projection.py`), Continuation 스케줄러(`continuation.py`), 표준/서브모델파트 경계조건(`boundary.py`).
- `wht_topopt.solvers`: `FEASolver` & `TopOptEvaluator` 인터페이스(`base.py`), `ScipySolver`, `PyAMGSolver`, `KratosSolver` (Python 3.12 `E:\kratos-build\install` In-Memory 및 Subprocess 지원), `FrontISTRSolver` (`D:\PythonCodeStudy\FrontISTR` SMP & Hybrid MS-MPI 지원).
- `wht_topopt.algorithms`: `GradientSource` 신뢰 계층 및 Trust-Region 수용 제어(`base.py`), `OptimalityCriteria`, `MMAOptimizer` (Svanberg 1987/2002 주-쌍대 뉴턴 내점법), `BESOOptimizer` (적응형 ER), `LevelSetOptimizer` (Hilbertian $H^1$ 속도 연장 + HJ Upwind + SLERP), `SEMDOTOptimizer`.
- `wht_topopt.functions`: 컴플라이언스(`compliance.py`), 체적/질량(`volume.py`), P-norm 응력 제약(`stress.py`), KS 선형 좌굴(`buckling.py`), 고유진동수 및 유연 메커니즘(`modal.py`).
- `wht_topopt.manufacturing`: 최소 길이 스케일 3-Field(`minimum_length.py`), 적층제조 오버행(`overhang.py`), 다축 밀링(`milling.py`), 주조(`casting.py`), 압출(`extrusion.py`).
- `wht_topopt.multiscale`: 주기 경계조건 수치 균질화(`homogenization.py`), 스핀오달 디호모지나이제이션(`dehomogenization.py`).
- `wht_topopt.robust`: Eroded/Intermediate/Dilated 강건 정식화(`robust_projection.py`), MuJoCo 다물체 동역학 하중 및 관성 완화(`mujoco_loads.py`).
- `wht_topopt.post` & `wht_topopt.neural`: Marching Cubes/Squares + Taubin 스무딩(`isosurface.py`), STL/VTK 익스포트(`stl_export.py`), B-Spline DXF/MMG/Build123d STEP 복원(`cad_reconstruction.py`), GINOT/TONIC 신경망 인터페이스.
- `wht_topopt.pipeline`: `TopOptConfig`, `TopOptRunner`, `TopOptReport`.
