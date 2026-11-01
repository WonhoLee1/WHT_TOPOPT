# `wht_topopt` 코어 Numba JIT 가속 및 5대 FEA 솔버 속도 프로파일링 리포트

## 1. 수치 무결성 교차 검증 (NumPy/SciPy vs Numba `@njit` 상대 오차)
- **강성행렬 조립 상대 오차 (`||K_numba - K_scipy||_F / ||K_scipy||_F`)**: `7.964e-17`
- **3D 단위 변형에너지 상대 오차 (`c_e0`)**: `1.822e-16`
- **3D DfAM 오버행 수반 민감도 상대 오차 (`backward`)**: `4.804e-16`

## 2. 코어 연산 Numba 가속비 (`ms/call`)
| 코어 연산 항목 | 기존 NumPy/SciPy COO (ms) | Numba `@njit` + Pre-indexed CSC (ms) | 가속 배율 (Speedup) |
|---|---:|---:|---:|
| 2D 강성행렬 조립 (180x60, 10.8k el) | 13.132 ms | 1.193 ms | **11.01x** |
| 3D 강성행렬 조립 (20x10x10, 2k Hex8) | 36.699 ms | 1.448 ms | **25.35x** |
| 3D 단위 변형에너지 (u_e^T Ke0 u_e) | 0.271 ms | 0.166 ms | **1.64x** |
| 3D von Mises 응력 (C0 B0 u_e) | 0.229 ms | 0.094 ms | **2.43x** |
| 3D DfAM 오버행 필터 (Forward+Backward) | 1.068 ms | 0.921 ms | **1.16x** |

## 3. 5대 FEA 선형 솔버 반복당 해석 시간 (`Half MBB 90x30`, 5,642 DOFs)
| 솔버 백엔드 | 구동 방식 | 1회 해석 시간 (ms) |
|---|---|---:|
| MKL PARDISO (pypardiso) | Native In-Memory / SMP | 18.75 ms |
| SciPy SuperLU (splu direct) | Native In-Memory / SMP | 47.56 ms |
| PyAMG (AMG-PCG) | Native In-Memory / SMP | 52.93 ms |
| Kratos C++ (Cached ModelPart) | Native In-Memory / SMP | 328.60 ms |
| FrontISTR v6.0 (fistr1.exe SMP) | Native In-Memory / SMP | 431.89 ms |
