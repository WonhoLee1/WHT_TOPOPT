| method | compliance (own model) | common re-analysis 2x (PARDISO) | common 2x (Kratos C++) | common 2x (FrontISTR) | design volume | gray | iterations | converged | time |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| SIMP (p = 3, density filter, OC) [PARDISO] | 272.00 | 216.02 | 216.02 | 215.59 | 0.394 | 0.064 | 220 | False | 6.6 s |
| SIMP + Heaviside (p = 3, beta 1->8, MMA) | 209.94 | 211.51 | 211.51 | 211.08 | 0.400 | 0.010 | 135 | True | 6.0 s |
| SEMDOT (non-penalised, smooth edges) | 210.15 | 218.95 | 218.95 | 218.68 | 0.400 | 0.043 | 60 | True | 3.7 s |
| BESO (adaptive evolutionary ratio) | 229.98 | 235.35 | 235.35 | 234.77 | 0.399 | 0.000 | 38 | True | 1.6 s |
| Level set (SIMP start, Hilbertian projection) | 198.46 | 268.81 | 268.81 | 266.99 | 0.400 | 0.047 | 20 | True | 1.6 s |
| SIMP + P-Norm Stress Constraint (MMA, peak σ_vM=1.71) | 263.93 | 213.89 | 213.89 | 213.50 | 0.400 | 0.060 | 120 | False | 88.0 s |
