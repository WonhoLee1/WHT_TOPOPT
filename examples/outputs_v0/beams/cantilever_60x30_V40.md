| method | compliance (own model) | common re-analysis 2x (PARDISO) | common 2x (Kratos C++) | common 2x (FrontISTR) | design volume | gray | iterations | converged | time |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| SIMP (p = 3, density filter, OC) [PARDISO] | 92.73 | 79.34 | 79.34 | 79.15 | 0.400 | 0.061 | 106 | True | 1.8 s |
| SIMP + Heaviside (p = 3, beta 1->8, MMA) | 83.55 | 79.33 | 79.33 | 79.12 | 0.400 | 0.036 | 110 | True | 2.5 s |
| SEMDOT (non-penalised, smooth edges) | 75.41 | 77.71 | 77.71 | 77.55 | 0.400 | 0.041 | 60 | True | 2.8 s |
| BESO (adaptive evolutionary ratio) | 80.47 | 85.31 | 85.31 | 84.86 | 0.399 | 0.000 | 28 | True | 0.6 s |
| Level set (SIMP start, Hilbertian projection) | 74.63 | 81.52 | 81.52 | 81.30 | 0.399 | 0.045 | 13 | True | 1.0 s |
