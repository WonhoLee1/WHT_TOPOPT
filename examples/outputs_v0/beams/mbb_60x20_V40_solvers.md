| method | compliance (own model) | common re-analysis 2x (PARDISO) | common 2x (Kratos C++) | common 2x (FrontISTR) | design volume | gray | iterations | converged | time |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| SIMP (p = 3, density filter, OC) [SCIPY] | 292.17 | 243.63 | 243.63 | 242.66 | 0.400 | 0.071 | 60 | False | 0.9 s |
| SIMP (p = 3, density filter, OC) [PARDISO] | 292.17 | 243.63 | 243.63 | 242.66 | 0.400 | 0.071 | 60 | False | 0.8 s |
| SIMP (p = 3, density filter, OC) [PYAMG] | 292.17 | 243.63 | 243.63 | 242.66 | 0.400 | 0.071 | 60 | False | 2.9 s |
| SIMP (p = 3, density filter, OC) [KRATOS] | 292.17 | 243.63 | 243.63 | 242.66 | 0.400 | 0.071 | 60 | False | 11.3 s |
| SIMP (p = 3, density filter, OC) [FRONTISTR] | 295.01 | 239.27 | 239.27 | 238.20 | 0.400 | 0.072 | 60 | False | 11.7 s |
