# GE Jet Engine Bracket High-Resolution Benchmark Summary

- **Mesh Resolution**: MC40 (40x15x25 grid, 15,000 elements, h=4.400 mm)
- **Material**: Ti-6Al-4V (Yield: 903.0 MPa, Density: 4.43 g/cm³)
- **Target Volume Fraction**: 35.0% (Nominal ~245 g, -25.2% vs 2013 Winner 327 g)
- **Total Algorithms Tested**: 2
- **Passive-Solid Skin**: none

## 1. Dual Evaluation Matrices

### (1) Static Stiffness Table
| Algorithm | Weighted Compliance (N*mm) | Vert (35.6kN) | Horiz (37.8kN) | 42° (42.3kN) | Torque (565N-m) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MMA (Svanberg 1987)** | 3,680,462,759.9 | 2,267,997,909.1 | 7,175,544,721.6 | 1,725,556,824.3 | 301,114,742.9 |
| **SE-AHCA (Fu & Tovar 2025/2026)** | 1,207,182,942.8 | 1,564,670,146.1 | 1,370,056,184.6 | 770,689,232.4 | 258,908,805.4 |

### (2) Stress, Mass & Safety Factor Table
| Algorithm | Final Mass (g) | Mass Reduction | 99.5% von Mises (MPa) | Peak von Mises (MPa) | Safety Factor (SF) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MMA (Svanberg 1987)** | 570.1 g | -72.0% | 836.9 MPa | 1376.9 MPa | **1.08** |
| **SE-AHCA (Fu & Tovar 2025/2026)** | 570.1 g | -72.0% | 302.6 MPa | 496.4 MPa | **2.98** |

## 2. Computational Profiling
| Algorithm | t_FEM (s) | t_Sens (s) | t_Opt (s) | t_Total (s) | Step Time (ms) | Speedup vs MMA |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MMA (Svanberg 1987)** | 5.07 s | 1.45 s | 0.72 s | **7.25 s** | 3624.0 ms | 1.00x |
| **SE-AHCA (Fu & Tovar 2025/2026)** | 4.83 s | 0.45 s | 0.10 s | **7.62 s** | 3808.8 ms | 0.95x |

## 3. Generated Artifacts
- Watertight Master STL: `ge_bracket_3d.stl`
- Progression GIF: `ge_bracket_progression.gif`
- 3-Panel Master Comparison View: `ge_bracket_master_comparison_3d.png`
- Hex8 Solid Voxel View: `ge_bracket_voxels_3d.png`
- CAD Surface 3D View: `ge_bracket_cad_3d.png`
- Convergence Curve: `ge_bracket_convergence.png`
- Interactive WebGL HTML Report: `ge_bracket_comprehensive_master_report.html`
- Master JSON Metrics: `ge_bracket_metrics.json`