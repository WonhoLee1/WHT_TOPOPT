# Topology-optimization references

PDFs live next to this file but are **not tracked by git** (`.gitignore`): most are the
authors' or publishers' copyright, saved here for local reading only. This index is tracked,
so a fresh checkout can re-download them from the links.

Downloaded 2026-10-30. Page counts are from the PDFs themselves.

## Level-set

| File | Reference | Used for |
|---|---|---|
| `2023_wegert_hilbertian-projection-levelset_arXiv-2305.03913.pdf` (23 p) | Z. J. Wegert, A. P. Roberts, V. J. Challis, *A Hilbertian projection method for constrained level set-based topology optimisation*, arXiv:2305.03913 (v4). https://arxiv.org/abs/2305.03913 | **Core algorithm of the level-set optimizer**: H¹ extension-regularisation of the sensitivities (`beta = 4 dx`), Gram-Schmidt projection onto the complement of the constraint sensitivities, normal velocity `v = sqrt(1-a^2) P g/|P g| + a mu/|mu|` with `a` from `<mu, v> = lambda C`, Godunov upwind Hamilton-Jacobi advection, CFL line search, reinitialisation. Table 1 has the default parameters. |
| `2026_jezek_sequential-simp-to-levelset_arXiv-2605.04735.pdf` (19 p) | O. Ježek, J. Kopačka, M. Isoz, D. Gabriel, *Sequential topology optimization: SIMP initialization for level-set boundary refinement*, arXiv:2605.04735 (2026-05). https://arxiv.org/abs/2605.04735 | SIMP result -> nodal density -> 0.5 iso-surface -> signed distance -> level-set refinement; adaptive lower bound `alpha_min,eff = alpha_min * min(1, |C|/tau)`. Reference code: https://github.com/jezekon/2026-Jezek-SeqTopOpt |
| `2024_wegert_gridaptopopt-jl_arXiv-2405.10478.pdf` (26 p) | Z. J. Wegert, J. Manyer, C. Mallon, S. Badia, V. J. Challis, *GridapTopOpt.jl: A scalable Julia toolbox for level set-based topology optimisation*, arXiv:2405.10478 (v2). https://arxiv.org/abs/2405.10478 | Reference implementation of the Hilbertian projection method (Julia); parallel/scalable design. |
| `2023_blauth_quasi-newton-levelset_arXiv-2303.15070.pdf` (22 p) | S. Blauth, K. Sturm, *Quasi-Newton Methods for Topology Optimization Using a Level-Set Method*, Struct. Multidisc. Optim. (2023), https://doi.org/10.1007/s00158-023-03653-2 , arXiv:2303.15070 (v3). | Possible later upgrade: L-BFGS in place of plain gradient steps. Not implemented. |
| `2025_oellerich_levelset-hamilton-principle_arXiv-2504.14892.pdf` (68 p) | J. Oellerich, T. Yamada, *A level set topology optimization theory based on Hamilton's principle*, arXiv:2504.14892 (v2). https://arxiv.org/abs/2504.14892 | Background only (level-set evolution as a modified wave equation). Not implemented. |

## BESO

| File | Reference | Used for |
|---|---|---|
| `2025_tian_beso-adaptive-evolutionary-ratio_CJME-38-122.pdf` (14 p) | Tian & Zhang, *Bi-Directional Evolutionary Topology Optimization with Adaptive Evolutionary Ratio for Nonlinear Structures*, Chinese J. Mech. Eng. 38:122 (2025), https://doi.org/10.1186/s10033-025-01276-w . **CC BY 4.0** (open access) | **Core algorithm of the BESO optimizer**: sensitivity number (Eq. 16), filter (17-18), history averaging (19), adaptive ER `ER = (ERmax-ERmin)/2 tanh(k(v_h-0.5)) + (ERmax+ERmin)/2` (20), `V_{k+1} = V_k (1 +- ER)` (21), convergence over the last N=5 iterations with tau=1e-3 (22). |
| `2007_huang-xie_beso-new-algorithm_JSME-49-4-1091.pdf` (9 p) | X. Huang, Y. M. Xie, M. C. Burry, *A New Algorithm for Bi-Directional Evolutionary Structural Optimization*, JSME Int. J. Series A 49(4) (2006), p. 1091. https://www.jstage.jst.go.jp/article/jsmec/49/4/49_4_1091/_pdf | The original BESO this line of work builds on (sensitivity filter, history averaging). |
| `2017_loeffelmann_beso-calculix_EngMech-0590.pdf` (4 p) | F. Löffelmann, *Failure Index Based Topology Optimization for Multiple Properties*, Engineering Mechanics 2017, paper 0590. https://engmechx.it.cas.cz/improc/2017/0590.pdf | Describes the `calculix/beso` code (https://github.com/calculix/beso, LGPLv3): mass add/removal ratios, decay after freezing, tolerance over the last 5 iterations. |

## HCA (hybrid cellular automata) and stress-based TO — SE-AHCA cited papers (added 2026-11-01)

Per-paper analysis: `investigations/se_ahca_cited_papers_summary_20261101.md`.

| File | Reference | Used for |
|---|---|---|
| `2010_penninger_hca-convergence-analysis_SMO-40.pdf` (12 p) | C. L. Penninger, L. T. Watson, A. Tovar, J. E. Renaud, *Convergence analysis of hybrid cellular automata for topology optimization*, SMO 40:271–282 (2010). https://eprints.cs.vt.edu/archive/00001068/01/hcaSMO08.pdf | HCA = fixed-point iteration; corrected KKT (stimulus `S_i=U_i/x_i`); neighbourhood averaging violates KKT (empty-neighbourhood verdict); disconnected `ρ(DG)<1` basins — convergence ≠ optimum. Read in full. |
| `2025_hca_transient-impact-stiffness-stress_Processes-13-3726.pdf` (27 p) | Z. Tong et al., *Application of Hybrid Cellular Automata Method for High-Precision Transient Stiffness Design of a Press Machine Frame*, Processes 13(11):3726 (2025). https://doi.org/10.3390/pr13113726 . **CC BY** | 3D→2D-shell HCA (SED homogenisation, 35 iters) → 2D→3D rib restore → CCD response surface + PSO sizing; relative displacement −31.02%. Direct precedent for SE-AHCA Mode 1 (external-solver gradient-free loop). Read in full. |

Not downloaded (with reason): Tovar et al. 2006 HCA original, JMD 128 (ASME paywall + ResearchGate block; triply cross-reconstructed from Penninger/Tong/survey); Raeisi et al. 2026 adaptive HCA, Eng Opt, DOI `10.1080/0305215X.2025.2608738` (T&F 403; cite as plan-level until verified); battery-pack multi-case HCA + von Mises 2026, ResearchGate 398115713 (bibliography unverifiable — citation ON HOLD); Wang–Song–Kan harmonic HCA, JMD 147(5):051702 (2025-05, not 2024; ASME/RG 403, abstract only); Le et al. 2010 stress qp-relaxation, SMO 41:605–620, DOI `10.1007/s00158-009-0440-y` (Springer paywall; abstract via Illinois Experts).

## Dynamic response and ESL (equivalent static loads)

See `DYNAMICS_SURVEY.md` for the survey, the ESL algorithm and how it would map onto `aieng.topopt`.

| File | Reference | Used for |
|---|---|---|
| `2025_buzaglo_first-order-ESL-dynamic_arXiv-2508.07222.pdf` (31 p) | M. Buzaglo, N. Pollini, *First-order equivalent static loads for dynamic response structural optimization*, arXiv:2508.07222 (2025-08). https://arxiv.org/abs/2508.07222 | ESL Algorithm 1 and eq. 3-13 (read in full); the first-order F-ESL fix; classification of dynamic optimisation (free / forced / transient). |
| `2017_stolpe_ESLM-does-not-converge-to-KKT_DTU-postprint.pdf` (18 p) | M. Stolpe, A. Verbart, S. Rojas-Labanda, *The equivalent static loads method for structural optimization does not in general generate optimal designs*, Struct. Multidisc. Optim. 58:139-154 (2018), https://doi.org/10.1007/s00158-017-1884-0 (DTU postprint) | Why plain ESL cannot be certified as an optimum of the dynamic problem (two-bar-truss counter-examples). |
| `2017_behrou-guest_transient-response-topopt_arXiv-1705.01538.pdf` (17 p) | R. Behrou, J. K. Guest, *Topology optimization for transient response of structures subjected to dynamic loads*, arXiv:1705.01538 (2017-05) | Direct time-domain adjoint (SIMP + Heaviside projection + MMA); the exact reference for linear transient problems. Only the abstract was read so far. |
| `2023_li_online-successive-reanalysis-dynamic-topopt_arXiv-2305.08110.pdf` (45 p) | S. Li, H. Wang, J. Yin, D. Guo, G. Li, *An efficient online successive reanalysis method for dynamic topology optimization*, arXiv:2305.08110 (2023-05) | Reanalysis to cut the cost of transient topology optimisation. Not read yet. |

| `2022_triller_DiESL-topology-SIMP_SMO-65-2022.pdf` (17 p) | J. Triller, R. Immel, L. Harzheim, *Topology optimization using difference-based equivalent static loads*, SMO 65:226 (2022), https://doi.org/10.1007/s00158-022-03309-7 . **CC BY** (open access) | **DiESL definition for geometric/material nonlinear dynamics** (eq. 1-14): deformed-geometry linear submodels, incremental loads `K_i Δu_i`, chained surrogate displacement, density interpretation, `p_NL=1`, move limits, stopping rules. Read in full; summary in `DYNAMICS_SURVEY.md` section 10. |

| `2023_reduced-basis-adjoint-dynamic-topopt_arXiv-2308.09864.pdf` | S. Li, H. Wang, J. Yin, X. Jiang, Y. Zhang et al., *A novel reduced basis method for adjoint sensitivity analysis of dynamic topology optimization*, arXiv:2308.09864 (2023-08) | Adjoint on a reduced basis (greedy-POD) for linear transient topology optimisation: the "exact gradients on a modal/reduced model" route that is the proposed reference for linear modal data. Not read yet; read first before implementing `adjoint_modal.py`. |

Known from search summaries only (paywalled or 403 on download, so not saved): Cao et al.,
CMAME 436 (2025), https://www.sciencedirect.com/science/article/abs/pii/S0045782525000325 ;
*Overview of the Equivalent Static Load Method for Nonlinear Response Topology Optimization*,
Acta Mech. Solida Sin. (2026), https://link.springer.com/article/10.1007/s10338-026-00838-2 ;
*Reduced-order methods for dynamic problems in topology optimization: A comparative study*,
CMAME (2021), https://orbit.dtu.dk/en/publications/reduced-order-methods-for-dynamic-problems-in-topology-optimizati/ ;
Altair OptiStruct ESLM, https://help.altair.com/2024/hwsolvers/os/topics/solvers/os/opt_composite_equiv_static_load_method_c.htm .

## Software referenced (not downloaded)

* `mma-dual` https://pypi.org/project/mma-dual/ and `mmapy` (GCMMA-MMA-Python) https://github.com/arjendeetman/GCMMA-MMA-Python : MMA / GCMMA (both GPLv3).
* `OpenLSTO` https://github.com/M2DOLab/OpenLSTO (C++, Apache-2.0) and `GridapTopOpt.jl`: level-set implementations consulted.
* SEMDOT references used for `aieng.topopt.semdot` (code read for the algorithm flow only, nothing copied):
  - `nedu10/optiworks`, `src/freeTO` https://github.com/nedu10/optiworks (MATLAB, **MIT**, last push 2024-09): 3D SEMDOT and SIMP on structured hex meshes with STL domains (FreeTO); 3D flow: OC update, nodal filter, trilinear grid points, tanh Heaviside with bisection threshold, beta capped at 2.
  - `ooibhadode/IbIPP` https://github.com/ooibhadode/IbIPP (MATLAB, **MIT**, last push 2024-02): 2D image-based initialisation and post-processing (extrude/revolve to STL) hosting Density, BESO, level set and SEMDOT; its `semdot.m` is adapted from the DTU 88-line code, whose own educational terms are separate from the repo's MIT license. Defaults: beta0 = 2, step 0.5, ngrid = 4, rmin = 2.
  - PMC12113074, *Multiscale topology design based on non-penalisation SEMDOT* (2025, open access; PDF not retrievable, HTML read): non-penalised stiffness `E = Psi E_s`, discrete sensitivity eq. 12-13, beta0 = 0.5 and step 0.5, rho_min = 1e-6.

## Not read in full

The extracts above for Wegert 2023 and Ježek 2026 come from the arXiv HTML pages (the first
100,000 characters of each); the BESO 2025 method section was read in full from the PDF text.
For the dynamics set, the ESL formulation sections of Buzaglo & Pollini 2025 and the abstract
of Stolpe et al. 2018 were read from the PDF text; the other PDFs were only located and saved.

## 2019-2026 methods beyond SIMP/BESO (added 2026-10-30)

Survey in `METHODS_SURVEY_2026.md`, framework proposal in `FRAMEWORK_PLAN.md`. All arXiv PDFs below were downloaded
the same day; only abstracts/introductions or first pages were read unless the survey says otherwise.

| File | Reference |
|---|---|
| `2019_hoyer_neural-reparameterization_arXiv-1909.04240.pdf` | Hoyer, Sohl-Dickstein, Greydanus, *Neural reparameterization improves structural optimization*, arXiv:1909.04240 |
| `2021_zehnder_ntopo-implicit-neural_arXiv-2102.10782.pdf` | Zehnder et al., *NTopo: mesh-free topology optimization using implicit neural representations*, arXiv:2102.10782 |
| `2024_sanu_neural-topopt-good-bad-ugly_arXiv-2407.13954.pdf` | Sanu, Aragon, Bessa, *Neural topology optimization: the good, the bad, and the ugly*, arXiv:2407.13954 |
| `2025_kuszczak_meta-neural-topopt_arXiv-2502.01830.pdf` | Kuszczak et al., *Meta-neural topology optimization*, arXiv:2502.01830 |
| `2022_xue_jax-fem_arXiv-2212.00964.pdf` | Xue et al., *JAX-FEM*, arXiv:2212.00964 (code GPL-3.0) |
| `2025_he_soptx-multibackend_arXiv-2505.02438.pdf` | He, Wei, Tian, *SOPTX: multi-backend framework for topology optimization*, arXiv:2505.02438 (read abstract + intro) |
| `2026_cattaneo_jetsci-jax-petsc_arXiv-2604.22087.pdf` | Cattaneo et al., *JetSCI: hybrid JAX-PETSc framework*, arXiv:2604.22087 |
| `2025_nobari_pgd-to-alternative-to-MMA_arXiv-2511.13905.pdf` | Nobari, Ahmed, *PGD-TO: scalable alternative to MMA*, arXiv:2511.13905 (code: pyFANTOM, license unknown) |
| `2025_ma_gaussian-ensemble-topology-GET_arXiv-2510.05572.pdf` | Ma et al., *Gaussian Ensemble Topology (GET)*, arXiv:2510.05572 |
| `2025_chen_phase-field-robust-stable_arXiv-2507.16519.pdf` | Chen et al., *A robust and stable phase field method for structural topology optimization*, arXiv:2507.16519 |
| `2025_phasefield-fourier-nn_arXiv-2511.14623.pdf` | *A unified phase-field Fourier neural network framework for topology optimization*, arXiv:2511.14623 (authors not extracted) |
| `2025_bastek_physics-informed-diffusion_ICLR_arXiv-2403.14404.pdf` | Bastek, Sun, Kochmann, *Physics-informed diffusion models*, ICLR 2025, arXiv:2403.14404 |
| `2025_lutheran_latent-diffusion-topopt_arXiv-2508.05624.pdf` | Lutheran, Das, Tabarraei, *Latent space diffusion for topology optimization*, arXiv:2508.05624 |
| `2026_lin_to-master-llm-agent_arXiv-2607.01812.pdf` | Lin et al., *TO-Master: an LLM-agent framework for automated topology optimization*, arXiv:2607.01812 (read abstract + intro) |
| `2026_park_topoptagents-llm-multiagent_arXiv-2605.23273.pdf` | Park, Chung, *Self-refining topology optimization via an LLM-based multi-agent framework*, arXiv:2605.23273 |
| `2026_yang_llm-simp-controller_arXiv-2603.25099.pdf` | Yang, Wang, Wang, *LLMs as optimization controllers: adaptive continuation for SIMP*, arXiv:2603.25099 |
| `2026_yang_matrixfree-gmg-single-gpu_arXiv-2604.26441.pdf` | Yang, Wang, Wang, *Matrix-free Galerkin multigrid solver and failure-mode screen for single-GPU 3D SIMP*, arXiv:2604.26441 |
| `2026_formopt-fenicsx-levelset_arXiv-2601.05709.pdf` | *FormOpt: a FEniCSx toolbox for level set-based shape optimization*, arXiv:2601.05709 |
| `2025_padhy_tomatoes-jax-checkpointed-transient_arXiv-2510.07057.pdf` | Padhy, Suresh, Chandrasekhar, *TOMATOES* (JAX, checkpointed transient AD), arXiv:2510.07057 |
| `2026_thombre_vasto-differentiable-to_arXiv-2606.05487.pdf` | Thombre et al., *VASTO: differentiable topology optimization for vascular geometry*, arXiv:2606.05487 |
| `2026_hosseinmardi_geometric-nonlinearity-thermomech-to_arXiv-2608.10344.pdf` | Hosseinmardi, Sun, Bostanabad, *Geometric nonlinearity and temperature-dependent properties in multi-material thermo-mechanical TO*, arXiv:2608.10344 |
| `2026_physics-informed-transformer-topopt_arXiv-2604.03522.pdf` | Lutheran, Das, Tabarraei, *Physics-informed transformer for real-time high-fidelity topology optimization*, arXiv:2604.03522 (first page) |
| `2025_tobaco-band-limited-coordinate-networks_arXiv-2508.10320.pdf` | Chandrasekhar et al., *TOBACO: TO via band-limited coordinate networks for compositionally graded alloys*, arXiv:2508.10320 (first page) |
| `2025_teixeira_isogeometric-topological-derivative_arXiv-2509.09236.pdf` | Teixeira, Krenn, Gangl, Marussig, *Isogeometric topology optimization based on topological derivatives*, arXiv:2509.09236 |
| `2022_chandrasekhar_gm-tounn-graded-multiscale_arXiv-2204.06682.pdf` | Chandrasekhar, Sridhara, Suresh, *GM-TOuNN: graded multiscale TO using neural networks*, arXiv:2204.06682 |
| `2024_chen_multiscale-topopt-neural-networks_arXiv-2404.08708.pdf` | Chen, Liu, Kara, *Multi-scale topology optimization using neural networks*, arXiv:2404.08708 |
| `2022_kumar_am-length-scale-overhang-orientation_arXiv-2204.07333.pdf` | Kumar, Fernandez, *TO for additive manufacturing with length scale, overhang and building orientation constraints*, arXiv:2204.07333 |
| `2024_gradient-free-neural-topopt-fracture_arXiv-2403.04937.pdf` | Kus, Bessa, *Gradient-free neural TO: towards effective fracture-resistant designs*, arXiv:2403.04937 |
| `2025_pozzi_adjoint-nonlinear-dynamics-spectral-submanifolds_arXiv-2503.17431.pdf` | Pozzi, Marconi, Jain, Li, Braghin, *Adjoint sensitivities for the optimization of nonlinear structural dynamics via spectral submanifolds*, arXiv:2503.17431 |
| `2017_chalmers-245215_multibody-topopt.pdf` | Ghandriz, Führer, Elmqvist, *Structural topology optimization of multibody systems*, Multibody Syst. Dyn. 39:1-14 (2017) (TOMBS; Chalmers open copy; abstract read) |
| `2018_tromme_system-based-approaches-flexible-multibody-topopt.pdf` | Tromme et al., *System-based approaches for structural optimization of flexible mechanisms*, 2018 review (weakly vs fully coupled; abstract read) |
| `2026_cosmic-structure-material-control-codesign_arXiv-2605.12654.pdf` | Guo, Wang et al., *COSMIC: concurrent optimization of structure, material and integrated control for robotic systems*, arXiv:2605.12654 (abstract read) |
| `2024_sdrs-shape-differentiable-robot-simulator_arXiv-2412.19127.pdf` | Ye, Gao, Wu, Pan, Komura, *SDRS: shape-differentiable robot simulator*, arXiv:2412.19127 (first page) |
| `2025_ghabraie_benchmarking-zhou-rozvany-binary-relaxed_CMES-143-3.pdf` | Zhou, Fu, Ghabraie, *Systematic benchmarking of TO methods using both binary and relaxed forms of the Zhou-Rozvany problem*, CMES 143(3) 2025 (CC BY; abstract + intro read) |
| `2020_fu_semdot-smooth-edged-material-distribution_arXiv-2005.09233.pdf` | Fu, Rolfe, *SEMDOT: smooth-edged material distribution for optimizing topology*, arXiv:2005.09233 (abstract read; includes 2D MATLAB code) |
| `2024_dunning_automatic-projection-parameter-increase_arXiv-2404.14111.pdf` | Dunning, *Achieving binary TO solutions via automatic projection parameter increase*, arXiv:2404.14111 (abstract read) |
| `2025_subpixel-smoothed-projection-levelset-density_arXiv-2503.20189.pdf` | *Unifying and accelerating level-set and density-based TO by subpixel-smoothed projection*, arXiv:2503.20189 (abstract read; photonics examples) |
| `2023_neural-level-set-unfitted-fem_arXiv-2303.13672.pdf` | *Neural level set topology optimization using unfitted finite elements*, arXiv:2303.13672 (not read) |
| `2026_multiobjective-nondominated-curves-compliance-volume_arXiv-2609.07149.pdf` | *Mapping locally nondominated curves in multiobjective TO of compliance and volume*, arXiv:2609.07149 (not read) |

Not downloaded (paywalled or not found): Triller et al. 2021 (sizing DiESL) and 2022 (adaptive ESL times), Ogawa 2021
(finite-deformation dynamic TO, IJNME), Pozzi et al. 2025 (SMO paper; its adjoint companion arXiv:2503.17431 is saved), Ivarsson 2018,
Weider 2018/19. Also: the 2022 Stolpe postprint fetched earlier was byte-identical to the 2017 file and was removed.

## Phase 0 downloads (2026-10-31, `download_missing_papers.py`)

Saved to `papers/` (all verified `%PDF-` header):

| File | Source |
|---|---|
| `04_Andreassen_2011_88line_SMO.pdf` (340794 B) | https://lirias.kuleuven.be/retrieve/614b80a6-c7ed-4ab2-a18a-884af2c51dfc |
| `12_Ferrari_Sigmund_Guest_2021_Buckling250_SMO.pdf` (518940 B) | https://arxiv.org/pdf/2010.03698.pdf |
| `21_Spinodal_Dehomogenization_2025.pdf` (5518724 B) | https://link.springer.com/content/pdf/10.1007/s00158-026-04248-3.pdf |
| `22a_GINOT_2025.pdf` (3406314 B) | https://arxiv.org/pdf/2606.16624 |
| `22b_TONIC_2026.pdf` (7459595 B) | https://link.springer.com/content/pdf/10.1007/s00366-026-02413-2.pdf |
| `25_Ferrari_Sigmund_2020_top99neo_top3d125_SMO.pdf` (3607433 B) | https://arxiv.org/pdf/2005.05436.pdf |

Still missing (paywall, no legal OA PDF found — cross-extract from follow-ups per instruction):
`01_Bendsoe_Sigmund_1999`, `06_Lazarov_Sigmund_2011_PDE_Filter`, `08_Le_2010_Stress`,
`14_Langelaar_2016_AM_3D`, `26_Langelaar_2019_MultiAxis_Machining`.
Fallbacks: `09_Holmberg_2013` / `10_daSilva_2019` (stress), `15_Langelaar_2017` (AM filter),
`16_Garaigordobil_2018` (overhang), DTU 88-line/top99neo codes for SIMP پایه.
