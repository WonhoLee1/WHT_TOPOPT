# Optimization for dynamic behaviour: survey and implementation notes

Prepared 2026-10-30 as groundwork for an ESL (equivalent static loads) implementation and for
the P2 dynamic responses. Nothing here is implemented yet. Statements are labelled by how I
know them: **[read]** I read the text of the paper; **[abstract]** only an abstract/snippet;
**[search]** only a search-result summary; **[proposal]** my own suggestion, not from a paper.

## 1. Is there code to learn from?

* **ESL: no usable open-source implementation found.** GitHub search (`gh search repos/code`)
  for "equivalent static load(s)", "ESLM", "ESL crashworthiness" returned one repository,
  `22110414-Guo/Crashworthiness-topology-optimization` (0 stars, no licence, a single 2-byte
  Python file), and otherwise unrelated hits (equivalent static *wind/seismic* loads). Altair
  OptiStruct implements ESLM commercially (https://help.altair.com/2024/hwsolvers/os/topics/solvers/os/opt_composite_equiv_static_load_method_c.htm).
  So the implementation has to come from the papers below.
* Dynamic topology optimization in general (transient, harmonic, eigenfrequency): `gh search
  repos` returned nothing relevant either. Closest published code I could identify: none
  verified. (The MATLAB/Python 99/88-line codes are static.)

## 2. Methods, grouped by what is being optimized

Following Buzaglo & Pollini's classification **[read]**: free vibration, forced vibration,
transient response.

| Method | Use when | Sensitivities | Cost / catch | Source |
|---|---|---|---|---|
| **Eigenfrequency optimisation** (maximise fundamental frequency, bound formulation for multiple eigenvalues) | free vibration; lowest natural frequency above a target | eigenvalue derivative `phi^T (dK - w^2 dM) phi`; repeated eigenvalues need special handling | spurious localized modes in void elements unless mass interpolation is modified (RAMP / lower mass exponent) | Díaz & Kikuchi 1992; Pedersen 2000; Ferrari 2018 (cited in Buzaglo-Pollini) **[read: citation only]** |
| **Harmonic / frequency-response, direct** | forced vibration at one or few frequencies | adjoint | one complex solve per frequency; exact | **[search]** |
| **Frequency response + mode superposition** (mode displacement MDM, mode acceleration MAM) | many frequencies, modest size | adjoint on the reduced model | MAM suits harmonic loads, loses accuracy for rapidly varying excitation; MDM too inaccurate for transients | Springer SMO 2018 (10.1007/s00158-018-1898-2) **[search]** |
| **Reduced-order models for broadband response** (Quasi-Static Ritz Vectors, Padé, second-order Krylov SOAR/SOMMG, adaptive SOAR) | wide frequency bands, large models | adjoint equations reduced the same way | a comparative study found second-order Krylov most accurate/stable; Ritz vectors usually most efficient in a TU Delft thesis | CMAME 2021 comparative study (DTU; 403 on download); MDPI Mathematics 13(10):1583 (2025) **[search]** |
| **Transient, direct time-domain adjoint** (SIMP + Heaviside projection + MMA; Newmark / HHT-alpha) | linear transient response, exact gradients | adjoint solved backwards in time | cost ~ 2x the forward time history per iteration; memory for the stored states | Behrou & Guest, arXiv:1705.01538 **[abstract]** (PDF saved); review of transient optimisation (direct differentiation vs adjoint) **[search]** |
| **ESL, zeroth order** (Choi & Park 1999/2002; Kang, Park & Arora 2006; Lee & Park 2015) | nonlinear static, nonlinear dynamic, crash: turns the problem into linear static multi-load-case optimisation | reuses the *static* adjoint; loads held constant in each sub-problem | **does not in general converge to a KKT point** (see section 3) | Stolpe et al. 2018 **[read]**; Buzaglo & Pollini 2025 **[read]** |
| **Difference-based ESL (DiESL)** (Triller et al. 2021) | nonlinear dynamic, better response approximation | as ESL | reported to improve displacement/strain approximation; equivalent to ESL for *linear* dynamics (Stolpe 2023) | **[search]** + **[read: statement in Buzaglo-Pollini]** |
| **First-order ESL (F-ESL)** (Buzaglo & Pollini 2025) | as ESL, but with first-order optimality guarantees | adds `grad f_eq` to the load definition (needs dynamic sensitivities) | few outer iterations in their examples; sizing examples only | **[read]** |
| **ESL + adjoint (ESL-AVM)** (Cao et al., CMAME 436, 2025) | transient dynamic TO with approximate response sensitivity from ESL | adjoint on the ESL system, HHT-alpha | paywalled | **[search]** |
| **Successive iteration of analysis and design** (CMAME 2024, S0045782524000434) and **online successive reanalysis** (Li et al., arXiv:2305.08110) | transient TO where reanalysis dominates the cost | reanalysis-based | reuse of the previous factorisation / response | **[abstract]** |
| **Hybrid cellular automata (HCA)** (Patel 2007/2009; Mozumder; modified & adaptive HCA 2019/2025) | crash / energy absorption | none: heuristic rule driving internal energy density to a target | no gradients, nonlinear explicit dynamics usable as a black box | **[search]** |
| **Topometry / metamodel-based** | crash | surrogate | many analyses | **[search]** |

A caution from the crash literature **[search]**: a design from an equivalent *static* load can
differ sharply from one found with the dynamic analysis (one flat-plate HCA study reports a
peak reaction force 44.1 kN for the static-equivalent versus 9.3 kN for the dynamic design).

## 3. ESL: the algorithm and its known weakness

Reference dynamic response problem **[read, Buzaglo-Pollini eq. PD]**

    min f(x, u_k(x))   s.t.   g_j(x, u_k(x)) <= 0,   M(x) u''_k + C(x) u'_k + K(x) u_k = f_k,   k = 1..nt

ESL loop **[read, Algorithm 1]**

1. Start from `x_0`. Run the (nonlinear) dynamic analysis at the current design `x_W`.
2. For every time step `k` in the window: `f_eq,k = K(x_W) u_k(x_W)`   (eq. 3).
3. Solve the *linear static* problem with one load case per `k`:
   `min f(x, u~_k)  s.t.  K(x) u~_k = f_eq,k` (loads constant inside the sub-problem).
4. `x_{W+1} = x~_W`; stop when `||x_{W+1} - x_W|| <= eps`.

For a nonlinear *static* response `K` is the linear stiffness and `u` the nonlinear
displacement; for dynamics, `f_eq,k = -M u''_k - C u'_k + f_k` is the same thing **[read, eq. 8]**.

**Weakness [read].** Park & Kang (2003) and later papers claimed that a converged ESL point is a
KKT point of the dynamic problem. Stolpe, Verbart & Rojas-Labanda (SMO 58:139-154, 2018) give
reproducible two-bar-truss counter-examples where ESLM fails to find optimal designs. The reason
is that `grad f_eq` is not zero at the optimum, but ESL treats the loads as constant.
**F-ESL** keeps the first-order term, `f_eq(x) ~ f_eq(x_W) + grad f_eq(x_W)(x - x_W)`
(eq. 4-6), and the sub-problem sensitivities then include `K^-1 grad f_eq` (eq. 13). Its
numerical examples are sizing problems (a truss, a two-storey shear frame) solved with SQP;
topology optimisation is not demonstrated there **[read]**.

## 4. How this would map onto `aieng.topopt` **[proposal]**

What already exists and applies directly:

* **Multi-load-case static optimisation with weights** (`LoadCase.weight`), **displacement
  limits per load case** (`DisplacementLimit.load_case`, adjoint), MMA / GCMMA, OC. That *is*
  the ESL inner problem (step 3), with `nt` load cases.
* `RefStiffness` gives `K(x) u` without assembling `K` (sum of `E_e Ke0 u_e`), so `f_eq,k` is cheap
  once `u_k` is known.
* The solver adapters accept any number of load cases; FrontISTR runs one case per process.

What is missing:

1. **A transient/nonlinear solve in the adapters**, returning `u_k` (and ideally `u'_k`, `u''_k`)
   for chosen steps: FrontISTR has dynamic analysis (`!SOLUTION, TYPE=DYNAMIC`, Newmark) and
   nonlinear options; Kratos has Newmark/Bossak schemes. Not wired up. For *linear* dynamics a
   small numpy Newmark integrator with consistent/lumped `M(x)` would also serve as a test
   oracle (needs a density-to-mass interpolation that avoids spurious void modes, see section 2).
2. **Time-step selection or weighting.** `nt` can be hundreds, and each step is a load case
   (one solve each). Choi & Park used all intervals; practical codes weight or subsample around
   response peaks. An honest first version: user-chosen steps/weights, reported.
3. **Convergence safeguard.** Because of section 3, an ESL result must not be reported as an
   optimum of the dynamic problem on the strength of `||dx|| < eps`. Proposed: at every outer
   iteration re-run the true dynamic analysis, compare the *true* objective and constraint
   values, shrink the move limit / reject the step if the true problem got worse, and label
   the final status `esl_stationary_not_certified` unless F-ESL (first-order) terms are used.
4. **Dynamic sensitivities for F-ESL** (adjoint through the time integration, as in Behrou &
   Guest) if the first-order variant is wanted.

Suggested order: (a) eigenfrequency response (P2, needs a mass matrix and an eigen solve in the
core; independent of ESL); (b) linear transient with a built-in Newmark integrator plus direct
adjoint (Behrou & Guest) as the exact reference; (c) ESL on top of (b) so it can be checked
against (b) on linear problems, where ESL should reproduce the direct answer or show its gap;
(d) ESL with an external nonlinear dynamic solver (FrontISTR / Kratos); (e) F-ESL.

## 6. ESL variants compared (added 2026-10-30)

There is **no head-to-head benchmark** of the variants in anything I found; the judgements below
come from abstracts and the two papers I read. Treat "best" as an engineering recommendation.

| Variant | Strength | Weakness | Evidence |
|---|---|---|---|
| **Classical ESL** (Choi & Park 2002, all time steps; Kang-Park-Arora 2006 nonlinear) | simplest; the inner problem is plain multi-load-case static optimisation, so any static optimizer/solver is reused; OptiStruct ships it (ESLM) | loads frozen per sub-problem, so no KKT guarantee (Stolpe 2018); a linear model on the *initial* geometry cannot follow a nonlinear displacement path; one load case per time step; time-step choice is manual | **[read]** Stolpe, Buzaglo-Pollini |
| **Modal ESL** (Choi & Park 1999: ESL from modal analysis) | for linear dynamics the response is a few modal coordinates: `f_eq(t) = K Phi q(t) = M Phi Omega^2 q(t)` (derivation: `K phi_j = w_j^2 M phi_j`), so no full time history or stiffness product is needed; cheap, and it matches results we already get from a modal solver | linear only; mode truncation error (static correction helps); modes change with the design, so they must be recomputed or tracked each outer iteration; same KKT caveat | **[search]** (titles/abstracts) + derivation **[proposal]** |
| **DiESL** (Triller et al. 2021; adaptive time selection + local stiffness adaption 2021; topology 2022) | each auxiliary load case uses the *deformed* geometry of its time step and reproduces only the displacement *increment*: better displacement/strain approximation and faster convergence on nonlinear dynamics; one paper reports it finding the optimum where ESL failed | needs the deformed state per step (more models and bookkeeping); **identical to ESL for linear dynamics** (Stolpe 2023), so no gain there; quantitative stability not found | **[search]** |
| **F-ESL** (Buzaglo & Pollini 2025) | first-order term in the load definition: a converged point satisfies first-order optimality; few outer iterations | needs `grad f_eq`, i.e. dynamic sensitivities (adjoint through time); demonstrated on small sizing problems only, not topology/nonlinear | **[read]** |
| **First-order ESL, nonlinear static** (Stolpe & Pollini 2023) | convergence to first-order points with a trust region | static nonlinear response only | **[search]** |
| **ESL + adjoint (ESL-AVM)** (Cao et al. 2025) | approximate dynamic sensitivities from the ESL system; HHT-alpha | paywalled; details unknown | **[search]** |

**Most stable / best performing.** For *nonlinear* dynamics the best-supported choice is
**DiESL (adaptive)**, because it is the only variant with reported gains in approximation quality
and convergence, and it already exists in a topology form. For *linear / modal* dynamics DiESL = ESL,
and **modal ESL** is the cheapest and the natural fit; its weakness (no KKT guarantee) is handled
by the safeguard in section 4 item 3 and, if needed, F-ESL's first-order term. F-ESL has the
strongest theory but the thinnest evidence. A **trust region / move limit plus acceptance on the
true dynamic response** is what makes any of them stable in practice **[proposal; Stolpe & Pollini
use a trust region for the static case]**.

## 7. Beyond HCA, and beyond ESL

* **HCA** (hybrid cellular automata): heuristic rule that drives internal energy density to a
  target. Gradient-free, so it can wrap a nonlinear explicit crash solver as a black box and needs
  one analysis per iteration. But it minimises no explicit objective and handles constraints only
  indirectly, and the static-equivalent vs dynamic designs can differ sharply **[search]**.
  Good for energy absorption of thin-walled structures; not a general dynamic optimizer.
* **For linear / modal results, adjoint with model reduction is better than ESL**: it gives *exact*
  gradients of the true dynamic response, so none of ESL's convergence issues arise. Reduced bases
  published: modal reduction plus static correction that removes the backward adjoint integration
  (Hooijkamp et al., IJNME 2018), reduced-basis adjoint with greedy-POD (arXiv:2308.09864), on-the-fly
  primal-dual ROM for large transient problems (CMAME 2024), Krylov/Ritz for harmonic response
  (CMAME 2021, MDPI 2025). All **[search]**; arXiv:2308.09864 is open and should be read first.
* **For nonlinear dynamics**: ROM for geometric nonlinearity (CMAME 2023, "up to 6x fewer full system
  updates") and POD + incremental SVD for primal and adjoint (CMAME 2025) exist but the literature
  itself notes that adjoint methods struggle with material nonlinearity and large deformation
  **[search]**. So ESL/DiESL remains the practical route when the nonlinear solver is a black box.
* Learning-based surrogates (autoencoder/GAN + evolutionary search, CMAME 2025) exist; no evidence
  they beat the above for our setting.

Suggested stack: **(1) adjoint on a modal ROM for linear modal data** (exact reference),
**(2) ESL core with pluggable load definitions for everything else** (modal ESL, zeroth ESL, DiESL,
optionally F-ESL), **(3) HCA only as an optional heuristic for crash-type energy absorption.**

## 8. General architecture for modal and nonlinear dynamics **[proposal]**

Goal: the same outer loop whether the response comes from a modal solution, a linear transient, or a
nonlinear dynamic solver (FrontISTR / Kratos / imported results).

```
aieng/topopt/dynamics/
  response.py     ResponseHistory: times, u(t), v(t), a(t) at selected steps, plus optional
                  state per step (deformed coordinates / tangent info for DiESL).
                  ModalData: Phi (ndof x nmodes), omega, q(t), normalisation, truncated flag.
  providers.py    DynamicProvider protocol (an extension of SolverAdapter):
                    solve_transient(request) -> ResponseHistory      (linear or nonlinear)
                    solve_modal(request, n_modes) -> ModalData
                    capabilities: {linear, nonlinear, returns_state, modal}
                  + ImportedModalProvider: loads modal results computed elsewhere (mode shapes
                    mapped to our node ids, omega, q(t), participation) so existing modal
                    analyses are usable without re-solving.
  matrices.py     M(x) (consistent / lumped) with RAMP-type mass interpolation (avoids spurious
                  void modes); C(x) Rayleigh.
  loads.py        LoadDefinition strategies, all returning one load vector per selected step:
                    ZerothESL  f = K(x_W) u_k
                    ModalESL   f = M Phi Omega^2 q_k           (no full history needed)
                    DiESL      f = increment on the deformed step model
                    FirstOrderESL  adds grad f_eq (needs dynamic sensitivities)
  steps.py        StepSelector: all | uniform | peak-of-response | adaptive (error-driven)
  functionals.py  response functionals mapped to the static sub-problem: dynamic compliance
                  sum_k w_k f_k.u_k, displacement at a dof, (later) stress.
  esl.py          the outer loop (below)
  adjoint_modal.py  exact direct/adjoint gradients on a modal ROM (linear reference)
```

Outer loop (`esl.py`), identical for every provider:
1. response = provider(x_W)  (modal, linear transient, or nonlinear)
2. loads_k = LoadDefinition(response, x_W) for steps chosen by StepSelector
3. inner problem = the **existing** `TopOptProblem` with `load_cases = loads_k`, weights, displacement
   limits, run with OC / MMA / GCMMA under a **trust region** on x
4. response' = provider(x_new); accept only if the *true* dynamic objective/constraints improved
   (else shrink the trust region and redo step 3); stop on small step *and* acceptance
5. report status `esl_stationary_not_certified` unless the first-order variant or the adjoint
   reference confirms stationarity. The final design always comes with its true dynamic response.

Reuse: everything in step 3 already exists (multi-load-case compliance, per-case displacement limits,
adjoint sensitivities, MMA/GCMMA, passive regions, symmetry links). New work is the provider layer,
`M(x)`, the load definitions, the step selector and the outer loop.

Build order: **D0** data model + builtin linear Newmark + builtin modal (scipy eigsh) + ZerothESL and
ModalESL + outer loop, validated on small linear problems against the adjoint-on-modal reference;
**D1** FrontISTR (`TYPE=DYNAMIC`, `TYPE=EIGEN`) and Kratos providers, imported-modal provider;
**D2** DiESL with a nonlinear provider; **D3** F-ESL / ESL-AVM if the safeguard proves insufficient.

## 9. Not read / not available

Behind paywalls or returning 403 on download, so known only from search summaries:
Cao et al. CMAME 436 (2025); the 2026 Springer overview of ESLM for nonlinear response TO
(Acta Mech. Solida Sin., 10.1007/s10338-026-00838-2); Lee & Park 2015 CMAME; Triller et al.
2021 (DiESL); Choi & Park 1999/2002; Kang, Park & Arora 2006; Stolpe & Pollini 2023;
the DTU comparative study of reduced-order methods (CMAME 2021, `orbit.dtu.dk`).
Behrou & Guest (arXiv:1705.01538) and Li et al. (arXiv:2305.08110) were saved but only their
search abstracts were used here; read them before relying on their details.

## 10. DiESL read in full (added 2026-10-30)

Source: Triller, Immel, Harzheim, *Topology optimization using difference-based equivalent static
loads*, SMO 65:226 (2022), open access CC BY; saved as
`2022_triller_DiESL-topology-SIMP_SMO-65-2022.pdf`. Sizing DiESL (Triller 2021) and adaptive
ESL-time selection (Triller 2022) are cited there but their PDFs were not retrievable (TUprints
returns an HTML block page); only this paper is read.

**Definitions (their eq. 1, 4-8, 12-14), geometric nonlinearity enters through coordinates only:**
* ESL: `f_i = K(x, X0) u(t_i)`: one linear model on the *undeformed* geometry.
* DiESL: one linear submodel `LSM_i` per selected time `t_i`, built on the *deformed* coordinates
  `X(t_i) = X0 + u(t_i)`; `f_i = K_i(x, X(t_i)) Δu_i`, `Δu_i = u(t_{i+1}) - u(t_i)`. The linear
  stiffness is the ordinary small-strain one on the deformed mesh (no geometric/initial-stress term).
* Surrogate responses accumulate along the chain: `δ_i` solves `K_i(x) δ_i = f_i`, then
  `u_i = u_{i-1} + δ_{i-1}`, `u_i = Σ_{j<i} δ_j`. The multi-model optimisation (MMO) therefore has
  *coupled* models: a response at `t_i` depends on every earlier model. Linear and nonlinear
  displacements agree only at the start of each inner loop.

**Procedure per outer cycle:** nonlinear dynamic analysis -> pick `n_T` times -> DiESLs -> inner
multi-model optimisation with move limits -> interpret densities -> repeat. Their settings: move
limit `Δ_ini = 0.2`, `maxiter = 4` inner iterations per cycle, stop when max constraint violation
<= 0.01, relative objective change <= 0.01 and discreteness change <= 0.02 (all three).

**Topology-specific pieces that an implementation needs:**
1. Density -> analysis model (eq. 9): `rho = 0` if `x <= 0.1`, `x` if `0.1 < x <= 0.9`, `1` above;
   `rho = 0` elements are *deleted*, disconnected islands are deleted. Intermediate densities are
   kept (continuity from cycle to cycle) rather than a 0/1 threshold.
2. Analysis-domain SIMP exponent `p_NL = 1` (design domain keeps `p = 3`): with `p_NL = 3` low-density
   elements have mass/stiffness too high and the dynamic run shows "hedgehog" nodes / distorted
   elements. They accept the stiffness inconsistency between the two domains.
3. Nodes of deleted elements have no nonlinear displacement; their deformed coordinates are
   reconstructed by a linear static run with the known displacements prescribed.
4. A deformed `LSM_i` can fail the element-quality check: distorted elements are deleted from that
   LSM for the current cycle only.
5. `n_T` and the times are chosen adaptively per cycle by fitting a polygonal line to a response
   curve (their contact force); breakpoints are the ESL times. The ESL-time selection method itself
   is in Triller 2022, not read.

**Consequences for aieng.topopt (they change section 8):**
* The core currently has ONE mesh and one reference `Ke0` per element type shared by every load
  case. DiESL needs a **per-load-case geometry**: `Ke0_i` per element on `X(t_i)` (for hex8 no longer
  a shared matrix), `ne x 24 x 24` per model, and per-model assembly.
* Responses are **chained across models** (`u_i = Σ δ_j`), so the objective/constraint sensitivities
  need the chain rule (adjoint through the sequence of linear models), not independent load cases.
  A compliance-style `Σ f_i·δ_i` per model decouples, but is a different problem from theirs
  (intrusion/displacement at a point).
* The inner problem is therefore not the existing `TopOptProblem` unchanged. Zeroth ESL (single
  geometry, independent load cases) still is.
* Their evidence is one beam-impact example with a commercial solver and OptiStruct; it supports
  "DiESL > ESL on that case" only, not general stability.
