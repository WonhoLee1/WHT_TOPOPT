# Topology-optimization methods beyond SIMP/BESO: survey (2026-10-30)

Purpose: decide what a general framework (`FRAMEWORK_PLAN.md`) has to be able to host.
Evidence tags: **[read]** = abstract/introduction/selected sections of the saved PDF were read;
**[search]** = only a web-search summary or first page seen. Nothing here was implemented or reproduced by us
except the DiESL/ESL gradient spike in section 7.

## 1. Taxonomy by what the method changes

| Axis | What varies | Methods found |
|---|---|---|
| A. Design parameterization | how the design is described | density grid (SIMP/RAMP), BESO 0/1, level set, explicit components (MMC; Gaussian Ensemble Topology), neural field (reparameterization, implicit neural representation), phase field, Fourier-network phase field |
| B. Physics / response | what is simulated | static linear, modal, linear transient, nonlinear (geometric) static, nonlinear transient, thermo-mechanical finite strain, multiphysics/fluid |
| C. Gradient source | where d(response)/d(design) comes from | hand-derived adjoint, automatic differentiation (JAX), reduced-basis adjoint, checkpointed AD through time, ESL-type surrogate, learned surrogate, none (gradient-free) |
| D. Update algorithm | how the design moves | OC, MMA, GCMMA, projected gradient (PGD-TO), Adam/L-BFGS (neural), CMA-ES / evolutionary |
| E. Generative priors | how a start or a whole design is proposed | meta-learned initial designs, diffusion models (physics-informed, latent), SIMP-then-refine pipelines |
| F. Orchestration | who sets the problem up | LLM agents driving deterministic FE/optimizer tools |
| G. Compute backend | where it runs | numpy/scipy, external FE solvers, JAX/PyTorch on GPU, JAX+PETSc, matrix-free GPU multigrid |

The axes are independent in principle. Most 2025-26 papers change one axis and keep the others fixed; that is the
structural argument for a framework with one interface per axis.

## 2. A. Parameterization

* **Neural reparameterization.** Hoyer, Sohl-Dickstein, Greydanus (2019) [search/first page]: optimise network weights
  instead of grid densities; reported better designs more often over 116 tasks. NTopo (Zehnder et al., 2021) [first
  page]: mesh-free, implicit neural representation. Sanu, Aragon, Bessa (2024, "the good, the bad, and the ugly")
  [read abstract]: the network makes the landscape non-convex even for convex problems, which can slow convergence on
  smooth problems and help exploration on non-convex ones; architecture/hyperparameter choice is hard. Meta-neural TO
  (Kuszczak et al., 2025) [first page + search]: meta-learns initial designs so new problems need fewer iterations.
  Take-away: a legitimate parameterization, not a free win; it needs gradients w.r.t. a *field*, which any of our
  evaluators already supplies (chain rule through the network).
* **Explicit geometry.** Moving Morphable Components (Guo et al., 2014; many 2025-26 extensions) [search]; Gaussian
  Ensemble Topology (Ma et al., 2025/26) [first page]: anisotropic Gaussians, smooth and curvature-continuous without
  post-processing, "comparable to MMC". Few design variables, CAD-friendly output. Needs a map components -> element
  densities with a gradient.
* **Phase field.** Chen et al. (2025) [first page]: claims monotonic decay of the original objective without
  modification. Fourier-neural-network phase field (2025/26) [first page]: state, adjoint and topology each as a
  network trained in an alternating scheme.
* **Level set (already implemented here):** Wegert 2023, Jezek 2026; FormOpt (2026) [first page] is a FEniCSx toolbox
  for level-set *shape* optimization with distributed shape derivatives.

## 3. B/C. Physics and gradients (the part that matters for dynamics)

* **Linear transient, exact gradients.** Discrete adjoint on a reduced basis (arXiv 2308.09864, greedy-POD) and modal
  reduction with static correction (Hooijkamp, van Keulen) [search]; Behrou & Guest 2017 time-domain adjoint [abstract
  only, saved]. Discrete adjoint is preferred over continuum adjoint because it is consistent with the time stepping.
* **Finite-deformation transient.** Ogawa (2021, IJNME) [search summary only, paywalled]: dynamic compliance for
  hyperelastic materials, sensitivities by an adjoint on Newmark-beta integration, checked against finite differences.
  Closest published match to "geometrically nonlinear transient with exact gradient"; not saved, not read.
  Nonlinear adjoints need the true tangent stiffness; Hosseinmardi et al. (2026, saved) [search summary; only the
  title block and a few derivation lines were seen] is said to state that a biased tangent biases every sensitivity
  (thermo-mechanical, steady state). Verify before citing.
* **Nonlinear structural dynamics with reduced models.** Pozzi et al. (2025, SMO; arXiv 2503.17431 for the adjoint
  part) [search]: invariant-manifold / spectral-submanifold ROM, backbone-curve objectives, not time-domain transients.
* **Autodiff.** JAX-FEM (Xue et al., 2022; GPL-3.0, active 2026-10) [first page + license check]: differentiable FE with
  automatic linearization for nonlinear statics, used for 3D TO of nonlinear materials. No transient/Newmark example
  was found in the search. SOPTX (He et al., 2025; built on FEALPy) [read abstract+intro]: NumPy/PyTorch/JAX backends,
  AD for sensitivities, "decouples analysis from optimization", non-intrusive: users replace filters, constraints,
  objectives without touching the solver. JetSCI (Cattaneo et al., 2026) [read abstract]: JAX for differentiable
  discretizations + PETSc for scalable solves.
* **Checkpointed AD through time** (reverse-mode through a transient with states stored only at selected steps):
  TOMATOES (Padhy et al., 2025) and VASTO (Thombre et al., 2026) [search]; both are thermal/flow, not structural
  dynamics, so the pattern transfers but no structural example is known to us.
* **ESL family** (see `DYNAMICS_SURVEY.md`): cheap, uses any black-box nonlinear solver, no gradient guarantee.
  Measured here, section 7.
* **Learned surrogates for crash/dynamic TO:** the search found none for 2025-26; the practical baseline is still
  ESL/DiESL (Triller et al. 2022, saved, read in full) and HCA.

## 4. D. Update algorithms

* **PGD-TO** (Nobari & Ahmed, 2025/26) [read abstract]: projected gradient descent with the projection posed as a
  regularised convex QP (semismooth Newton for multiple constraints, bisection for single/independent ones), spectral
  step size, nonlinear CG directions; reports compliance comparable to MMA and OC with lower per-iteration cost; code in
  `pyFANTOM` (GitHub has no license field set: **license unknown, do not vendor**). Four benchmark families incl.
  multi-material and a nonlinear compliance constraint.
* A magnetic-actuator benchmark (INTERMAG 2025) compared MMA, IPOPT, SNOPT [search]; a trust-region MMA variant for
  stress constraints exists (2025) [search]; gradient-free CMA-ES variants perform worse than MMA within a budget
  (arXiv 2403.04937) [search].
* Already here: OC, MMA (mma-dual/mmapy), GCMMA.

## 5. E. Generative priors and F. LLM orchestration

* **Diffusion.** Physics-Informed Diffusion Models (Bastek, Sun, Kochmann, ICLR 2025) [first page]: residual-based
  training loss, residual error cut by up to two orders of magnitude; TO is one benchmark. Latent-space diffusion for TO
  (Lutheran, Das, Tabarraei, 2025) [first page]: VAE + latent diffusion, conditioned on stress/strain-energy/volume
  fraction/loads, auxiliary losses against floating material and volume error. HPG-Diff (2026) [search]. All are 2D
  compliance-style results; none gives a guarantee, so a generated design must be treated as a **proposal to be
  re-analysed**, never as a result.
* **LLM agents.** TO-Master (Lin et al., 2026; Xue is an author of JAX-FEM) [read abstract+intro]: agent selects tools,
  builds FE models, checks meshes/BCs, launches sensitivity-based optimisation with typed solver arguments; ablation
  shows tool-usage rules, reasoning guidance and few-shot examples are what make formulation robust.
  TopOptAgents (Park & Chung, 2026) [first page]: multi-agent self-refinement; "80% success rate" on a complex L-beam
  (search summary). LLM as SIMP continuation controller (Yang et al., 2026) [first page]. This repo already exposes
  deterministic tools to an external agent over MCP; what these papers add is evidence that *typed arguments and
  validation* decide robustness, which is the repo's existing design.

## 6. G. Compute backends

* GPU/matrix-free: pyFANTOM (8M+ elements on a 16 GB GPU) [search]; single-GPU matrix-free Galerkin multigrid
  (Yang et al., 2026) [first page]; GPUTopOpt (Julia/CUDA) [search]; 50M-element GPU eigenfrequency TO [search].
  Billion-element results are older MPI/HPC work.
* Python TO frameworks: SOPTX (FEALPy; license NOASSERTION), FEniTop (FEniCSx, GPL-3.0, last push 2024-11),
  FormOpt, open-topolegy (Apache-2.0), JAX-FEM (GPL-3.0). OpenLSTO (Apache-2.0). Kratos (NOASSERTION on GitHub API;
  has a Topology app (SIMP compliance) and adjoint sensitivities via StructuralMechanics/ShapeOptimization; transient
  adjoint support not confirmed).

## 7. What we measured ourselves (throwaway spike, 6x2x2 cantilever, 24 elements, Kratos TotalLagrangian + Bossak)

Gradient of tip displacement at four times: surrogate (DiESL or zeroth ESL) vs central finite difference of the true
nonlinear transient. Cosine similarity per time:

| Setting | DiESL | ESL |
|---|---|---|
| Dynamic step load (rho=0.01), F=0.05 / 0.2 / 0.5 | 0.56-0.77 at 3 of 4 times, ~0 at one | about the same, ESL better at F=0.5 |
| Quasi-static ramp (rho=1e-6), F=0.05 | 0.9999 | 0.9993-0.9999 |
| Quasi-static ramp, F=0.5 (tip 14% of L) | 0.995 at all times | 0.99 -> 0.94 (sign agreement 0.71) |
| Quasi-static ramp, F=2.0 (tip 45% of L) | 0.94 at all times (rel. err 0.38-0.80, sign 0.71) | 0.94 -> 0.62 (rel. err up to 2.1) |

Reading: (1) the surrogate assembly and adjoint are right (reproduction 1e-12, adjoint vs FD of the surrogate 1e-5);
(2) with inertia, a frozen equivalent load is a poor gradient proxy even for a nearly linear response; (3) DiESL beats
zeroth ESL as soon as geometry changes, as claimed in the paper, but is not accurate at large deformation, plausibly
because K_i has no initial-stress term (**not tested**). One mesh, one random design, one response: indicative only.

## 8. Implications for a framework

1. Gradient source must be a **declared, tiered property of a response**, because the same optimizer is happy with an
   exact adjoint and silently wrong with an ESL proxy. Tiers: exact (hand adjoint / discrete adjoint), AD, surrogate
   (ESL/DiESL), learned, finite-difference, none. Anything below "exact/AD" needs acceptance on the true response.
2. Parameterization and update algorithm are independent plug-ins around one evaluation contract
   (z -> physical field -> response + gradient -> pull-back to z). The existing filters/projection/variable-group code
   already is a parameterization layer.
3. Generative models and LLM agents sit **outside** the numerical core: they produce a start design or a problem
   specification; the core re-evaluates with deterministic physics and reports the result with its own credibility
   tier.
4. JAX/GPL-licensed components (JAX-FEM, FEniTop) can only be optional dependencies, imported at run time, never
   vendored (same policy as mma-dual); pyFANTOM's license must be clarified first.

## 9. Not covered / open

Stress constraints, fatigue, multi-scale/lattice, additive-manufacturing constraints, fluid/thermal multiphysics were
not surveyed (out of the stated scope). Ogawa 2021, Pozzi 2025 (SMO), Weider 2018/19 and Ivarsson 2018 (adjoint for
finite-strain/viscoplastic transients) are paywalled or unread and are the first things to obtain before implementing a
nonlinear transient adjoint. HPG-Diff and several 2026 arXiv items were seen only in search listings.

## 10. Field-wide additions (added later the same day, after the user clarified the scope is topology optimization as a field, not only dynamics)

Section 1-9 was biased toward the dynamics question. The areas below were searched afterwards; evidence is mostly
**[search]** because most hits are journal abstracts. Reviews found: composite fibre-path + topology (Materials & Design
2025), connectivity promotion (Cool, Aage, Sigmund, SMO 2025), shellular structures (2026), thermal energy storage
(2026), a 2026 *Computation* special issue; none read.

* **Constraints and manufacturing.**
  - Stress: trust-region MMA to stabilise stress-aggregation settings (Hu et al. 2025); p-norm global stress for
    multi-material with per-material limits (Nguyen et al. 2025); aggregation-free local stress by augmented Lagrangian
    even under dynamic loads; explicit-component methods avoid intermediate-density stress artefacts (Ma et al. 2024);
    binary TOBS-GT with a body-fitted mesh (CILAMCE 2025). Local constraints are the hard part for any optimizer API.
  - Additive manufacturing: length scale + overhang + build orientation (Kumar & Fernandez 2022, saved, first page);
    global length/overhang control by perimeter minimisation for level-set and density methods (IJNME 2025); AM
    overhang + reliability (2025). Casting/milling: only 2006/2011 references surfaced, nothing recent found.
  - Connectivity: practical review (Cool, Aage, Sigmund 2025). Relevant to our BESO load-path check.
* **Multi-material and graded multiscale.** GM-TOuNN (Chandrasekhar et al. 2022, saved), multiscale TO with neural
  networks encoding a cell's microstructure and its distribution as a differentiable field (Chen, Liu, Kara 2024, saved),
  physics-augmented neural-network material models for graded lattices (Stollberg et al., CMAME 2025), compositionally
  graded alloys via band-limited coordinate networks (TOBACO, 2025, saved). Common structure: a *field* network plus a
  *material model* (homogenised or learned); the physics solve is unchanged.
* **Geometry-conforming discretizations.** Isogeometric TO with topological derivatives on an immersed spline basis
  (Teixeira et al. 2025, saved): higher-degree bases improve accuracy while linear bases suffice for the level-set field.
  Topological-derivative methods in general are a distinct update rule (insert holes where the derivative is
  favourable), orthogonal to density methods.
* **Robust / reliability / multi-objective.** Pareto-tracing with polynomial chaos for expected-compliance vs
  deviation; reliability-based with FORM, gradient-free proportional TO (2024-26 items, none on arXiv). Structure: an
  outer loop over load/geometry uncertainty around the same inner evaluator.
* **ML as a solver substitute.** Physics-informed transformer for real-time TO (Lutheran et al. 2026, saved, first
  page), a "foundation model for shape- and resolution-free TO" (2025, only seen in a reference list, not located).
  Same caveat as diffusion: a prediction, not a result.

## 11. Robot / multibody coupling (user: "MuJoCo dynamic-motion results should feed the optimization; understand it as robotics")

No paper coupling MuJoCo (or MJX) to structural topology optimization was found; the user reports such attempts exist,
source not yet given. What the literature does offer for *motion-driven loads on a flexible component*:

* **Weakly vs fully coupled system-based methods** (Tromme et al., 2018 review, saved, abstract read). Weakly coupled:
  simulate the mechanism, turn the time history into loads, optimise the component, update, re-simulate (ESL is the
  typical conversion). Fully coupled: sensitivities include the dynamics. This is the same three-way split as our
  gradient tiers.
* **Component-based loads break down in multibody systems** because joint forces change when the design changes
  (search summary of Ghandriz et al.). TOMBS (Ghandriz, Führer, Elmqvist, 2017, saved, abstract read): simultaneous
  topology optimisation of all flexible bodies with large rotation, FEM + modal reduction, DAE integration, approximate
  sensitivities, and **element mass penalisation in addition to stiffness penalisation** to avoid instability and mesh
  distortion in the dynamic case.
* **Equivalent static displacement** (Gupta & Tovar, SAE 2025, search only): prescribe nodal displacements per time step
  instead of loads; connecting-rod/piston example. ESL for multibody flexible components: Hong et al. 2010, OptiStruct
  ESLM for flexible bodies (search).
* **Robotics side.** Topology optimisation of robot links under dynamic loads and stiffness requirements (Wanninger et
  al., DESIGN 2024, search); BESO with self-weight and inertial loads for serial arms under time-varying loads (SMO 2025,
  search). Differentiable simulation for co-design: COSMIC (Guo et al. 2026, saved, abstract read: gradient-based
  co-design of topology, material and a neural controller of a truss-lattice robot inside a differentiable simulator),
  SDRS shape-differentiable simulator (Ye et al. 2024, saved, first page) which notes that fixed-topology deformable
  models cannot represent link-topology change.
* **What this means.** MuJoCo is a rigid-body simulator with contacts; its useful outputs for a component are joint
  reaction forces, contact forces, body accelerations, and (via the design's mass/inertia) the motion itself. A coupling
  therefore needs (1) a **LoadProvider** returning time-varying load sets in the component frame (incl. inertial loads,
  inertia-relief-style balancing), (2) an **update path from design to rigid-body properties** (mass, COM, inertia) so
  that re-simulation reflects the new design, and (3) a statement of which coupling tier is used: loads frozen from one
  simulation (weak, no guarantee, like ESL), loop with re-simulation and true-response acceptance, or sensitivities
  through the simulator (needs a differentiable one, e.g. MJX). All three can sit behind the same contract as the
  structural dynamics work; the structural solve is then quasi-static per load set, with the linear/nonlinear transient
  provider as a separate, more expensive option.

## 12. Algorithms that replace SIMP and BESO (focused pass, 2026-10-30)

Question: which published algorithms replace SIMP (density + penalty + OC/MMA) or BESO (0/1 evolutionary), as opposed to
tuning them? Split into **replacements** (different design description or update rule) and **upgrades** (keep the
density field, fix its binarization).

**Evidence on which is better: there is no head-to-head benchmark across families.** What exists:
* Zhou, Fu, Ghabraie, *Systematic benchmarking ... Zhou-Rozvany problem*, CMES 143(3), 2025 (saved, abstract and
  introduction read; CC BY): SIMP, SIMP + continuation and SEMDOT against the **known global solutions of a 100-element
  tie-beam problem, in both its binary and relaxed forms**; a greyness measure; "in most cases the grayscale solutions of
  SEMDOT perform better than SIMP", attributed to Heaviside smoothing versus the power-law penalty. ESO fails on this
  problem and Rozvany states BESO does too (introduction of the paper). 2D, one problem, three methods.
* One aerospace pitch-arm study (single case, venue unverified): level set about 6 % lower compliance than SIMP at 40 %
  weight reduction at roughly 3.75x cost; SIMP left 8-12 % intermediate densities. Treat as anecdote.
* Our own 24x8x6 result: SIMP initialisation + level-set refinement costs about 4 % compliance relative to the SIMP
  geometry at the target volume and removes grey (Jezek 2026 reports < 4 % spread across settings).

**Replacements for SIMP**

| Method | What replaces what | Evidence | Code / license | Fit with our core |
|---|---|---|---|---|
| Level set (Hilbertian projection, parametric RBF, topological-derivative nucleation) | boundary evolution instead of density; crisp boundaries | built here; Wegert 2023; hole nucleation is the known weak point (Barrera et al. 2019 seed holes from density fields) | OpenLSTO Apache-2.0, GridapTopOpt.jl | done; limits: no variable groups, no displacement limits |
| SEMDOT (Fu & Rolfe) | non-penalised, grid-point densities -> smooth-edged element volume fractions via Heaviside | paper 2020 (saved, read) + ZR benchmark 2025; **implemented here as `aieng.topopt.semdot`** (2026-10-30) | 2D MATLAB in the paper; MIT-licensed 3D code `nedu10/optiworks` and 2D `ooibhadode/IbIPP` (see INDEX); non-penalisation form with discrete sensitivity: PMC12113074 | built; see `aieng/docs/topopt_solver_adapters.md` for measured behaviour |
| MMC / Gaussian Ensemble Topology | explicit components instead of a field; few variables, CAD-friendly | GET paper (saved, first page); MMC papers (search) | none confirmed | Representation plug-in; needs components -> density map with gradient |
| Phase field | Allen-Cahn / Cahn-Hilliard evolution of an order parameter | Chen 2025: monotone objective decay (saved, first page) | none confirmed | Representation + Optimizer |
| Topological-derivative methods | insert holes where the derivative is favourable; no penalty | Teixeira 2025 (IGA, saved); Gangl multi-material (2019) | none confirmed | update rule orthogonal to density methods |
| Neural reparameterization (TOuNN, NTopo, meta-neural) | network weights as design variables | Sanu 2024: reshapes the landscape, can slow convergence on smooth problems | research code | Representation plug-in; needs autodiff pull-back |
| Generative / transformer predictors | learned map loads -> design | diffusion 2025, transformer 2026 | checkpoints | not a replacement: a proposal that must be re-analysed |

**Upgrades to SIMP (not replacements, cheap and likely worth taking first)**
* Automatic projection-parameter increase (Dunning 2024, saved, abstract read): beta grows exponentially at a rate tied to
  the objective change; claimed to need no per-problem tuning; examples include linear buckling and geometrically
  nonlinear problems (2D). Our `driver.py` uses a fixed beta schedule.
* Subpixel-smoothed projection (SSP, 2025, saved, abstract read): drop-in projection that depends on the filtered field
  and its gradient so that it moves smoothly between density-based and level-set behaviour; demonstrated on photonics
  problems, not structures.

**Replacements for BESO**

| Method | Idea | Evidence | Code / license | Notes |
|---|---|---|---|---|
| TOBS (Sivapuram & Picelli 2018; 101-line code 2020) | binary variables; sequential linearisation with move limits; each step solved as an integer linear program; any differentiable objective/constraints | search summaries only (papers paywalled; not saved) | MATLAB repo `renatopicelli/tobs` on GitHub (GPL-3.0, last push 2020-10); needs `intlinprog` or CPLEX | the closest formal replacement for BESO; body-fitted variant TOBS-GT (2024) exists. The ILP solver is a new dependency (scipy.optimize.milp / HiGHS might serve; unverified for this use) |
| PTO (proportional TO, Biyikli & To) | no sensitivity derivation: material redistributed in proportion to element response | Zheng et al., *Mathematics* 2025: PTO faster than BESO for stress minimisation, BESO sometimes lower peak stress but higher compliance (search summary, one problem type) | published pseudocode | trivial to host on the contract |
| SERA / SESO (ESO family) | rejection sets stiffness near zero; admission BESO-like | comparison of ESO/SESO/SERA in 3D (2023) | - | not a successor to BESO, an alternative member of the family |

**Exploratory / not recommended as replacements now:** quantum annealing for TO (QUBO updater; Ye, Qian, Pan 2023;
Sukulthanasorn 2024-25; fault-tolerant algorithm only simulated classically, Hoelscher 2025), reinforcement learning
(pre-2025, matched gradient methods on small 2D grids), gradient-free CMA-ES (worse than MMA within budget, arXiv
2403.04937, saved).

**Implications**
1. The only verifiable benchmark with known global answers is Zhou-Rozvany (2D, 100 elements). Our core has no 2D element,
   so reproducing it needs either a plane-stress quad in the core or a thin 3D slab whose solution differs; decide before
   promising a comparison. Without such a benchmark any claim that X replaces SIMP made here would be untested.
2. Cheapest real gains: projection upgrade (Dunning), PTO (a few dozen lines), TOBS via an MILP solver. Largest
   representation changes: SEMDOT, MMC/GET, neural, phase field.
3. None of the replacements removes the need for the evaluation contract and the gradient-tier labelling in
   `FRAMEWORK_PLAN.md`; TOBS and PTO in particular change what "converged" means (binary designs, no gradient for PTO).
