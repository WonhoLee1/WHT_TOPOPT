# Framework plan: one evaluation contract, nine plug-in points (DRAFT v2, 2026-10-30)

Status: proposal for review, nothing implemented. Replaces v1, which was biased toward the dynamics question.
Evidence base: `METHODS_SURVEY_2026.md` (sections 1-11), `DYNAMICS_SURVEY.md`, the opus architect review of the dynamics
design, and the gradient spike (survey section 7).

## 1. Why a different structure than today

`aieng.topopt` is one pipeline: `TopOptProblem` (mesh + loads + SIMP settings) -> `TopOptEvaluator` (linear static solve
through an adapter, adjoint sensitivities, filters) -> OC/MMA/GCMMA or a bespoke driver (BESO, level set). Each new method
so far forked the driver or was refused (level set rejects variable groups and displacement limits). The literature
changes **one thing at a time** (survey section 1): a representation (neural, explicit, phase field, graded multiscale), a
constraint family (stress, overhang, connectivity), a load source (multibody/robot motion), an optimizer (PGD-TO), or a
prior (diffusion, meta-learning). A framework should give each its own plug-in point around one contract.

## 2. The contract

```
z --Representation.to_field--> field (density / geometry / graded material)
loads = LoadProvider.loads(field)            # given, ESL from a solver, or motion-driven (MuJoCo / multibody)
resp  = ResponseModel.evaluate(field, loads) # static, modal, transient, nonlinear
Evaluation = { objective, constraints[], gradients wrt field, gradient_source tier, provenance }
gradients wrt z = Representation.pull_back(gradients wrt field)
z' = Optimizer.step(values, gradients wrt z, bounds, trust_region)
```

Plug-in points:
1. **Representation**: density grid with filter/projection/variable groups (today's code), level set, explicit components
   (MMC, Gaussian ensemble), neural field, phase field, graded multiscale/lattice with a material model, isogeometric
   level set. Owns `to_field` and `pull_back` (analytic or autodiff) and its own design-side constraints (symmetry,
   extrusion, passive regions, minimum length scale).
2. **LoadProvider**: fixed nodal loads (today), load history imported from a file, ESL built from a nonlinear solver
   history, and **motion-driven loads from a rigid/multibody simulator (MuJoCo)**: joint reaction forces, contact forces,
   body accelerations as inertial loads, with inertia-relief balancing. Also declares whether loads depend on the design
   (mass, COM, inertia change the motion) and, if so, how that dependence is updated.
3. **ResponseModel**: static compliance, node displacement, modal, linear transient, nonlinear (geometric) transient.
   Solver adapters stay thin oracles; the core keeps Ke0, mass interpolation and sensitivity assembly.
4. **Constraint registry** (generalises `displacement_limits`): volume, displacement, stress (p-norm aggregate or local,
   augmented Lagrangian), minimum length scale, overhang/build orientation, connectivity, casting/milling. Each returns
   value + gradient + its own gradient tier.
5. **GradientSource tier** (section 3), attached to every objective/constraint evaluation.
6. **Optimizer**: OC, MMA, GCMMA now; PGD-TO, Adam/L-BFGS (neural fields), CMA-ES (values only) later. Declares which
   constraint types and gradient tiers it can handle, and respects a trust region.
7. **Generator**: `propose(problem) -> z0` (uniform, SIMP result as for level set, meta-learned, diffusion/transformer).
   Always re-evaluated by the ResponseModel; a prediction is never presented as a result.
8. **Orchestrator**: stays outside the numerical core: the existing MCP tools with typed arguments and refusals (the lesson
   of TO-Master/TopOptAgents: typed arguments and validation decide robustness).
9. **Backend**: numpy/scipy builtin, external solver adapters (FrontISTR, Kratos), optional JAX/GPU.

## 3. GradientSource: the one new concept

| Tier | Meaning | Required of the driver |
|---|---|---|
| `exact_adjoint` | discrete adjoint consistent with the solver's discretization | FD check passes in CI |
| `autodiff` | AD through the solve, incl. checkpointed transients | FD check passes in CI |
| `surrogate` | ESL / DiESL / reduced model / frozen motion-derived loads: not the gradient of the true response | accept a step only if the TRUE response improves; never report `converged` |
| `learned` | neural surrogate or generative proposal | as `surrogate`, plus out-of-distribution flag |
| `finite_difference` | brute force | cost warning; tiny problems only |
| `none` | gradient-free | optimizer must declare it can run without gradients |

Why first-class: the repo's credibility stamp already ranks evidence this way, and our spike showed the failure: a frozen-
load ESL gradient had cosine 0.5-0.77 to the true gradient for a nearly linear dynamic response while an optimizer would
have run happily. The same applies to loads frozen from one MuJoCo run (weak coupling, Tromme et al. 2018).

## 4. Where each surveyed family lands

| Family | Plug-in point | New code | Reused |
|---|---|---|---|
| PGD-TO | Optimizer | implementation from the paper (pyFANTOM license unknown) | evaluator |
| Neural reparameterization, meta-init, GM-TOuNN, TOBACO | Representation (+ Generator) | network + autodiff pull-back (torch/JAX optional) | evaluator, MMA/Adam |
| MMC, Gaussian ensemble | Representation | components -> density map with gradient | evaluator |
| Phase field | Representation + Optimizer | Allen-Cahn / Cahn-Hilliard update | evaluator |
| Isogeometric + topological derivative | Representation + Backend | immersed spline basis | - |
| Level set, BESO | Representation + Optimizer | already built; re-host only when a second consumer exists | - |
| Stress, overhang, length scale, connectivity | Constraint | one class each; stress needs its own adjoint | adjoint kernels, `beso.load_path_connected` |
| Robust / reliability / Pareto | outer loop around the evaluator | uncertainty sampler | evaluator |
| Diffusion, transformer predictors | Generator | checkpoint loader | evaluator re-analysis |
| LLM agents | Orchestrator | none in core | MCP tools |
| MuJoCo / multibody loads | LoadProvider | MuJoCo adapter (optional dependency), frame transforms, inertia relief, mass-property update | load cases, evaluator |
| ESL / DiESL, transient adjoint | ResponseModel + GradientSource | see section 5 | evaluator kernels |
| GPU / matrix-free multigrid | Backend | solver backend | adapters |

## 5. Dynamics and robot coupling

Measured (survey section 7): DiESL is a good proxy only for quasi-static nonlinear response (cos 0.94-0.995); ESL-type
gradients are poor under inertia. Hence:

* **Robot/multibody path first-class.** For robot links the structural solve is naturally quasi-static per load set
  (loads from the multibody simulation, including inertial loads), which keeps the exact-adjoint tier available. Needed:
  LoadProvider for MuJoCo output; coupling tier stated explicitly (loads frozen from one run = `surrogate`; re-simulate
  after each design update with acceptance on the true response; sensitivities through the simulator only with a
  differentiable one such as MJX); mass-property update; TOMBS-style mass penalisation when mass is design dependent.
* **Structural transient (modal/nonlinear)** stays an optional ResponseModel with the options of the v1 plan: linear
  transient discrete adjoint (builtin, CI oracle), total-Lagrangian Newmark adjoint (Ogawa 2021 unread), JAX AD with
  checkpointing, reduced-basis adjoint for linear modal data, Kratos adjoint (unverified), DiESL + geometric-stiffness fix
  (untested), ESL/DiESL multi-model evaluator as the cheap `surrogate` tier with trust region (architect design).

## 6. Proposed order

* **R0**: `GradientSource` tier + driver rules, constraint registry (re-express volume and displacement limits as
  constraints), FD-gradient harness over (representation, response, constraint, tier). No behaviour change for today's
  optimizers.
* **R1**: LoadProvider with imported load histories and several quasi-static load sets (weighted/envelope), inertia relief;
  tests need no MuJoCo. Then the MuJoCo adapter as an optional dependency, validated on a case with a closed-form answer
  (rigid link swinging: joint force vs. analytic).
* **R2**: first new constraint beyond volume/displacement: aggregated stress with adjoint (largest practical gap for
  manufacturing-relevant results), then minimum length scale / overhang.
* **R3**: new optimizer (PGD-TO) and one new representation (neural field or explicit components) to prove the contract
  hosts them without touching the evaluator.
* **R4**: transient structural responses with exact gradients (v1 options), ESL/DiESL multi-model as `surrogate`.
* **R5**: generators, multiscale/graded materials, isogeometric, robust/reliability.

## 7. Decisions / information needed from the user

1. The **MuJoCo attempts** the user found: source(s) (paper/GitHub), what is taken from MuJoCo (joint reaction forces,
   contact forces, accelerations, whole trajectories), and whether the design feeds back into the simulation. This sets
   the coupling tier in R1.
2. Priority among: constraints (stress, manufacturing), robot/multibody loads, new representations (neural, MMC, GET),
   new optimizers, structural transient. The order above is a guess.
3. Is a **second FE implementation** acceptable as a dynamics oracle (builtin or JAX), given the earlier preference for
   existing OSS? JAX as an optional backend (GPL-3.0 JAX-FEM stays out)?
4. Keep the architect's cut list (no FrontISTR dynamic provider, no F-ESL, no adaptive time selection)?

## 8. Risks

* The contract may over-generalise: level set and BESO work and need not be re-hosted without a second consumer.
* Neural and generative methods are young (Sanu et al. 2024: reparameterization can slow convergence); options, not defaults.
* The dynamic evidence is one 24-element spike; repeat on a second geometry and design before relying on it.
* Licenses: pyFANTOM unknown, JAX-FEM/FEniTop GPL-3.0, FEALPy/Kratos not asserted by the GitHub API; MuJoCo is believed
  Apache-2.0 (verify before adding).
* Most field-wide evidence is journal abstracts (survey section 10); stress-constraint and manufacturing choices need a
  proper paper read before R2.
