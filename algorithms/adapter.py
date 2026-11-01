"""Unified Optimization Algorithm Adapter & Registry Framework.

Provides a plug-and-play adapter layer allowing topology optimization
workflows to seamlessly switch, test, and benchmark multiple algorithms:
    - MMA (Method of Moving Asymptotes, Svanberg 1987) [Default]
    - GCMMA (Globally Convergent MMA, Svanberg 2002)
    - OC (Optimality Criteria Bisection)
    - BESO (Bi-directional Evolutionary Structural Optimization)
    - SEMDOT (Smooth-Edged Material Distribution)
    - LevelSet (Reaction-Diffusion Level Set)

References:
    - Svanberg, K. (1987), "The method of moving asymptotes—a new method for structural optimization",
      Int. J. Numer. Meth. Engng., 24(2), 359-373.
    - Svanberg, K. (2002), "A class of globally convergent optimization methods based on conservative
      convex approximations", SIAM J. Optim., 12(2), 555-573.
    - Bendsøe & Sigmund (2003), "Topology Optimization: Theory, Methods and Applications", Springer.
    - Huang & Xie (2007), "Convergent and mesh-independent solutions for the bi-directional evolutionary
      structural optimization method", Finite Elements in Analysis and Design, 43(14), 1039-1049.
    - Fu, Rolvink & Sigmund (2022), "Smooth-edged material distribution for optimizing topology (SEMDOT)",
      Structural and Multidisciplinary Optimization, 65(5), 1-22.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

try:
    import mmapy
    from mmapy import asymp, concheck, gcmmasub, kktcheck, mmasub, raaupdate
    HAS_MMAPY = True
except ImportError:
    HAS_MMAPY = False

from .base import Optimizer
from .beso import BESOOptimizer
from .level_set import LevelSetOptimizer
from .mma_wrapper import MMAOptimizer
from .oc import OCOptimizer
from .semdot import SEMDOTOptimizer


class FormulationType(str, Enum):
    """Classification of the material and geometric representation model."""
    SIMP = "simp"
    BESO = "beso"
    SEMDOT = "semdot"
    LEVEL_SET = "level_set"


class OptimizerType(str, Enum):
    """Classification of the mathematical optimization update engine."""
    MMA = "mma"
    GCMMA = "gcmma"
    OC = "oc"
    BESO_RANKING = "beso_ranking"


@dataclass
class TopOptPairing:
    """Validated pairing of a formulation model and its mathematical optimizer."""
    formulation: FormulationType
    optimizer: OptimizerType
    note: str = ""


@dataclass
class AlgorithmCapabilities:
    """Metadata describing the traits and constraints handling of an optimization algorithm.

    Attributes:
        name: Canonical algorithm identifier.
        display_name: Human-readable display title.
        supports_multiple_constraints: True if arbitrary m >= 1 constraints are handled simultaneously.
        supports_stress_constraint: True if local or aggregated stress constraints are supported.
        supports_frequency_constraint: True if eigenvalue/frequency constraints are supported.
        supports_com_constraint: True if center of mass / inertia constraints are supported.
        variable_type: Type of design representation ('continuous', 'binary', or 'level_set').
        convergence_guarantee: Mathematical convergence type ('strictly_globally_convergent',
            'locally_convergent', or 'heuristic').
        recommended_for: Detailed engineering recommendation description.
    """
    name: str
    display_name: str
    supports_multiple_constraints: bool = True
    supports_stress_constraint: bool = True
    supports_frequency_constraint: bool = True
    supports_com_constraint: bool = True
    variable_type: str = "continuous"
    convergence_guarantee: str = "locally_convergent"
    recommended_for: str = ""


@dataclass
class OptimizationStepInput:
    """Standardized input packet passed to an algorithm adapter at each optimization step.

    Attributes:
        iteration: 1-based outer iteration index.
        rho: Current mathematical design density array of shape (N_elem,).
        df_drho: Objective sensitivity vector of shape (N_elem,).
        g_vals: Evaluated constraint values g_i(rho) <= 0 of shape (m,).
        dg_drho: Constraint sensitivity matrix of shape (m, N_elem) or (N_elem,).
        elem_volumes: Optional element volume array of shape (N_elem,).
        passive_solid: Optional mask or indices of elements locked to solid (rho=rho_max).
        passive_void: Optional mask or indices of elements locked to void (rho=rho_min).
        f0_val: Optional objective function value at current step.
        eval_fn: Optional black-box evaluator returning (f0_val, g_vals) for conservative inner loops.
        extra_metrics: Dictionary of diagnostic metrics (e.g. max_stress, com_shift, freq1).
    """
    iteration: int
    rho: np.ndarray
    df_drho: np.ndarray
    g_vals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    dg_drho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    elem_volumes: Optional[np.ndarray] = None
    passive_solid: Optional[np.ndarray] = None
    passive_void: Optional[np.ndarray] = None
    f0_val: float = 0.0
    eval_fn: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStepOutput:
    """Standardized result packet returned by an algorithm adapter after each step.

    Attributes:
        rho_next: Updated mathematical design density array of shape (N_elem,).
        change: Maximum absolute step change ||rho^(k+1) - rho^(k)||_inf.
        dual_multipliers: Optional dual Lagrange multiplier vector lambda of shape (m,).
        subproblem_converged: True if the mathematical subproblem successfully converged.
        subproblem_time_ms: Wall-clock execution time of the subproblem solve in milliseconds.
        algorithm: Name of the algorithm that performed the update.
        diagnostics: Additional mathematical diagnostics (e.g. KKT residual, inner iterations).
    """
    rho_next: np.ndarray
    change: float
    dual_multipliers: Optional[np.ndarray] = None
    subproblem_converged: bool = True
    subproblem_time_ms: float = 0.0
    algorithm: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TopOptAlgorithmAdapter(Optimizer, ABC):
    """Abstract base adapter unifying diverse topology optimization solvers."""

    name: str = "base_adapter"
    capabilities: AlgorithmCapabilities

    @abstractmethod
    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        """Initialize the algorithm internal state for a given problem size and volume."""
        pass

    @abstractmethod
    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        """Execute one optimization update step using the standardized data contract."""
        pass

    def reset(self) -> None:
        """Reset internal history, filters, or subproblem caches."""
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics from the most recent iteration."""
        return {}

    # Backward compatibility with Optimizer.update()
    def update(
        self,
        rho: np.ndarray,
        df_drho: np.ndarray,
        g_vals: np.ndarray,
        dg_drho: np.ndarray,
        elem_volumes: Optional[np.ndarray] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
        iteration: int = 1,
        f0_val: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Drop-in compatible update method executing through the standardized adapter."""
        inp = OptimizationStepInput(
            iteration=iteration,
            rho=rho,
            df_drho=df_drho,
            g_vals=g_vals,
            dg_drho=dg_drho,
            elem_volumes=elem_volumes,
            passive_solid=passive_solid,
            passive_void=passive_void,
            f0_val=f0_val,
            extra_metrics=kwargs,
        )
        out = self.step(inp)
        return out.rho_next


# ==============================================================================
# 1. MMA Adapter (Universal Default Recommendation)
# ==============================================================================

class MMAAdapter(TopOptAlgorithmAdapter):
    """Adapter for Svanberg (1987) Method of Moving Asymptotes (MMA).

    Universal default optimizer for multi-physics topology optimization.
    Features:
        - High-speed dual subproblem solving with Svanberg moving asymptotes.
        - Backends: 'mmapy' (GitHub Svanberg official Python port), 'scipy' dual NR.
        - Robust support for volume, von Mises stress, CoM, and natural frequencies.
    """

    name = "mma"
    capabilities = AlgorithmCapabilities(
        name="mma",
        display_name="Method of Moving Asymptotes (MMA)",
        supports_multiple_constraints=True,
        supports_stress_constraint=True,
        supports_frequency_constraint=True,
        supports_com_constraint=True,
        variable_type="continuous",
        convergence_guarantee="locally_convergent",
        recommended_for="Default universal optimizer for multi-constraint continuum topology optimization.",
    )

    def __init__(
        self,
        move: float = 0.2,
        backend: str = "mmapy",
        rho_min: float = 1e-3,
        rho_max: float = 1.0,
        **kwargs,
    ) -> None:
        self.move = move
        self.backend = backend
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.kwargs = {k: v for k, v in kwargs.items() if k not in ("target_volfrac", "volfrac")}
        self._solver = MMAOptimizer(
            move=self.move,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            backend=self.backend,
            **self.kwargs,
        )

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self._solver.reset()

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)

        rho_new = self._solver.update(
            rho=rho_curr,
            df_drho=step_input.df_drho,
            g_vals=step_input.g_vals,
            dg_drho=step_input.dg_drho,
            elem_volumes=step_input.elem_volumes,
            passive_solid=step_input.passive_solid,
            passive_void=step_input.passive_void,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            dual_multipliers=None,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"backend": self.backend, "iter": self._solver.iter},
        )

    def reset(self) -> None:
        self._solver.reset()


# ==============================================================================
# 2. GCMMA Adapter (Globally Convergent MMA with Conservatism Loop)
# ==============================================================================

class GCMMAAdapter(TopOptAlgorithmAdapter):
    """Adapter for Svanberg (2002) Globally Convergent MMA (GCMMA).

    Features:
        - Conservatism parameter raa updating ensuring strict convex upper bounds.
        - Guaranteed global convergence even for severe non-convex stress/eigenvalue problems.
    """

    name = "gcmma"
    capabilities = AlgorithmCapabilities(
        name="gcmma",
        display_name="Globally Convergent MMA (GCMMA)",
        supports_multiple_constraints=True,
        supports_stress_constraint=True,
        supports_frequency_constraint=True,
        supports_com_constraint=True,
        variable_type="continuous",
        convergence_guarantee="strictly_globally_convergent",
        recommended_for="Non-convex, stress-constrained, or mode-switching optimization requiring global convergence.",
    )

    def __init__(
        self,
        move: float = 0.2,
        rho_min: float = 1e-3,
        rho_max: float = 1.0,
        max_inner_iter: int = 15,
        **kwargs,
    ) -> None:
        self.move = move
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.max_inner_iter = max_inner_iter
        self.kwargs = kwargs

        self.outer_iter: int = 0
        self.xold1: Optional[np.ndarray] = None
        self.xold2: Optional[np.ndarray] = None
        self.low: Optional[np.ndarray] = None
        self.upp: Optional[np.ndarray] = None
        self.raa0: float = 0.01
        self.raa: Optional[np.ndarray] = None

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self.reset()

    def reset(self) -> None:
        self.outer_iter = 0
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.raa0 = 0.01
        self.raa = None

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)
        n = len(rho_curr)

        g_arr = np.atleast_1d(np.asarray(step_input.g_vals, dtype=np.float64))
        m = len(g_arr)

        if not HAS_MMAPY:
            # Fallback to standard MMA if mmapy is not installed
            fallback = MMAAdapter(move=self.move, rho_min=self.rho_min, rho_max=self.rho_max)
            return fallback.step(step_input)

        self.outer_iter += 1

        xval = rho_curr.reshape(-1, 1)
        xmin = np.full((n, 1), self.rho_min, dtype=np.float64)
        xmax = np.full((n, 1), self.rho_max, dtype=np.float64)

        if self.xold1 is None:
            self.xold1 = xval.copy()
        if self.xold2 is None:
            self.xold2 = self.xold1.copy()
        if self.low is None:
            self.low = xmin.copy()
        if self.upp is None:
            self.upp = xmax.copy()
        if self.raa is None:
            self.raa = 0.01 * np.ones((m, 1), dtype=np.float64)

        raa0eps = 1e-6
        raaeps = 1e-6 * np.ones((m, 1), dtype=np.float64)
        epsimin = 1e-7

        df0dx = np.asarray(step_input.df_drho, dtype=np.float64).reshape(-1, 1).copy()
        fval = g_arr.reshape(-1, 1) if m > 0 else np.zeros((0, 1), dtype=np.float64)

        if m > 0:
            dfdx = np.asarray(step_input.dg_drho, dtype=np.float64)
            if dfdx.ndim == 1:
                dfdx = dfdx.reshape(1, n)
        else:
            dfdx = np.zeros((0, n), dtype=np.float64)

        # Handle passive regions by zeroing sensitivities
        solid_mask = np.zeros(n, dtype=bool)
        if step_input.passive_solid is not None:
            if np.asarray(step_input.passive_solid).dtype == bool:
                solid_mask = np.asarray(step_input.passive_solid, dtype=bool)
            else:
                solid_mask[np.asarray(step_input.passive_solid, dtype=np.int64)] = True

        void_mask = np.zeros(n, dtype=bool)
        if step_input.passive_void is not None:
            if np.asarray(step_input.passive_void).dtype == bool:
                void_mask = np.asarray(step_input.passive_void, dtype=bool)
            else:
                void_mask[np.asarray(step_input.passive_void, dtype=np.int64)] = True

        passive_any = solid_mask | void_mask
        if np.any(passive_any):
            df0dx[passive_any] = 0.0
            if m > 0:
                dfdx[:, passive_any] = 0.0

        a0 = 1.0
        a = np.zeros((m, 1), dtype=np.float64)
        c = 1000.0 * np.ones((m, 1), dtype=np.float64)
        d = np.ones((m, 1), dtype=np.float64)

        # 1. Update moving asymptotes and raa parameters
        self.low, self.upp, self.raa0, self.raa = asymp(
            self.outer_iter,
            n,
            xval,
            self.xold1,
            self.xold2,
            xmin,
            xmax,
            self.low,
            self.upp,
            self.raa0,
            self.raa,
            raa0eps,
            raaeps,
            df0dx,
            dfdx,
        )

        # 2. Solve GCMMA subproblem
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
            m,
            n,
            self.outer_iter,
            epsimin,
            xval,
            xmin,
            xmax,
            self.low,
            self.upp,
            self.raa0,
            self.raa,
            step_input.f0_val,
            df0dx,
            fval,
            dfdx,
            a0,
            a,
            c,
            d,
        )

        # 3. Conservative inner loop if evaluator function is provided
        inner_count = 0
        if step_input.eval_fn is not None:
            f0_new, f_new = step_input.eval_fn(xmma.ravel())
            f_new_arr = np.atleast_1d(f_new).reshape(-1, 1)
            conserv = concheck(m, epsimin, f0app, f0_new, fapp, f_new_arr)

            while conserv == 0 and inner_count < self.max_inner_iter:
                inner_count += 1
                self.raa0, self.raa = raaupdate(
                    xmma,
                    xval,
                    xmin,
                    xmax,
                    self.low,
                    self.upp,
                    f0_new,
                    f_new_arr,
                    f0app,
                    fapp,
                    self.raa0,
                    self.raa,
                    raa0eps,
                    raaeps,
                    epsimin,
                )
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                    m,
                    n,
                    self.outer_iter,
                    epsimin,
                    xval,
                    xmin,
                    xmax,
                    self.low,
                    self.upp,
                    self.raa0,
                    self.raa,
                    step_input.f0_val,
                    df0dx,
                    fval,
                    dfdx,
                    a0,
                    a,
                    c,
                    d,
                )
                f0_new, f_new = step_input.eval_fn(xmma.ravel())
                f_new_arr = np.atleast_1d(f_new).reshape(-1, 1)
                conserv = concheck(m, epsimin, f0app, f0_new, fapp, f_new_arr)

        # 4. Update history
        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()

        rho_new = xmma.ravel()
        if np.any(solid_mask):
            rho_new[solid_mask] = self.rho_max
        if np.any(void_mask):
            rho_new[void_mask] = self.rho_min

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            dual_multipliers=lam.ravel() if lam is not None else None,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"outer_iter": self.outer_iter, "inner_iter": inner_count},
        )


# ==============================================================================
# 3. OC Adapter (Optimality Criteria Bisection)
# ==============================================================================

class OCAdapter(TopOptAlgorithmAdapter):
    """Adapter for Bisection Optimality Criteria (OC) volume-constrained optimizer."""

    name = "oc"
    capabilities = AlgorithmCapabilities(
        name="oc",
        display_name="Optimality Criteria (OC)",
        supports_multiple_constraints=False,
        supports_stress_constraint=False,
        supports_frequency_constraint=False,
        supports_com_constraint=False,
        variable_type="continuous",
        convergence_guarantee="heuristic",
        recommended_for="Ultra-fast baseline for single volume-constrained compliance problems.",
    )

    def __init__(
        self,
        volfrac: float = 0.5,
        move: float = 0.2,
        eta: float = 0.5,
        rho_min: float = 1e-3,
        tol: float = 1e-9,
        **kwargs,
    ) -> None:
        clean_kw = dict(kwargs)
        if "target_volfrac" in clean_kw:
            volfrac = float(clean_kw.pop("target_volfrac"))
        self.volfrac = volfrac
        self.move = move
        self.eta = eta
        self.rho_min = rho_min
        self.tol = tol
        self.kwargs = clean_kw
        self._solver = OCOptimizer(
            volfrac=self.volfrac,
            move=self.move,
            eta=self.eta,
            rho_min=self.rho_min,
            tol=self.tol,
            **self.kwargs,
        )

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self.volfrac = target_vol
        self._solver.volfrac = target_vol
        if hasattr(self._solver, "reset"):
            self._solver.reset()

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)

        rho_new = self._solver.update(
            rho=rho_curr,
            df_drho=step_input.df_drho,
            g_vals=step_input.g_vals,
            dg_drho=step_input.dg_drho,
            elem_volumes=step_input.elem_volumes,
            passive_solid=step_input.passive_solid,
            passive_void=step_input.passive_void,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            dual_multipliers=None,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"move": self.move, "eta": self.eta},
        )

    def reset(self) -> None:
        if hasattr(self._solver, "reset"):
            self._solver.reset()


# ==============================================================================
# 4. BESO Adapter (Bi-directional Evolutionary Structural Optimization)
# ==============================================================================

class BESOAdapter(TopOptAlgorithmAdapter):
    """Adapter for Bi-directional Evolutionary Structural Optimization (BESO)."""

    name = "beso"
    capabilities = AlgorithmCapabilities(
        name="beso",
        display_name="Bi-directional Evolutionary Structural Optimization (BESO)",
        supports_multiple_constraints=False,
        supports_stress_constraint=False,
        supports_frequency_constraint=False,
        supports_com_constraint=False,
        variable_type="binary",
        convergence_guarantee="heuristic",
        recommended_for="Sharp 0/1 binary structural topologies without gray elements.",
    )

    def __init__(
        self,
        target_volfrac: float = 0.5,
        er: float = 0.02,
        rho_min: float = 1e-3,
        move: float = 0.2,
        **kwargs,
    ) -> None:
        clean_kw = dict(kwargs)
        if "volfrac" in clean_kw:
            target_volfrac = float(clean_kw.pop("volfrac"))
        self.target_volfrac = target_volfrac
        self.er = er
        self.rho_min = rho_min
        self.move = move
        self.kwargs = clean_kw
        self._solver = BESOOptimizer(
            target_volfrac=self.target_volfrac,
            er=self.er,
            rho_min=self.rho_min,
            move=self.move,
            **self.kwargs,
        )

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self.target_volfrac = target_vol
        self._solver.target_volfrac = target_vol
        self._solver.reset()

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)

        rho_new = self._solver.update(
            rho=rho_curr,
            df_drho=step_input.df_drho,
            g_vals=step_input.g_vals,
            dg_drho=step_input.dg_drho,
            elem_volumes=step_input.elem_volumes,
            passive_solid=step_input.passive_solid,
            passive_void=step_input.passive_void,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"er": self.er, "iter": self._solver.iter},
        )

    def reset(self) -> None:
        self._solver.reset()


# ==============================================================================
# 5. SEMDOT Adapter (Smooth-Edged Material Distribution)
# ==============================================================================

class SEMDOTAdapter(TopOptAlgorithmAdapter):
    """Adapter for Smooth-Edged Material Distribution for Optimizing Topology (SEMDOT)."""

    name = "semdot"
    capabilities = AlgorithmCapabilities(
        name="semdot",
        display_name="Smooth-Edged Material Distribution (SEMDOT)",
        supports_multiple_constraints=True,
        supports_stress_constraint=True,
        supports_frequency_constraint=False,
        supports_com_constraint=True,
        variable_type="continuous",
        convergence_guarantee="locally_convergent",
        recommended_for="Smooth CAD-ready boundary topologies via sub-element mapping.",
    )

    def __init__(
        self,
        volfrac: float = 0.5,
        n_sub: int = 4,
        move: float = 0.2,
        rho_min: float = 1e-3,
        **kwargs,
    ) -> None:
        clean_kw = dict(kwargs)
        if "target_volfrac" in clean_kw:
            volfrac = float(clean_kw.pop("target_volfrac"))
        self.volfrac = volfrac
        self.n_sub = n_sub
        self.move = move
        self.rho_min = rho_min
        self.kwargs = clean_kw
        self._solver = SEMDOTOptimizer(
            volfrac=self.volfrac,
            n_sub=self.n_sub,
            move=self.move,
            rho_min=self.rho_min,
            **self.kwargs,
        )

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self.volfrac = target_vol
        self._solver.volfrac = target_vol
        self._solver.reset()

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)

        rho_new = self._solver.update(
            rho=rho_curr,
            df_drho=step_input.df_drho,
            g_vals=step_input.g_vals,
            dg_drho=step_input.dg_drho,
            elem_volumes=step_input.elem_volumes,
            passive_solid=step_input.passive_solid,
            passive_void=step_input.passive_void,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"move": self.move, "iter": self._solver.iter},
        )

    def reset(self) -> None:
        self._solver.reset()


# ==============================================================================
# 6. Level-Set Adapter (Reaction-Diffusion Level Set)
# ==============================================================================

class LevelSetAdapter(TopOptAlgorithmAdapter):
    """Adapter for Reaction-Diffusion Reaction Level Set Optimizer."""

    name = "level_set"
    capabilities = AlgorithmCapabilities(
        name="level_set",
        display_name="Reaction-Diffusion Level-Set",
        supports_multiple_constraints=False,
        supports_stress_constraint=False,
        supports_frequency_constraint=False,
        supports_com_constraint=False,
        variable_type="level_set",
        convergence_guarantee="locally_convergent",
        recommended_for="Implicit smooth-boundary topology optimization with topological nucleation.",
    )

    def __init__(
        self,
        volfrac: float = 0.5,
        ell: float = 0.1,
        gamma: float = 0.1,
        rho_min: float = 1e-3,
        move: float = 0.2,
        **kwargs,
    ) -> None:
        self.volfrac = volfrac
        self.ell = ell
        self.gamma = gamma
        self.rho_min = rho_min
        self.move = move
        self.kwargs = kwargs
        self._solver = LevelSetOptimizer(
            volfrac=self.volfrac,
            ell=self.ell,
            gamma=self.gamma,
            rho_min=self.rho_min,
            move=self.move,
            **self.kwargs,
        )

    def initialize(self, n_elem: int, target_vol: float, **kwargs) -> None:
        self.volfrac = target_vol
        self._solver.volfrac = target_vol
        self._solver.reset()

    def step(self, step_input: OptimizationStepInput) -> OptimizationStepOutput:
        t0 = time.perf_counter()
        rho_curr = np.asarray(step_input.rho, dtype=np.float64)

        rho_new = self._solver.update(
            rho=rho_curr,
            df_drho=step_input.df_drho,
            g_vals=step_input.g_vals,
            dg_drho=step_input.dg_drho,
            elem_volumes=step_input.elem_volumes,
            passive_solid=step_input.passive_solid,
            passive_void=step_input.passive_void,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        change = float(np.max(np.abs(rho_new - rho_curr)))

        return OptimizationStepOutput(
            rho_next=rho_new,
            change=change,
            subproblem_converged=True,
            subproblem_time_ms=dt_ms,
            algorithm=self.name,
            diagnostics={"ell": self.ell, "iter": self._solver.iter},
        )

    def reset(self) -> None:
        self._solver.reset()


# ==============================================================================
# 7. Algorithm Registry (Factory & Dynamic Discovery)
# ==============================================================================

class AlgorithmRegistry:
    """Central registry and factory for topology optimization algorithm adapters."""

    _registry: Dict[str, Type[TopOptAlgorithmAdapter]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        adapter_cls: Type[TopOptAlgorithmAdapter],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register an algorithm adapter with a canonical name and optional aliases."""
        canon = name.strip().lower()
        cls._registry[canon] = adapter_cls
        if aliases:
            for alias in aliases:
                cls._aliases[alias.strip().lower()] = canon

    @classmethod
    def resolve_name(cls, name: str) -> str:
        """Resolve an alias or raw name to its canonical identifier."""
        clean = name.strip().lower()
        return cls._aliases.get(clean, clean)

    @classmethod
    def get(cls, name: str, **kwargs) -> TopOptAlgorithmAdapter:
        """Instantiate an algorithm adapter by name (or alias)."""
        canon = cls.resolve_name(name)
        if canon not in cls._registry:
            avail = sorted(list(cls._registry.keys()))
            raise KeyError(f"Unknown optimization algorithm '{name}' (resolved: '{canon}'). Available: {avail}")
        adapter_cls = cls._registry[canon]
        return adapter_cls(**kwargs)

    @classmethod
    def get_default(cls, **kwargs) -> TopOptAlgorithmAdapter:
        """Return the default recommended algorithm (MMA with mmapy backend)."""
        return cls.get("mma", **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered canonical algorithm names."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def get_capabilities(cls, name: str) -> AlgorithmCapabilities:
        """Retrieve the capabilities metadata for a registered algorithm."""
        canon = cls.resolve_name(name)
        if canon not in cls._registry:
            raise KeyError(f"Unknown algorithm '{name}'")
        return cls._registry[canon].capabilities

    @classmethod
    def create_system(
        cls,
        formulation: Union[str, FormulationType] = FormulationType.SIMP,
        optimizer: Optional[Union[str, OptimizerType]] = None,
        **kwargs,
    ) -> TopOptAlgorithmAdapter:
        """Create a decoupled optimization pipeline pairing Formulation and Optimizer.

        Args:
            formulation: Material/geometric formulation ('simp', 'beso', 'semdot', 'level_set').
            optimizer: Mathematical optimizer engine ('mma', 'gcmma', 'oc'). If None,
                the optimal default for the formulation is selected (e.g. MMA for SIMP).
            **kwargs: Extra parameters passed to the adapter.

        Returns:
            Configured TopOptAlgorithmAdapter instance with formulation and optimizer metadata.
        """
        f_str = str(formulation.value if isinstance(formulation, FormulationType) else formulation).lower().strip()

        # Handle compound legacy names like 'simp_mma' or 'simp_oc'
        if "_" in f_str and optimizer is None:
            parts = f_str.split("_", 1)
            if parts[0] == "simp" and parts[1] in {"mma", "oc"}:
                f_str = parts[0]
                optimizer = parts[1]

        if f_str == FormulationType.BESO.value:
            adapter = cls.get("beso", **kwargs)
            adapter.formulation = FormulationType.BESO
            adapter.optimizer_type = OptimizerType.BESO_RANKING
            return adapter

        elif f_str == FormulationType.SEMDOT.value:
            opt_str = "mma" if optimizer is None else str(optimizer.value if isinstance(optimizer, OptimizerType) else optimizer).lower().strip()
            adapter = cls.get("semdot", **kwargs)
            adapter.formulation = FormulationType.SEMDOT
            adapter.optimizer_type = OptimizerType(cls.resolve_name(opt_str))
            return adapter

        elif f_str == FormulationType.LEVEL_SET.value:
            adapter = cls.get("level_set", **kwargs)
            adapter.formulation = FormulationType.LEVEL_SET
            adapter.optimizer_type = OptimizerType.MMA
            return adapter

        elif f_str == FormulationType.SIMP.value:
            opt_str = "mma" if optimizer is None else str(optimizer.value if isinstance(optimizer, OptimizerType) else optimizer).lower().strip()
            canon_opt = cls.resolve_name(opt_str)
            if canon_opt not in {"mma", "gcmma", "oc"}:
                canon_opt = "mma"
            adapter = cls.get(canon_opt, **kwargs)
            adapter.formulation = FormulationType.SIMP
            adapter.optimizer_type = OptimizerType(canon_opt)
            return adapter

        else:
            adapter = cls.get(f_str, **kwargs)
            adapter.formulation = FormulationType.SIMP
            adapter.optimizer_type = OptimizerType.MMA
            return adapter


# Register built-in standard algorithms
AlgorithmRegistry.register("mma", MMAAdapter, aliases=["simp_mma", "svanberg_mma", "nlopt_mma", "simp"])
AlgorithmRegistry.register("gcmma", GCMMAAdapter, aliases=["globally_convergent_mma"])
AlgorithmRegistry.register("oc", OCAdapter, aliases=["simp_oc", "optimality_criteria"])
AlgorithmRegistry.register("beso", BESOAdapter, aliases=["evolutionary"])
AlgorithmRegistry.register("semdot", SEMDOTAdapter, aliases=["smooth_edged"])
AlgorithmRegistry.register("level_set", LevelSetAdapter, aliases=["reaction_diffusion", "ls"])

__all__ = [
    "FormulationType",
    "OptimizerType",
    "TopOptPairing",
    "AlgorithmCapabilities",
    "OptimizationStepInput",
    "OptimizationStepOutput",
    "TopOptAlgorithmAdapter",
    "MMAAdapter",
    "GCMMAAdapter",
    "OCAdapter",
    "BESOAdapter",
    "SEMDOTAdapter",
    "LevelSetAdapter",
    "AlgorithmRegistry",
]
