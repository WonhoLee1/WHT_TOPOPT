"""Topology optimization algorithms and mathematical programming update engines.

Exports:
    - GradientSource (Enum indicating source of sensitivities)
    - OptState (Design state snapshot dataclass)
    - TrustRegionController (Adaptive step size controller for surrogate/learned gradients)
    - Optimizer (Abstract base class)
    - OptimalityCriteria / OCOptimizer (Volume-weighted OC bisection optimizer)
    - MMAOptimizer (Method of Moving Asymptotes nonlinearly constrained optimizer)
    - BESOOptimizer (Bi-directional Evolutionary Structural Optimization with adaptive ER)
    - LevelSetOptimizer (Hilbertian projection Level-Set optimizer)
    - SEMDOTOptimizer (Smooth-Edged Material Distribution for Optimizing Topology)
"""
from __future__ import annotations

from .base import (
    GradientSource,
    OptState,
    Optimizer,
    TrustRegionController,
)
from .oc import OCOptimizer, OptimalityCriteria
from .mma_wrapper import MMAOptimizer
from .beso import BESOOptimizer
from .level_set import LevelSetOptimizer
from .semdot import SEMDOTOptimizer
from .hca import SEAHCAConfig, StressEnergyAdaptiveHCA, HCAOptimizer
from .adapter import (
    FormulationType,
    OptimizerType,
    TopOptPairing,
    AlgorithmCapabilities,
    OptimizationStepInput,
    OptimizationStepOutput,
    TopOptAlgorithmAdapter,
    MMAAdapter,
    GCMMAAdapter,
    OCAdapter,
    BESOAdapter,
    SEMDOTAdapter,
    LevelSetAdapter,
    AlgorithmRegistry,
)

__all__ = [
    "FormulationType",
    "OptimizerType",
    "TopOptPairing",
    "GradientSource",
    "OptState",
    "TrustRegionController",
    "Optimizer",
    "OptimalityCriteria",
    "OCOptimizer",
    "MMAOptimizer",
    "BESOOptimizer",
    "LevelSetOptimizer",
    "SEMDOTOptimizer",
    "SEAHCAConfig",
    "StressEnergyAdaptiveHCA",
    "HCAOptimizer",
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
