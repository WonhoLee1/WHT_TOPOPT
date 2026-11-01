"""Declarative pipeline configuration, 13-stage runner, and visualization report generator."""
from __future__ import annotations

from .config import (
    BCConfig,
    ContinuationConfig,
    FilterConfig,
    ManufacturingConfig,
    MaterialConfig,
    MeshConfig,
    MuJoCoConfig,
    OptimizerConfig,
    PhysicsConstraintsConfig,
    PostConfig,
    ProjectionConfig,
    RobustConfig,
    SolverConfig,
    TopOptConfig,
)
from .report import (
    DEFAULT_FIGSIZE_CONVERGENCE,
    DEFAULT_FIGSIZE_DENSITY_2D,
    DEFAULT_FONT_SIZE,
    DEFAULT_PYVISTA_FONT_SIZE,
    TopOptReport,
)
from .runner import TopOptPipelineResult, TopOptRunner

__all__ = [
    "BCConfig",
    "ContinuationConfig",
    "FilterConfig",
    "ManufacturingConfig",
    "MaterialConfig",
    "MeshConfig",
    "MuJoCoConfig",
    "OptimizerConfig",
    "PhysicsConstraintsConfig",
    "PostConfig",
    "ProjectionConfig",
    "RobustConfig",
    "SolverConfig",
    "TopOptConfig",
    "DEFAULT_FIGSIZE_CONVERGENCE",
    "DEFAULT_FIGSIZE_DENSITY_2D",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_PYVISTA_FONT_SIZE",
    "TopOptReport",
    "TopOptPipelineResult",
    "TopOptRunner",
]
