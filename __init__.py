"""wht_topopt: Next-Generation Multi-Solver Topology Optimization Framework (v3.0)."""
from __future__ import annotations

__version__ = "3.0.0"

from . import (
    algorithms,
    core,
    functions,
    manufacturing,
    multiscale,
    neural,
    pipeline,
    post,
    robust,
    solvers,
)
from .pipeline import TopOptConfig, TopOptPipelineResult, TopOptReport, TopOptRunner

__all__ = [
    "__version__",
    "algorithms",
    "core",
    "functions",
    "manufacturing",
    "multiscale",
    "neural",
    "pipeline",
    "post",
    "robust",
    "solvers",
    "TopOptConfig",
    "TopOptPipelineResult",
    "TopOptReport",
    "TopOptRunner",
]
