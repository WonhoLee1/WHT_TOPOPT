"""Finite element solver adapters and interfaces.

Exports:
    - LoadCase, InertiaReliefConfig, SolverResult, FEASolver, build_rigid_body_modes
    - ScipySolver (Direct splu & iterative CG/MINRES)
    - PyAMGSolver (Smoothed aggregation AMG with rigid-body nullspace)
    - KratosSolver (In-memory ModelPart execution & subprocess fallback)
    - FrontISTRSolver (SMP & Hybrid MPI with exact/quantized materials)
"""
from __future__ import annotations

from .base import (
    FEASolver,
    InertiaReliefConfig,
    LoadCase,
    SolverResult,
    build_rigid_body_modes,
)
from .frontistr_adapter import FrontISTRSolver
from .kratos_solver import KratosSolver
from .pyamg_solver import PyAMGSolver
from .scipy_solver import ScipySolver

__all__ = [
    "LoadCase",
    "InertiaReliefConfig",
    "SolverResult",
    "FEASolver",
    "build_rigid_body_modes",
    "ScipySolver",
    "PyAMGSolver",
    "KratosSolver",
    "FrontISTRSolver",
]
