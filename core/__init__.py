"""Core finite element, mesh, filter, interpolation, and boundary condition modules."""

from __future__ import annotations

from .mesh import (
    StructuredMesh2D,
    StructuredMesh3D,
    UnstructuredMesh,
)

from .fe_builders import (
    isotropic_elasticity_2d,
    isotropic_elasticity_3d,
    tri3_ke0_me0,
    q4_ke0_me0,
    tet4_ke0_me0,
    tet4_nodal_smoothed_strains,
    tet10_ke0_me0,
    hex8_ke0_me0,
    hex20_ke0_me0,
    mitc3_ke0_me0,
    mitc4_ke0_me0,
)

from .fem import FEMEngine

from .filter import (
    FilterOperator,
    DensityFilter,
    SensitivityFilter,
    HelmholtzPDEFilter,
)

from .interpolation import (
    MaterialInterpolation,
    SIMP,
    ClassicSIMP,
    RAMP,
    MultiMaterialSIMP,
)

from .projection import HeavisideProjection

from .continuation import ContinuationScheduler

from .boundary import BoundaryConditionManager

__all__ = [
    # Mesh
    "StructuredMesh2D",
    "StructuredMesh3D",
    "UnstructuredMesh",
    # Element & FE Builders
    "isotropic_elasticity_2d",
    "isotropic_elasticity_3d",
    "tri3_ke0_me0",
    "q4_ke0_me0",
    "tet4_ke0_me0",
    "tet4_nodal_smoothed_strains",
    "tet10_ke0_me0",
    "hex8_ke0_me0",
    "hex20_ke0_me0",
    "mitc3_ke0_me0",
    "mitc4_ke0_me0",
    # FEM Engine
    "FEMEngine",
    # Filters
    "FilterOperator",
    "DensityFilter",
    "SensitivityFilter",
    "HelmholtzPDEFilter",
    # Interpolation
    "MaterialInterpolation",
    "SIMP",
    "ClassicSIMP",
    "RAMP",
    "MultiMaterialSIMP",
    # Projection
    "HeavisideProjection",
    # Continuation
    "ContinuationScheduler",
    # Boundary Conditions
    "BoundaryConditionManager",
]
