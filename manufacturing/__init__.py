"""Manufacturing constraints and projections for topology optimization.

Provides:
- ThreeFieldProjection: Sigmund (2009) & Wang et al. (2011) robust 3-field Heaviside projection.
- compute_eta_from_length_scale: Threshold calculation for target minimum length scale.
- GeometricLengthScaleConstraint: Zhou et al. (2015) structural indicator minimum length scale constraint.
- OverhangFilter: Langelaar (2016/2017) Additive Manufacturing self-supporting filter.
- MillingFilter: Langelaar (2018) multi-axis subtractive CNC milling filter.
- CastingFilter: Gersborg & Sigmund (2006) single-draw and parting-plane split mold casting filter.
- ExtrusionFilter: Volume-weighted constant cross-section extrusion filter.
"""

from __future__ import annotations

from .casting import CastingFilter
from .extrusion import ExtrusionFilter, BottomPlateZExtrusionFilter, ShellRibExtrusionFilter
from .milling import MillingFilter
from .minimum_length import (
    GeometricLengthScaleConstraint,
    ThreeFieldProjection,
    compute_eta_from_length_scale,
)
from .overhang import OverhangFilter

__all__ = [
    "ThreeFieldProjection",
    "compute_eta_from_length_scale",
    "GeometricLengthScaleConstraint",
    "OverhangFilter",
    "MillingFilter",
    "CastingFilter",
    "ExtrusionFilter",
    "BottomPlateZExtrusionFilter",
    "ShellRibExtrusionFilter",
]
