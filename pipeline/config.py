"""Declarative Pydantic v2 configuration schema for the wht_topopt 13-stage pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MeshConfig(BaseModel):
    """Finite element mesh configuration."""

    model_config = ConfigDict(extra="forbid")

    dimension: Literal[2, 3] = 2
    nelx: int = Field(default=30, ge=1)
    nely: int = Field(default=10, ge=1)
    nelz: int = Field(default=1, ge=1)
    lx: Optional[float] = Field(default=None, gt=0.0)
    ly: Optional[float] = Field(default=None, gt=0.0)
    lz: Optional[float] = Field(default=None, gt=0.0)
    elem_type: Optional[str] = None


class MaterialConfig(BaseModel):
    """Material interpolation and constitutive properties."""

    model_config = ConfigDict(extra="forbid")

    e0: float = Field(default=1.0, gt=0.0)
    emin: float = Field(default=1e-9, gt=0.0)
    nu: float = Field(default=0.3, gt=-1.0, lt=0.5)
    rho0: float = Field(default=1.0, gt=0.0)
    model: Literal["simp", "simp_modified", "ramp", "hs_upper", "halpin_tsai"] = "simp_modified"
    penal: float = Field(default=3.0, ge=1.0)
    ramp_q: float = Field(default=3.0, ge=0.0)
    plane_stress: bool = True


class BCConfig(BaseModel):
    """Boundary condition, load, and passive domain configuration."""

    model_config = ConfigDict(extra="forbid")

    preset: Literal[
        "cantilever_2d",
        "mbb_2d",
        "column_2d",
        "cantilever_3d",
        "custom",
        "mujoco",
    ] = "cantilever_2d"
    fixed_dofs: Optional[List[int]] = None
    loads: Optional[List[Tuple[int, float]]] = None
    force_magnitude: float = 1.0
    passive_solid: Optional[List[int]] = None
    passive_void: Optional[List[int]] = None


class FilterConfig(BaseModel):
    """Spatial regularization filter configuration."""

    model_config = ConfigDict(extra="forbid")

    filter_type: Literal["cone", "gaussian", "helmholtz", "none"] = "cone"
    rmin: float = Field(default=1.5, gt=0.0)


class ProjectionConfig(BaseModel):
    """Smoothed Heaviside threshold projection configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    beta_init: float = Field(default=1.0, gt=0.0)
    eta: float = Field(default=0.5, ge=0.0, le=1.0)


class ContinuationConfig(BaseModel):
    """Parameter continuation schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    penal_init: float = Field(default=1.0, ge=1.0)
    penal_max: float = Field(default=3.0, ge=1.0)
    penal_step: float = Field(default=0.5, gt=0.0)
    beta_init: float = Field(default=1.0, gt=0.0)
    beta_max: float = Field(default=16.0, gt=0.0)
    beta_factor: float = Field(default=2.0, gt=1.0)
    interval: int = Field(default=15, ge=1)


class ManufacturingConfig(BaseModel):
    """Manufacturing constraint filters and geometric length-scale control."""

    model_config = ConfigDict(extra="forbid")

    overhang: bool = False
    build_dir: str = "+y"
    milling: bool = False
    tool_dirs: List[str] = Field(default_factory=lambda: ["+y"])
    casting: bool = False
    draw_dir: str = "+y"
    casting_mode: Literal["single_draw", "parting_plane"] = "single_draw"
    extrusion: bool = False
    extrude_axis: str = "z"
    min_length_scale: Optional[float] = Field(default=None, gt=0.0)


class SolverConfig(BaseModel):
    """Linear FEA solver configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["scipy", "pyamg", "kratos", "frontistr"] = "scipy"
    mode: str = "in_memory"
    tol: float = Field(default=1e-8, gt=0.0)
    maxiter: int = Field(default=1000, ge=1)


class OptimizerConfig(BaseModel):
    """Mathematical programming / topology evolution algorithm configuration."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["oc", "mma", "beso", "level_set", "semdot"] = "oc"
    volfrac: float = Field(default=0.5, gt=0.0, le=1.0)
    move: float = Field(default=0.2, gt=0.0, le=1.0)
    max_iter: int = Field(default=40, ge=1)
    tol_change: float = Field(default=1e-3, gt=0.0)
    rho_min: float = Field(default=1e-3, gt=0.0, lt=1.0)
    use_trust_region: bool = False
    gradient_source: Literal[
        "exact_adjoint",
        "semi_analytic",
        "finite_diff",
        "neural_surrogate",
    ] = "exact_adjoint"


class PhysicsConstraintsConfig(BaseModel):
    """Stress, linear buckling, and modal frequency constraint settings."""

    model_config = ConfigDict(extra="forbid")

    stress_enabled: bool = False
    sigma_allow: float = Field(default=2.0, gt=0.0)
    stress_p: float = Field(default=8.0, ge=1.0)
    stress_q: float = Field(default=0.5, ge=0.0)

    buckling_enabled: bool = False
    lambda_min: float = Field(default=1.0, gt=0.0)
    n_buckling_modes: int = Field(default=4, ge=1)
    buckling_ks_rho: float = Field(default=20.0, gt=0.0)

    modal_enabled: bool = False
    omega_min: float = Field(default=0.1, gt=0.0)
    n_modal_modes: int = Field(default=3, ge=1)


class MuJoCoConfig(BaseModel):
    """MuJoCo multibody dynamics trajectory load and inertia relief configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    xml_path: Optional[str] = None
    xml_string: Optional[str] = None
    body_name: str = "link1"
    duration: float = Field(default=0.4, gt=0.0)
    dt: float = Field(default=0.002, gt=0.0)
    use_inertia_relief: bool = True
    update_mjcf_inertials: bool = False
    updated_xml_output_path: Optional[str] = None


class RobustConfig(BaseModel):
    """3-field eroded/intermediate/dilated robust topology optimization configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    eta_erode: float = Field(default=0.7, gt=0.5, lt=1.0)
    eta_inter: float = Field(default=0.5, gt=0.0, lt=1.0)
    eta_dilate: float = Field(default=0.3, gt=0.0, lt=0.5)
    aggregation: Literal["worst_case", "ks", "weighted_sum"] = "worst_case"


class PostConfig(BaseModel):
    """Post-processing, smoothing, CAD reconstruction, and reporting configuration."""

    model_config = ConfigDict(extra="forbid")

    extract_isosurface: bool = True
    iso_level: float = Field(default=0.5, gt=0.0, lt=1.0)
    taubin_iterations: int = Field(default=5, ge=0)
    taubin_lam: float = Field(default=0.5, gt=0.0)
    taubin_mu: float = Field(default=-0.53, lt=0.0)
    export_stl_path: Optional[str] = None
    export_step_path: Optional[str] = None
    export_report_dir: Optional[str] = None


class TopOptConfig(BaseModel):
    """Master configuration schema for the 13-stage wht_topopt pipeline."""

    model_config = ConfigDict(extra="forbid")

    name: str = "topopt_run"
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    material: MaterialConfig = Field(default_factory=MaterialConfig)
    bc: BCConfig = Field(default_factory=BCConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    continuation: ContinuationConfig = Field(default_factory=ContinuationConfig)
    manufacturing: ManufacturingConfig = Field(default_factory=ManufacturingConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    physics: PhysicsConstraintsConfig = Field(default_factory=PhysicsConstraintsConfig)
    mujoco: MuJoCoConfig = Field(default_factory=MuJoCoConfig)
    robust: RobustConfig = Field(default_factory=RobustConfig)
    post: PostConfig = Field(default_factory=PostConfig)

    @field_validator("mesh")
    @classmethod
    def _validate_mesh(cls, v: MeshConfig) -> MeshConfig:
        if v.dimension == 2 and v.nelz != 1:
            v.nelz = 1
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a Python dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopOptConfig":
        """Instantiate configuration from a Python dictionary."""
        return cls.model_validate(data)

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize configuration to a UTF-8 JSON string (and optionally write to disk)."""
        json_str = self.model_dump_json(indent=indent)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> "TopOptConfig":
        """Load configuration from a JSON file path or raw JSON string."""
        if isinstance(path_or_str, Path) or (
            isinstance(path_or_str, str)
            and not path_or_str.lstrip().startswith("{")
            and Path(path_or_str).exists()
        ):
            with open(Path(path_or_str), "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        return cls.model_validate_json(str(path_or_str))
