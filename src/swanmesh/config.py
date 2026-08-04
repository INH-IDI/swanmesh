"""Configuration models for swanmesh using Pydantic v2."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from swanmesh.errors import ConfigurationError


class InterestPointConfig(BaseModel):
    """Interest point or zone to locally refine mesh size."""
    name: str = "interest_point"
    x: float
    y: float
    hmin: float
    hmax: float
    n_power: float = 1.0

class MeshConfig(BaseModel):
    """Full configuration model for swanmesh pipeline."""

    project_name: str = "swan_mesh"
    output_dir: str = "./output"
    overwrite: bool = False

    # CRS Configuration
    input_crs: str | None = None
    work_crs: str = "EPSG:4326"
    output_crs: str = "EPSG:4326"
    is_utm: bool = False
    utm_zone: str | None = None

    # Z / Elevation Convention
    z_convention: Literal["elevation_negative_down", "depth_positive_down"] = "elevation_negative_down"
    clamp_z_min: float | None = None
    clamp_z_max: float | None = None

    # Input Files
    bathy_xyz_path: Optional[str] = None
    bathy_tif_path: Optional[str] = None
    domain_path: str = ""
    slope_tif_path: Optional[str] = None
    coastline_path: Optional[str] = None

    # Online Bathymetry & Multi-source Blending
    use_online_bathymetry: bool = False
    online_bathy_provider: str = "auto"
    blend_online_bathymetry: bool = False
    blend_width_pixels: float = 10.0

    # DEM Generation Settings
    dx: float = 0.002
    dy: float = 0.002
    buffer_cells: int = 20
    interp_method: str = "linear"

    # Mesh Size Field Settings
    hmin: float = 0.001
    hmax: float = 0.05
    wave_period: float = 30.0
    l_constant: float = 1.5613
    strategy: Literal["product", "mean", "depth_weighted", "hybrid_smooth"] = "product"
    slope_mean: float | None = None
    slope_std: float | None = None
    hybrid_exponent_a: float = 1.0
    hybrid_exponent_b: float = 1.0
    smooth_sigma_pixels: float = 0.0
    combiner: Literal["min", "product"] = "min"
    interest_points: list[InterestPointConfig] = Field(default_factory=list)

    # Boundary & Marker Settings
    depth_limit: float = -50.0
    marker_strategy: Literal["depth_limit", "open_boundary_lines"] = "depth_limit"
    open_boundary_lines_path: str | None = None

    # Export & Sidecars
    export_sidecar_tifs: bool = True
    export_plots: bool = True
    export_msh: bool = True

    @model_validator(mode="after")
    def validate_inputs_and_sizes(self) -> "MeshConfig":
        if not self.bathy_xyz_path and not self.bathy_tif_path and not self.use_online_bathymetry:
            raise ConfigurationError("At least one of bathy_xyz_path, bathy_tif_path, or use_online_bathymetry must be provided.")
        if self.hmin <= 0 or self.hmax <= 0:
            raise ConfigurationError(f"hmin ({self.hmin}) and hmax ({self.hmax}) must be positive.")
        if self.hmin > self.hmax:
            raise ConfigurationError(f"hmin ({self.hmin}) cannot be greater than hmax ({self.hmax}).")
        if self.dx <= 0 or self.dy <= 0:
            raise ConfigurationError(f"dx ({self.dx}) and dy ({self.dy}) must be positive.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MeshConfig":
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to parse YAML config file {path}: {e}") from e

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
