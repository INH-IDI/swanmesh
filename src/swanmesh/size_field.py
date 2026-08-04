"""Mesh size field computation for swanmesh."""

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from swanmesh.bathymetry import DEMData
from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.errors import SwanMeshError
from swanmesh.slope import SlopeData


class MeshSizeField:
    """Stores generated mesh-size raster grid, spatial transform, and metadata."""

    def __init__(
        self,
        grid: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: str,
        bounds: tuple[float, float, float, float],
        hmin: float,
        hmax: float,
        strategy: str,
    ):
        self.grid = grid.astype(np.float32)
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.height, self.width = grid.shape
        self.hmin = hmin
        self.hmax = hmax
        self.strategy = strategy

    def save_geotiff(self, output_path: str | Path) -> None:
        """Write mesh-size field raster to GeoTIFF file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=1,
            dtype=self.grid.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(self.grid, 1)

def compute_interest_point_field(
    ip: InterestPointConfig,
    shape: tuple[int, int],
    transform: rasterio.transform.Affine,
) -> np.ndarray:
    """Compute radial mesh size field around an interest point."""
    height, width = shape
    # Transform point coordinate to pixel coordinate (col px, row py)
    inv_tr = ~transform
    px, py = inv_tr * (ip.x, ip.y)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    
    # Distance in native spatial units
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dist = np.sqrt(((rows - py) * res_y) ** 2 + ((cols - px) * res_x) ** 2)

    dist_power = dist**ip.n_power
    min_d, max_d = np.min(dist_power), np.max(dist_power)
    if max_d > min_d:
        norm_dist = (dist_power - min_d) / (max_d - min_d)
    else:
        norm_dist = np.zeros_like(dist_power)

    h_point = ip.hmin + norm_dist * (ip.hmax - ip.hmin)
    return h_point.astype(np.float32)

def build_size_field(config: MeshConfig, dem: DEMData, slope: SlopeData) -> MeshSizeField:
    """Build background mesh-size field H from bathymetry and slope."""
    hmin, hmax = config.hmin, config.hmax
    h_span = hmax - hmin

    # Wavelength scale L = T^2 * 1.5613
    l_scale = (config.wave_period**2) * config.l_constant
    bathy_abs = np.abs(dem.grid)
    escala1 = np.abs(np.tanh(2.0 * np.pi * bathy_abs / l_scale))

    # Slope normalization
    slope_mean = config.slope_mean if config.slope_mean is not None else slope.mean
    slope_std = config.slope_std if config.slope_std is not None else slope.std
    if slope_std <= 1e-12:
        slope_std = 1.0

    slope_norm = (slope.grid - slope_mean) / slope_std
    escala2 = 1.0 - norm.cdf(slope_norm)

    # Base strategy calculation
    if config.strategy == "product":
        h_base = hmin + escala1 * escala2 * h_span
    elif config.strategy == "mean":
        h_base = hmin + 0.5 * (escala1 + escala2) * h_span
    elif config.strategy == "depth_weighted":
        h1 = hmin + escala1 * escala2 * h_span
        h3 = hmin + 0.5 * (escala1 + escala2) * h_span
        peso_aux = max(1.0, float(np.max(bathy_abs)))
        peso = np.clip(bathy_abs, 4.0, peso_aux)
        h_base = h1 * (1.0 - peso / peso_aux) + h3 * (peso / peso_aux)
    elif config.strategy == "hybrid_smooth":
        a, b = config.hybrid_exponent_a, config.hybrid_exponent_b
        e1_mod = np.power(np.clip(escala1, 1e-6, 1.0), a)
        e2_mod = np.power(np.clip(escala2, 1e-6, 1.0), b)
        h_base = hmin + e1_mod * e2_mod * h_span
        if config.smooth_sigma_pixels > 0:
            # Gaussian smooth in log space for scale consistency
            log_h = np.log(np.clip(h_base, 1e-6, None))
            log_h_smooth = gaussian_filter(log_h, sigma=config.smooth_sigma_pixels)
            h_base = np.exp(log_h_smooth)
    else:
        raise SwanMeshError(f"Unknown size field strategy: {config.strategy}")

    # Combine interest points if any
    for ip in config.interest_points:
        ip_h = compute_interest_point_field(ip, (dem.height, dem.width), dem.transform)
        if config.combiner == "min":
            h_base = np.minimum(h_base, ip_h)
        elif config.combiner == "product":
            norm_ip = ip_h / max(1e-6, np.max(ip_h))
            h_base = h_base * norm_ip

    # Final clamping to [hmin, hmax]
    h_final = np.clip(h_base, hmin, hmax)

    return MeshSizeField(
        grid=h_final,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        hmin=hmin,
        hmax=hmax,
        strategy=config.strategy,
    )
