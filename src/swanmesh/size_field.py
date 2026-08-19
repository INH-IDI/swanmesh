"""Mesh size field computation for swanmesh."""

import heapq
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from swanmesh.bathymetry import DEMData
from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.errors import SwanMeshError
from swanmesh.geo_units import is_geographic_crs, length_to_native, meters_per_degree
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
    base_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute SPH compact kernel (Wendland C2) radial mesh refinement around control point."""
    height, width = shape
    inv_tr = ~transform
    px, py = inv_tr * (ip.x, ip.y)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dist = np.sqrt(((rows - py) * res_y) ** 2 + ((cols - px) * res_x) ** 2)

    h_target_max = base_grid if base_grid is not None else (ip.hmax if ip.hmax > 0 else 0.05)

    if ip.radius > 0:
        q = np.clip(dist / ip.radius, 0.0, 1.0)
        w_kernel = ((1.0 - q) ** 4) * (1.0 + 4.0 * q)
    else:
        q = np.clip(dist / max(1e-6, np.max(dist)), 0.0, 1.0)
        w_kernel = ((1.0 - q) ** 4) * (1.0 + 4.0 * q)

    h_point = h_target_max * (1.0 - w_kernel) + ip.hmin * w_kernel
    return h_point.astype(np.float32)


def apply_size_gradation(
    grid: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
) -> np.ndarray:
    """Limit maximum geometric growth of the mesh size between adjacent cells.

    For every pair of neighboring cells (i, j) at distance d in CRS-native
    units, the returned field satisfies

        H_i <= H_j * gamma ** (d / H_j)

    so the size may grow by at most `gamma` over a distance equal to the
    local size. Only decreases H (fills transition zones outward from the
    minima); hmin floors are preserved. gamma <= 1.0 returns the input
    unchanged.
    """
    if gamma <= 1.0:
        return grid.astype(np.float64, copy=True)

    g = np.ascontiguousarray(grid, dtype=np.float64).copy()
    finite_max = float(np.nanmax(g)) if np.isfinite(g).any() else 1.0
    g[~np.isfinite(g)] = finite_max

    height, width = g.shape
    flat = g.ravel()
    heap = [(value, idx) for idx, value in enumerate(flat)]
    heapq.heapify(heap)

    d_row, d_col = abs(float(dy)), abs(float(dx))
    d_diag = math.hypot(d_col, d_row)

    while heap:
        value, idx = heapq.heappop(heap)
        if value > flat[idx]:
            continue
        r, c = divmod(idx, width)
        for dr in (-1, 0, 1):
            nr = r + dr
            if not 0 <= nr < height:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nc = c + dc
                if not 0 <= nc < width:
                    continue
                if dr != 0 and dc != 0:
                    dist = d_diag
                elif dc != 0:
                    dist = d_col
                else:
                    dist = d_row
                cand = value * (gamma ** (dist / value))
                j = nr * width + nc
                if cand < flat[j]:
                    flat[j] = cand
                    heapq.heappush(heap, (cand, j))

    return g


def build_size_field_steps(
    config: MeshConfig, dem: DEMData, slope: SlopeData
) -> tuple[MeshSizeField, dict[str, MeshSizeField]]:
    """Build background mesh-size field H from bathymetry, slope, and control points."""
    hmin, hmax = config.hmin, config.hmax
    h_span = hmax - hmin

    bathy_abs = np.abs(dem.grid)
    from swanmesh.lc_profile import compute_wave_length

    wavelength_m = compute_wave_length(bathy_abs, period=config.wave_period)

    center_y = 0.5 * (dem.bounds[1] + dem.bounds[3])
    is_geo = is_geographic_crs(dem.crs, is_utm=config.is_utm) or (
        abs(dem.transform.a) < 0.1 and not config.is_utm
    )

    # Wavelength in CRS-native units (degrees or meters)
    if is_geo:
        m_lon, m_lat = meters_per_degree(center_y)
        m_per_deg = 0.5 * (m_lon + m_lat)
        wavelength_native = wavelength_m / max(m_per_deg, 1.0)
    else:
        wavelength_native = wavelength_m

    n_lambda = max(float(config.n_lambda), 1e-6)
    max_wl = float(np.max(wavelength_native)) if np.max(wavelength_native) > 0 else 1.0
    escala1 = np.clip(wavelength_native / max_wl, 0.0, 1.0)

    # Slope is expected in m/m (dimensionless)
    slope_mean = config.slope_mean if config.slope_mean is not None else slope.mean
    slope_std = config.slope_std if config.slope_std is not None else slope.std
    if slope_std <= 1e-12:
        slope_std = 1.0

    slope_norm = (slope.grid - slope_mean) / slope_std
    escala2 = 1.0 - norm.cdf(slope_norm)

    w_total = config.weight_slope + config.weight_depth
    w_s = config.weight_slope / w_total if w_total > 0 else 0.7
    w_d = config.weight_depth / w_total if w_total > 0 else 0.3

    if config.strategy == "dispersion_gradient":
        # Physical base size: L / n_lambda in CRS-native units, clipped to [hmin, hmax]
        grid1_raw = wavelength_native / n_lambda
        alpha = config.alpha_grad if config.alpha_grad >= 0 else 1.0
        grad_h = np.maximum(slope.grid.astype(np.float64), 0.0)
        grid2_raw = grid1_raw / (1.0 + alpha * grad_h)
    elif config.strategy == "product":
        grid1_raw = hmin + (escala1 ** w_d) * h_span
        grid2_raw = hmin + (escala1 ** w_d) * (escala2 ** w_s) * h_span
    elif config.strategy == "mean":
        grid1_raw = hmin + (w_d * escala1) * h_span
        grid2_raw = hmin + (w_d * escala1 + w_s * escala2) * h_span
    elif config.strategy == "depth_weighted":
        h1 = hmin + (escala1 ** w_d) * (escala2 ** w_s) * h_span
        h3 = hmin + (w_d * escala1 + w_s * escala2) * h_span
        peso_aux = max(1.0, float(np.max(bathy_abs)))
        peso = np.clip(bathy_abs, 4.0, peso_aux)
        grid1_raw = h1
        grid2_raw = h1 * (1.0 - peso / peso_aux) + h3 * (peso / peso_aux)
    elif config.strategy == "hybrid_smooth":
        a, b = config.hybrid_exponent_a, config.hybrid_exponent_b
        e1_mod = np.power(np.clip(escala1, 1e-6, 1.0), a * w_d)
        e2_mod = np.power(np.clip(escala2, 1e-6, 1.0), b * w_s)
        grid1_raw = hmin + e1_mod * h_span
        grid2_raw = hmin + e1_mod * e2_mod * h_span
        if config.smooth_sigma_pixels > 0:
            log_h = np.log(np.clip(grid2_raw, 1e-6, None))
            log_h_smooth = gaussian_filter(log_h, sigma=config.smooth_sigma_pixels)
            grid2_raw = np.exp(log_h_smooth)
    elif config.strategy == "relative_depth":
        # Deep-water reference wavelength for the swell period
        l0_m = 9.81 * config.wave_period**2 / (2.0 * np.pi)
        rel_depth = bathy_abs / max(float(l0_m), 1.0)

        # Linear mapping of relative depth to [hmin, hmax]:
        #   h/L >= 0.5  -> hmax (deep water)
        #   h/L <= 0.05 -> hmin (shallow water)
        rel_shallow = 0.05
        rel_deep = 0.5
        frac = np.clip(
            (rel_depth - rel_shallow) / (rel_deep - rel_shallow), 0.0, 1.0
        )
        grid1_raw = hmin + frac * h_span

        # Slope ponderator: unchanged below the slope reference, refines above it
        s_ref = config.slope_mean if config.slope_mean is not None else slope.mean
        s_ref = max(float(s_ref), 1e-12)
        alpha = config.alpha_grad if config.alpha_grad >= 0 else 1.0
        slope_excess = (
            np.maximum(slope.grid.astype(np.float64) - s_ref, 0.0) / s_ref
        )
        ponderador = 1.0 / (1.0 + alpha * slope_excess)
        grid2_raw = grid1_raw * ponderador
    else:
        raise SwanMeshError(f"Unknown size field strategy: {config.strategy}")

    grid1 = np.clip(grid1_raw, hmin, hmax)
    grid2 = np.clip(grid2_raw, hmin, hmax)

    grid3 = grid2.copy()
    min_allowable_h = hmin

    for ip in config.interest_points:
        ip_hmin = float(ip.hmin)
        # Auto-scale accidental metric hmin values when working in geographic CRS
        if is_geo and ip_hmin >= 1.0:
            ip_hmin = length_to_native(ip_hmin, center_y, dem.crs, is_utm=config.is_utm)
        min_allowable_h = min(min_allowable_h, ip_hmin)
        ip_config = ip.model_copy(update={"hmin": ip_hmin}) if hasattr(ip, "model_copy") else ip
        ip_h = compute_interest_point_field(
            ip_config, (dem.height, dem.width), dem.transform, base_grid=grid3
        )
        if config.combiner == "min":
            grid3 = np.minimum(grid3, ip_h)
        elif config.combiner == "product":
            norm_ip = ip_h / max(1e-6, np.max(ip_h))
            grid3 = grid3 * norm_ip

    grid3 = np.clip(grid3, min_allowable_h, hmax)

    # Slow geometric growth (hgrad): preserve hmin, fill transitions outward
    if config.mesh_growth > 1.0:
        grid4 = apply_size_gradation(
            grid3,
            dx=abs(float(dem.transform.a)),
            dy=abs(float(dem.transform.e)),
            gamma=config.mesh_growth,
        )
    else:
        grid4 = grid3

    sf1 = MeshSizeField(
        grid=grid1,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        hmin=hmin,
        hmax=hmax,
        strategy="step1_depth",
    )
    sf2 = MeshSizeField(
        grid=grid2,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        hmin=hmin,
        hmax=hmax,
        strategy="step2_slope",
    )
    sf3 = MeshSizeField(
        grid=grid3,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        hmin=min_allowable_h,
        hmax=hmax,
        strategy="step3_control",
    )
    sf4 = MeshSizeField(
        grid=grid4,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        hmin=min_allowable_h,
        hmax=hmax,
        strategy="step4_gradation",
    )

    steps_dict = {
        "step1_depth": sf1,
        "step2_slope": sf2,
        "step3_control": sf3,
        "step4_gradation": sf4,
    }
    return sf4, steps_dict


def build_size_field(config: MeshConfig, dem: DEMData, slope: SlopeData) -> MeshSizeField:
    """Build background mesh-size field H from bathymetry, slope, and control points."""
    final_sf, _ = build_size_field_steps(config, dem, slope)
    return final_sf


def estimate_mesh_nodes(
    size_grid: np.ndarray,
    dx: float,
    dy: float,
    domain_mask: Optional[np.ndarray] = None,
) -> dict[str, int | float]:
    """
    Estimate expected number of 2D triangular mesh elements and nodes
    based on spatial integration of element size H(x,y):
      N_triangles ≈ ∫ 2 / (sqrt(3)/2 * H^2) dA  ≈ ∫ 4 / (sqrt(3) * H^2) dA
      N_nodes     ≈ N_triangles / 2
    """
    grid = np.asanyarray(size_grid, dtype=np.float64)
    valid_mask = ~np.isnan(grid) & (grid > 1e-8)
    if domain_mask is not None:
        valid_mask = valid_mask & domain_mask

    if not np.any(valid_mask):
        return {"est_nodes": 0, "est_elements": 0, "min_h": 0.0, "max_h": 0.0, "mean_h": 0.0}

    valid_h = grid[valid_mask]
    cell_area = abs(float(dx) * float(dy))

    node_density = (2.0 / (np.sqrt(3.0) * (valid_h**2))) * cell_area
    total_nodes = int(np.round(np.sum(node_density)))
    total_elements = int(np.round(total_nodes * 2))

    return {
        "est_nodes": total_nodes,
        "est_elements": total_elements,
        "min_h": float(np.min(valid_h)),
        "max_h": float(np.max(valid_h)),
        "mean_h": float(np.mean(valid_h)),
    }
