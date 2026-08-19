"""Bathymetry and DEM handling for swanmesh."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.interpolate import griddata

from swanmesh.config import MeshConfig
from swanmesh.errors import DataInputError


class DEMData:
    """Stores DEM raster array, affine transform, CRS, and extent."""

    def __init__(
        self,
        grid: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: str,
        bounds: tuple[float, float, float, float],
    ):
        self.grid = grid.astype(np.float32)
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.height, self.width = grid.shape

    def save_geotiff(self, output_path: str | Path) -> None:
        """Write DEM raster to GeoTIFF file."""
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

    def sample_profile(
        self,
        center: tuple[float, float] | None = None,
        dir_vector: tuple[float, float] = (1.0, 0.0),
        num_points: int = 250,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract bathymetric cross-section profile h(s) along a directional line through domain center (xc, yc).

        Args:
            center: (xc, yc) center point. If None, uses center of self.bounds.
            dir_vector: Direction vector (i, j) along profile line. Default is (1.0, 0.0).
            num_points: Number of sampling points along the profile.

        Returns:
            (s, x_coords, y_coords, h_profile):
                - s: distance along profile starting at 0.0
                - x_coords: absolute X coordinates along profile
                - y_coords: absolute Y coordinates along profile
                - h_profile: seabed depth profile h(s) in meters
        """
        minx, miny, maxx, maxy = self.bounds
        if center is None:
            xc = (minx + maxx) / 2.0
            yc = (miny + maxy) / 2.0
        else:
            xc, yc = center

        i_val, j_val = dir_vector
        norm = np.hypot(i_val, j_val)
        if norm == 0:
            i_val, j_val = 1.0, 0.0
            norm = 1.0
        vx, vy = i_val / norm, j_val / norm

        dx_span = abs(maxx - minx)
        dy_span = abs(maxy - miny)
        half_l = max(dx_span, dy_span) * 0.5

        s_raw = np.linspace(-half_l, half_l, num_points, dtype=np.float32)
        x_coords = xc + s_raw * vx
        y_coords = yc + s_raw * vy

        inv_transform = ~self.transform
        cols, rows = inv_transform * (x_coords, y_coords)

        cols_clipped = np.clip(cols, 0, self.width - 1)
        rows_clipped = np.clip(rows, 0, self.height - 1)

        from scipy.ndimage import map_coordinates
        sampled_z = map_coordinates(
            self.grid, [rows_clipped, cols_clipped], order=1, mode="nearest"
        )

        h_profile = np.abs(sampled_z).astype(np.float32)
        # Use actual X coordinates if profile is along X, or offset s_raw to start at minx
        if abs(vx) >= abs(vy):
            s = x_coords.astype(np.float32)
        else:
            s = (s_raw - s_raw.min() + minx).astype(np.float32)
        return s, x_coords, y_coords, h_profile

def _load_single_xyz_file(p: Path) -> pd.DataFrame:
    """Read a single XYZ CSV/space-delimited file into a clean DataFrame [X, Y, Z]."""
    df = pd.read_csv(p, sep=r"[;,\s\t]+", engine="python")
    df.columns = [str(col).strip() for col in df.columns]

    col_map = {}
    for col in df.columns:
        c_up = col.upper()
        if c_up in ["X", "LON", "LONGITUDE", "EASTING"]:
            col_map[col] = "X"
        elif c_up in ["Y", "LAT", "LATITUDE", "NORTHING"]:
            col_map[col] = "Y"
        elif c_up in ["Z", "DEPTH", "COTA", "ELEV", "ELEVATION", "ALTITUDE"]:
            col_map[col] = "Z"

    if len(col_map) >= 3:
        df = df.rename(columns=col_map)[["X", "Y", "Z"]]
    else:
        df = pd.read_csv(p, sep=r"[;,\s\t]+", header=None, engine="python")
        df = df.iloc[:, :3]
        df.columns = ["X", "Y", "Z"]

    for col in ["X", "Y", "Z"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    return df.dropna(subset=["X", "Y", "Z"])

def _reduce_points_per_cell(df: pd.DataFrame, max_points_per_cell: int = 5) -> pd.DataFrame:
    """Reduce points in DataFrame to at most max_points_per_cell per grid cell (ix, iy)."""
    if max_points_per_cell <= 1:
        return (
            df.groupby(["ix", "iy"])
            .agg(X=pd.NamedAgg(column="X", aggfunc="median"),
                 Y=pd.NamedAgg(column="Y", aggfunc="median"),
                 Z=pd.NamedAgg(column="Z", aggfunc="median"))
            .reset_index()
        )

    def sample_cell(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        if n <= max_points_per_cell:
            return g[["X", "Y", "Z"]]
        quantiles = np.linspace(0, 1, max_points_per_cell)
        z_vals = g["Z"].to_numpy()
        q_target = np.quantile(z_vals, quantiles)
        selected_idx = []
        for q in q_target:
            idx = (np.abs(z_vals - q)).argmin()
            selected_idx.append(idx)
        return g.iloc[list(set(selected_idx))][["X", "Y", "Z"]]

    return df.groupby(["ix", "iy"])[["X", "Y", "Z"]].apply(sample_cell).reset_index(drop=True)

def load_xyz_to_dem(
    xyz_path: str | Path | list[str | Path],
    bounds: tuple[float, float, float, float],
    dx: float,
    dy: float,
    work_crs: str,
    input_crs: str | None = None,
    is_utm: bool = False,
    interp_method: str = "linear",
    clamp_z_min: float | None = None,
    clamp_z_max: float | None = None,
    enable_point_reduction: bool = True,
    max_points_per_cell: int = 5,
) -> DEMData:
    """Build regular DEM grid from one or multiple XYZ CSV/space-delimited files with cell point reduction."""
    paths = [xyz_path] if isinstance(xyz_path, (str, Path)) else xyz_path
    if not paths:
        raise DataInputError("No XYZ bathymetry files provided.")

    dfs = []
    for p_item in paths:
        p = Path(p_item)
        if not p.exists():
            raise DataInputError(f"XYZ bathymetry file not found: {p}")
        dfs.append(_load_single_xyz_file(p))

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if df.empty:
        raise DataInputError("Provided XYZ files contain no valid point data.")

    # Reproject coordinates if input_crs or is_utm is set
    if input_crs or is_utm:
        src_crs = input_crs if input_crs else ("EPSG:32718" if is_utm else "EPSG:4326")
        if src_crs != work_crs:
            transformer = pyproj.Transformer.from_crs(src_crs, work_crs, always_xy=True)
            x_out, y_out = transformer.transform(df["X"].to_numpy(), df["Y"].to_numpy())
            df["X"] = x_out
            df["Y"] = y_out

    minx, miny, maxx, maxy = bounds
    width = max(2, int(np.ceil((maxx - minx) / dx)))
    height = max(2, int(np.ceil((maxy - miny) / dy)))

    # Point reduction to at most max_points_per_cell per grid cell
    df["ix"] = np.floor((df["X"] - minx) / dx).astype(int)
    df["iy"] = np.floor((maxy - df["Y"]) / dy).astype(int)

    if enable_point_reduction:
        df_red = _reduce_points_per_cell(df, max_points_per_cell=max_points_per_cell)
    else:
        df_red = df

    pts = df_red[["X", "Y"]].to_numpy()
    vals = df_red["Z"].to_numpy()

    # Create grid mesh
    grid_x_1d = np.linspace(minx + dx / 2, maxx - dx / 2, width)
    grid_y_1d = np.linspace(maxy - dy / 2, miny + dy / 2, height)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    grid_z = griddata(pts, vals, (grid_x, grid_y), method=interp_method)
    
    # Fill remaining NaNs with smooth Gaussian diffused extrapolation to eliminate Voronoi pie-slice artifacts
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        from scipy.ndimage import gaussian_filter
        grid_z_near = griddata(pts, vals, (grid_x, grid_y), method="nearest")
        sigma = max(3.0, float(min(width, height)) / 40.0)
        smooth_filled = gaussian_filter(grid_z_near, sigma=sigma)
        grid_z[nan_mask] = smooth_filled[nan_mask]

    # Optional Z clamping
    if clamp_z_min is not None:
        grid_z = np.maximum(grid_z, clamp_z_min)
    if clamp_z_max is not None:
        grid_z = np.minimum(grid_z, clamp_z_max)

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return DEMData(grid=grid_z, transform=transform, crs=work_crs, bounds=bounds)

def load_tif_to_dem(
    tif_path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    dx: float | None = None,
    dy: float | None = None,
    work_crs: str = "EPSG:4326",
    clamp_z_min: float | None = None,
    clamp_z_max: float | None = None,
) -> DEMData:
    """Load DEM raster from GeoTIFF file, optionally warp to bounds and work_crs."""
    p = Path(tif_path)
    if not p.exists():
        raise DataInputError(f"GeoTIFF DEM file not found: {tif_path}")

    try:
        with rasterio.open(p) as src:
            grid = src.read(1)
            src_crs = src.crs.to_string() if src.crs else work_crs
            src_transform = src.transform
            src_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

            # Check if reproject / crop is needed
            if src_crs == work_crs and bounds is None:
                grid = grid.astype(np.float32)
                if clamp_z_min is not None:
                    grid = np.maximum(grid, clamp_z_min)
                if clamp_z_max is not None:
                    grid = np.minimum(grid, clamp_z_max)
                return DEMData(grid=grid, transform=src_transform, crs=work_crs, bounds=src_bounds)

            # Reproject to target bounds and work_crs
            minx, miny, maxx, maxy = bounds if bounds else src_bounds
            step_x = dx if dx else abs(src_transform.a)
            step_y = dy if dy else abs(src_transform.e)

            width = max(2, int(np.ceil((maxx - minx) / step_x)))
            height = max(2, int(np.ceil((maxy - miny) / step_y)))
            dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
            dst_grid = np.empty((height, width), dtype=np.float32)

            reproject(
                source=grid,
                destination=dst_grid,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=work_crs,
                resampling=Resampling.bilinear,
            )

            if clamp_z_min is not None:
                dst_grid = np.maximum(dst_grid, clamp_z_min)
            if clamp_z_max is not None:
                dst_grid = np.minimum(dst_grid, clamp_z_max)

            return DEMData(grid=dst_grid, transform=dst_transform, crs=work_crs, bounds=(minx, miny, maxx, maxy))
    except Exception as e:
        raise DataInputError(f"Error loading GeoTIFF DEM file {tif_path}: {e}") from e

def build_dem(config: MeshConfig, bounds: Tuple[float, float, float, float] | None = None) -> DEMData:
    """High level builder for DEM using configuration settings and online bathymetry sources."""
    from swanmesh.online_bathy import blend_bathymetry, download_online_bathymetry

    # Auto-determine bounds if not provided
    if bounds is None:
        if config.domain_path and Path(config.domain_path).exists():
            try:
                from swanmesh.domain import load_domain
                dom = load_domain(config.domain_path, work_crs=config.work_crs)
                if config.subdomain_bbox and len(config.subdomain_bbox) == 4:
                    dom = dom.crop_to_subdomain(config.subdomain_bbox)
                bounds = dom.get_bounds(buffer_cells=config.buffer_cells, dx=config.dx, dy=config.dy)
            except Exception:
                pass

        if bounds is None:
            xyz_list_tmp = config.bathy_xyz_paths or ([config.bathy_xyz_path] if config.bathy_xyz_path else [])
            if xyz_list_tmp and Path(xyz_list_tmp[0]).exists():
                try:
                    df0 = _load_single_xyz_file(Path(xyz_list_tmp[0]))
                    bounds = (float(df0["X"].min()), float(df0["Y"].min()), float(df0["X"].max()), float(df0["Y"].max()))
                except Exception:
                    pass

        if bounds is None:
            crs_up = (config.work_crs or "").upper()
            is_geo = ("4326" in crs_up) or (not config.is_utm and "UTM" not in crs_up)
            bounds = (-70.5, -23.7, -70.3, -23.5) if is_geo else (350000.0, 7380000.0, 370000.0, 7400000.0)

    primary_dem = None

    # Collect XYZ paths
    xyz_list = []
    if config.bathy_xyz_paths:
        xyz_list.extend(config.bathy_xyz_paths)
    elif config.bathy_xyz_path:
        xyz_list.append(config.bathy_xyz_path)

    # Collect TIF paths
    tif_list = []
    if config.bathy_tif_paths:
        tif_list.extend(config.bathy_tif_paths)
    elif config.bathy_tif_path:
        tif_list.append(config.bathy_tif_path)

    if tif_list:
        primary_dem = load_tif_to_dem(
            tif_path=tif_list[0],
            bounds=bounds,
            dx=config.dx,
            dy=config.dy,
            work_crs=config.work_crs,
            clamp_z_min=config.clamp_z_min,
            clamp_z_max=config.clamp_z_max,
        )
    elif xyz_list:
        primary_dem = load_xyz_to_dem(
            xyz_path=xyz_list,
            bounds=bounds,
            dx=config.dx,
            dy=config.dy,
            work_crs=config.work_crs,
            input_crs=config.input_crs,
            is_utm=config.is_utm,
            interp_method=config.interp_method,
            clamp_z_min=config.clamp_z_min,
            clamp_z_max=config.clamp_z_max,
            enable_point_reduction=config.enable_point_reduction,
            max_points_per_cell=config.max_points_per_cell,
        )

    if config.use_online_bathymetry or config.blend_online_bathymetry:
        online_dem = download_online_bathymetry(
            bounds=bounds,
            provider=config.online_bathy_provider,
            dx=config.dx,
            dy=config.dy,
            work_crs=config.work_crs,
        )

        if primary_dem is None:
            return online_dem

        if config.blend_online_bathymetry:
            return blend_bathymetry(
                primary_dem=primary_dem,
                background_dem=online_dem,
                blend_width_pixels=config.blend_width_pixels,
            )

    if primary_dem is not None:
        return primary_dem

    raise DataInputError("No bathymetry file (XYZ or TIF) or online bathymetry defined in configuration.")
