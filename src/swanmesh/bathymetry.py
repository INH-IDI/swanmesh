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

def load_xyz_to_dem(
    xyz_path: str | Path,
    bounds: tuple[float, float, float, float],
    dx: float,
    dy: float,
    work_crs: str,
    input_crs: str | None = None,
    is_utm: bool = False,
    interp_method: str = "linear",
    clamp_z_min: float | None = None,
    clamp_z_max: float | None = None,
) -> DEMData:
    """Build regular DEM grid from XYZ CSV/space-delimited file."""
    p = Path(xyz_path)
    if not p.exists():
        raise DataInputError(f"XYZ bathymetry file not found: {xyz_path}")

    try:
        df = pd.read_csv(p, sep=r"[;,\s]+", engine="python")
        
        # If columns missing or headerless
        if len(df.columns) < 3 or not any(col.upper() in ["X", "Y", "Z", "LON", "LAT"] for col in df.columns):
            df = pd.read_csv(p, sep=r"[;,\s]+", header=None, engine="python")
            df = df.iloc[:, :3]
            df.columns = ["X", "Y", "Z"]
        else:
            col_map = {}
            for col in df.columns:
                c_up = str(col).upper()
                if c_up in ["X", "LON", "LONGITUDE"]:
                    col_map[col] = "X"
                elif c_up in ["Y", "LAT", "LATITUDE"]:
                    col_map[col] = "Y"
                elif c_up in ["Z", "DEPTH", "COTA"]:
                    col_map[col] = "Z"
            df = df.rename(columns=col_map)
            df = df[["X", "Y", "Z"]]

    except Exception as e:
        raise DataInputError(f"Could not read XYZ file {xyz_path}: {e}") from e

    df = df.dropna(subset=["X", "Y", "Z"])
    if df.empty:
        raise DataInputError(f"XYZ file {xyz_path} contains no valid point data.")

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

    # Point reduction to cell centroids (reducir_puntos)
    df["ix"] = np.floor((df["X"] - minx) / dx).astype(int)
    df["iy"] = np.floor((maxy - df["Y"]) / dy).astype(int)

    df_red = (
        df.groupby(["ix", "iy"])
        .agg(X=pd.NamedAgg(column="X", aggfunc="median"),
             Y=pd.NamedAgg(column="Y", aggfunc="median"),
             Z=pd.NamedAgg(column="Z", aggfunc="median"))
        .reset_index()
    )

    pts = df_red[["X", "Y"]].to_numpy()
    vals = df_red["Z"].to_numpy()

    # Create grid mesh
    grid_x_1d = np.linspace(minx + dx / 2, maxx - dx / 2, width)
    grid_y_1d = np.linspace(maxy - dy / 2, miny + dy / 2, height)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    grid_z = griddata(pts, vals, (grid_x, grid_y), method=interp_method)
    
    # Fill remaining NaNs with nearest neighbor interpolation
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_near = griddata(pts, vals, (grid_x, grid_y), method="nearest")
        grid_z[nan_mask] = grid_z_near[nan_mask]

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

def build_dem(config: MeshConfig, bounds: Tuple[float, float, float, float]) -> DEMData:
    """High level builder for DEM using configuration settings and online bathymetry sources."""
    from swanmesh.online_bathy import blend_bathymetry, download_online_bathymetry

    primary_dem = None

    if config.bathy_tif_path:
        primary_dem = load_tif_to_dem(
            tif_path=config.bathy_tif_path,
            bounds=bounds,
            dx=config.dx,
            dy=config.dy,
            work_crs=config.work_crs,
            clamp_z_min=config.clamp_z_min,
            clamp_z_max=config.clamp_z_max,
        )
    elif config.bathy_xyz_path:
        primary_dem = load_xyz_to_dem(
            xyz_path=config.bathy_xyz_path,
            bounds=bounds,
            dx=config.dx,
            dy=config.dy,
            work_crs=config.work_crs,
            input_crs=config.input_crs,
            is_utm=config.is_utm,
            interp_method=config.interp_method,
            clamp_z_min=config.clamp_z_min,
            clamp_z_max=config.clamp_z_max,
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
