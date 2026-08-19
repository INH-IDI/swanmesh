"""Slope calculation and processing for swanmesh."""

from pathlib import Path

import numpy as np
import rasterio

from swanmesh.bathymetry import DEMData
from swanmesh.errors import DataInputError
from swanmesh.geo_units import cell_size_meters


class SlopeData:
    """Stores slope raster array, transform, CRS, mean, and std statistics."""

    def __init__(
        self,
        grid: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: str,
        bounds: tuple[float, float, float, float],
        units: str = "m/m",
    ):
        self.grid = grid.astype(np.float32)
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.height, self.width = grid.shape
        self.units = units
        self.mean = float(np.mean(grid))
        self.std = float(np.std(grid))

    def save_geotiff(self, output_path: str | Path) -> None:
        """Write slope raster to GeoTIFF file."""
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
            dst.update_tags(units=self.units)


def compute_slope_from_dem(dem: DEMData, is_utm: bool = False) -> SlopeData:
    """Compute dimensionless slope |grad h| using metric spacing when CRS is geographic."""
    dx_native = abs(dem.transform.a)
    dy_native = abs(dem.transform.e)
    center_y = 0.5 * (dem.bounds[1] + dem.bounds[3])
    dx_m, dy_m = cell_size_meters(dx_native, dy_native, center_y, dem.crs, is_utm=is_utm)

    grad_y, grad_x = np.gradient(dem.grid.astype(np.float64), dy_m, dx_m)
    slope_grid = np.sqrt(grad_x**2 + grad_y**2)

    return SlopeData(
        grid=slope_grid,
        transform=dem.transform,
        crs=dem.crs,
        bounds=dem.bounds,
        units="m/m",
    )


def load_slope_tif(
    slope_path: str | Path,
    dem: DEMData,
) -> SlopeData:
    """Load pre-computed slope GeoTIFF."""
    p = Path(slope_path)
    if not p.exists():
        raise DataInputError(f"Slope file not found: {slope_path}")
    try:
        with rasterio.open(p) as src:
            grid = src.read(1).astype(np.float32)
            units = src.tags().get("units", "m/m")
            return SlopeData(
                grid=grid,
                transform=src.transform,
                crs=src.crs.to_string() if src.crs else dem.crs,
                bounds=dem.bounds,
                units=units,
            )
    except Exception as e:
        raise DataInputError(f"Error loading slope GeoTIFF {slope_path}: {e}") from e


def build_slope(
    dem: DEMData,
    slope_tif_path: str | Path | None = None,
    is_utm: bool = False,
) -> SlopeData:
    """Build slope raster either by loading GeoTIFF or calculating from DEM."""
    if slope_tif_path:
        return load_slope_tif(slope_tif_path, dem)
    return compute_slope_from_dem(dem, is_utm=is_utm)
