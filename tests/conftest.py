"""Pytest fixtures for swanmesh test suite."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture
def synthetic_domain_shp(temp_dir: Path) -> Path:
    """Create a 0 to 10 square domain shapefile with a hole 4 to 6."""
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    poly = Polygon(outer.exterior, [hole.exterior])

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    shp_path = temp_dir / "synthetic_domain.shp"
    gdf.to_file(shp_path)
    return shp_path

@pytest.fixture
def synthetic_bathy_xyz(temp_dir: Path) -> Path:
    """Create synthetic XYZ bathymetry file (x 0..10, y 0..10, depth z = -x*2 - y)."""
    x = np.linspace(-1, 11, 25)
    y = np.linspace(-1, 11, 25)
    xx, yy = np.meshgrid(x, y)
    zz = -2.0 * xx - 1.0 * yy - 5.0  # Seabed depth < 0

    df = pd.DataFrame({"X": xx.flatten(), "Y": yy.flatten(), "Z": zz.flatten()})
    xyz_path = temp_dir / "synthetic_bathy.xyz"
    df.to_csv(xyz_path, index=False, sep=" ")
    return xyz_path

@pytest.fixture
def synthetic_dem_tif(temp_dir: Path) -> Path:
    """Create synthetic DEM GeoTIFF."""
    width, height = 50, 50
    transform = from_bounds(0, 0, 10, 10, width, height)
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    xx, yy = np.meshgrid(x, y)
    zz = (-2.0 * xx - 1.0 * yy - 5.0).astype(np.float32)

    tif_path = temp_dir / "synthetic_dem.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(zz, 1)

    return tif_path
