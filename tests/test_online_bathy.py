"""Tests for online bathymetry downloading, caching, and blending."""

from pathlib import Path
from swanmesh.bathymetry import DEMData, load_tif_to_dem
from swanmesh.config import MeshConfig
from swanmesh.online_bathy import blend_bathymetry, download_online_bathymetry
from swanmesh.pipeline import run

def test_download_online_bathymetry(tmp_path: Path):
    bounds = (0.0, 0.0, 1.0, 1.0)
    dem = download_online_bathymetry(bounds, provider="auto", cache_dir=tmp_path / "cache", dx=0.1, dy=0.1)
    assert dem.grid.shape[0] > 0
    assert dem.grid.shape[1] > 0
    # Test caching
    cached_file = list((tmp_path / "cache").glob("*.tif"))
    assert len(cached_file) > 0

def test_blend_bathymetry(synthetic_dem_tif: Path, tmp_path: Path):
    primary_dem = load_tif_to_dem(synthetic_dem_tif)
    bg_dem = download_online_bathymetry(primary_dem.bounds, cache_dir=tmp_path / "cache", dx=0.1, dy=0.1)

    blended = blend_bathymetry(primary_dem, bg_dem, blend_width_pixels=5.0)
    assert blended.grid.shape == primary_dem.grid.shape
    assert blended.grid.min() < 0
