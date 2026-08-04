"""Tests for mesh size field generation and strategies."""

from swanmesh.bathymetry import load_tif_to_dem
from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.size_field import build_size_field
from swanmesh.slope import build_slope


def test_size_field_strategies(synthetic_dem_tif):
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    for strat in ["product", "mean", "depth_weighted", "hybrid_smooth"]:
        cfg = MeshConfig(
            domain_path="dummy.shp",
            bathy_tif_path=str(synthetic_dem_tif),
            hmin=0.01,
            hmax=0.1,
            strategy=strat,
            smooth_sigma_pixels=1.0 if strat == "hybrid_smooth" else 0.0,
        )
        sf = build_size_field(cfg, dem=dem, slope=slope)
        assert sf.grid.shape == dem.grid.shape
        assert sf.grid.min() >= cfg.hmin - 1e-6
        assert sf.grid.max() <= cfg.hmax + 1e-6

def test_interest_point_radial_refinement(synthetic_dem_tif):
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    ip = InterestPointConfig(name="center_point", x=5.0, y=5.0, hmin=0.005, hmax=0.1, n_power=2.0)
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.01,
        hmax=0.1,
        strategy="product",
        interest_points=[ip],
        combiner="min",
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)
    
    # Check size at center (5.0, 5.0) is smaller than corners
    h, w = sf.grid.shape
    center_val = sf.grid[h // 2, w // 2]
    corner_val = sf.grid[0, 0]
    assert center_val < corner_val
