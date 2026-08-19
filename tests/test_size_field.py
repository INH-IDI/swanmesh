"""Tests for mesh size field generation and strategies."""

import numpy as np
import pytest
from rasterio.transform import from_bounds

from swanmesh.bathymetry import DEMData, load_tif_to_dem
from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.size_field import apply_size_gradation, build_size_field, build_size_field_steps
from swanmesh.slope import SlopeData, build_slope


def test_size_field_strategies(synthetic_dem_tif):
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    for strat in ["dispersion_gradient", "product", "mean", "depth_weighted", "hybrid_smooth", "relative_depth"]:
        cfg = MeshConfig(
            domain_path="dummy.shp",
            bathy_tif_path=str(synthetic_dem_tif),
            hmin=0.01,
            hmax=0.1,
            n_lambda=15.0,
            alpha_grad=1.0,
            strategy=strat,
            smooth_sigma_pixels=1.0 if strat == "hybrid_smooth" else 0.0,
        )
        sf = build_size_field(cfg, dem=dem, slope=slope)
        assert sf.grid.shape == dem.grid.shape
        assert sf.grid.min() >= cfg.hmin - 1e-6
        assert sf.grid.max() <= cfg.hmax + 1e-6


def test_estimate_mesh_nodes(synthetic_dem_tif):
    from swanmesh.size_field import estimate_mesh_nodes

    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.01,
        hmax=0.1,
        strategy="dispersion_gradient",
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)
    est = estimate_mesh_nodes(sf.grid, dx=0.1, dy=0.1)

    assert est["est_nodes"] > 0
    assert est["est_nodes"] > 0
    assert est["est_elements"] == est["est_nodes"] * 2
    assert est["min_h"] >= cfg.hmin - 1e-6

def test_interest_point_radial_refinement(synthetic_dem_tif):
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    ip = InterestPointConfig(name="center_point", x=5.0, y=5.0, hmin=0.005, hmax=0.1, n_power=2.0)
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.01,
        hmax=0.1,
        strategy="dispersion_gradient",
        interest_points=[ip],
        combiner="min",
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)
    
    # Check size at center (5.0, 5.0) is smaller than corners
    h, w = sf.grid.shape
    center_val = sf.grid[h // 2, w // 2]
    corner_val = sf.grid[0, 0]
    assert center_val < corner_val


def test_relative_depth_strategy_linear_thresholds():
    """h/L<=0.05 -> hmin, h/L>=0.5 -> hmax, linear in between; slope only refines above reference."""
    width, height = 5, 5
    transform = from_bounds(0.0, 0.0, 10000.0, 10000.0, width, height)

    # Swell T=20s -> L0 = 9.81*400/(2pi) ~ 624.5 m
    # Rel=0.05 -> depth 31.2 m, Rel=0.5 -> depth 312.25 m
    l0 = 9.81 * 20.0**2 / (2.0 * np.pi)
    depths = np.array([[0.0, 20.0, 100.0, 300.0, 500.0]], dtype=np.float32)
    dem = DEMData(
        grid=np.tile(-depths, (height, 1)),
        transform=transform,
        crs="EPSG:32719",
        bounds=(0.0, 0.0, 10000.0, 10000.0),
    )

    # Flat slope -> ponderador must be 1 everywhere
    slope_flat = SlopeData(
        grid=np.zeros((height, width), dtype=np.float32),
        transform=transform,
        crs="EPSG:32719",
        bounds=(0.0, 0.0, 10000.0, 10000.0),
    )
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path="dummy.tif",
        is_utm=True,
        work_crs="EPSG:32719",
        hmin=50.0,
        hmax=2000.0,
        wave_period=20.0,
        strategy="relative_depth",
        alpha_grad=1.0,
        mesh_growth=1.0,
    )
    sf, _ = build_size_field_steps(cfg, dem=dem, slope=slope_flat)
    row = sf.grid[2, :]
    assert row[0] == pytest.approx(50.0, abs=1e-3)      # rel=0 -> hmin
    assert row[4] == pytest.approx(2000.0, abs=1e-3)    # rel=500/624>0.5 -> hmax

    rel_mid = 100.0 / l0
    expected_mid = 50.0 + (rel_mid - 0.05) / 0.45 * (2000.0 - 50.0)
    assert row[2] == pytest.approx(expected_mid, abs=1e-2)


def test_relative_depth_slope_ponderator_refines_only_above_reference():
    width, height = 4, 4
    transform = from_bounds(0.0, 0.0, 1000.0, 1000.0, width, height)
    depth = 100.0
    dem = DEMData(
        grid=np.full((height, width), -depth, dtype=np.float32),
        transform=transform,
        crs="EPSG:32719",
        bounds=(0.0, 0.0, 1000.0, 1000.0),
    )
    slope_grid = np.zeros((height, width), dtype=np.float32)
    slope_grid[:, 2:] = 0.15  # above reference
    slope_data = SlopeData(
        grid=slope_grid,
        transform=transform,
        crs="EPSG:32719",
        bounds=(0.0, 0.0, 1000.0, 1000.0),
    )
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path="dummy.tif",
        is_utm=True,
        work_crs="EPSG:32719",
        hmin=50.0,
        hmax=2000.0,
        wave_period=20.0,
        strategy="relative_depth",
        alpha_grad=1.0,
        slope_mean=0.05,
        mesh_growth=1.0,
    )
    sf, steps = build_size_field_steps(cfg, dem=dem, slope=slope_data)

    base = steps["step1_depth"].grid
    final = sf.grid
    # slope == reference: unchanged
    assert np.allclose(final[:, :2], base[:, :2])
    # slope > reference: refinement (smaller H)
    assert np.all(final[:, 2:] < base[:, 2:])


def test_gradation_no_op_when_growth_one():
    grid = np.array([[10.0, 10.0], [10.0, 1000.0]])
    out = apply_size_gradation(grid, dx=100.0, dy=100.0, gamma=1.0)
    assert np.array_equal(out, grid)


def test_gradation_preserves_minima_and_limits_adjacent_growth():
    height, width = 40, 1
    grid = np.full((height, width), 1000.0)
    grid[0, 0] = 10.0
    out = apply_size_gradation(grid, dx=100.0, dy=100.0, gamma=1.2)
    assert out[0, 0] == 10.0
    # Slow growth: after 40 cells the transition is still below hmax
    assert float(np.max(out)) < 1000.0
    assert np.all(np.diff(out[:, 0]) >= 0.0)
    for r in range(height - 1):
        a = out[r, 0]
        b = out[r + 1, 0]
        assert a <= b * 1.2 ** (100.0 / b) + 1e-9
        assert b <= a * 1.2 ** (100.0 / a) + 1e-9


def test_gradation_step4_integrated(synthetic_dem_tif):
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.01,
        hmax=0.1,
        strategy="dispersion_gradient",
        mesh_growth=1.2,
    )
    final, steps = build_size_field_steps(cfg, dem=dem, slope=slope)
    raw = steps["step3_control"].grid
    graded = final.grid
    assert np.array_equal(steps["step4_gradation"].grid, graded)
    assert np.all(graded <= raw + 1e-12)
    assert float(np.min(graded)) >= cfg.hmin - 1e-6
