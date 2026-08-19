"""Operational hardening tests: units, n_lambda, guards, overwrite, output_crs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_bounds

from swanmesh.bathymetry import DEMData, load_tif_to_dem
from swanmesh.config import MeshConfig
from swanmesh.errors import ConfigurationError, MeshSizeGuardError
from swanmesh.geo_units import is_geographic_crs, meters_per_degree
from swanmesh.lc_profile import generate_lc_raster
from swanmesh.pipeline import run
from swanmesh.qa import run_mesh_qa
from swanmesh.size_field import build_size_field
from swanmesh.slope import compute_slope_from_dem


def test_slope_units_geographic_are_dimensionless():
    """Synthetic gentle slope in lon/lat must yield O(0.01-1) m/m, not thousands."""
    width, height = 40, 40
    transform = from_bounds(-70.5, -23.7, -70.3, -23.5, width, height)
    cols = np.linspace(0, 1, width)
    xx, _ = np.meshgrid(cols, np.linspace(0, 1, height))
    zz = (-20.0 - 200.0 * xx).astype(np.float32)
    dem = DEMData(grid=zz, transform=transform, crs="EPSG:4326", bounds=(-70.5, -23.7, -70.3, -23.5))
    slope = compute_slope_from_dem(dem, is_utm=False)
    assert slope.units == "m/m"
    assert float(np.max(slope.grid)) < 5.0
    assert float(np.mean(slope.grid)) > 1e-4


def test_n_lambda_affects_dispersion_size():
    """In metric CRS, larger n_lambda must refine the mesh size field."""
    width, height = 40, 40
    bounds = (0.0, 0.0, 20000.0, 20000.0)
    transform = from_bounds(*bounds, width, height)
    x = np.linspace(0, 1, width)
    xx, _ = np.meshgrid(x, np.linspace(0, 1, height))
    zz = (-20.0 - 200.0 * xx).astype(np.float32)
    dem = DEMData(grid=zz, transform=transform, crs="EPSG:32719", bounds=bounds)
    slope = compute_slope_from_dem(dem, is_utm=True)
    cfg_lo = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path="dummy.tif",
        is_utm=True,
        work_crs="EPSG:32719",
        hmin=10.0,
        hmax=5000.0,
        n_lambda=5.0,
        strategy="dispersion_gradient",
        alpha_grad=0.0,
        wave_period=14.0,
    )
    cfg_hi = cfg_lo.model_copy(update={"n_lambda": 20.0})
    sf_lo = build_size_field(cfg_lo, dem=dem, slope=slope)
    sf_hi = build_size_field(cfg_hi, dem=dem, slope=slope)
    assert float(sf_hi.grid.mean()) < float(sf_lo.grid.mean()) * 0.9


def test_dispersion_no_global_collapse_on_gentle_geo_bathy():
    """With n_lambda chosen for lon/lat coastal scales, field must not pin to hmin."""
    width, height = 30, 30
    bounds = (-70.5, -23.7, -70.3, -23.5)
    transform = from_bounds(*bounds, width, height)
    x = np.linspace(0, 1, width)
    xx, _ = np.meshgrid(x, np.linspace(0, 1, height))
    zz = (-50.0 - 400.0 * xx).astype(np.float32)
    dem = DEMData(grid=zz, transform=transform, crs="EPSG:4326", bounds=bounds)
    slope = compute_slope_from_dem(dem)
    cfg = MeshConfig(
        domain_path="dummy.shp",
        bathy_tif_path="dummy.tif",
        hmin=0.002,
        hmax=0.05,
        n_lambda=0.25,
        alpha_grad=1.0,
        strategy="dispersion_gradient",
        wave_period=14.0,
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)
    assert float(sf.grid.mean()) > cfg.hmin * 1.5
    assert float(sf.grid.max()) > cfg.hmin * 2.0


def test_lc_profile_uses_n_lambda():
    x = np.linspace(0, 10000, 200)
    h = np.full_like(x, 100.0)
    r5 = generate_lc_raster(x, h, period=14.0, n_lambda=5.0, min_lc=1.0, max_lc=5000.0, is_geographic=False)
    r20 = generate_lc_raster(x, h, period=14.0, n_lambda=20.0, min_lc=1.0, max_lc=5000.0, is_geographic=False)
    assert float(np.mean(r20.lc1)) < float(np.mean(r5.lc1))


def test_overwrite_guard(synthetic_domain_shp, synthetic_bathy_xyz, tmp_path):
    out = tmp_path / "ow"
    cfg = MeshConfig(
        project_name="ow_test",
        output_dir=str(out),
        domain_path=str(synthetic_domain_shp),
        bathy_xyz_path=str(synthetic_bathy_xyz),
        hmin=0.5,
        hmax=1.0,
        dx=0.5,
        dy=0.5,
        buffer_cells=2,
        strategy="product",
        overwrite=True,
        export_plots=False,
        export_sidecar_tifs=False,
        export_msh=False,
        enforce_min_node_degree=False,
        max_est_nodes=1_000_000,
    )
    run(cfg)
    cfg2 = cfg.model_copy(update={"overwrite": False})
    with pytest.raises(ConfigurationError):
        run(cfg2)


def test_max_est_nodes_guard(synthetic_domain_shp, synthetic_bathy_xyz, tmp_path):
    cfg = MeshConfig(
        project_name="guard_test",
        output_dir=str(tmp_path / "g"),
        domain_path=str(synthetic_domain_shp),
        bathy_xyz_path=str(synthetic_bathy_xyz),
        hmin=0.01,
        hmax=0.02,
        dx=0.2,
        dy=0.2,
        buffer_cells=2,
        strategy="product",
        overwrite=True,
        export_plots=False,
        export_sidecar_tifs=False,
        export_msh=False,
        max_est_nodes=10,
        abort_on_est_nodes=True,
    )
    with pytest.raises(MeshSizeGuardError):
        run(cfg)


def test_output_crs_reproject(synthetic_domain_shp, synthetic_bathy_xyz, tmp_path):
    cfg = MeshConfig(
        project_name="crs_test",
        output_dir=str(tmp_path / "crs"),
        domain_path=str(synthetic_domain_shp),
        bathy_xyz_path=str(synthetic_bathy_xyz),
        work_crs="EPSG:4326",
        output_crs="EPSG:3857",
        hmin=0.5,
        hmax=1.0,
        dx=0.5,
        dy=0.5,
        buffer_cells=2,
        strategy="product",
        overwrite=True,
        export_plots=False,
        export_sidecar_tifs=False,
        export_msh=True,
        enforce_min_node_degree=False,
        max_est_nodes=1_000_000,
    )
    res = run(cfg)
    assert Path(res.files_generated["node"]).exists()
    assert Path(res.files_generated["msh"]).exists()
    xs = []
    with open(res.files_generated["node"]) as f:
        f.readline()
        for line in f:
            xs.append(abs(float(line.split()[1])))
    assert max(xs) > 100.0


def test_qa_vectorized_on_simple_mesh():
    nodes = pd.DataFrame(
        {
            "N": [1, 2, 3, 4],
            "X": [0.0, 1.0, 0.0, 1.0],
            "Y": [0.0, 0.0, 1.0, 1.0],
            "Z": [-10.0, -12.0, -11.0, -13.0],
            "Borde": [1, 1, 1, 1],
        }
    )
    elems = pd.DataFrame(
        {
            "ID": [1, 2],
            "ELEMENT1": [1, 2],
            "ELEMENT2": [2, 4],
            "ELEMENT3": [3, 3],
        }
    )
    qa = run_mesh_qa(nodes, elems)
    assert qa["total_triangles"] == 2
    assert qa["cw_triangles_count"] == 0
    assert qa["orphaned_nodes_count"] == 0
    assert qa["total_nans"] == 0


def test_geo_units_helpers():
    assert is_geographic_crs("EPSG:4326") is True
    assert is_geographic_crs("EPSG:32719", is_utm=True) is False
    m_lon, m_lat = meters_per_degree(-23.5)
    assert 90000 < m_lon < 110000
    assert 110000 < m_lat < 112000


def test_gradation_scale_invariant_native_units():
    """Gradation uses CRS-native units: scaling grid+spacing must not change it."""
    from swanmesh.size_field import apply_size_gradation

    grid = np.array([[10.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0]])
    out_metric = apply_size_gradation(grid, dx=100.0, dy=100.0, gamma=1.2)
    out_geo = apply_size_gradation(grid * 1e-5, dx=1e-3, dy=1e-3, gamma=1.2) / 1e-5
    assert np.allclose(out_metric, out_geo)
