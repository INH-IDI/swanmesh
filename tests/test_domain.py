"""Tests for domain model loading and hole extraction."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from swanmesh.domain import DomainModel, load_domain


def test_load_domain(synthetic_domain_shp: Path):
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    assert dom.work_crs == "EPSG:4326"
    assert len(dom.exterior_coords) >= 4
    assert len(dom.holes_coords) == 1
    assert len(dom.holes_coords[0]) >= 4

    bounds = dom.get_bounds(buffer_cells=10, dx=0.1, dy=0.1)
    assert bounds == (-1.0, -1.0, 11.0, 11.0)


def test_resample_boundary(synthetic_domain_shp: Path):
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    orig_pts_count = len(dom.exterior_coords)
    # Resample at fine spacing = 0.5 degrees
    resampled_dom = dom.resample_boundary(spacing=0.5)
    assert len(resampled_dom.exterior_coords) > orig_pts_count
    assert len(resampled_dom.holes_coords[0]) > len(dom.holes_coords[0])


def test_resample_boundary_with_spacing_refines_local_zones():
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        crs="EPSG:4326",
    )
    dom = DomainModel(gdf, "EPSG:4326")

    def spacing_fn(x, y):
        return 0.05 if x < 2.0 else 0.5

    out = dom.resample_boundary_with_spacing(spacing_fn, sample_step=0.01)
    coords = np.array(out.exterior_coords)
    left_edge = np.sum(coords[:, 0] < 0.01)
    right_edge = np.sum(coords[:, 0] > 9.99)
    assert left_edge > 4 * right_edge


def test_resample_boundary_with_spacing_respects_point_cap():
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        crs="EPSG:4326",
    )
    dom = DomainModel(gdf, "EPSG:4326")

    capped = dom.resample_boundary_with_spacing(
        lambda x, y: 0.001, sample_step=0.001, max_points_per_ring=500
    )
    assert 3 <= len(capped.exterior_coords) <= 500
