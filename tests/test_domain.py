"""Tests for domain model loading and hole extraction."""

from pathlib import Path

from swanmesh.domain import load_domain


def test_load_domain(synthetic_domain_shp: Path):
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    assert dom.work_crs == "EPSG:4326"
    assert len(dom.exterior_coords) >= 4
    assert len(dom.holes_coords) == 1
    assert len(dom.holes_coords[0]) >= 4

    bounds = dom.get_bounds(buffer_cells=10, dx=0.1, dy=0.1)
    assert bounds == (-1.0, -1.0, 11.0, 11.0)
