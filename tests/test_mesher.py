"""Tests for Gmsh meshing engine and CCW triangle verification."""

import numpy as np

from swanmesh.bathymetry import load_tif_to_dem
from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.domain import load_domain
from swanmesh.mesher.gmsh_mesher import GmshMesher
from swanmesh.size_field import build_size_field
from swanmesh.slope import build_slope


def test_gmsh_mesher_synthetic(synthetic_domain_shp, synthetic_dem_tif):
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)
    cfg = MeshConfig(
        domain_path=str(synthetic_domain_shp),
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.5,
        hmax=1.0,
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)

    mesher = GmshMesher()
    res = mesher.generate_mesh(domain=dom, size_field=sf)

    assert len(res.nodes) > 0
    assert len(res.triangles) > 0
    assert len(res.boundary_edges) > 0

    # Verify CCW orientation of all triangles
    node_dict = dict(zip(res.nodes["N"], zip(res.nodes["X"], res.nodes["Y"])))
    for _, row in res.triangles.iterrows():
        n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
        x1, y1 = node_dict[n1]
        x2, y2 = node_dict[n2]
        x3, y3 = node_dict[n3]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        assert cross > 0, f"Triangle {row['ID']} is not CCW!"

def test_gmsh_mesher_threaded(synthetic_domain_shp, synthetic_dem_tif):
    import threading

    def run_worker():
        dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
        dem = load_tif_to_dem(synthetic_dem_tif)
        slope = build_slope(dem)
        cfg = MeshConfig(
            domain_path=str(synthetic_domain_shp),
            bathy_tif_path=str(synthetic_dem_tif),
            hmin=0.5,
            hmax=1.0,
        )
        sf = build_size_field(cfg, dem=dem, slope=slope)
        mesher = GmshMesher()
        res = mesher.generate_mesh(domain=dom, size_field=sf)
        assert len(res.nodes) > 0

    t = threading.Thread(target=run_worker)
    t.start()
    t.join()


def test_resample_boundary_respects_budget_and_field(synthetic_domain_shp, synthetic_dem_tif):
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)
    cfg = MeshConfig(
        domain_path=str(synthetic_domain_shp),
        bathy_tif_path=str(synthetic_dem_tif),
        hmin=0.5,
        hmax=1.0,
    )
    sf = build_size_field(cfg, dem=dem, slope=slope)

    mesher = GmshMesher(max_boundary_points=20)
    resampled = mesher._resample_boundary_to_field(dom, sf)
    total = len(resampled.exterior_coords) + sum(len(h) for h in resampled.holes_coords)
    assert 3 <= total <= 20


def test_sample_size_field_bilinear():
    import numpy as np
    from rasterio.transform import from_bounds

    from swanmesh.size_field import MeshSizeField

    transform = from_bounds(0, 0, 10, 10, 11, 11)
    grid = np.ones((11, 11), dtype=np.float32)
    sf = MeshSizeField(
        grid=grid, transform=transform, crs="EPSG:4326",
        bounds=(0, 0, 10, 10), hmin=0.1, hmax=1.0, strategy="test",
    )
    assert GmshMesher._sample_size_field(sf, 4.5, 4.5) == 1.0
    assert GmshMesher._sample_size_field(sf, -1.0, 5.0) is None


def test_resample_boundary_considers_interest_point_kernel(synthetic_domain_shp, synthetic_dem_tif):
    """Boundary spacing must react to the Wendland kernel of interest points."""
    dom = load_domain(synthetic_domain_shp, work_crs="EPSG:4326")
    dem = load_tif_to_dem(synthetic_dem_tif)
    slope = build_slope(dem)

    def build(cp_hmin):
        interest_points = (
            [InterestPointConfig(name="cp_left", x=0.0, y=5.0, hmin=cp_hmin, radius=2.0)]
            if cp_hmin is not None
            else []
        )
        cfg = MeshConfig(
            domain_path=str(synthetic_domain_shp),
            bathy_tif_path=str(synthetic_dem_tif),
            hmin=0.5,
            hmax=1.0,
            strategy="product",
            interest_points=interest_points,
            combiner="min",
        )
        return build_size_field(cfg, dem=dem, slope=slope)

    mesher = GmshMesher(max_boundary_points=20000)
    with_cp = mesher._resample_boundary_to_field(dom, build(0.1))
    without_cp = mesher._resample_boundary_to_field(dom, build(None))

    def left_edge_spacings(domain):
        coords = np.array(domain.exterior_coords)
        left = np.sort(coords[coords[:, 0] < 0.01][:, 1])
        return np.diff(left)

    d_with = left_edge_spacings(with_cp)
    d_without = left_edge_spacings(without_cp)
    assert d_with.min() < d_without.min() * 0.6
