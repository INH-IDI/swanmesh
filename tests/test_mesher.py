"""Tests for Gmsh meshing engine and CCW triangle verification."""

from swanmesh.bathymetry import load_tif_to_dem
from swanmesh.config import MeshConfig
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
