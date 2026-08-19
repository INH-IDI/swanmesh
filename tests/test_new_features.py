"""Tests for new features: Topology enforcer, ROI cropping, control point radius, and bathymetry interpolation."""

from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd

from swanmesh.config import InterestPointConfig, MeshConfig
from swanmesh.domain import DomainModel
from swanmesh.interpolate import interpolate_z_to_nodes
from swanmesh.bathymetry import DEMData
from swanmesh.mesher.topology import enforce_min_node_degree
from swanmesh.size_field import build_size_field, compute_interest_point_field
from swanmesh.slope import SlopeData
from rasterio.transform import from_bounds


def test_topology_enforce_min_degree():
    # Construct a mesh with a degree-2 boundary node
    # Triangular domain: nodes 1 (0,0), 2 (1,0), 3 (0,1).
    nodes = pd.DataFrame(
        {
            "N": [1, 2, 3],
            "X": [0.0, 1.0, 0.0],
            "Y": [0.0, 0.0, 1.0],
            "Z": [-10.0, -10.0, -10.0],
            "Borde": [1, 1, 1],
        }
    )
    triangles = pd.DataFrame(
        {
            "ID": [1],
            "ELEMENT1": [1],
            "ELEMENT2": [2],
            "ELEMENT3": [3],
        }
    )

    new_nodes, new_triangles, new_b_edges = enforce_min_node_degree(nodes, triangles, min_degree=3)

    # Verify no boundary nodes have degree < 3 in the resulting mesh
    edge_list = []
    for _, row in new_triangles.iterrows():
        n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
        edge_list.append(tuple(sorted([n1, n2])))
        edge_list.append(tuple(sorted([n2, n3])))
        edge_list.append(tuple(sorted([n3, n1])))

    node_neighbors = {}
    for edge in set(edge_list):
        node_neighbors.setdefault(edge[0], set()).add(edge[1])
        node_neighbors.setdefault(edge[1], set()).add(edge[0])

    for n in new_nodes["N"]:
        assert len(node_neighbors[n]) >= 3


def test_subdomain_roi_cropping():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    domain = DomainModel(gdf, work_crs="EPSG:4326")

    cropped = domain.crop_to_subdomain([2, 2, 8, 8])
    bounds = cropped.geometry.bounds
    assert bounds == (2.0, 2.0, 8.0, 8.0)


def test_interest_point_radius():
    tr = from_bounds(0, 0, 10, 10, 100, 100)
    ip = InterestPointConfig(name="cp1", x=5.0, y=5.0, hmin=0.01, hmax=0.1, radius=2.0, n_power=1.0)
    grid = compute_interest_point_field(ip, (100, 100), tr)

    # At center (5,5), size should be hmin (0.01)
    # At distance > 2.0 (e.g. at (0,0)), size should be hmax (0.1)
    assert np.isclose(grid[50, 50], 0.01, atol=1e-3)
    assert np.isclose(grid[0, 0], 0.1, atol=1e-3)


def test_robust_bathymetry_interpolation():
    grid_dem = np.full((10, 10), -50.0, dtype=np.float32)
    grid_dem[0, 0] = -9999.0  # Nodata point
    tr = from_bounds(0, 0, 10, 10, 10, 10)
    dem = DEMData(grid=grid_dem, transform=tr, crs="EPSG:4326", bounds=(0, 0, 10, 10))

    nodes = pd.DataFrame(
        {
            "N": [1, 2],
            "X": [0.5, 5.5],
            "Y": [0.5, 5.5],
        }
    )

    interp_df = interpolate_z_to_nodes(nodes, dem, z_convention="elevation_negative_down")

    # None of the nodes should be NaN or -9999.0
    assert not interp_df["Z"].isna().any()
    assert (interp_df["Z"] != -9999.0).all()
    assert (interp_df["Z"] == -50.0).all()


def test_multiple_xyz_files_and_point_reduction(tmp_path: Path):
    from swanmesh.bathymetry import load_xyz_to_dem

    f1 = tmp_path / "b1.xyz"
    f2 = tmp_path / "b2.xyz"

    df1 = pd.DataFrame({"X": [1.0, 2.0, 1.0, 2.0, 1.02, 1.01], "Y": [1.0, 1.0, 2.0, 2.0, 1.05, 1.01], "Z": [-10, -20, -11, -12, -13, -14]})
    df2 = pd.DataFrame({"X": [3.0, 4.0, 3.0, 4.0], "Y": [3.0, 3.0, 4.0, 4.0], "Z": [-30, -40, -35, -45]})

    df1.to_csv(f1, index=False, sep=" ")
    df2.to_csv(f2, index=False, sep=" ")

    dem = load_xyz_to_dem(
        xyz_path=[f1, f2],
        bounds=(0.0, 0.0, 5.0, 5.0),
        dx=1.0,
        dy=1.0,
        work_crs="EPSG:4326",
        enable_point_reduction=True,
        max_points_per_cell=5,
    )

    assert dem.grid.shape == (5, 5)
    assert not np.isnan(dem.grid).any()


def test_dem_sample_profile(tmp_path: Path):
    from swanmesh.bathymetry import load_xyz_to_dem

    f1 = tmp_path / "b1.xyz"
    df1 = pd.DataFrame({"X": [0.0, 10.0, 0.0, 10.0], "Y": [0.0, 0.0, 10.0, 10.0], "Z": [-5, -50, -5, -50]})
    df1.to_csv(f1, index=False, sep=" ")

    dem = load_xyz_to_dem(
        xyz_path=f1,
        bounds=(0.0, 0.0, 10.0, 10.0),
        dx=1.0,
        dy=1.0,
        work_crs="EPSG:4326",
    )

    s, x, y, h = dem.sample_profile(dir_vector=(1.0, 0.0), num_points=50)
    assert len(s) == 50
    assert len(h) == 50
    assert (h >= 0).all()
