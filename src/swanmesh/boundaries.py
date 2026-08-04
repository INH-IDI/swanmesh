"""Boundary marker classification for swanmesh."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from swanmesh.errors import DataInputError


def classify_boundary_markers(
    nodes_df: pd.DataFrame,
    boundary_edges: list[tuple[int, int]],
    depth_limit: float = -50.0,
    z_convention: str = "elevation_negative_down",
    marker_strategy: str = "depth_limit",
    open_boundary_lines_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Classify mesh nodes with boundary markers:
    0: Interior node
    1: Closed boundary / land shoreline
    2: Open ocean / wave boundary
    """
    df = nodes_df.copy()

    # Identify boundary node IDs from boundary edges
    bnd_node_set: set[int] = set()
    for n1, n2 in boundary_edges:
        bnd_node_set.add(n1)
        bnd_node_set.add(n2)

    df["Borde"] = df["N"].apply(lambda node_id: 1 if node_id in bnd_node_set else 0)

    if marker_strategy == "depth_limit":
        if z_convention == "elevation_negative_down":
            # Seabed depth is Z < depth_limit (e.g. Z = -60 < -50)
            open_mask = (df["Borde"] == 1) & (df["Z"] < depth_limit)
        else:
            # depth_positive_down: depth > depth_limit (e.g. Z = 60 > 50)
            open_mask = (df["Borde"] == 1) & (df["Z"] > depth_limit)
        df.loc[open_mask, "Borde"] = 2

    elif marker_strategy == "open_boundary_lines":
        if not open_boundary_lines_path or not Path(open_boundary_lines_path).exists():
            raise DataInputError(f"Open boundary lines file missing: {open_boundary_lines_path}")
        
        lines_gdf = gpd.read_file(open_boundary_lines_path)
        # Buffer open boundary lines slightly to intersect nodes
        lines_geom = lines_gdf.geometry.union_all() if hasattr(lines_gdf.geometry, "union_all") else lines_gdf.geometry.unary_union
        lines_buffered = lines_geom.buffer(1e-4)

        bnd_nodes = df[df["Borde"] == 1]
        nodes_gdf = gpd.GeoDataFrame(
            bnd_nodes,
            geometry=gpd.points_from_xy(bnd_nodes["X"], bnd_nodes["Y"]),
        )
        open_node_ids = set(nodes_gdf[nodes_gdf.geometry.intersects(lines_buffered)]["N"])
        df.loc[df["N"].isin(open_node_ids), "Borde"] = 2

    return df
