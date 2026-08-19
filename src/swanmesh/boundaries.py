"""Boundary marker classification for swanmesh."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from swanmesh.errors import DataInputError


def classify_boundary_markers(
    nodes_df: pd.DataFrame,
    boundary_edges: list[tuple[int, int]],
    depth_limit: float = -50.0,
    z_convention: str = "elevation_negative_down",
    marker_strategy: str = "depth_limit",
    open_boundary_lines_path: str | Path | None = None,
    open_boundary_buffer: float | None = None,
) -> pd.DataFrame:
    """
    Classify mesh nodes with boundary markers:
    0: Interior node
    1: Closed boundary / land shoreline
    2: Open ocean / wave boundary
    """
    df = nodes_df.copy()

    bnd_node_set: set[int] = set()
    for n1, n2 in boundary_edges:
        bnd_node_set.add(int(n1))
        bnd_node_set.add(int(n2))

    bnd_node_arr = np.array(list(bnd_node_set), dtype=np.int64)
    df["Borde"] = np.isin(df["N"].to_numpy(dtype=np.int64), bnd_node_arr).astype(int)

    if marker_strategy == "depth_limit":
        if z_convention == "elevation_negative_down":
            open_mask = (df["Borde"] == 1) & (df["Z"] < depth_limit)
        else:
            open_mask = (df["Borde"] == 1) & (df["Z"] > abs(depth_limit))
        df.loc[open_mask, "Borde"] = 2

    elif marker_strategy == "open_boundary_lines":
        if not open_boundary_lines_path or not Path(open_boundary_lines_path).exists():
            raise DataInputError(f"Open boundary lines file missing: {open_boundary_lines_path}")

        lines_gdf = gpd.read_file(open_boundary_lines_path)
        lines_geom = (
            lines_gdf.geometry.union_all()
            if hasattr(lines_gdf.geometry, "union_all")
            else lines_gdf.geometry.unary_union
        )

        if open_boundary_buffer is not None and open_boundary_buffer > 0:
            buf = float(open_boundary_buffer)
        else:
            # Adaptive buffer from coordinate magnitude (degrees vs meters)
            span = max(
                float(df["X"].max() - df["X"].min()) if len(df) else 1.0,
                float(df["Y"].max() - df["Y"].min()) if len(df) else 1.0,
            )
            buf = 1e-4 if span < 20.0 else 50.0

        lines_buffered = lines_geom.buffer(buf)

        bnd_nodes = df[df["Borde"] == 1]
        nodes_gdf = gpd.GeoDataFrame(
            bnd_nodes,
            geometry=gpd.points_from_xy(bnd_nodes["X"], bnd_nodes["Y"]),
        )
        open_node_ids = set(nodes_gdf[nodes_gdf.geometry.intersects(lines_buffered)]["N"])
        df.loc[df["N"].isin(open_node_ids), "Borde"] = 2

    return df
