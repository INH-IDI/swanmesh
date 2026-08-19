"""Depth/elevation interpolation from DEM to mesh nodes."""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from swanmesh.bathymetry import DEMData


def interpolate_z_to_nodes(
    nodes_df: pd.DataFrame,
    dem: DEMData,
    z_convention: str = "elevation_negative_down",
) -> pd.DataFrame:
    """
    Interpolate DEM elevation Z to node locations (X, Y).

    z_convention:
      - 'elevation_negative_down': Z remains elevation (e.g. seabed is Z < 0)
      - 'depth_positive_down': Z is depth (e.g. seabed is Z > 0)
    """
    df = nodes_df.copy()
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    # Build 1D coordinate vectors for RegularGridInterpolator
    h, w = dem.height, dem.width
    tr = dem.transform

    cols = np.arange(w)
    rows = np.arange(h)
    x_coords = tr.a * cols + tr.c + tr.a / 2.0
    y_coords = tr.e * rows + tr.f + tr.e / 2.0

    # Ensure y_coords are strictly increasing for RegularGridInterpolator
    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        grid_data = dem.grid[::-1, :]
    else:
        grid_data = dem.grid

    # Mask nodata or non-finite values from grid_data
    valid_mask = np.isfinite(grid_data) & (grid_data != -9999.0)
    grid_clean = np.where(valid_mask, grid_data, np.nan)

    interp = RegularGridInterpolator(
        (y_coords, x_coords),
        grid_clean,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([y, x])
    z_vals = interp(pts)

    # Fill NaNs with cKDTree nearest neighbor among VALID dem points
    nan_mask = np.isnan(z_vals)
    if np.any(nan_mask):
        from scipy.spatial import cKDTree
        gy, gx = np.meshgrid(y_coords, x_coords, indexing="ij")
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > 0:
            dem_pts = np.column_stack([gy[valid_indices], gx[valid_indices]])
            dem_vals = grid_clean[valid_indices]
            tree = cKDTree(dem_pts)
            _, nearest_idx = tree.query(pts[nan_mask])
            z_vals[nan_mask] = dem_vals[nearest_idx]
        else:
            z_vals[nan_mask] = 0.0

    if z_convention == "depth_positive_down":
        z_vals = -z_vals

    df["Z"] = z_vals.astype(np.float32)
    return df
