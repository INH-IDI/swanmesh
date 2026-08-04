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

    interp = RegularGridInterpolator(
        (y_coords, x_coords),
        grid_data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([y, x])
    z_vals = interp(pts)

    # Fill NaNs with nearest neighbor if any outside exact bounds
    nan_mask = np.isnan(z_vals)
    if np.any(nan_mask):
        interp_near = RegularGridInterpolator(
            (y_coords, x_coords),
            grid_data,
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        z_vals[nan_mask] = interp_near(pts[nan_mask])

    if z_convention == "depth_positive_down":
        z_vals = -z_vals

    df["Z"] = z_vals.astype(np.float32)
    return df
