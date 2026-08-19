"""Visualization and PNG plot generation for swanmesh."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swanmesh.bathymetry import DEMData
from swanmesh.domain import DomainModel
from swanmesh.size_field import MeshSizeField


def plot_pipeline_previews(
    domain: DomainModel,
    dem: DEMData,
    size_field: MeshSizeField,
    nodes_df: pd.DataFrame,
    elem_df: pd.DataFrame,
    output_dir: str | Path,
    base_name: str = "swan_mesh",
) -> Path:
    """Generate PNG preview figures for pipeline outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "previews"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Size field & DEM figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    minx, miny, maxx, maxy = dem.bounds
    extent_bbox = [minx, maxx, miny, maxy]
    im0 = axes[0].imshow(dem.grid, cmap="cividis", extent=extent_bbox, origin="lower")
    axes[0].set_title("Bathymetry DEM (Z)")
    axes[0].set_xlim(minx, maxx)
    axes[0].set_ylim(miny, maxy)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(size_field.grid, cmap="viridis", extent=extent_bbox, origin="lower")
    axes[1].set_title(f"Mesh Size Field H ({size_field.strategy})")
    axes[1].set_xlim(minx, maxx)
    axes[1].set_ylim(miny, maxy)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig.savefig(fig_dir / f"{base_name}_size_field.png", dpi=150)
    plt.close(fig)

    # 2. Mesh & Markers figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mesh Triangles & Z
    x_coords = nodes_df["X"].to_numpy()
    y_coords = nodes_df["Y"].to_numpy()
    z_coords = nodes_df["Z"].to_numpy()

    # Convert triangle node IDs to 0-based index for tripcolor
    id_to_idx = {nid: i for i, nid in enumerate(nodes_df["N"])}
    tri_indices = np.array(
        [
            [id_to_idx[r["ELEMENT1"]], id_to_idx[r["ELEMENT2"]], id_to_idx[r["ELEMENT3"]]]
            for _, r in elem_df.iterrows()
        ]
    )

    tc = axes[0].tripcolor(x_coords, y_coords, tri_indices, facecolors=z_coords[tri_indices].mean(axis=1), cmap="cividis")
    axes[0].set_title("Generated Triangular Mesh (Elevation Z)")
    axes[0].set_aspect("equal")
    fig.colorbar(tc, ax=axes[0])

    # Markers Scatter
    scatter = axes[1].scatter(x_coords, y_coords, c=nodes_df["Borde"], cmap="tab10", s=6)
    axes[1].set_title("Boundary Markers (0: Interior, 1: Shore, 2: Open)")
    axes[1].set_aspect("equal")
    fig.colorbar(scatter, ax=axes[1])

    plt.tight_layout()
    fig.savefig(fig_dir / f"{base_name}_mesh.png", dpi=150)
    plt.close(fig)

    return fig_dir
