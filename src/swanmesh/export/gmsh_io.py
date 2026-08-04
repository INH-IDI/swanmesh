"""Legacy Gmsh .msh parser and converter to SWAN format."""

from collections import Counter
from pathlib import Path

import pandas as pd

from swanmesh.bathymetry import load_tif_to_dem
from swanmesh.boundaries import classify_boundary_markers
from swanmesh.errors import ConversionError
from swanmesh.export.swan_triangle import export_swan_triangle
from swanmesh.interpolate import interpolate_z_to_nodes


def parse_gmsh2_ascii(msh_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust parser for Gmsh MSH 2.0 ASCII files."""
    p = Path(msh_path)
    if not p.exists():
        raise ConversionError(f"Gmsh .msh file not found: {msh_path}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find $Nodes
        try:
            nodes_idx = lines.index("$Nodes\n")
        except ValueError:
            nodes_idx = lines.index("$Nodes\r\n")

        num_nodes = int(lines[nodes_idx + 1].strip())
        node_lines = lines[nodes_idx + 2 : nodes_idx + 2 + num_nodes]

        nodes = []
        for line in node_lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                nodes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        nodes_df = pd.DataFrame(nodes, columns=["N", "X", "Y", "Z_raw"])

        # Find $Elements
        try:
            elem_idx = lines.index("$Elements\n")
        except ValueError:
            elem_idx = lines.index("$Elements\r\n")

        num_elements = int(lines[elem_idx + 1].strip())
        elem_lines = lines[elem_idx + 2 : elem_idx + 2 + num_elements]

        triangles = []
        for line in elem_lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                e_id = int(parts[0])
                e_type = int(parts[1])
                num_tags = int(parts[2])
                # Triangle element type in MSH 2 is 2 (3 nodes)
                if e_type == 2:
                    nodes_start = 3 + num_tags
                    n1 = int(parts[nodes_start])
                    n2 = int(parts[nodes_start + 1])
                    n3 = int(parts[nodes_start + 2])
                    triangles.append([e_id, n1, n2, n3])

        if not triangles:
            raise ConversionError(f"No 2D triangle elements found in {msh_path}")

        elem_df = pd.DataFrame(triangles, columns=["ID", "ELEMENT1", "ELEMENT2", "ELEMENT3"])

        # Check and enforce CCW orientation
        node_dict = dict(zip(nodes_df["N"], zip(nodes_df["X"], nodes_df["Y"])))
        ccw_triangles = []
        for _, row in elem_df.iterrows():
            e_id = int(row["ID"])
            n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
            x1, y1 = node_dict[n1]
            x2, y2 = node_dict[n2]
            x3, y3 = node_dict[n3]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross < 0:
                ccw_triangles.append([e_id, n1, n3, n2])
            else:
                ccw_triangles.append([e_id, n1, n2, n3])

        elem_df = pd.DataFrame(ccw_triangles, columns=["ID", "ELEMENT1", "ELEMENT2", "ELEMENT3"])
        return nodes_df, elem_df

    except Exception as e:
        if isinstance(e, ConversionError):
            raise
        raise ConversionError(f"Failed to parse Gmsh file {msh_path}: {e}") from e

def convert_gmsh_to_swan(
    msh_path: str | Path,
    dem_path: str | Path,
    output_dir: str | Path,
    base_name: str = "converted_mesh",
    depth_limit: float = -50.0,
    z_convention: str = "elevation_negative_down",
) -> Path:
    """
    Convert legacy Gmsh .msh file + DEM GeoTIFF into SWAN .node, .ele, .bot files.
    """
    nodes_df, elem_df = parse_gmsh2_ascii(msh_path)

    # Compute boundary edges (frequency 1)
    edges = []
    for _, row in elem_df.iterrows():
        n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
        edges.append(tuple(sorted([n1, n2])))
        edges.append(tuple(sorted([n2, n3])))
        edges.append(tuple(sorted([n3, n1])))

    edge_counts = Counter(edges)
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

    # Load DEM and interpolate Z
    dem = load_tif_to_dem(dem_path)
    nodes_df = interpolate_z_to_nodes(nodes_df, dem, z_convention=z_convention)

    # Classify markers
    nodes_df = classify_boundary_markers(
        nodes_df,
        boundary_edges=boundary_edges,
        depth_limit=depth_limit,
        z_convention=z_convention,
    )

    # Export SWAN
    return export_swan_triangle(
        nodes_df=nodes_df,
        elem_df=elem_df,
        output_dir=output_dir,
        base_name=base_name,
    )
