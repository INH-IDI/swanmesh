"""Mesh quality assurance and verification checks."""

from typing import Any

import numpy as np
import pandas as pd


def run_mesh_qa(nodes_df: pd.DataFrame, elem_df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform QA checks on generated mesh:
    - NaN coordinate/depth checks
    - Triangle CCW orientation verification
    - Orphaned nodes check
    - Edge length & size statistics
    - Marker histogram
    """
    # 1. NaN check
    total_nans = int(nodes_df[["X", "Y", "Z"]].isna().sum().sum())

    # 2. Referenced nodes vs total nodes
    tri = elem_df[["ELEMENT1", "ELEMENT2", "ELEMENT3"]].to_numpy(dtype=np.int64)
    used_node_ids = set(np.unique(tri).tolist())
    all_node_ids = set(nodes_df["N"].astype(np.int64).tolist())
    orphaned_nodes = list(all_node_ids - used_node_ids)

    # 3. Vectorized CCW orientation + edge lengths
    node_ids = nodes_df["N"].to_numpy(dtype=np.int64)
    xs = nodes_df["X"].to_numpy(dtype=np.float64)
    ys = nodes_df["Y"].to_numpy(dtype=np.float64)
    max_id = int(node_ids.max()) if len(node_ids) else 0
    x_of = np.zeros(max_id + 1, dtype=np.float64)
    y_of = np.zeros(max_id + 1, dtype=np.float64)
    x_of[node_ids] = xs
    y_of[node_ids] = ys

    n1, n2, n3 = tri[:, 0], tri[:, 1], tri[:, 2]
    x1, y1 = x_of[n1], y_of[n1]
    x2, y2 = x_of[n2], y_of[n2]
    x3, y3 = x_of[n3], y_of[n3]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    cw_triangles = int(np.count_nonzero(cross <= 0))

    d12 = np.hypot(x2 - x1, y2 - y1)
    d23 = np.hypot(x3 - x2, y3 - y2)
    d31 = np.hypot(x1 - x3, y1 - y3)
    edge_lengths = np.concatenate([d12, d23, d31])

    # 4. Node degree & boundary connectivity
    e1 = np.sort(tri[:, [0, 1]], axis=1)
    e2 = np.sort(tri[:, [1, 2]], axis=1)
    e3 = np.sort(tri[:, [2, 0]], axis=1)
    all_edges = np.vstack([e1, e2, e3])
    edges_unique, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    boundary_nodes = set(boundary_edges.ravel().tolist()) if len(boundary_edges) else set()

    node_neighbors: dict[int, set[int]] = {}
    for a, b in edges_unique:
        a_i, b_i = int(a), int(b)
        node_neighbors.setdefault(a_i, set()).add(b_i)
        node_neighbors.setdefault(b_i, set()).add(a_i)

    node_degrees = {node: len(neighbors) for node, neighbors in node_neighbors.items()}
    deg2_boundary_count = sum(1 for n in boundary_nodes if node_degrees.get(n, 0) < 3)

    # 5. Marker histogram
    marker_hist = nodes_df["Borde"].value_counts().to_dict() if "Borde" in nodes_df.columns else {}

    qa_passed = (
        total_nans == 0
        and len(orphaned_nodes) == 0
        and cw_triangles == 0
        and deg2_boundary_count == 0
    )

    return {
        "qa_passed": qa_passed,
        "total_nodes": len(nodes_df),
        "total_triangles": len(elem_df),
        "orphaned_nodes_count": len(orphaned_nodes),
        "cw_triangles_count": cw_triangles,
        "deg2_boundary_nodes_count": deg2_boundary_count,
        "min_node_degree": min(node_degrees.values()) if node_degrees else 0,
        "total_nans": total_nans,
        "marker_histogram": {int(k): int(v) for k, v in marker_hist.items()},
        "edge_length_stats": {
            "min": float(np.min(edge_lengths)) if len(edge_lengths) else 0.0,
            "max": float(np.max(edge_lengths)) if len(edge_lengths) else 0.0,
            "mean": float(np.mean(edge_lengths)) if len(edge_lengths) else 0.0,
            "std": float(np.std(edge_lengths)) if len(edge_lengths) else 0.0,
        },
        "z_stats": {
            "min": float(nodes_df["Z"].min()) if "Z" in nodes_df.columns else 0.0,
            "max": float(nodes_df["Z"].max()) if "Z" in nodes_df.columns else 0.0,
            "mean": float(nodes_df["Z"].mean()) if "Z" in nodes_df.columns else 0.0,
        },
    }
