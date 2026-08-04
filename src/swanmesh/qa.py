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
    used_node_ids = set(elem_df["ELEMENT1"]).union(set(elem_df["ELEMENT2"])).union(set(elem_df["ELEMENT3"]))
    all_node_ids = set(nodes_df["N"])
    orphaned_nodes = list(all_node_ids - used_node_ids)

    # 3. CCW Orientation check
    node_dict = dict(zip(nodes_df["N"], zip(nodes_df["X"], nodes_df["Y"])))
    cw_triangles = 0
    edge_lengths = []

    for _, row in elem_df.iterrows():
        n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
        x1, y1 = node_dict[n1]
        x2, y2 = node_dict[n2]
        x3, y3 = node_dict[n3]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross <= 0:
            cw_triangles += 1

        d12 = np.hypot(x2 - x1, y2 - y1)
        d23 = np.hypot(x3 - x2, y3 - y2)
        d31 = np.hypot(x1 - x3, y1 - y3)
        edge_lengths.extend([d12, d23, d31])

    # 4. Marker histogram
    marker_hist = nodes_df["Borde"].value_counts().to_dict()

    qa_passed = (total_nans == 0) and (len(orphaned_nodes) == 0) and (cw_triangles == 0)

    return {
        "qa_passed": qa_passed,
        "total_nodes": len(nodes_df),
        "total_triangles": len(elem_df),
        "orphaned_nodes_count": len(orphaned_nodes),
        "cw_triangles_count": cw_triangles,
        "total_nans": total_nans,
        "marker_histogram": {int(k): int(v) for k, v in marker_hist.items()},
        "edge_length_stats": {
            "min": float(np.min(edge_lengths)) if edge_lengths else 0.0,
            "max": float(np.max(edge_lengths)) if edge_lengths else 0.0,
            "mean": float(np.mean(edge_lengths)) if edge_lengths else 0.0,
            "std": float(np.std(edge_lengths)) if edge_lengths else 0.0,
        },
        "z_stats": {
            "min": float(nodes_df["Z"].min()),
            "max": float(nodes_df["Z"].max()),
            "mean": float(nodes_df["Z"].mean()),
        },
    }
