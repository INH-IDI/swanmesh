"""Tests for boundary markers classification."""

import pandas as pd

from swanmesh.boundaries import classify_boundary_markers


def test_boundary_markers_classification():
    nodes = pd.DataFrame(
        {
            "N": [1, 2, 3, 4],
            "X": [0.0, 10.0, 10.0, 5.0],
            "Y": [0.0, 0.0, 10.0, 5.0],
            "Z": [-10.0, -60.0, -5.0, -20.0],
        }
    )
    # 1-2-3 form boundary loop. Node 4 is interior.
    boundary_edges = [(1, 2), (2, 3), (3, 1)]

    res = classify_boundary_markers(
        nodes,
        boundary_edges=boundary_edges,
        depth_limit=-50.0,
        z_convention="elevation_negative_down",
    )

    markers = dict(zip(res["N"], res["Borde"]))
    assert markers[4] == 0  # Interior node
    assert markers[1] == 1  # Boundary node with Z = -10 (shallow water)
    assert markers[3] == 1  # Boundary node with Z = -5 (shallow water)
    assert markers[2] == 2  # Open ocean boundary with Z = -60 < -50
