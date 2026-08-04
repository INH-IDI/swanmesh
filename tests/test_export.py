"""Tests for SWAN Triangle file export."""

from pathlib import Path

import pandas as pd

from swanmesh.export.swan_triangle import export_swan_triangle


def test_export_swan_triangle(tmp_path: Path):
    nodes = pd.DataFrame(
        {
            "N": [1, 2, 3],
            "X": [0.0, 1.0, 0.5],
            "Y": [0.0, 0.0, 1.0],
            "Z": [-10.5, -12.0, -11.0],
            "Borde": [1, 2, 0],
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

    out_dir = export_swan_triangle(nodes, triangles, output_dir=tmp_path, base_name="test_swan")

    node_file = out_dir / "test_swan.node"
    ele_file = out_dir / "test_swan.ele"
    bot_file = out_dir / "test_swan.bot"

    assert node_file.exists()
    assert ele_file.exists()
    assert bot_file.exists()

    # Verify .node header
    with open(node_file) as f:
        first_line = f.readline().strip()
        assert first_line == "3 2 0 1"

    # Verify .ele header
    with open(ele_file) as f:
        first_line = f.readline().strip()
        assert first_line == "1 3 0"

    # Verify .bot line count
    with open(bot_file) as f:
        bot_lines = f.readlines()
        assert len(bot_lines) == 3
        assert "-10.500" in bot_lines[0]
