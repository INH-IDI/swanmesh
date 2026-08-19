"""SWAN Triangle-style export functions (.node, .ele, .bot)."""

from pathlib import Path

import numpy as np
import pandas as pd

from swanmesh.errors import ExportError


def export_swan_triangle(
    nodes_df: pd.DataFrame,
    elem_df: pd.DataFrame,
    output_dir: str | Path,
    base_name: str = "swan_mesh",
) -> Path:
    """
    Write .node, .ele, and .bot files for SWAN.

    nodes_df columns: N (id), X, Y, Z, Borde (marker)
    elem_df columns: ID, ELEMENT1, ELEMENT2, ELEMENT3

    Element IDs are renumbered consecutively from 1 to the number of
    elements (SWAN requires increasing element numbers).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_file = out_dir / f"{base_name}.node"
    ele_file = out_dir / f"{base_name}.ele"
    bot_file = out_dir / f"{base_name}.bot"

    try:
        nodes_sorted = nodes_df.sort_values("N").reset_index(drop=True)
        if nodes_sorted["Z"].isna().any():
            raise ExportError(
                f"Found {int(nodes_sorted['Z'].isna().sum())} NaN values in node Z "
                f"coordinates for {base_name}."
            )

        n_ids = nodes_sorted["N"].to_numpy(dtype=np.int64)
        xs = nodes_sorted["X"].to_numpy(dtype=np.float64)
        ys = nodes_sorted["Y"].to_numpy(dtype=np.float64)
        markers = nodes_sorted["Borde"].to_numpy(dtype=np.int32)
        zs = nodes_sorted["Z"].to_numpy(dtype=np.float64)

        total_nodes = len(nodes_sorted)
        with open(node_file, "w", encoding="utf-8") as f:
            f.write(f"{total_nodes} 2 0 1\n")
            np.savetxt(
                f,
                np.column_stack([n_ids, xs, ys, markers]),
                fmt=["%d", "%.7f", "%.7f", "%d"],
            )

        e1 = elem_df["ELEMENT1"].to_numpy(dtype=np.int64)
        e2 = elem_df["ELEMENT2"].to_numpy(dtype=np.int64)
        e3 = elem_df["ELEMENT3"].to_numpy(dtype=np.int64)
        total_elems = len(elem_df)
        e_ids = np.arange(1, total_elems + 1, dtype=np.int64)
        with open(ele_file, "w", encoding="utf-8") as f:
            f.write(f"{total_elems} 3 0\n")
            np.savetxt(
                f,
                np.column_stack([e_ids, e1, e2, e3]),
                fmt="%d",
            )

        with open(bot_file, "w", encoding="utf-8") as f:
            np.savetxt(f, zs, fmt="%.3f")

        return out_dir

    except Exception as e:
        raise ExportError(f"Failed to write SWAN Triangle files ({base_name}): {e}") from e
