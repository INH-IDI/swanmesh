"""SWAN Triangle-style export functions (.node, .ele, .bot)."""

from pathlib import Path

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
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_file = out_dir / f"{base_name}.node"
    ele_file = out_dir / f"{base_name}.ele"
    bot_file = out_dir / f"{base_name}.bot"

    try:
        # Write .node
        total_nodes = len(nodes_df)
        with open(node_file, "w", encoding="utf-8") as f:
            f.write(f"{total_nodes} 2 0 1\n")
            f.writelines(f"{int(row['N'])} {row['X']:.7f} {row['Y']:.7f} {int(row['Borde'])}\n" for _, row in nodes_df.iterrows())

        # Write .ele
        total_elems = len(elem_df)
        with open(ele_file, "w", encoding="utf-8") as f:
            f.write(f"{total_elems} 3 0\n")
            f.writelines(f"{int(row['ID'])} {int(row['ELEMENT1'])} {int(row['ELEMENT2'])} {int(row['ELEMENT3'])}\n" for _, row in elem_df.iterrows())

        # Write .bot
        with open(bot_file, "w", encoding="utf-8") as f:
            f.writelines(f"{z:.3f}\n" for z in nodes_df["Z"])

        return out_dir

    except Exception as e:
        raise ExportError(f"Failed to write SWAN Triangle files ({base_name}): {e}") from e
