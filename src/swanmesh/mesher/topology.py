"""Mesh topology post-processing and boundary node connectivity enforcement."""

from collections import Counter
import numpy as np
import pandas as pd


def enforce_min_node_degree(
    nodes_df: pd.DataFrame,
    elem_df: pd.DataFrame,
    min_degree: int = 3,
    max_iterations: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int]]]:
    """
    Ensure all boundary nodes have degree >= min_degree (minimum number of connected mesh edges).

    Degree-2 boundary nodes occur when a node belongs to only 1 triangle (forming a corner with only 2 boundary edges).
    This function splits the interior edge opposite to the degree-2 boundary node, introducing an interior connection
    that increases the node's degree from 2 to 3 without introducing degenerate triangles.
    """
    nodes = nodes_df.copy()
    elements = elem_df.copy()

    for iteration in range(max_iterations):
        # Build node neighbors and edge counts
        edge_list = []
        for _, row in elements.iterrows():
            n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
            edge_list.append(tuple(sorted([n1, n2])))
            edge_list.append(tuple(sorted([n2, n3])))
            edge_list.append(tuple(sorted([n3, n1])))

        edge_counts = Counter(edge_list)
        boundary_edges = set(edge for edge, count in edge_counts.items() if count == 1)

        # Map node to connected neighbor nodes
        node_neighbors: dict[int, set[int]] = {}
        for edge in edge_counts.keys():
            n1, n2 = edge
            node_neighbors.setdefault(n1, set()).add(n2)
            node_neighbors.setdefault(n2, set()).add(n1)

        boundary_nodes = set()
        for edge in boundary_edges:
            boundary_nodes.add(edge[0])
            boundary_nodes.add(edge[1])

        # Find degree-2 boundary nodes
        deg2_boundary_nodes = [
            node for node in boundary_nodes if len(node_neighbors.get(node, set())) < min_degree
        ]

        if not deg2_boundary_nodes:
            # All boundary nodes meet minimum degree condition
            break

        # Process the first degree-2 boundary node
        target_node = deg2_boundary_nodes[0]

        # Find element containing target_node
        elem_matches = elements[
            (elements["ELEMENT1"] == target_node)
            | (elements["ELEMENT2"] == target_node)
            | (elements["ELEMENT3"] == target_node)
        ]

        if elem_matches.empty:
            break

        t1_row = elem_matches.iloc[0]
        t1_id = int(t1_row["ID"])
        t1_nodes = [int(t1_row["ELEMENT1"]), int(t1_row["ELEMENT2"]), int(t1_row["ELEMENT3"])]

        # Reorder t1_nodes so target_node is first: (v, v1, v2)
        idx_v = t1_nodes.index(target_node)
        v = target_node
        v1 = t1_nodes[(idx_v + 1) % 3]
        v2 = t1_nodes[(idx_v + 2) % 3]

        opp_edge = tuple(sorted([v1, v2]))

        # Check if opp_edge is an interior edge (count == 2)
        t2_matches = elements[
            (elements["ID"] != t1_id)
            & (
                ((elements["ELEMENT1"] == v1) & (elements["ELEMENT2"] == v2))
                | ((elements["ELEMENT1"] == v2) & (elements["ELEMENT2"] == v1))
                | ((elements["ELEMENT2"] == v1) & (elements["ELEMENT3"] == v2))
                | ((elements["ELEMENT2"] == v2) & (elements["ELEMENT3"] == v1))
                | ((elements["ELEMENT3"] == v1) & (elements["ELEMENT1"] == v2))
                | ((elements["ELEMENT3"] == v2) & (elements["ELEMENT1"] == v1))
            )
        ]

        # Create new node at midpoint of (v1, v2)
        node_dict = dict(zip(nodes["N"], zip(nodes["X"], nodes["Y"])))
        x1, y1 = node_dict[v1]
        x2, y2 = node_dict[v2]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        new_n = int(nodes["N"].max()) + 1
        new_row = {"N": new_n, "X": xm, "Y": ym}

        if "Z" in nodes.columns:
            z_dict = dict(zip(nodes["N"], nodes["Z"]))
            new_row["Z"] = (z_dict[v1] + z_dict[v2]) / 2.0
        if "Borde" in nodes.columns:
            new_row["Borde"] = 0  # Interior node

        nodes = pd.concat([nodes, pd.DataFrame([new_row])], ignore_index=True)

        # Replace T1 with T1a (v, v1, m) and T1b (v, m, v2)
        # Ensure CCW orientation
        def make_ccw(n_a, n_b, n_c):
            node_map = dict(zip(nodes["N"], zip(nodes["X"], nodes["Y"])))
            xa, ya = node_map[n_a]
            xb, yb = node_map[n_b]
            xc, yc = node_map[n_c]
            cross = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
            if cross < 0:
                return [n_a, n_c, n_b]
            return [n_a, n_b, n_c]

        t1a_nodes = make_ccw(v, v1, new_n)
        t1b_nodes = make_ccw(v, new_n, v2)

        # Remove T1
        elements = elements[elements["ID"] != t1_id]

        max_elem_id = int(elements["ID"].max()) if not elements.empty else 0
        new_t1a = {"ID": max_elem_id + 1, "ELEMENT1": t1a_nodes[0], "ELEMENT2": t1a_nodes[1], "ELEMENT3": t1a_nodes[2]}
        new_t1b = {"ID": max_elem_id + 2, "ELEMENT1": t1b_nodes[0], "ELEMENT2": t1b_nodes[1], "ELEMENT3": t1b_nodes[2]}
        new_elems = [new_t1a, new_t1b]

        if not t2_matches.empty:
            t2_row = t2_matches.iloc[0]
            t2_id = int(t2_row["ID"])
            t2_nodes = [int(t2_row["ELEMENT1"]), int(t2_row["ELEMENT2"]), int(t2_row["ELEMENT3"])]
            v3 = [n for n in t2_nodes if n not in (v1, v2)][0]

            t2a_nodes = make_ccw(v3, v2, new_n)
            t2b_nodes = make_ccw(v3, new_n, v1)

            elements = elements[elements["ID"] != t2_id]
            new_t2a = {"ID": max_elem_id + 3, "ELEMENT1": t2a_nodes[0], "ELEMENT2": t2a_nodes[1], "ELEMENT3": t2a_nodes[2]}
            new_t2b = {"ID": max_elem_id + 4, "ELEMENT1": t2b_nodes[0], "ELEMENT2": t2b_nodes[1], "ELEMENT3": t2b_nodes[2]}
            new_elems.extend([new_t2a, new_t2b])

        elements = pd.concat([elements, pd.DataFrame(new_elems)], ignore_index=True)

    # Re-calculate final boundary edges
    final_edges = []
    for _, row in elements.iterrows():
        n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
        final_edges.append(tuple(sorted([n1, n2])))
        final_edges.append(tuple(sorted([n2, n3])))
        final_edges.append(tuple(sorted([n3, n1])))

    final_counts = Counter(final_edges)
    boundary_edges_res = [edge for edge, count in final_counts.items() if count == 1]

    nodes = nodes.sort_values("N").reset_index(drop=True)
    elements = elements.sort_values("ID").reset_index(drop=True)

    return nodes, elements, boundary_edges_res
