"""Gmsh meshing engine implementation for swanmesh."""

from collections import Counter

import numpy as np
import pandas as pd

from swanmesh.domain import DomainModel
from swanmesh.errors import MeshingError
from swanmesh.mesher.base import BaseMesher, MeshResult
from swanmesh.size_field import MeshSizeField

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

import signal
import threading


def _safe_gmsh_initialize() -> None:
    """Initialize Gmsh in a thread-safe manner preventing signal registration errors."""
    if threading.current_thread() is threading.main_thread():
        gmsh.initialize()
    else:
        orig_signal = signal.signal
        def dummy_signal(signalnum, handler):
            try:
                return orig_signal(signalnum, handler)
            except ValueError:
                return signal.SIG_DFL
        try:
            signal.signal = dummy_signal
            gmsh.initialize()
        finally:
            signal.signal = orig_signal

def _safe_gmsh_finalize() -> None:
    """Finalize Gmsh in a thread-safe manner preventing signal registration errors."""
    if threading.current_thread() is threading.main_thread():
        gmsh.finalize()
    else:
        orig_signal = signal.signal
        def dummy_signal(signalnum, handler):
            try:
                return orig_signal(signalnum, handler)
            except ValueError:
                return signal.SIG_DFL
        try:
            signal.signal = dummy_signal
            gmsh.finalize()
        finally:
            signal.signal = orig_signal

class GmshMesher(BaseMesher):
    """Gmsh 2D mesh generator using background size field."""

    def __init__(self, algorithm_2d: int = 6):
        """
        algorithm_2d: 1: MeshAdapt, 2: Automatic, 5: Delaunay, 6: Frontal-Delaunay (default)
        """
        if not HAS_GMSH:
            raise MeshingError("gmsh Python package is not installed.")
        self.algorithm_2d = algorithm_2d

    def generate_mesh(self, domain: DomainModel, size_field: MeshSizeField) -> MeshResult:
        """Build Gmsh model from domain polygon and apply background size field raster."""
        try:
            if not gmsh.isInitialized():
                _safe_gmsh_initialize()
            
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("swanmesh_model")
            gmsh.option.setNumber("Mesh.Algorithm", self.algorithm_2d)

            curve_loops = []

            # 1. Build Outer Loop
            ext_pts = self._clean_coords(domain.exterior_coords)
            ext_loop = self._add_polygon_loop(ext_pts)
            curve_loops.append(ext_loop)

            # 2. Build Hole Loops
            for hole_coords in domain.holes_coords:
                h_pts = self._clean_coords(hole_coords)
                if len(h_pts) >= 3:
                    hole_loop = self._add_polygon_loop(h_pts)
                    curve_loops.append(hole_loop)

            # 3. Create Surface
            gmsh.model.geo.addPlaneSurface(curve_loops)
            gmsh.model.geo.synchronize()

            # 4. Create Background Size Field View (Scalar Quads SQ)
            view_tag = gmsh.view.add("bg_size_field")
            sq_data = self._build_sq_data(size_field)
            num_quads = (size_field.width - 1) * (size_field.height - 1)
            gmsh.view.addListData(view_tag, "SQ", num_quads, sq_data)

            field_tag = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field_tag, "ViewIndex", 0)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

            # 5. Generate 2D Mesh
            gmsh.model.mesh.generate(2)

            # 6. Extract Nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            coords_2d = node_coords.reshape(-1, 3)[:, :2]
            nodes_df = pd.DataFrame(
                {
                    "N": node_tags.astype(np.int32),
                    "X": coords_2d[:, 0],
                    "Y": coords_2d[:, 1],
                }
            ).sort_values("N").reset_index(drop=True)

            # 7. Extract Triangles (element type 2 in Gmsh is 3-node triangle)
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
            tri_indices = [i for i, t in enumerate(elem_types) if t == 2]
            if not tri_indices:
                raise MeshingError("Gmsh generated no 2D triangular elements.")

            tri_tags = elem_tags[tri_indices[0]].astype(np.int32)
            tri_nodes = elem_node_tags[tri_indices[0]].reshape(-1, 3).astype(np.int32)

            # Map node IDs to coordinates for CCW orientation check
            node_dict = dict(zip(nodes_df["N"], zip(nodes_df["X"], nodes_df["Y"])))
            ccw_tri_nodes = []
            for n1, n2, n3 in tri_nodes:
                x1, y1 = node_dict[n1]
                x2, y2 = node_dict[n2]
                x3, y3 = node_dict[n3]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross < 0:
                    # Clockwise -> swap n2 and n3 to make CCW
                    ccw_tri_nodes.append((n1, n3, n2))
                else:
                    ccw_tri_nodes.append((n1, n2, n3))

            ccw_arr = np.array(ccw_tri_nodes, dtype=np.int32)
            elem_df = pd.DataFrame(
                {
                    "ID": tri_tags,
                    "ELEMENT1": ccw_arr[:, 0],
                    "ELEMENT2": ccw_arr[:, 1],
                    "ELEMENT3": ccw_arr[:, 2],
                }
            ).sort_values("ID").reset_index(drop=True)

            # 8. Extract Boundary Edges (frequency == 1)
            edges = []
            for _, row in elem_df.iterrows():
                n1, n2, n3 = int(row["ELEMENT1"]), int(row["ELEMENT2"]), int(row["ELEMENT3"])
                edges.append(tuple(sorted([n1, n2])))
                edges.append(tuple(sorted([n2, n3])))
                edges.append(tuple(sorted([n3, n1])))

            edge_counts = Counter(edges)
            boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

            return MeshResult(nodes=nodes_df, triangles=elem_df, boundary_edges=boundary_edges)

        except Exception as e:
            if isinstance(e, MeshingError):
                raise
            raise MeshingError(f"Error during Gmsh mesh generation: {e}") from e

        finally:
            if gmsh.isInitialized():
                _safe_gmsh_finalize()

    @staticmethod
    def _clean_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Remove trailing duplicate point if closed polyline."""
        if len(coords) > 1 and coords[0] == coords[-1]:
            return coords[:-1]
        return coords

    @staticmethod
    def _add_polygon_loop(pts: list[tuple[float, float]]) -> int:
        """Add points and lines for a closed polygon loop in Gmsh geo."""
        p_tags = [gmsh.model.geo.addPoint(x, y, 0.0) for x, y in pts]
        num_p = len(p_tags)
        l_tags = []
        for i in range(num_p):
            p1 = p_tags[i]
            p2 = p_tags[(i + 1) % num_p]
            l_tags.append(gmsh.model.geo.addLine(p1, p2))
        return gmsh.model.geo.addCurveLoop(l_tags)

    @staticmethod
    def _build_sq_data(size_field: MeshSizeField) -> list[float]:
        """Convert MeshSizeField raster into Scalar Quad (SQ) array for Gmsh PostView."""
        h, w = size_field.height, size_field.width
        tr = size_field.transform

        # Compute cell corner coordinates
        # Note: rasterio affine transform: (col c, row r) -> (x, y)
        cols = np.arange(w)
        rows = np.arange(h)
        x_coords = tr.a * cols + tr.c
        y_coords = tr.e * rows + tr.f

        sq_list = []
        grid = size_field.grid

        for r in range(h - 1):
            for c in range(w - 1):
                x1, y1 = x_coords[c], y_coords[r]
                x2, y2 = x_coords[c + 1], y_coords[r]
                x3, y3 = x_coords[c + 1], y_coords[r + 1]
                x4, y4 = x_coords[c], y_coords[r + 1]

                v1 = float(grid[r, c])
                v2 = float(grid[r, c + 1])
                v3 = float(grid[r + 1, c + 1])
                v4 = float(grid[r + 1, c])

                # SQ format: x1..x4, y1..y4, z1..z4, v1..v4
                sq_list.extend([x1, x2, x3, x4, y1, y2, y3, y4, 0.0, 0.0, 0.0, 0.0, v1, v2, v3, v4])

        return sq_list
