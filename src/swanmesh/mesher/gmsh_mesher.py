"""Gmsh meshing engine implementation for swanmesh."""

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
    """Gmsh 2D mesh generator using background size field and quality optimization."""

    def __init__(
        self,
        algorithm_2d: int = 6,
        optimize_netgen: bool = True,
        smoothing_steps: int = 3,
        enforce_min_node_degree: bool = True,
        min_node_degree: int = 3,
        max_boundary_points: int = 5000,
    ):
        """
        algorithm_2d: 1: MeshAdapt, 2: Automatic, 5: Delaunay, 6: Frontal-Delaunay (default), 7: BAMG
        """
        if not HAS_GMSH:
            raise MeshingError("gmsh Python package is not installed.")
        self.algorithm_2d = algorithm_2d
        self.optimize_netgen = optimize_netgen
        self.smoothing_steps = smoothing_steps
        self.enforce_min_degree = enforce_min_node_degree
        self.min_node_degree = min_node_degree
        self.max_boundary_points = max_boundary_points

    def generate_mesh(self, domain: DomainModel, size_field: MeshSizeField) -> MeshResult:
        """Build Gmsh model from domain polygon and apply background size field raster."""
        try:
            if not gmsh.isInitialized():
                _safe_gmsh_initialize()
            
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("swanmesh_model")
            gmsh.option.setNumber("Mesh.Algorithm", self.algorithm_2d)
            if self.smoothing_steps > 0:
                gmsh.option.setNumber("Mesh.Smoothing", self.smoothing_steps)

            # 0. Resample domain boundary with the local size field
            domain = self._resample_boundary_to_field(domain, size_field)

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

            # Avoid extending tiny boundary sizes into the interior
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            # 5. Generate 2D Mesh and optimize (fallback algorithm if empty)
            gmsh.model.mesh.generate(2)
            etypes, etags, _ = gmsh.model.mesh.getElements(dim=2)
            if not any(t == 2 for t in etypes):
                # Fallback: coarser Delaunay without aggressive boundary densification
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
                gmsh.model.mesh.generate(2)

            if self.optimize_netgen:
                try:
                    gmsh.model.mesh.optimize("Netgen")
                    gmsh.model.mesh.optimize("Relocate2D")
                except Exception:
                    pass

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

            # Vectorized CCW orientation check using NumPy
            tag_to_idx = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int32)
            tag_to_idx[nodes_df["N"].values] = np.arange(len(nodes_df))

            idx1 = tag_to_idx[tri_nodes[:, 0]]
            idx2 = tag_to_idx[tri_nodes[:, 1]]
            idx3 = tag_to_idx[tri_nodes[:, 2]]

            x_all = nodes_df["X"].values
            y_all = nodes_df["Y"].values

            x1, y1 = x_all[idx1], y_all[idx1]
            x2, y2 = x_all[idx2], y_all[idx2]
            x3, y3 = x_all[idx3], y_all[idx3]

            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            flip_mask = cross < 0

            ccw_tri_nodes = tri_nodes.copy()
            ccw_tri_nodes[flip_mask, 1] = tri_nodes[flip_mask, 2]
            ccw_tri_nodes[flip_mask, 2] = tri_nodes[flip_mask, 1]

            elem_df = pd.DataFrame(
                {
                    "ID": tri_tags,
                    "ELEMENT1": ccw_tri_nodes[:, 0],
                    "ELEMENT2": ccw_tri_nodes[:, 1],
                    "ELEMENT3": ccw_tri_nodes[:, 2],
                }
            ).sort_values("ID").reset_index(drop=True)

            # 8. Fast Vectorized Extract Boundary Edges (frequency == 1)
            e1 = np.sort(ccw_tri_nodes[:, [0, 1]], axis=1)
            e2 = np.sort(ccw_tri_nodes[:, [1, 2]], axis=1)
            e3 = np.sort(ccw_tri_nodes[:, [2, 0]], axis=1)
            all_edges = np.vstack([e1, e2, e3])

            edges_unique, counts = np.unique(all_edges, axis=0, return_counts=True)
            bnd_arr = edges_unique[counts == 1]
            boundary_edges = [tuple(e) for e in bnd_arr]

            # 9. Enforce Minimum Node Degree Topology (min degree >= 3)
            if self.enforce_min_degree:
                from swanmesh.mesher.topology import enforce_min_node_degree
                nodes_df, elem_df, boundary_edges = enforce_min_node_degree(
                    nodes_df, elem_df, min_degree=self.min_node_degree
                )

            return MeshResult(nodes=nodes_df, triangles=elem_df, boundary_edges=boundary_edges)

        except Exception as e:
            if isinstance(e, MeshingError):
                raise
            raise MeshingError(f"Error during Gmsh mesh generation: {e}") from e

        finally:
            if gmsh.isInitialized():
                _safe_gmsh_finalize()

    def _resample_boundary_to_field(self, domain: DomainModel, size_field: MeshSizeField) -> DomainModel:
        """Resample the domain boundary with spacing driven by the local size field.

        - Spacing is sampled (bilinear) from the size field at each boundary position,
          so refined zones (coast, interest points) get denser boundary nodes.
        - Spacing is never smaller than the raster resolution (the field cannot
          represent finer detail) nor larger than a robust fallback.
        - The global point budget `max_boundary_points` includes holes; if exceeded,
          all spacings are rescaled uniformly.
        """
        h_vals = size_field.grid[np.isfinite(size_field.grid) & (size_field.grid > 0)]
        fallback_h = float(np.percentile(h_vals, 10)) if h_vals.size else 1.0
        raster_min = max(
            min(abs(size_field.transform.a), abs(size_field.transform.e)), 1e-9
        )

        def make_spacing(factor: float):
            def spacing_fn(x: float, y: float) -> float:
                h = self._sample_size_field(size_field, x, y)
                if h is None:
                    h = fallback_h
                h = max(float(h), raster_min)
                return h * factor

            return spacing_fn

        factor = 1.0
        resampled = domain
        if self.max_boundary_points <= 0:
            return domain.resample_boundary_with_spacing(
                make_spacing(1.0),
                sample_step=raster_min,
                simplify_tol=max(raster_min * 0.25, 1e-8),
            )
        for _ in range(3):
            resampled = domain.resample_boundary_with_spacing(
                make_spacing(factor),
                sample_step=raster_min,
                simplify_tol=max(raster_min * 0.25, 1e-8),
            )
            total = len(resampled.exterior_coords) + sum(
                len(h) for h in resampled.holes_coords
            )
            if total <= self.max_boundary_points:
                return resampled
            factor = max(factor * total / float(self.max_boundary_points), 1e-6)
        return resampled

    @staticmethod
    def _sample_size_field(size_field: MeshSizeField, x: float, y: float) -> float | None:
        """Bilinear sample of the size field at (x, y) in CRS units; None outside grid."""
        tr = size_field.transform
        det = tr.a * tr.e - tr.b * tr.d
        if det == 0:
            return None
        col = (tr.e * (x - tr.c) - tr.b * (y - tr.f)) / det
        row = (-tr.d * (x - tr.c) + tr.a * (y - tr.f)) / det
        c0 = int(np.floor(col))
        r0 = int(np.floor(row))
        if c0 < 0 or r0 < 0 or c0 >= size_field.width - 1 or r0 >= size_field.height - 1:
            return None
        grid = size_field.grid
        fx = col - c0
        fy = row - r0
        value = (
            grid[r0, c0] * (1.0 - fx) * (1.0 - fy)
            + grid[r0, c0 + 1] * fx * (1.0 - fy)
            + grid[r0 + 1, c0] * (1.0 - fx) * fy
            + grid[r0 + 1, c0 + 1] * fx * fy
        )
        if not np.isfinite(value) or value <= 0:
            return None
        return float(value)

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
        """Convert MeshSizeField raster into Scalar Quad (SQ) array for Gmsh PostView with fast vectorization."""
        h, w = size_field.height, size_field.width
        tr = size_field.transform

        cols = np.arange(w)
        rows = np.arange(h)
        x_coords = (tr.a * cols + tr.c).astype(np.float64)
        y_coords = (tr.e * rows + tr.f).astype(np.float64)

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        x1 = x_mesh[:-1, :-1].ravel()
        x2 = x_mesh[:-1, 1:].ravel()
        x3 = x_mesh[1:, 1:].ravel()
        x4 = x_mesh[1:, :-1].ravel()

        y1 = y_mesh[:-1, :-1].ravel()
        y2 = y_mesh[:-1, 1:].ravel()
        y3 = y_mesh[1:, 1:].ravel()
        y4 = y_mesh[1:, :-1].ravel()

        z0 = np.zeros_like(x1)

        grid = size_field.grid.astype(np.float64)
        v1 = grid[:-1, :-1].ravel()
        v2 = grid[:-1, 1:].ravel()
        v3 = grid[1:, 1:].ravel()
        v4 = grid[1:, :-1].ravel()

        sq_matrix = np.column_stack([x1, x2, x3, x4, y1, y2, y3, y4, z0, z0, z0, z0, v1, v2, v3, v4])
        return sq_matrix.ravel().tolist()
