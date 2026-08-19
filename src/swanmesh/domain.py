"""Domain geometry handling for swanmesh."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from swanmesh.errors import DataInputError


class DomainModel:
    """Encapsulates domain polygon, holes, CRS, and bounding box."""

    def __init__(self, gdf: gpd.GeoDataFrame, work_crs: str):
        if gdf.empty:
            raise DataInputError("Domain shapefile/geodataframe is empty.")
        
        # Ensure CRS is defined
        if gdf.crs is None:
            gdf = gdf.set_crs(work_crs)
        else:
            gdf = gdf.to_crs(work_crs)

        self.gdf = gdf
        self.work_crs = work_crs
        self.geometry = self._union_geometries(gdf)
        self.exterior_coords, self.holes_coords = self._extract_polygons()

    @staticmethod
    def _union_geometries(gdf: gpd.GeoDataFrame) -> Polygon | MultiPolygon:
        """Merge all geometries in the GeoDataFrame into a single Polygon or MultiPolygon."""
        combined = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
        if not combined.is_valid:
            combined = make_valid(combined)
        return combined

    def _extract_polygons(self) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
        """Extract exterior coordinates and hole lists from geometry."""
        geom = self.geometry
        holes = []
        if isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            for interior in geom.interiors:
                holes.append(list(interior.coords))
        elif isinstance(geom, MultiPolygon):
            # Take exterior of largest polygon as main exterior
            largest_poly = max(geom.geoms, key=lambda p: p.area)
            ext = list(largest_poly.exterior.coords)
            for poly in geom.geoms:
                for interior in poly.interiors:
                    holes.append(list(interior.coords))
        else:
            raise DataInputError(f"Unsupported domain geometry type: {type(geom)}")
        return ext, holes

    def get_bounds(self, buffer_cells: int = 0, dx: float = 0.0, dy: float = 0.0) -> tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) with optional buffer applied."""
        minx, miny, maxx, maxy = self.geometry.bounds
        if buffer_cells > 0 and dx > 0 and dy > 0:
            minx -= buffer_cells * dx
            maxx += buffer_cells * dx
            miny -= buffer_cells * dy
            maxy += buffer_cells * dy
        return minx, miny, maxx, maxy

    def resample_boundary(self, spacing: float) -> "DomainModel":
        """Resample domain exterior/holes at ~spacing, then force a valid simple polygon."""
        if spacing <= 0:
            return self

        import copy
        import numpy as np
        from shapely.geometry import LineString, Polygon
        from shapely.validation import make_valid

        def _resample_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if len(coords) < 3:
                return coords
            clean_pts = coords[:-1] if (len(coords) > 1 and coords[0] == coords[-1]) else coords
            if len(clean_pts) < 3:
                return coords

            line = LineString(clean_pts + [clean_pts[0]])
            length = line.length
            if length <= 0 or length <= spacing:
                return clean_pts

            num_pts = max(3, int(np.round(length / spacing)))
            distances = np.linspace(0, length, num_pts, endpoint=False)
            return [tuple(line.interpolate(d).coords[0]) for d in distances]

        def _dedupe(coords: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
            if not coords:
                return coords
            out = [coords[0]]
            for pt in coords[1:]:
                if abs(pt[0] - out[-1][0]) > tol or abs(pt[1] - out[-1][1]) > tol:
                    out.append(pt)
            if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
                out = out[:-1]
            return out

        # Prefer geometric simplify on the true polygon, then light densification.
        geom = self.geometry
        if not geom.is_valid:
            geom = make_valid(geom)
        # Collapse near-duplicate vertices / micro spikes that break Gmsh 1D recovery
        simp_tol = max(spacing * 0.25, 1e-8)
        geom = geom.simplify(simp_tol, preserve_topology=True)
        if not geom.is_valid:
            geom = make_valid(geom)

        # Extract largest polygon if multipolygon after clean
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        if geom.geom_type != "Polygon" or geom.is_empty:
            # fallback to coordinate resampling of original exterior
            resampled_ext = _dedupe(_resample_coords(self.exterior_coords), spacing * 0.1)
            resampled_holes = [_dedupe(_resample_coords(h), spacing * 0.1) for h in self.holes_coords]
        else:
            ext = list(geom.exterior.coords)
            holes = [list(r.coords) for r in geom.interiors]
            resampled_ext = _dedupe(_resample_coords(ext), spacing * 0.1)
            resampled_holes = [_dedupe(_resample_coords(h), spacing * 0.1) for h in holes]
            # Final validity check; if broken, use simplified exterior without densify
            try:
                poly = Polygon(resampled_ext, resampled_holes)
                if (not poly.is_valid) or poly.is_empty or poly.area <= 0:
                    resampled_ext = _dedupe(list(geom.exterior.coords)[:-1], spacing * 0.1)
                    resampled_holes = [
                        _dedupe(list(r.coords)[:-1], spacing * 0.1) for r in geom.interiors
                    ]
            except Exception:
                resampled_ext = _dedupe(list(geom.exterior.coords)[:-1], spacing * 0.1)
                resampled_holes = []

        if len(resampled_ext) < 3:
            return self

        new_domain = copy.copy(self)
        new_domain.exterior_coords = resampled_ext
        new_domain.holes_coords = [h for h in resampled_holes if len(h) >= 3]
        return new_domain

    def resample_boundary_with_spacing(
        self,
        spacing_fn,
        sample_step: float,
        simplify_tol: float = 0.0,
        max_points_per_ring: int = 8000,
    ) -> "DomainModel":
        """Resample exterior and holes with a local spacing function.

        `spacing_fn(x, y)` returns the target point spacing at that location.
        `sample_step` controls the resolution of the spacing profile; it should
        not exceed the smallest spacing returned by `spacing_fn`.
        """
        import copy

        import numpy as np
        from shapely.geometry import LineString
        from shapely.validation import make_valid

        geom = self.geometry
        if not geom.is_valid:
            geom = make_valid(geom)
        if simplify_tol > 0:
            geom = geom.simplify(simplify_tol, preserve_topology=True)
            if not geom.is_valid:
                geom = make_valid(geom)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        def _dedupe(coords: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
            if not coords:
                return coords
            out = [coords[0]]
            for pt in coords[1:]:
                if abs(pt[0] - out[-1][0]) > tol or abs(pt[1] - out[-1][1]) > tol:
                    out.append(pt)
            return out

        def _resample_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if len(coords) < 3:
                return coords
            clean = coords[:-1] if (len(coords) > 1 and coords[0] == coords[-1]) else list(coords)
            if len(clean) < 3:
                return clean

            line = LineString(clean + [clean[0]])
            length = float(line.length)
            if length <= 0:
                return clean

            n_samples = int(np.clip(length / max(float(sample_step), 1e-9), 16, max_points_per_ring))
            s_vals = np.linspace(0.0, length, n_samples, endpoint=False)
            sample_pts = np.array([line.interpolate(float(s)).coords[0] for s in s_vals])
            h_vals = np.array(
                [max(float(spacing_fn(px, py)), 1e-12) for px, py in sample_pts]
            )
            if not np.all(np.isfinite(h_vals)):
                return clean

            # Place points where the integral of 1/h(s) crosses integers
            dt = length / n_samples
            acc = np.cumsum(dt / h_vals)
            total = float(acc[-1])
            if not np.isfinite(total) or total < 3:
                return clean

            k_max = min(int(np.floor(total)), max_points_per_ring)
            if k_max < 3:
                return clean

            s_placed = np.interp(np.arange(0.0, float(k_max)), acc, s_vals)
            out = [tuple(line.interpolate(float(s)).coords[0]) for s in s_placed]
            return _dedupe(out, sample_step * 0.1)

        ext = list(geom.exterior.coords) if geom.geom_type == "Polygon" and not geom.is_empty else list(self.exterior_coords)
        holes = [list(r.coords) for r in geom.interiors] if geom.geom_type == "Polygon" else self.holes_coords

        resampled_ext = _resample_ring(ext)
        resampled_holes = [_resample_ring(h) for h in holes]

        if len(resampled_ext) < 3:
            return self

        new_domain = copy.copy(self)
        new_domain.exterior_coords = resampled_ext
        new_domain.holes_coords = [h for h in resampled_holes if len(h) >= 3]
        return new_domain

    def crop_to_subdomain(self, bbox: tuple[float, float, float, float] | list[float]) -> "DomainModel":
        """Crop domain geometry to a ROI bounding box [xmin, ymin, xmax, ymax]."""
        from shapely.geometry import box
        crop_box = box(*bbox)
        cropped_geom = self.geometry.intersection(crop_box)
        if cropped_geom.is_empty:
            raise DataInputError(f"Subdomain ROI {bbox} does not intersect domain geometry.")
        cropped_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[cropped_geom], crs=self.work_crs)
        return DomainModel(cropped_gdf, work_crs=self.work_crs)

def load_domain(domain_path: str | Path, work_crs: str) -> DomainModel:
    """Load domain shapefile/GeoJSON/GPKG and reproject to work_crs."""
    p = Path(domain_path)
    if not p.exists():
        raise DataInputError(f"Domain file not found: {domain_path}")
    try:
        gdf = gpd.read_file(p)
        return DomainModel(gdf, work_crs=work_crs)
    except Exception as e:
        if isinstance(e, DataInputError):
            raise
        raise DataInputError(f"Error loading domain file {domain_path}: {e}") from e
