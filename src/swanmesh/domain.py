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
