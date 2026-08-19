"""Geographic / projected unit helpers for swanmesh."""

from __future__ import annotations

import math
from typing import Any


def is_geographic_crs(crs: Any, is_utm: bool = False) -> bool:
    """Return True when CRS is geographic (degrees) rather than projected meters."""
    if is_utm:
        return False
    crs_str = str(crs or "").upper()
    if "UTM" in crs_str:
        return False
    if "4326" in crs_str or "4269" in crs_str or "CRS84" in crs_str:
        return True
    if "GEOGCS" in crs_str or "GEOGRAPHIC" in crs_str:
        return True
    # Heuristic: EPSG codes in 4xxx geographic family often appear as EPSG:4xxx
    try:
        if "EPSG:" in crs_str:
            code = int(crs_str.split("EPSG:")[-1].split()[0].strip(",)"))
            if 4000 <= code < 5000:
                return True
    except (ValueError, IndexError):
        pass
    return False


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Approximate meters per degree longitude/latitude at a given latitude."""
    lat_rad = math.radians(float(lat_deg))
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2.0 * lat_rad) + 1.175 * math.cos(4.0 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3.0 * lat_rad)
    return abs(m_per_deg_lon), abs(m_per_deg_lat)


def cell_size_meters(
    dx: float,
    dy: float,
    center_y: float,
    crs: Any,
    is_utm: bool = False,
) -> tuple[float, float]:
    """Convert native cell sizes (dx, dy) to meters."""
    if is_geographic_crs(crs, is_utm=is_utm):
        m_lon, m_lat = meters_per_degree(center_y)
        return abs(dx) * m_lon, abs(dy) * m_lat
    return abs(float(dx)), abs(float(dy))


def length_to_native(
    length_m: float,
    center_y: float,
    crs: Any,
    is_utm: bool = False,
) -> float:
    """Convert a metric length to CRS-native units (degrees or meters)."""
    if is_geographic_crs(crs, is_utm=is_utm):
        m_lon, m_lat = meters_per_degree(center_y)
        # isotropic average suitable for mesh size fields
        m_per_deg = 0.5 * (m_lon + m_lat)
        if m_per_deg <= 0:
            m_per_deg = 101900.0
        return float(length_m) / m_per_deg
    return float(length_m)


def native_to_meters(
    length_native: float,
    center_y: float,
    crs: Any,
    is_utm: bool = False,
) -> float:
    """Convert CRS-native length to meters."""
    if is_geographic_crs(crs, is_utm=is_utm):
        m_lon, m_lat = meters_per_degree(center_y)
        return float(length_native) * 0.5 * (m_lon + m_lat)
    return float(length_native)
