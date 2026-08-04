# ADR 0005: Default Output Coordinate Reference System (EPSG:4326)

## Context
SWAN models commonly use geographic coordinates (longitude, latitude in degrees) for regional coastal wave simulation domains.

## Decision
The default output CRS for `.node` coordinates is `EPSG:4326` (WGS84 lon/lat). Working in UTM or metric projections is fully supported by configuring `work_crs`.

## Consequences
- Native output compatibility with SWAN inputs.
- Clear reporting of units in `report.json`.
