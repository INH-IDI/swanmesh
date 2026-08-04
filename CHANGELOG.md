# CHANGELOG

## [0.1.0] - 2026-07-30

### Added
- Modular library `swanmesh` for generating 2D unstructured SWAN meshes (.node, .ele, .bot) using Gmsh.
- Support for bathymetry XYZ (CSV/space/semicolon delimited) and GeoTIFF inputs.
- Domain polygon loading with hole/island support (Shapefile, GeoJSON, GPKG).
- Four mesh size strategies: `product` (default), `mean`, `depth_weighted`, and `hybrid_smooth`.
- Radial size field refinement for points of interest (`Distancia_y_Peso`).
- Boundary marker classification (0: interior, 1: land boundary, 2: open ocean boundary).
- Legacy Gmsh `.msh` conversion tool (`convert_gmsh_to_swan` and `swanmesh convert-msh`).
- FreeSimpleGUI desktop app `swanmesh-gui`.
- Typer-based command line interface `swanmesh`.
- Comprehensive QA suite and report generation (`_report.json`).
- Architecture Decision Records in `docs/adr/`.
