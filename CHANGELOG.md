# Changelog

## [0.1.1] - 2026-08-07
### Fixed
- Slope now computed in metric units (m/m) for geographic CRS (was m/deg → mesh collapse).
- `n_lambda` now drives dispersion size field: `lc = clip(L_native / n_lambda, hmin, hmax)`.
- Domain boundary resampling simplified/validated to avoid Gmsh 1D self-intersections.
- `overwrite`, `output_crs`, `export_msh` now have real effects.
- QA/export vectorized for large meshes.
### Added
- `max_est_nodes` / `abort_on_est_nodes` safety guards.
- `max_boundary_points` cap for contour densification.
- Operational config `configs/examples/antofagasta_sane.yaml`.
- Hardening test suite (`tests/test_hardening.py`).

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
