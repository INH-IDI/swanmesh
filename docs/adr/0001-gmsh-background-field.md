# ADR 0001: Gmsh Background Mesh Size Field via PostView Scalar Quads (SQ)

## Context
SWAN requires 2D unstructured triangular meshes with spatially varying cell resolution. To control Gmsh mesh density according to bathymetry and slope gradients, a background size field must be supplied to Gmsh.

## Decision
We pass the 2D mesh size raster $H(x, y)$ directly to Gmsh using an in-memory `PostView` created via `gmsh.view.addListData(tag, "SQ", num_quads, sq_data)` and setting it as background mesh via `gmsh.model.mesh.field.add("PostView")`.

## Consequences
- Fast and fully in-memory without needing temporary files on disk.
- Provides smooth bilinear spatial interpolation across elements.
- Clean fallback and cross-platform compatibility.
