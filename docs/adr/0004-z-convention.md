# ADR 0004: Z Elevation Convention (`elevation_negative_down`)

## Context
Bathymetric DEMs represent sea bottom depth either as positive depth below sea level or negative elevation relative to datum.

## Decision
By default, `swanmesh` adopts `z_convention = "elevation_negative_down"`, where seabed points have $Z < 0$ (e.g. $-20\text{ m}$). An explicit option `z_convention = "depth_positive_down"` is supported for positive depth export.

## Consequences
- Prevents silent sign inversions in `.bot` files.
- Clear marker thresholding (`depth_limit = -50.0`).
