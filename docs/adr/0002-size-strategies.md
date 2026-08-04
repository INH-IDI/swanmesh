# ADR 0002: Mesh Size Field Strategies

## Context
Mesh size density depends on physical water wave propagation (shallow water requires smaller elements to resolve wavelength) and bottom topography slope.

## Decision
We implement four distinct named strategies in `swanmesh`:
1. `product` (DEFAULT): $h = h_{\min} + E_1 \cdot E_2 \cdot (h_{\max} - h_{\min})$ preserving original notebook logic.
2. `mean`: $h = h_{\min} + 0.5 (E_1 + E_2) \cdot (h_{\max} - h_{\min})$.
3. `depth_weighted`: Blends product and mean based on depth weight.
4. `hybrid_smooth` (NEW): Exponent-based combination $E_1^a \cdot E_2^b$ with log-space Gaussian smoothing to prevent abrupt size jumps.

## Consequences
- Maintains full parity with author notebooks.
- Provides `hybrid_smooth` for improved numerical stability in Gmsh mesh generation.
