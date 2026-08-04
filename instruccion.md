Build a production-quality Python software suite named **swanmesh** that AUTOMATICALLY generates unstructured 2D triangular meshes for SWAN (Triangle-style .node / .ele / .bot), by porting and hardening three reference Jupyter notebooks into: (1) an installable library, (2) a thin FreeSimpleGUI desktop app, and (3) a small CLI. Mesh generation MUST use the Gmsh Python API with a background mesh-size field derived from the notebooks’ formulas.

================================================================================
SOURCE NOTEBOOKS (authoritative logic — port, do not ship notebooks as the product)
================================================================================
1) interpolarGrid.ipynb
   - XYZ bathymetry → optional UTM→lon/lat (pyproj; flag is_utm; default UTM 18S / WGS84 south)
   - Domain shapefile bounds define raster extent (+ buffer)
   - scipy.interpolate.griddata(method="linear") → regular grid
   - Write GeoTIFF (from_origin; row flip as notebook); CRS documented (often EPSG:4326)

2) gridSize.ipynb
   - Mesh-size raster from bathymetry + slope
   - minGSize / maxGSize in **native CRS units of the size raster** (degrees if working in lon/lat; meters if UTM)
   - Core scales (MUST preserve as named strategies):
       L = T**2 * 1.5613   # T default 30 s; constant configurable
       Escala1 = abs(tanh(2*pi*abs(Bathy)/L))     # shallow → small size
       Escala2 = 1 - Φ((Slope-μ)/σ)               # steep → small size (scipy.stats.norm.sf or 1-cdf)
   - Combiners from notebook:
       gridsize1 PRODUCT (DEFAULT):  h = hmin + Escala1*Escala2*(hmax-hmin)
       gridsize3 MEAN:               h = hmin + 0.5*(Escala1+Escala2)*(hmax-hmin)
       gridsize4 DEPTH_WEIGHTED:     blend as in notebook (weight by depth/Escala1)
   - Interest points: Distancia_y_Peso(xc,yc,n_power,...) → radial field; combine via elementwise minimum (or configurable)
   - Optional clips on bathy/slope before scales

3) Gmsh2SWAN.ipynb
   - Read Gmsh ASCII 2.0 .msh → nodes / triangles
   - Boundary edges = edges with frequency 1
   - Sample Z to nodes from DEM
   - Markers: 0 interior; 1 boundary default; 2 open/wave if boundary AND Z < depth_limit (default -50; configurable)
   - Export:
       .node  header "N 2 0 1" ; "id x y marker"
       .ele   header "M 3 0"  ; "id n1 n3 n2"  # CCW convention from notebook — keep + test
       .bot   one value per node, same id order as .node

NEW product MUST generate the mesh (not only convert). Legacy convert-only mode remains a first-class feature.

================================================================================
PRODUCT DELIVERABLES
================================================================================
A) Library package: swanmesh  (importable; NO GUI dependency)
B) GUI package/entry: swanmesh-gui  (FreeSimpleGUI; thin orchestration only)
C) CLI entrypoint: swanmesh  (Typer or argparse) — same API as library
   Examples:
     swanmesh run path/to/config.yaml
     swanmesh convert-msh mesh.msh --dem dem.tif --out outdir
     swanmesh build-dem ...
     swanmesh build-size-field ...
     swanmesh validate-config ...

Hard rules:
- No hardcoded absolute paths
- No notebook-style globals (fix AA/grilla/CC bugs)
- Config-driven YAML/JSON + Python API + CLI overrides
- Code/identifiers/docstrings in English; **README.md in Spanish** (optional short README_EN.md later)
- Explicit CRS + units for mesh size (degrees vs meters) everywhere in docs and report

================================================================================
CRS & COORDINATE POLICY (flexible; SWAN often lon/lat)
================================================================================
- Support working CRS and output CRS independently when feasible.
- **Default path (most user SWAN cases):**
  - domain/bathy may be 4326 or UTM
  - mesh generation may run in lon/lat OR in a metric CRS
  - **SWAN outputs (.node x y) default to EPSG:4326 lon/lat**
- **UTM option:** allow work_crs = UTM (user EPSG or auto from domain centroid); size field in meters; optional output_crs = UTM or 4326
- Config fields: input assumptions, work_crs, output_crs, utm_zone / epsg, is_utm for XYZ
- Never treat degrees as meters silently. report.json must record CRS and h units.
- Reproject with pyproj Transformer on arrays (vectorized), not row-wise apply.

================================================================================
Z / .bot CONVENTION
================================================================================
- Internal DEM stores bottom elevation **cota Z** with **negative downward** (notebook default).
- **Default .bot export = cota Z negativa** (same sign as internal).
- Alternative export: depth_positive = -Z (or max(0,-Z) only if explicitly configured — do not surprise the user).
- Config: z_convention: "elevation_negative_down" | "depth_positive_down"
- depth_limit for marker 2 is interpreted in the **same convention as node Z used for the test**; document clearly.
- No silent Z clamps; any clamp_z_min/max must be explicit in config.

================================================================================
INPUTS
================================================================================
Required:
1) Bathymetry: XYZ and/or GeoTIFF
2) Domain polygon: shapefile / GPKG / GeoJSON (exterior + holes/islands)

Optional:
- Slope GeoTIFF (else compute from DEM; document algorithm, e.g. numpy gradient / Horn, in work CRS units)
- Interest points/zones: CSV or GeoJSON (x,y or lon,lat, n_power, hmin, hmax, name)
- Coastline polyline (extension; MVP may rely on shallow-water + slope + domain boundary)
- Open-boundary polylines (extension beyond depth_limit strategy)
- Existing .msh for legacy convert mode

================================================================================
OUTPUTS
================================================================================
- {name}.node, {name}.ele, {name}.bot
- {name}_config.yaml (full resolved config + package version)
- {name}_report.json (counts, h stats, marker histogram, CRS, bounds, z_convention, QA flags)
- Optional: DEM/slope/size-field GeoTIFFs, .msh, PNG previews (domain, H, mesh, Z, markers)
- Log file

================================================================================
MESH SIZE MODEL (modular)
================================================================================
Implement MeshSizeField → regular raster H used as Gmsh Background mesh-size field.

Strategies (name them in config; default = "product"):
1) product          # gridsize1 — DEFAULT (importance weights Escala1*Escala2)
2) mean             # gridsize3
3) depth_weighted   # gridsize4 notebook blend
4) **hybrid_smooth** (NEW — invent and document):
   - h_raw = hmin + (Escala1**a) * (Escala2**b) * (hmax-hmin)  with a,b configurable (default a=1,b=1 → product)
   - optional multiplicative coast factor if coast distance available: f_coast in (0,1]
   - optional interest fields via min()
   - then Gaussian smooth of log(H) or H (configurable sigma in pixels) to avoid abrupt size jumps
   - clamp to [hmin, hmax]
   Rationale: keeps product-style importance weights, adds controllable exponents + smoothing for Gmsh stability.

Interest points:
- Port Distancia_y_Peso; combine with base field by elementwise minimum (default) or multiply normalized weights.

Also expose hooks for future Gmsh Distance/Threshold coast fields without breaking API.

Parameters (MeshConfig / YAML): hmin, hmax, T, L_constant, strategy, slope_mean/std overrides, clips, interest list, smooth_sigma, exponents a/b, combiner.

MVP must reproduce notebook product field on a synthetic case (tests).

================================================================================
MESHING ENGINE (mandatory: Gmsh Python)
================================================================================
- import gmsh behind interface Mesher (base + gmsh_mesher.py)
- Flow:
  1) Build/load DEM (interpolarGrid logic or GeoTIFF)
  2) Slope
  3) Size-field raster H
  4) Gmsh model from domain polygon (+ holes)
  5) Background field from H (PostView / field API — choose robust approach, test it)
  6) generate 2D triangular mesh; extract nodes, triangles, boundary edges
  7) interpolate Z; markers; export SWAN
- Actionable error if gmsh missing
- Library must not import GUI modules

================================================================================
PIPELINE
================================================================================
swanmesh.pipeline.run(config: MeshConfig | path) -> Result

1. ValidateConfig
2. LoadDomain (geopandas/shapely), CRS, optional repair/simplify
3. Load/Build DEM (XYZ/raster)
4. BuildSlope
5. BuildMeshSizeField
6. GenerateMeshGmsh
7. InterpolateDepthToNodes (no globals; robust nodata)
8. ClassifyBoundaryMarkers (strategy pattern):
   - default: notebook rule depth_limit on boundary nodes
   - optional: open_boundary_lines shapefile (if provided)
9. QualityChecks (CCW, NaN, orphans, markers, h stats)
10. ExportSWAN + sidecars + plots
11. WriteReport

Legacy (first-class):
  swanmesh.convert_gmsh_to_swan(...) / CLI convert-msh
  Parity with Gmsh2SWAN + regression tests
  Fix brittle parsing (don’t assume fixed token counts); triangles only; consistent Z for markers

================================================================================
ARCHITECTURE
================================================================================
swanmesh/
  pyproject.toml
  README.md                 # Spanish
  CHANGELOG.md
  configs/examples/*.yaml
  examples/
  src/swanmesh/
    __init__.py
    config.py               # pydantic v2 preferred
    domain.py
    bathymetry.py
    slope.py
    size_field.py
    mesher/base.py
    mesher/gmsh_mesher.py
    boundaries.py
    interpolate.py
    export/swan_triangle.py
    export/gmsh_io.py
    qa.py
    plot.py
    pipeline.py
    cli.py
    errors.py
  src/swanmesh_gui/
    __init__.py
    app.py
    widgets.py
    workers.py              # long runs off main thread
  tests/
  .github/workflows/ci.yml

Principles: thin GUI; DI for Mesher; typed API; logging; overwrite=False by default; reproducible dumps.

================================================================================
GUI (FreeSimpleGUI — easiest MVP)
================================================================================
Single-window or tabbed MVP:
- Project: name, output_dir, work_crs, output_crs, z_convention
- Inputs: domain, bathy xyz/tif, is_utm, slope optional, interest table
- DEM: resolution/pixel_size, buffer, interp method
- Size field: hmin/hmax, T, strategy (product default), hybrid params, clips
- Boundary: depth_limit, marker strategy
- Run / log / cancel-if-feasible / open output
- Save-load YAML (same schema as library)
- Optional simple previews (matplotlib to PNG file + path) — visualization quality secondary to simplicity

No duplicated meshing logic.

================================================================================
SOFTWARE ENGINEERING (mandatory)
================================================================================
1) pyproject.toml extras: [gui], [dev] (gmsh as install/docs requirement)
2) Python >= 3.10
3) Deps: numpy scipy pandas geopandas shapely pyproj rasterio matplotlib pyyaml pydantic; gmsh; FreeSimpleGUI; typer (CLI)
4) ruff + format + pytest; mypy gradual optional
5) Tests:
   - Escala1/Escala2 sanity + product default
   - hybrid_smooth clamp/smooth
   - interest radial minimum at center
   - DEM griddata smoke
   - CCW .ele order
   - marker depth rule under elevation_negative_down
   - CRS units recorded
   - synthetic integration: rectangle + analytic bathy/slope → .node/.ele/.bot parseable
   - legacy msh convert golden/regression (tiny fixture)
6) CI GitHub Actions: lint + tests on Linux
7) Semver; SwanMeshError with actionable messages
8) README.md **in Spanish**: install (including Gmsh), quickstart lib, CLI, GUI, CRS/units, z convention, markers 0/1/2, example YAML, limitations
9) Short ADRs in docs/adr/:
   - Gmsh background field approach
   - default size strategy product + hybrid_smooth
   - FreeSimpleGUI choice
   - z_convention default elevation_negative_down
   - default output lon/lat for SWAN

================================================================================
DEFINITION OF DONE
================================================================================
1. pip install -e ".[dev,gui]" on clean env (documented in Spanish README)
2. Synthetic example writes valid .node/.ele/.bot
3. Size field densifies shallow water, steep slope, and an interest point (stats + optional PNG)
4. Default SWAN xy in lon/lat path works; UTM work/output path documented and smoke-tested
5. .bot default negative elevation; positive-depth option works
6. GUI runs one full case without code edits
7. CLI run + convert-msh work
8. Legacy convert first-class + tested
9. CI green; no local paths/secrets; notebook bugs fixed
10. ADRs + Spanish README + example configs

================================================================================
EXECUTION PLAN (agent must implement, not only analyze)
================================================================================
Phase 0 — Notebook summary, bugs, conventions (CRS, z, markers, h units)
Phase 1 — Scaffold repo, pydantic config, logging, CLI stubs, package imports
Phase 2 — Domain + DEM (interpolarGrid) + slope + tests
Phase 3 — size_field (product default, mean, depth_weighted, hybrid_smooth) + tests
Phase 4 — Gmsh mesher + pipeline MVP synthetic end-to-end
Phase 5 — SWAN export + legacy convert + golden tests
Phase 6 — markers/QA/plots/report
Phase 7 — FreeSimpleGUI MVP
Phase 8 — examples, CI, Spanish README, ADR, hardening (holes, large XYZ)

Atomic commits, tests green before advancing. Prefer working MVP over perfect abstraction.
If blocked, choose a reasonable default, document in ADR, continue.

================================================================================
CONSTRAINTS
================================================================================
- Local/offline only
- Gmsh is the mesh generator
- Preserve SWAN Triangle conventions from Gmsh2SWAN unless config opts out
- Clarity over premature optimization
- Start Phase 0–1 immediately and continue through MVP without waiting for the user unless truly blocked
