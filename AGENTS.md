# AGENTS.md — swanmesh (`grillas/`)

Python package that builds 2D unstructured SWAN meshes (`.node` / `.ele` / `.bot`) via Gmsh. Package lives here; parent `../` holds Antofagasta domain/bathy data and run outputs.

## Layout that matters

| Path | Role |
|------|------|
| `src/swanmesh/` | Core library. Entry: `pipeline.run` ← `cli.py` |
| `src/swanmesh_gui/` | FreeSimpleGUI app (`swanmesh-gui`) |
| `configs/examples/` | YAML configs (use `antofagasta_sane.yaml` as operational reference) |
| `tests/` | pytest; fixtures in `conftest.py` (synthetic domain/bathy) |
| `docs/adr/` | Architecture decisions (size strategies, Z convention, CRS) |

Parent workspace (not this package): `../dominio.shp`, `Bathy_ANT.csv` (also copied here), `../output_*`.

## Environment & commands

```bash
source ~/ambientes/ML/bin/activate   # project venv (Python 3.11)
cd /home/model2/trabajo/antofa/grillas
pip install -e ".[dev,gui]"          # after dependency changes

pytest -v                            # full suite (~36 tests, fast, no network)
pytest tests/test_hardening.py -v    # single file
pytest tests/test_pipeline.py::test_full_pipeline_synthetic -v

ruff check src/ tests/               # CI lint (line-length 100)
swanmesh run configs/examples/antofagasta_sane.yaml
swanmesh validate-config path.yaml
swanmesh-gui
```

CI (`.github/workflows/ci.yml`): `pip install -e ".[dev,gui]"` → `ruff check src/ tests/` → `pytest -v`. No mypy in CI.

Numpy is pinned `<2` in `pyproject.toml`.

## Pipeline flow (do not reorder mentally)

`domain → DEM (XYZ/TIF/online) → slope (m/m) → size_field → Gmsh mesh → Z interp → boundary markers → QA → export`

Orchestrator: `src/swanmesh/pipeline.py`. Config model: `src/swanmesh/config.py` (Pydantic v2).

## Units & physics traps (read before changing size field)

- **`hmin` / `hmax` / mesh sizes are in `work_crs` native units**: degrees for EPSG:4326, metres for UTM.
- **Slope is always dimensionless m/m**, even on geographic CRS (`geo_units.py` + `slope.py`). Never feed degree-spacing into `np.gradient` without metric conversion — that historically collapsed meshes to millions of nodes.
- **`dispersion_gradient` uses `lc = clip(L_native / n_lambda, hmin, hmax)`** (`size_field.py`).
  - lon/lat coastal: `n_lambda ≈ 0.2–0.5` (not 15).
  - UTM metres: `n_lambda ≈ 10–30`.
  - Default `n_lambda=15.0` in `MeshConfig` is UTM-oriented; lon/lat configs must override.
- **`max_est_nodes`** (default 500k) + **`abort_on_est_nodes`** abort before Gmsh if the size field would explode. Raise only deliberately.
- **Boundary markers**: `0` interior, `1` closed/land, `2` open ocean. With `elevation_negative_down`, marker 2 needs boundary `Z < depth_limit` (default −50). Flat/wrong DEM → no marker 2.
- **`output_crs`**, **`overwrite`**, **`export_msh`** are real pipeline flags (not dead config).

## Operational gotchas

- Domain boundary is simplified/resampled before Gmsh (`domain.resample_boundary`). Self-intersecting densified contours fail with “no 2D triangular elements”. Cap with **`max_boundary_points`** (sane ≈ 800).
- Prefer `configs/examples/antofagasta_sane.yaml` over `../config_debug.yaml` / ad-hoc YAMLs that previously produced multi‑GB meshes.
- Full Antofagasta runs write large files under `output_dir`; keep `overwrite: true` only when intentional.
- GUI (`swanmesh_gui/app.py`) must stay in sync with new `MeshConfig` fields when you add them.
- Legacy notebooks (`Gmsh2SWAN.ipynb`, `gridSize.ipynb`, …) are historical sources; **code in `src/` is authoritative**.

## When changing code

1. Touch units/size field → update `tests/test_hardening.py` and run it first.
2. Touch pipeline/export/QA → `pytest tests/test_pipeline.py tests/test_export.py tests/test_hardening.py`.
3. Keep comments out of code unless asked.
4. Do not commit secrets; large bathy CSVs / mesh outputs are local data, not package source.
