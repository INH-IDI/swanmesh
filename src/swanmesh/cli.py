"""CLI entrypoint for swanmesh using typer."""

from pathlib import Path

import typer

from swanmesh import __version__
from swanmesh.bathymetry import build_dem
from swanmesh.config import MeshConfig
from swanmesh.domain import load_domain
from swanmesh.export.gmsh_io import convert_gmsh_to_swan
from swanmesh.pipeline import run as run_pipeline
from swanmesh.size_field import build_size_field
from swanmesh.slope import build_slope

app = typer.Typer(help="swanmesh CLI: Automatic SWAN 2D unstructured mesh generation & legacy conversion.")

@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML configuration file"),
):
    """Run full swanmesh mesh generation pipeline from YAML config."""
    typer.echo(f"Starting swanmesh pipeline from {config_path}...")
    res = run_pipeline(config_path)
    typer.echo(f"Mesh generation completed successfully! Outputs written to: {res.output_dir}")
    typer.echo(f"QA Status: {'PASSED' if res.qa_report['qa_passed'] else 'FAILED'}")

@app.command(name="convert-msh")
def convert_msh(
    msh_path: Path = typer.Argument(..., help="Path to input Gmsh .msh file"),
    dem: Path = typer.Option(..., "--dem", "-d", help="Path to bathymetry DEM GeoTIFF"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    base_name: str = typer.Option("converted_mesh", "--base-name", "-b", help="Base name for SWAN files"),
    depth_limit: float = typer.Option(-50.0, "--depth-limit", help="Depth limit for marker 2 (open ocean)"),
    z_convention: str = typer.Option("elevation_negative_down", "--z-convention", help="Z convention"),
):
    """Convert legacy Gmsh .msh file to SWAN .node, .ele, .bot files."""
    typer.echo(f"Converting legacy .msh file {msh_path}...")
    out_dir = convert_gmsh_to_swan(
        msh_path=msh_path,
        dem_path=dem,
        output_dir=out,
        base_name=base_name,
        depth_limit=depth_limit,
        z_convention=z_convention,
    )
    typer.echo(f"Conversion complete! SWAN mesh written to: {out_dir}")

@app.command(name="validate-config")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to YAML configuration file"),
):
    """Validate YAML configuration file schema and parameter constraints."""
    try:
        cfg = MeshConfig.from_yaml(config_path)
        typer.echo(f"Configuration file {config_path} is VALID.")
        typer.echo(f"Project: {cfg.project_name}, Strategy: {cfg.strategy}, CRS: {cfg.work_crs}")
    except Exception as e:
        typer.echo(f"Configuration ERRORS found in {config_path}:\n{e}")
        raise typer.Exit(code=1)

@app.command(name="build-dem")
def build_dem_cmd(
    config_path: Path = typer.Argument(..., help="Path to YAML configuration file"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output GeoTIFF path"),
):
    """Generate bathymetry DEM GeoTIFF from configuration."""
    cfg = MeshConfig.from_yaml(config_path)
    dom = load_domain(cfg.domain_path, work_crs=cfg.work_crs)
    bounds = dom.get_bounds(buffer_cells=cfg.buffer_cells, dx=cfg.dx, dy=cfg.dy)
    dem_data = build_dem(cfg, bounds=bounds)
    target = out if out else Path(cfg.output_dir) / f"{cfg.project_name}_dem.tif"
    dem_data.save_geotiff(target)
    typer.echo(f"DEM GeoTIFF written to: {target}")

@app.command(name="build-size-field")
def build_size_field_cmd(
    config_path: Path = typer.Argument(..., help="Path to YAML configuration file"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output GeoTIFF path"),
):
    """Generate background mesh-size raster GeoTIFF from configuration."""
    cfg = MeshConfig.from_yaml(config_path)
    dom = load_domain(cfg.domain_path, work_crs=cfg.work_crs)
    bounds = dom.get_bounds(buffer_cells=cfg.buffer_cells, dx=cfg.dx, dy=cfg.dy)
    dem_data = build_dem(cfg, bounds=bounds)
    slope_data = build_slope(dem_data, slope_tif_path=cfg.slope_tif_path)
    sf_data = build_size_field(cfg, dem=dem_data, slope=slope_data)
    target = out if out else Path(cfg.output_dir) / f"{cfg.project_name}_size_field.tif"
    sf_data.save_geotiff(target)
    typer.echo(f"Mesh size field GeoTIFF written to: {target}")

@app.command()
def version():
    """Display swanmesh software version."""
    typer.echo(f"swanmesh version {__version__}")

if __name__ == "__main__":
    app()
