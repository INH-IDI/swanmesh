"""Master orchestration pipeline for swanmesh."""

import json
from pathlib import Path
from typing import Any

from swanmesh import __version__
from swanmesh.bathymetry import build_dem
from swanmesh.boundaries import classify_boundary_markers
from swanmesh.config import MeshConfig
from swanmesh.domain import load_domain
from swanmesh.export.swan_triangle import export_swan_triangle
from swanmesh.interpolate import interpolate_z_to_nodes
from swanmesh.mesher.gmsh_mesher import GmshMesher
from swanmesh.plot import plot_pipeline_previews
from swanmesh.qa import run_mesh_qa
from swanmesh.size_field import build_size_field
from swanmesh.slope import build_slope


class PipelineResult:
    """Stores result metadata, output directory, and QA report of a pipeline run."""

    def __init__(self, output_dir: Path, qa_report: dict[str, Any], files_generated: dict[str, str]):
        self.output_dir = output_dir
        self.qa_report = qa_report
        self.files_generated = files_generated

def run(config_or_path: MeshConfig | str | Path) -> PipelineResult:
    """Execute complete swanmesh workflow from configuration."""
    if isinstance(config_or_path, (str, Path)):
        config = MeshConfig.from_yaml(config_or_path)
    else:
        config = config_or_path

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.project_name

    # 1. Load Domain
    domain = load_domain(config.domain_path, work_crs=config.work_crs)
    bounds = domain.get_bounds(buffer_cells=config.buffer_cells, dx=config.dx, dy=config.dy)

    # 2. Build DEM
    dem = build_dem(config, bounds=bounds)

    # 3. Build Slope
    slope = build_slope(dem, slope_tif_path=config.slope_tif_path)

    # 4. Build Size Field
    size_field = build_size_field(config, dem=dem, slope=slope)

    # 5. Generate Mesh with Gmsh
    mesher = GmshMesher()
    mesh_res = mesher.generate_mesh(domain=domain, size_field=size_field)

    # 6. Interpolate Elevation Z to Nodes
    nodes_z = interpolate_z_to_nodes(
        mesh_res.nodes,
        dem=dem,
        z_convention=config.z_convention,
    )

    # 7. Classify Boundary Markers
    nodes_final = classify_boundary_markers(
        nodes_z,
        boundary_edges=mesh_res.boundary_edges,
        depth_limit=config.depth_limit,
        z_convention=config.z_convention,
        marker_strategy=config.marker_strategy,
        open_boundary_lines_path=config.open_boundary_lines_path,
    )

    # 8. Run QA Checks
    qa_report = run_mesh_qa(nodes_final, mesh_res.triangles)

    # 9. Export SWAN Triangle (.node, .ele, .bot)
    export_swan_triangle(
        nodes_df=nodes_final,
        elem_df=mesh_res.triangles,
        output_dir=out_dir,
        base_name=base_name,
    )

    files_gen = {
        "node": str(out_dir / f"{base_name}.node"),
        "ele": str(out_dir / f"{base_name}.ele"),
        "bot": str(out_dir / f"{base_name}.bot"),
    }

    # 10. Export Sidecar TIFs
    if config.export_sidecar_tifs:
        dem_path = out_dir / f"{base_name}_dem.tif"
        slope_path = out_dir / f"{base_name}_slope.tif"
        size_path = out_dir / f"{base_name}_size_field.tif"
        dem.save_geotiff(dem_path)
        slope.save_geotiff(slope_path)
        size_field.save_geotiff(size_path)
        files_gen["dem_tif"] = str(dem_path)
        files_gen["slope_tif"] = str(slope_path)
        files_gen["size_tif"] = str(size_path)

    # 11. Export Plots
    if config.export_plots:
        plot_dir = plot_pipeline_previews(
            domain=domain,
            dem=dem,
            size_field=size_field,
            nodes_df=nodes_final,
            elem_df=mesh_res.triangles,
            output_dir=out_dir,
            base_name=base_name,
        )
        files_gen["plots_dir"] = str(plot_dir)

    # 12. Write Resolved Config & Report JSON
    config_out = out_dir / f"{base_name}_config.yaml"
    config.to_yaml(config_out)
    files_gen["config_yaml"] = str(config_out)

    report_data = {
        "swanmesh_version": __version__,
        "project_name": base_name,
        "work_crs": config.work_crs,
        "output_crs": config.output_crs,
        "z_convention": config.z_convention,
        "hmin": config.hmin,
        "hmax": config.hmax,
        "bounds": bounds,
        "qa_report": qa_report,
        "files_generated": files_gen,
    }
    report_out = out_dir / f"{base_name}_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    files_gen["report_json"] = str(report_out)

    return PipelineResult(output_dir=out_dir, qa_report=qa_report, files_generated=files_gen)
