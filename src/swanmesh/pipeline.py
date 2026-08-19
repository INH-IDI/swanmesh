"""Master orchestration pipeline for swanmesh."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyproj

from swanmesh import __version__
from swanmesh.bathymetry import build_dem
from swanmesh.boundaries import classify_boundary_markers
from swanmesh.config import MeshConfig
from swanmesh.domain import load_domain
from swanmesh.errors import ConfigurationError, MeshSizeGuardError
from swanmesh.export.swan_triangle import export_swan_triangle
from swanmesh.interpolate import interpolate_z_to_nodes
from swanmesh.mesher.gmsh_mesher import GmshMesher
from swanmesh.plot import plot_pipeline_previews
from swanmesh.qa import run_mesh_qa
from swanmesh.size_field import build_size_field_steps, estimate_mesh_nodes
from swanmesh.slope import build_slope


class PipelineResult:
    """Stores result metadata, output directory, and QA report of a pipeline run."""

    def __init__(self, output_dir: Path, qa_report: dict[str, Any], files_generated: dict[str, str]):
        self.output_dir = output_dir
        self.qa_report = qa_report
        self.files_generated = files_generated


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_overwrite(out_dir: Path, base_name: str, overwrite: bool) -> None:
    markers = [
        out_dir / f"{base_name}.node",
        out_dir / f"{base_name}.ele",
        out_dir / f"{base_name}.bot",
        out_dir / f"{base_name}_report.json",
    ]
    existing = [str(p) for p in markers if p.exists()]
    if existing and not overwrite:
        raise ConfigurationError(
            "Output already exists and overwrite=false. Existing files: "
            + ", ".join(existing)
        )


def _reproject_nodes(nodes_df, src_crs: str, dst_crs: str):
    if not dst_crs or src_crs == dst_crs:
        return nodes_df
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x_out, y_out = transformer.transform(nodes_df["X"].to_numpy(), nodes_df["Y"].to_numpy())
    out = nodes_df.copy()
    out["X"] = x_out
    out["Y"] = y_out
    return out


def _export_msh(nodes_df, elem_df, out_path: Path) -> None:
    """Write a minimal Gmsh MSH 2.2 ASCII file from triangle mesh tables."""
    nodes_sorted = nodes_df.sort_values("N")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{len(nodes_sorted)}\n")
        for _, row in nodes_sorted.iterrows():
            f.write(f"{int(row['N'])} {row['X']:.10g} {row['Y']:.10g} 0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{len(elem_df)}\n")
        for _, row in elem_df.iterrows():
            # type 2 triangle, 2 tags (0 0)
            f.write(
                f"{int(row['ID'])} 2 2 0 0 "
                f"{int(row['ELEMENT1'])} {int(row['ELEMENT2'])} {int(row['ELEMENT3'])}\n"
            )
        f.write("$EndElements\n")


def run(config_or_path: MeshConfig | str | Path) -> PipelineResult:
    """Execute complete swanmesh workflow from configuration."""
    t0 = time.perf_counter()
    if isinstance(config_or_path, (str, Path)):
        config = MeshConfig.from_yaml(config_or_path)
    else:
        config = config_or_path

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = config.project_name
    _check_overwrite(out_dir, base_name, config.overwrite)

    # 1. Domain
    domain = load_domain(config.domain_path, work_crs=config.work_crs)
    if config.subdomain_bbox and len(config.subdomain_bbox) == 4:
        domain = domain.crop_to_subdomain(config.subdomain_bbox)
    bounds = domain.get_bounds(buffer_cells=config.buffer_cells, dx=config.dx, dy=config.dy)
    _log(f"[1/12] Domain loaded. bounds={bounds}")

    # 2. DEM
    dem = build_dem(config, bounds=bounds)
    zmin, zmax, zmean = float(np.nanmin(dem.grid)), float(np.nanmax(dem.grid)), float(np.nanmean(dem.grid))
    _log(f"[2/12] DEM shape={dem.grid.shape} Z=[{zmin:.2f}, {zmax:.2f}] mean={zmean:.2f}")

    # 3. Slope (metric units)
    slope = build_slope(dem, slope_tif_path=config.slope_tif_path, is_utm=config.is_utm)
    _log(
        f"[3/12] Slope units={getattr(slope, 'units', 'm/m')} "
        f"mean={slope.mean:.6g} std={slope.std:.6g}"
    )

    # 4. Size field
    from swanmesh.lc_profile import plot_lc_preview_plotly

    size_field, size_steps = build_size_field_steps(config, dem=dem, slope=slope)
    _log("\n" + "=" * 60)
    _log(" [SIZE FIELD DEBUG STEPS]")
    _log(
        f"   • step1_depth: min={size_steps['step1_depth'].grid.min():.6g}, "
        f"max={size_steps['step1_depth'].grid.max():.6g}, "
        f"mean={size_steps['step1_depth'].grid.mean():.6g}"
    )
    _log(
        f"   • step2_slope: min={size_steps['step2_slope'].grid.min():.6g}, "
        f"max={size_steps['step2_slope'].grid.max():.6g}, "
        f"mean={size_steps['step2_slope'].grid.mean():.6g}"
    )
    _log(
        f"   • step3_control: min={size_steps['step3_control'].grid.min():.6g}, "
        f"max={size_steps['step3_control'].grid.max():.6g}, "
        f"mean={size_steps['step3_control'].grid.mean():.6g}"
    )
    _log(
        f"   • step4_gradation (mesh_growth={config.mesh_growth:g}): "
        f"min={size_steps['step4_gradation'].grid.min():.6g}, "
        f"max={size_steps['step4_gradation'].grid.max():.6g}, "
        f"mean={size_steps['step4_gradation'].grid.mean():.6g}"
    )
    _log("=" * 60)

    node_est = estimate_mesh_nodes(size_field.grid, dx=config.dx, dy=config.dy)
    _log(
        f"[4/12] Size field ready. est_nodes~{node_est['est_nodes']:,} "
        f"est_elements~{node_est['est_elements']:,} "
        f"H=[{node_est['min_h']:.6g}, {node_est['max_h']:.6g}] mean={node_est.get('mean_h', 0):.6g}"
    )
    if config.abort_on_est_nodes and int(node_est["est_nodes"]) > int(config.max_est_nodes):
        raise MeshSizeGuardError(
            f"Estimated nodes ({node_est['est_nodes']:,}) exceed max_est_nodes "
            f"({config.max_est_nodes:,}). Increase hmin/hmax, reduce n_lambda, "
            f"or raise max_est_nodes / set abort_on_est_nodes=false."
        )

    # Depth-limit feasibility warning
    if config.marker_strategy == "depth_limit":
        if config.z_convention == "elevation_negative_down" and zmin >= config.depth_limit:
            _log(
                f"WARNING: DEM Zmin={zmin:.2f} >= depth_limit={config.depth_limit}. "
                "Open-boundary marker 2 may be absent."
            )
        if config.z_convention == "depth_positive_down" and zmax <= abs(config.depth_limit):
            _log(
                f"WARNING: DEM Zmax={zmax:.2f} <= |depth_limit|={abs(config.depth_limit)}. "
                "Open-boundary marker 2 may be absent."
            )

    # 5. Mesh
    mesher = GmshMesher(
        algorithm_2d=config.gmsh_algorithm_2d,
        optimize_netgen=config.gmsh_optimize_netgen,
        smoothing_steps=config.gmsh_smoothing_steps,
        enforce_min_node_degree=config.enforce_min_node_degree,
        min_node_degree=config.min_node_degree,
        max_boundary_points=config.max_boundary_points,
    )
    t_mesh = time.perf_counter()
    mesh_res = mesher.generate_mesh(domain=domain, size_field=size_field)
    _log(
        f"[5/12] Mesh generated in {time.perf_counter()-t_mesh:.1f}s: "
        f"{len(mesh_res.nodes):,} nodes, {len(mesh_res.triangles):,} triangles"
    )

    # 6. Z interpolation
    nodes_z = interpolate_z_to_nodes(
        mesh_res.nodes,
        dem=dem,
        z_convention=config.z_convention,
    )
    _log("[6/12] Z interpolated to nodes")

    # 7. Boundary markers
    nodes_marked = classify_boundary_markers(
        nodes_z,
        boundary_edges=mesh_res.boundary_edges,
        depth_limit=config.depth_limit,
        z_convention=config.z_convention,
        marker_strategy=config.marker_strategy,
        open_boundary_lines_path=config.open_boundary_lines_path,
        open_boundary_buffer=config.open_boundary_buffer,
    )
    _log("[7/12] Boundary markers classified")

    # 8. Optional output CRS reproject (after Z/markers; Z is CRS-invariant)
    nodes_final = _reproject_nodes(nodes_marked, config.work_crs, config.output_crs)
    if config.output_crs != config.work_crs:
        _log(f"[8/12] Reprojected nodes {config.work_crs} -> {config.output_crs}")
    else:
        _log("[8/12] output_crs == work_crs (no reprojection)")

    # 9. QA
    qa_report = run_mesh_qa(nodes_final, mesh_res.triangles)
    qa_report["node_estimate"] = node_est
    qa_report["slope_stats"] = {
        "mean": slope.mean,
        "std": slope.std,
        "units": getattr(slope, "units", "m/m"),
    }
    _log(f"[9/12] QA {'PASSED' if qa_report['qa_passed'] else 'FAILED'} markers={qa_report.get('marker_histogram')}")

    # 10. Export SWAN
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
    _log("[10/12] SWAN triangle files exported")

    if config.export_msh:
        msh_path = out_dir / f"{base_name}.msh"
        _export_msh(nodes_final, mesh_res.triangles, msh_path)
        files_gen["msh"] = str(msh_path)

    # 11. Sidecar rasters + debug steps
    bathy_path = out_dir / f"{base_name}_bathy.tif"
    dem_path = out_dir / f"{base_name}_dem.tif"
    dem.save_geotiff(bathy_path)
    dem.save_geotiff(dem_path)
    files_gen["bathy_tif"] = str(bathy_path)
    files_gen["dem_tif"] = str(dem_path)

    debug_dir = out_dir / "debug_steps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for step_name, step_sf in size_steps.items():
        step_tif = debug_dir / f"{base_name}_{step_name}_size_field.tif"
        step_sf.save_geotiff(step_tif)
        files_gen[f"{step_name}_tif"] = str(step_tif)

    if config.export_sidecar_tifs:
        slope_path = out_dir / f"{base_name}_slope.tif"
        size_path = out_dir / f"{base_name}_size_field.tif"
        slope.save_geotiff(slope_path)
        size_field.save_geotiff(size_path)
        files_gen["slope_tif"] = str(slope_path)
        files_gen["size_tif"] = str(size_path)

    # 12. Plots
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

        try:
            s_prof, _, _, h_sampled = dem.sample_profile(num_points=250)
            # Use already-computed step grids (column-mean proxy profile)
            plotly_html = out_dir / f"{base_name}_lc_profile.html"
            plot_lc_preview_plotly(
                x=s_prof,
                h=h_sampled,
                lc1=size_steps["step1_depth"].grid.mean(axis=0),
                lc2=size_steps["step2_slope"].grid.mean(axis=0),
                lc3=size_steps["step3_control"].grid.mean(axis=0),
                min_lc=config.hmin,
                max_lc=config.hmax,
                period=config.wave_period,
                title=f"Análisis Vectorial e Interactivo de Perfil — {base_name}",
                output_html=plotly_html,
            )
            files_gen["profile_plotly_html"] = str(plotly_html)
            _log(f"Plotly profile saved: {plotly_html}")
        except Exception as p_err:
            _log(f"Aviso Plotly: {p_err}")

    config_out = out_dir / f"{base_name}_config.yaml"
    config.to_yaml(config_out)
    files_gen["config_yaml"] = str(config_out)

    elapsed = time.perf_counter() - t0
    report_data = {
        "swanmesh_version": __version__,
        "project_name": base_name,
        "work_crs": config.work_crs,
        "output_crs": config.output_crs,
        "z_convention": config.z_convention,
        "hmin": config.hmin,
        "hmax": config.hmax,
        "n_lambda": config.n_lambda,
        "alpha_grad": config.alpha_grad,
        "strategy": config.strategy,
        "mesh_growth": config.mesh_growth,
        "bounds": bounds,
        "elapsed_seconds": elapsed,
        "qa_report": qa_report,
        "files_generated": files_gen,
    }
    report_out = out_dir / f"{base_name}_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    files_gen["report_json"] = str(report_out)
    _log(f"[12/12] Done in {elapsed:.1f}s. Report: {report_out}")

    return PipelineResult(output_dir=out_dir, qa_report=qa_report, files_generated=files_gen)
