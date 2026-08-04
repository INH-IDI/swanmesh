"""FreeSimpleGUI application for swanmesh."""

import os
import subprocess
import sys
from pathlib import Path

import FreeSimpleGUI as sg

from swanmesh.config import MeshConfig
from swanmesh_gui.workers import ConvertWorker, PipelineWorker


def show_copyable_error_popup(title: str, error_text: str):
    """Display a popup window with copyable multiline error text."""
    layout = [
        [sg.Text(title, font=("Helvetica", 11, "bold"), text_color="red")],
        [sg.Multiline(error_text, size=(80, 20), read_only=True, font=("Courier", 10), key="-ERR_TEXT-")],
        [sg.Button("Copy to Clipboard", key="-COPY-"), sg.Button("Close", key="-CLOSE-")],
    ]
    win = sg.Window(title, layout, modal=True, finalize=True)
    while True:
        ev, val = win.read()
        if ev in (sg.WIN_CLOSED, "-CLOSE-"):
            break
        elif ev == "-COPY-":
            sg.clipboard_set(error_text)
            sg.popup_quick_message("Copied to clipboard!", background_color="green", text_color="white")
    win.close()

def create_main_window():
    sg.theme("DarkTeal6")

    # Tab 1: Project & Inputs
    tab1_layout = [
        [sg.Text("Project Name:", size=(18, 1)), sg.Input("swan_mesh", key="-PROJ_NAME-")],
        [sg.Text("Output Directory:", size=(18, 1)), sg.Input("./output", key="-OUT_DIR-"), sg.FolderBrowse()],
        [sg.Text("Work CRS:", size=(18, 1)), sg.Input("EPSG:4326", key="-WORK_CRS-")],
        [sg.Text("Output CRS:", size=(18, 1)), sg.Input("EPSG:4326", key="-OUT_CRS-")],
        [sg.Text("Domain Shapefile:", size=(18, 1)), sg.Input("", key="-DOMAIN_PATH-"), sg.FileBrowse(file_types=(("Vector Files", "*.shp *.geojson *.gpkg"),))],
        [sg.Text("Bathymetry XYZ:", size=(18, 1)), sg.Input("", key="-XYZ_PATH-"), sg.FileBrowse(file_types=(("XYZ / CSV Files", "*.csv *.xyz *.txt"),))],
        [sg.Text("Bathymetry DEM TIF:", size=(18, 1)), sg.Input("", key="-TIF_PATH-"), sg.FileBrowse(file_types=(("GeoTIFF Files", "*.tif *.tiff"),))],
        [sg.Checkbox("XYZ is in UTM coordinates", default=False, key="-IS_UTM-")],
        [sg.Checkbox("Use Online Bathymetry (GEBCO / Open-Elevation)", default=False, key="-USE_ONLINE_BATHY-")],
        [sg.Checkbox("Blend Local Bathymetry with Online Background", default=False, key="-BLEND_ONLINE_BATHY-")],
    ]

    # Tab 2: DEM & Mesh Size Field
    tab2_layout = [
        [sg.Text("Resolution dx (native CRS):", size=(22, 1)), sg.Input("0.002", key="-DX-")],
        [sg.Text("Resolution dy (native CRS):", size=(22, 1)), sg.Input("0.002", key="-DY-")],
        [sg.Text("Buffer Cells:", size=(22, 1)), sg.Input("20", key="-BUFFER-")],
        [sg.Text("Minimum Mesh Size hmin:", size=(22, 1)), sg.Input("0.001", key="-HMIN-")],
        [sg.Text("Maximum Mesh Size hmax:", size=(22, 1)), sg.Input("0.05", key="-HMAX-")],
        [sg.Text("Wave Period T (s):", size=(22, 1)), sg.Input("30.0", key="-WAVE_T-")],
        [sg.Text("Size Strategy:", size=(22, 1)), sg.Combo(["product", "mean", "depth_weighted", "hybrid_smooth"], default_value="product", key="-STRATEGY-")],
        [sg.Text("Hybrid Smooth Sigma (px):", size=(22, 1)), sg.Input("0.0", key="-SMOOTH_SIGMA-")],
    ]

    # Tab 3: Boundaries & Z Convention
    tab3_layout = [
        [sg.Text("Z Convention:", size=(22, 1)), sg.Combo(["elevation_negative_down", "depth_positive_down"], default_value="elevation_negative_down", key="-Z_CONV-")],
        [sg.Text("Depth Limit for Marker 2:", size=(22, 1)), sg.Input("-50.0", key="-DEPTH_LIMIT-")],
        [sg.Text("Marker Strategy:", size=(22, 1)), sg.Combo(["depth_limit", "open_boundary_lines"], default_value="depth_limit", key="-MARKER_STRAT-")],
    ]

    # Tab 4: Legacy MSH Convert
    tab4_layout = [
        [sg.Text("Input Gmsh .msh File:", size=(20, 1)), sg.Input("", key="-CONV_MSH-"), sg.FileBrowse(file_types=(("Gmsh Mesh Files", "*.msh"),))],
        [sg.Text("Bathymetry DEM .tif:", size=(20, 1)), sg.Input("", key="-CONV_DEM-"), sg.FileBrowse(file_types=(("GeoTIFF Files", "*.tif *.tiff"),))],
        [sg.Text("Output Base Name:", size=(20, 1)), sg.Input("converted_mesh", key="-CONV_BASE-")],
        [sg.Button("Run Legacy Convert", key="-RUN_CONVERT-")],
    ]

    layout = [
        [
            sg.TabGroup(
                [
                    [sg.Tab("Project & Inputs", tab1_layout)],
                    [sg.Tab("DEM & Size Field", tab2_layout)],
                    [sg.Tab("Boundaries & Z", tab3_layout)],
                    [sg.Tab("Legacy Convert", tab4_layout)],
                ]
            )
        ],
        [
            sg.Button("Load YAML Config", key="-LOAD_YAML-"),
            sg.Button("Save YAML Config", key="-SAVE_YAML-"),
            sg.Button("Run Mesh Pipeline", key="-RUN_PIPELINE-", button_color=("white", "green")),
            sg.Button("Open Output Dir", key="-OPEN_OUT-"),
            sg.Button("Exit", key="-EXIT-"),
        ],
        [sg.Text("Execution Log:")],
        [sg.Output(size=(80, 10), key="-LOG-")],
    ]

    return sg.Window("swanmesh — 2D SWAN Mesh Generator", layout, finalize=True)

def read_config_from_gui(values) -> MeshConfig:
    return MeshConfig(
        project_name=values["-PROJ_NAME-"],
        output_dir=values["-OUT_DIR-"],
        work_crs=values["-WORK_CRS-"],
        output_crs=values["-OUT_CRS-"],
        domain_path=values["-DOMAIN_PATH-"],
        bathy_xyz_path=values["-XYZ_PATH-"] if values["-XYZ_PATH-"].strip() else None,
        bathy_tif_path=values["-TIF_PATH-"] if values["-TIF_PATH-"].strip() else None,
        is_utm=values["-IS_UTM-"],
        use_online_bathymetry=values["-USE_ONLINE_BATHY-"],
        blend_online_bathymetry=values["-BLEND_ONLINE_BATHY-"],
        dx=float(values["-DX-"]),
        dy=float(values["-DY-"]),
        buffer_cells=int(values["-BUFFER-"]),
        hmin=float(values["-HMIN-"]),
        hmax=float(values["-HMAX-"]),
        wave_period=float(values["-WAVE_T-"]),
        strategy=values["-STRATEGY-"],
        smooth_sigma_pixels=float(values["-SMOOTH_SIGMA-"]),
        z_convention=values["-Z_CONV-"],
        depth_limit=float(values["-DEPTH_LIMIT-"]),
        marker_strategy=values["-MARKER_STRAT-"],
    )

def fill_gui_from_config(window, cfg: MeshConfig):
    window["-PROJ_NAME-"].update(cfg.project_name)
    window["-OUT_DIR-"].update(cfg.output_dir)
    window["-WORK_CRS-"].update(cfg.work_crs)
    window["-OUT_CRS-"].update(cfg.output_crs)
    window["-DOMAIN_PATH-"].update(cfg.domain_path)
    window["-XYZ_PATH-"].update(cfg.bathy_xyz_path or "")
    window["-TIF_PATH-"].update(cfg.bathy_tif_path or "")
    window["-IS_UTM-"].update(cfg.is_utm)
    window["-USE_ONLINE_BATHY-"].update(cfg.use_online_bathymetry)
    window["-BLEND_ONLINE_BATHY-"].update(cfg.blend_online_bathymetry)
    window["-DX-"].update(str(cfg.dx))
    window["-DY-"].update(str(cfg.dy))
    window["-BUFFER-"].update(str(cfg.buffer_cells))
    window["-HMIN-"].update(str(cfg.hmin))
    window["-HMAX-"].update(str(cfg.hmax))
    window["-WAVE_T-"].update(str(cfg.wave_period))
    window["-STRATEGY-"].update(cfg.strategy)
    window["-SMOOTH_SIGMA-"].update(str(cfg.smooth_sigma_pixels))
    window["-Z_CONV-"].update(cfg.z_convention)
    window["-DEPTH_LIMIT-"].update(str(cfg.depth_limit))
    window["-MARKER_STRAT-"].update(cfg.marker_strategy)

def main():
    window = create_main_window()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        elif event == "-LOAD_YAML-":
            f_path = sg.popup_get_file("Select YAML config file", file_types=(("YAML Files", "*.yaml *.yml"),))
            if f_path:
                try:
                    cfg = MeshConfig.from_yaml(f_path)
                    fill_gui_from_config(window, cfg)
                    print(f"Loaded config from {f_path}")
                except Exception as e:
                    show_copyable_error_popup("Error loading YAML config", str(e))

        elif event == "-SAVE_YAML-":
            f_path = sg.popup_get_file("Save YAML config file", save_as=True, file_types=(("YAML Files", "*.yaml *.yml"),))
            if f_path:
                try:
                    cfg = read_config_from_gui(values)
                    cfg.to_yaml(f_path)
                    print(f"Saved config to {f_path}")
                except Exception as e:
                    show_copyable_error_popup("Error saving YAML config", str(e))

        elif event == "-RUN_PIPELINE-":
            try:
                cfg = read_config_from_gui(values)
                print(f"Launching swanmesh pipeline for '{cfg.project_name}'...")
                window["-RUN_PIPELINE-"].update(disabled=True)
                worker = PipelineWorker(cfg, window)
                worker.start()
            except Exception as e:
                show_copyable_error_popup("Configuration Error", str(e))

        elif event == "-THREAD_PIPELINE_DONE-":
            res, err = values[event]
            window["-RUN_PIPELINE-"].update(disabled=False)
            if err:
                print(f"PIPELINE ERROR: {err}")
                show_copyable_error_popup("Pipeline Failed", err)
            else:
                print(f"PIPELINE COMPLETED! Outputs in {res.output_dir}")
                sg.popup("Pipeline Success", f"Mesh written to {res.output_dir}")

        elif event == "-RUN_CONVERT-":
            msh = values["-CONV_MSH-"]
            dem = values["-CONV_DEM-"]
            out_d = values["-OUT_DIR-"]
            bname = values["-CONV_BASE-"]
            dlim = float(values["-DEPTH_LIMIT-"])
            zconv = values["-Z_CONV-"]

            if not msh or not dem:
                sg.popup_error("Please specify both .msh file and DEM GeoTIFF.")
                continue

            print(f"Launching legacy conversion for {msh}...")
            window["-RUN_CONVERT-"].update(disabled=True)
            cw = ConvertWorker(msh, dem, out_d, bname, dlim, zconv, window)
            cw.start()

        elif event == "-THREAD_CONVERT_DONE-":
            out_p, err = values[event]
            window["-RUN_CONVERT-"].update(disabled=False)
            if err:
                print(f"CONVERSION ERROR: {err}")
                show_copyable_error_popup("Conversion Failed", err)
            else:
                print(f"CONVERSION COMPLETED! SWAN files in {out_p}")
                sg.popup("Conversion Success", f"SWAN files created in {out_p}")

        elif event == "-OPEN_OUT-":
            out_d = values["-OUT_DIR-"]
            if Path(out_d).exists():
                if sys.platform == "win32":
                    os.startfile(out_d)
                elif sys.platform == "darwin":
                    subprocess.run(["open", out_d], check=False)
                else:
                    subprocess.run(["xdg-open", out_d], check=False)

    window.close()

if __name__ == "__main__":
    main()
