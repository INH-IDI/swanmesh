"""FreeSimpleGUI application for swanmesh."""

import math
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
        [sg.Multiline(error_text, size=(80, 20), disabled=True, font=("Courier", 10), key="-ERR_TEXT-")],
        [sg.Button("Copy to Clipboard", key="-COPY-"), sg.Button("Close", key="-CLOSE-")],
    ]
    win = sg.Window(title, layout, modal=True, finalize=True)
    while True:
        ev, _ = win.read()
        if ev in (sg.WIN_CLOSED, "-CLOSE-"):
            break
        elif ev == "-COPY-":
            sg.clipboard_set(error_text)
            sg.popup_quick_message("Copied to clipboard!", background_color="green", text_color="white")
    win.close()

# State store for dynamic control points list in GUI session
current_control_points = []
loaded_config: MeshConfig | None = None


def _is_geographic_config(config: MeshConfig) -> bool:
    from swanmesh.geo_units import is_geographic_crs

    return is_geographic_crs(config.work_crs, is_utm=config.is_utm)


def _recommended_n_lambda(config: MeshConfig) -> float | None:
    if config.strategy != "dispersion_gradient" or not _is_geographic_config(config) or config.hmax <= 0:
        return None

    m_lon, m_lat = 101900.0, 111000.0
    m_per_degree = 0.5 * (m_lon + m_lat)
    deep_wavelength_native = 9.81 * config.wave_period**2 / (2.0 * math.pi * m_per_degree)
    target = deep_wavelength_native / (0.9 * config.hmax)
    return round(min(0.5, max(0.2, target)), 2)


def _configuration_warnings(config: MeshConfig) -> list[str]:
    warnings = []
    recommended = _recommended_n_lambda(config)
    if recommended is not None and config.n_lambda < recommended:
        warnings.append(
            f"n_lambda={config.n_lambda:g} satura el campo en hmax={config.hmax:g} "
            f"para T={config.wave_period:g} s en CRS geográfico. "
            f"Se recomienda n_lambda={recommended:g}."
        )

    if _is_geographic_config(config) and min(config.dx, config.dy) > 2.0 * config.hmin:
        warnings.append(
            "La resolución del raster dx/dy es demasiado gruesa para hmin; "
            "el campo de tamaño no podrá representar el refinamiento solicitado."
        )
    return warnings


SIZE_FIELD_HELP = """CAMPO DE TAMANO DE MALLA

El raster H(x,y) contiene el tamano objetivo de cada elemento que Gmsh
recibe como campo de fondo. H, hmin, hmax, dx y dy usan unidades nativas del
CRS de trabajo: grados para EPSG:4326 y metros para UTM.

VARIABLES COMUNES

D = abs(DEM): profundidad o magnitud de la elevacion del fondo [m].
T: periodo usado por la relacion de dispersion [s].
omega = 2*pi/T.
k: numero de onda que resuelve omega^2 = g*k*tanh(k*D).
L = 2*pi/k: longitud de onda [m], convertida a unidades del CRS.
E1 = clip(L_native / max(L_native), 0, 1): indicador relativo de profundidad.
S = |grad Z|: pendiente del fondo [m/m], calculada con dx_m y dy_m.
E2 = 1 - Phi((S - mu_S) / sigma_S): indicador relativo de pendiente.
Phi: funcion de distribucion acumulada normal.
w_s = weight_slope / (weight_slope + weight_depth).
w_d = weight_depth / (weight_slope + weight_depth).
Delta_h = hmax - hmin.

Todas las estrategias recortan el resultado a [hmin, hmax]. Una pendiente
alta reduce H; un fondo mas profundo normalmente aumenta H.

ESTRATEGIAS

1) product

H1 = hmin + E1^w_d * Delta_h
H2 = hmin + E1^w_d * E2^w_s * Delta_h

Usa profundidad relativa y pendiente. Es la estrategia recomendada para un
modelo espectral cuando la resolucion se debe controlar con hmin/hmax. No usa
n_lambda para fijar el tamano absoluto.

2) mean

H1 = hmin + w_d * E1 * Delta_h
H2 = hmin + (w_d*E1 + w_s*E2) * Delta_h

Produce una transicion mas suave y menos agresiva que product.

3) depth_weighted

H1 = hmin + E1^w_d * E2^w_s * Delta_h
H3 = hmin + (w_d*E1 + w_s*E2) * Delta_h
P = clip(D, 4, Pmax), donde Pmax = max(1, max(D)).
H2 = H1*(1 - P/Pmax) + H3*(P/Pmax)

Mezcla product y mean dando mas peso a la profundidad absoluta.

4) hybrid_smooth

A = clip(E1, 1e-6, 1)^(hybrid_exponent_a*w_d)
B = clip(E2, 1e-6, 1)^(hybrid_exponent_b*w_s)
H1 = hmin + A*Delta_h
H2 = hmin + A*B*Delta_h

Si smooth_sigma_pixels > 0, H2 se suaviza en espacio logaritmico mediante
un filtro gaussiano con ese sigma en pixeles del raster.

5) dispersion_gradient

H1 = clip(L_native / n_lambda, hmin, hmax)
H2 = clip(H1 / (1 + alpha_grad*S), hmin, hmax)

n_lambda solo controla esta estrategia: valores mayores producen H menor.
alpha_grad controla cuanto refina la pendiente. En un modelo espectral no
debe usarse para representar el espectro; si se elige esta estrategia queda
como un ajuste numerico de resolucion.

6) relative_depth (swell, recomendada para este modelo)

L0 = g*T^2/(2*pi): longitud de onda de referencia en aguas profundas [m].
rel = D / L0: profundidad relativa con la cota de fondo.
F = clip((rel - 0.05)/(0.5 - 0.05), 0, 1)

H1 = hmin + F*(hmax - hmin)
  rel >= 0.5  (aguas profundas) -> H1 = hmax
  rel <= 0.05 (aguas someras)    -> H1 = hmin
  resto: interpolacion lineal entre hmin y hmax.

Pasada de pendiente (ponderador con zona muerta):
s_ref = slope_mean si esta definido, si no la media del raster de pendiente.
E = max(S - s_ref, 0) / s_ref
W = 1 / (1 + alpha_grad*E)
H2 = H1 * W

Donde la pendiente esta por debajo de s_ref el ponderador vale 1 y la
malla queda igual; donde la pendiente supera s_ref se refina en forma
proporcional al exceso, controlado por alpha_grad.

PUNTOS DE CONTROL

Para distancia r y radio R:
q = clip(r/R, 0, 1)
W = (1-q)^4 * (1+4q)       (kernel Wendland C2)
Hcp = Hbase*(1-W) + hmin_cp*W

combiner=min aplica el minimo entre H y Hcp. En CRS geografico, un hmin_cp
>= 1 se interpreta como metros y se convierte a grados; el radio R siempre
se interpreta en unidades nativas del CRS.

GRADACION (mesh_growth)

mesh_growth impone un crecimiento geometrico maximo entre celdas vecinas:
H_i = min(H_i, H_j * mesh_growth^(d/H_j)), donde d es la distancia entre
celdas en unidades nativas del CRS. Solo reduce H en las zonas de transicion
(rellena con tamanos intermedios desde hmin hacia afuera) y preserva hmin.
1.2 produce un crecimiento lento; mesh_growth=1.0 desactiva la gradacion.

REGLA PRACTICA PARA ESTE MODELO

Para swell use strategy=relative_depth con el periodo del swell (p. ej.
T=20 s): la profundidad relativa fija sola el rango hmin/hmax y el
ponderador de pendiente refina el talud. Revise la estimacion de nodos
antes de ejecutar Gmsh. Subir hmin o reducir alpha_grad baja la cantidad
de nodos.
"""


def size_field_help_text() -> str:
    """Return the GUI help text for mesh-size strategies and variables."""
    return SIZE_FIELD_HELP


def create_main_window():
    sg.theme("DarkTeal6")

    # Tab 1: Project & Inputs & ROI
    tab1_layout = [
        [sg.Text("Project Name:", size=(18, 1)), sg.Input("swan_mesh", key="-PROJ_NAME-")],
        [sg.Text("Output Directory:", size=(18, 1)), sg.Input("./output", key="-OUT_DIR-"), sg.FolderBrowse()],
        [sg.Checkbox("Overwrite existing outputs", default=True, key="-OVERWRITE-"),
         sg.Checkbox("Export .msh sidecar", default=True, key="-EXPORT_MSH-"),
         sg.Checkbox("Export plots", default=True, key="-EXPORT_PLOTS-")],
        [sg.Text("Work CRS:", size=(18, 1)), sg.Input("EPSG:4326", key="-WORK_CRS-")],
        [sg.Text("Output CRS:", size=(18, 1)), sg.Input("EPSG:4326", key="-OUT_CRS-")],
        [sg.Text("Domain Shapefile:", size=(18, 1)), sg.Input("", key="-DOMAIN_PATH-"), sg.FileBrowse(file_types=(("Vector Files", "*.shp *.geojson *.gpkg"),))],
        [sg.Text("Bathymetry XYZ Files:", size=(18, 1)), sg.Input("", key="-XYZ_PATHS-"), sg.FilesBrowse(file_types=(("XYZ / CSV Files", "*.csv *.xyz *.txt"),))],
        [sg.Text("Bathymetry TIF Files:", size=(18, 1)), sg.Input("", key="-TIF_PATHS-"), sg.FilesBrowse(file_types=(("GeoTIFF Files", "*.tif *.tiff"),))],
        [sg.Checkbox("XYZ is in UTM coordinates", default=False, key="-IS_UTM-")],
        [sg.Checkbox("Use Online Bathymetry (GEBCO / Open-Elevation)", default=False, key="-USE_ONLINE_BATHY-")],
        [sg.Checkbox("Blend Local Bathymetry with Online Background", default=False, key="-BLEND_ONLINE_BATHY-")],
        [sg.Frame("Subdomain ROI (Test Area Box)", [
            [sg.Checkbox("Crop to Subdomain ROI", default=False, key="-USE_ROI-")],
            [
                sg.Text("Xmin:"), sg.Input("", size=(10, 1), key="-ROI_XMIN-"),
                sg.Text("Ymin:"), sg.Input("", size=(10, 1), key="-ROI_YMIN-"),
                sg.Text("Xmax:"), sg.Input("", size=(10, 1), key="-ROI_XMAX-"),
                sg.Text("Ymax:"), sg.Input("", size=(10, 1), key="-ROI_YMAX-"),
            ]
        ])]
    ]

    # Tab 2: DEM & Mesh Size Field (2-Pass Wave Dispersion + Slope Gradient + TAM Radial Refinement)
    tab2_layout = [
        [sg.Text("Resolution dx (native CRS):", size=(24, 1)), sg.Input("0.002", key="-DX-"), sg.Text("[grados/m]", key="-UNIT_DX-", text_color="cyan")],
        [sg.Text("Resolution dy (native CRS):", size=(24, 1)), sg.Input("0.002", key="-DY-"), sg.Text("[grados/m]", key="-UNIT_DY-", text_color="cyan")],
        [sg.Text("Buffer Cells:", size=(24, 1)), sg.Input("20", key="-BUFFER-")],
        [sg.Text("Metodología / Estrategia:", size=(24, 1)), sg.Combo(["product", "relative_depth", "dispersion_gradient", "mean", "depth_weighted", "hybrid_smooth"], default_value="product", key="-STRATEGY-")],
        [sg.Text("Wave Period T (s):", size=(24, 1)), sg.Input("30.0", key="-WAVE_T-")],
        [sg.Text("N_lambda (solo dispersion_gradient):", size=(28, 1)), sg.Input("0.3", key="-NLAMBDA-"),
         sg.Text("no controla product/mean/depth_weighted", text_color="yellow")],
        [sg.Text("Gradient Factor (alpha_grad):", size=(24, 1)), sg.Input("1.0", key="-ALPHA_GRAD-")],
        [sg.Text("Minimum Mesh Size hmin:", size=(24, 1)), sg.Input("0.001", key="-HMIN-"), sg.Text("[grados/m]", key="-UNIT_HMIN-", text_color="cyan")],
        [sg.Text("Maximum Mesh Size hmax:", size=(24, 1)), sg.Input("0.05", key="-HMAX-"), sg.Text("[grados/m]", key="-UNIT_HMAX-", text_color="cyan")],
        [sg.Frame("Estimación Aproximada de Malla (Nodos y Elementos)", [
            [sg.Button("Calcular Estimación Nodos", key="-ESTIMATE_NODES-"), sg.Text("Haga clic para estimar cantidad...", key="-EST_TEXT-", font=("Helvetica", 10, "bold"), text_color="yellow")],
            [sg.Text("max_est_nodes:"), sg.Input("500000", size=(10, 1), key="-MAX_EST_NODES-"),
             sg.Checkbox("Abortar si se excede", default=True, key="-ABORT_EST-")],
        ])],
        [sg.Frame("Limpieza y Reducción de Puntos por Cuadrícula", [
            [sg.Checkbox("Activar reducción de puntos en batimetría", default=True, key="-ENABLE_POINT_RED-")],
            [sg.Text("Máximo puntos por cuadrícula (dx/dy):"), sg.Input("5", size=(8, 1), key="-MAX_PTS_PER_CELL-")],
        ])],
        [
            sg.Button("Previsualizar Campo de Tamaño (1D/2D lc)", key="-OPEN_LC_PREVIEW-", button_color=("white", "blue")),
            sg.Button("Abrir Perfil en Plotly (Vectorial / Zoom)", key="-OPEN_PLOTLY-", button_color=("white", "purple")),
            sg.Button("Ayuda: formulas y variables", key="-HELP_SIZE_FIELD-", button_color=("white", "darkblue")),
        ],
        [sg.Frame("TAM Control Points (Optional Radial Refinement)", [
            [
                sg.Text("X:"), sg.Input("", size=(10, 1), key="-CP_X-"),
                sg.Text("Y:"), sg.Input("", size=(10, 1), key="-CP_Y-"),
                sg.Text("hmin (m si geo):"), sg.Input("", size=(8, 1), key="-CP_HMIN-"),
                sg.Text("Radio R (CRS):"), sg.Input("", size=(8, 1), key="-CP_R-"),
                sg.Button("Add Point", key="-ADD_CP-"),
                sg.Button("Clear Points", key="-CLEAR_CPS-"),
            ],
            [sg.Listbox(values=[], size=(65, 3), key="-CPS_LIST-")],
        ])]
    ]

    # Tab 3: Boundaries, Mesher & Topology Quality
    tab3_layout = [
        [sg.Text("Z Convention:", size=(22, 1)), sg.Combo(["elevation_negative_down", "depth_positive_down"], default_value="elevation_negative_down", key="-Z_CONV-")],
        [sg.Text("Depth Limit for Marker 2:", size=(22, 1)), sg.Input("-50.0", key="-DEPTH_LIMIT-")],
        [sg.Text("Marker Strategy:", size=(22, 1)), sg.Combo(["depth_limit", "open_boundary_lines"], default_value="depth_limit", key="-MARKER_STRAT-")],
        [sg.Text("Gmsh 2D Algorithm:", size=(22, 1)), sg.Combo(["6: Frontal-Delaunay", "5: Delaunay", "1: MeshAdapt", "7: BAMG"], default_value="6: Frontal-Delaunay", key="-GMSH_ALG-")],
        [sg.Checkbox("Optimize with Netgen + Relocate2D", default=True, key="-OPTIM_NETGEN-")],
        [sg.Checkbox("Enforce Minimum Node Degree >= 3 on Boundary", default=True, key="-ENFORCE_DEGREE-")],
        [sg.Text("Gmsh smoothing steps:", size=(32, 1)), sg.Input("3", size=(8, 1), key="-GMSH_SMOOTHING-"),
         sg.Text("Minimum node degree:", size=(22, 1)), sg.Input("3", size=(8, 1), key="-MIN_NODE_DEGREE-")],
        [sg.Text("Max boundary points (resample cap):", size=(32, 1)), sg.Input("800", key="-MAX_BND_PTS-")],
        [sg.Text("Mesh growth factor (1.2 = slow):", size=(32, 1)), sg.Input("1.2", size=(8, 1), key="-MESH_GROWTH-")],
        [sg.Text("Open-boundary buffer (optional):", size=(32, 1)), sg.Input("", key="-OPEN_BND_BUF-")],
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
                    [sg.Tab("Boundaries & Quality", tab3_layout)],
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
    if not values or not isinstance(values, dict):
        raise ValueError("No se pudieron leer los valores de la interfaz gráfica.")

    def parse_float(key, default):
        val = str(values.get(key, default)).strip().replace(",", ".")
        try:
            return float(val)
        except Exception:
            return float(default)

    def parse_int(key, default):
        val = str(values.get(key, default)).strip()
        try:
            return int(val)
        except Exception:
            return int(default)

    subdomain_bbox = None
    if values.get("-USE_ROI-") and str(values.get("-ROI_XMIN-", "")).strip():
        try:
            subdomain_bbox = [
                parse_float("-ROI_XMIN-", 0.0),
                parse_float("-ROI_YMIN-", 0.0),
                parse_float("-ROI_XMAX-", 0.0),
                parse_float("-ROI_YMAX-", 0.0),
            ]
        except Exception as e:
            raise ValueError(f"Coordenadas de ROI inválidas: {e}")

    alg_str = values.get("-GMSH_ALG-", "6")
    alg_code = 6
    if "1:" in alg_str:
        alg_code = 1
    elif "5:" in alg_str:
        alg_code = 5
    elif "7:" in alg_str:
        alg_code = 7

    raw_xyz = values.get("-XYZ_PATHS-", "").strip()
    xyz_paths = [p.strip() for p in raw_xyz.replace(";", " ").split() if p.strip()]

    raw_tif = values.get("-TIF_PATHS-", "").strip()
    tif_paths = [p.strip() for p in raw_tif.replace(";", " ").split() if p.strip()]

    base_data = loaded_config.model_dump() if loaded_config is not None else {}
    max_pts = parse_int("-MAX_PTS_PER_CELL-", 5)
    n_lambda = parse_float("-NLAMBDA-", 0.3)
    alpha_grad = parse_float("-ALPHA_GRAD-", 1.0)
    open_buf_raw = str(values.get("-OPEN_BND_BUF-", "")).strip()
    open_buf = float(open_buf_raw.replace(",", ".")) if open_buf_raw else None

    gui_data = {
        "project_name": values.get("-PROJ_NAME-", "swan_mesh"),
        "output_dir": values.get("-OUT_DIR-", "./output"),
        "overwrite": bool(values.get("-OVERWRITE-", True)),
        "work_crs": values.get("-WORK_CRS-", "EPSG:4326"),
        "output_crs": values.get("-OUT_CRS-", "EPSG:4326"),
        "domain_path": values.get("-DOMAIN_PATH-", ""),
        "bathy_xyz_paths": xyz_paths,
        "bathy_tif_paths": tif_paths,
        "bathy_xyz_path": xyz_paths[0] if xyz_paths else None,
        "bathy_tif_path": tif_paths[0] if tif_paths else None,
        "is_utm": bool(values.get("-IS_UTM-", False)),
        "use_online_bathymetry": bool(values.get("-USE_ONLINE_BATHY-", False)),
        "blend_online_bathymetry": bool(values.get("-BLEND_ONLINE_BATHY-", False)),
        "subdomain_bbox": subdomain_bbox,
        "dx": parse_float("-DX-", 0.002),
        "dy": parse_float("-DY-", 0.002),
        "buffer_cells": parse_int("-BUFFER-", 20),
        "enable_point_reduction": bool(values.get("-ENABLE_POINT_RED-", True)),
        "max_points_per_cell": max_pts,
        "hmin": parse_float("-HMIN-", 0.001),
        "hmax": parse_float("-HMAX-", 0.05),
        "wave_period": parse_float("-WAVE_T-", 30.0),
        "n_lambda": n_lambda,
        "alpha_grad": alpha_grad,
        "strategy": values.get("-STRATEGY-", "product"),
        "interest_points": current_control_points,
        "z_convention": values.get("-Z_CONV-", "elevation_negative_down"),
        "depth_limit": parse_float("-DEPTH_LIMIT-", -50.0),
        "marker_strategy": values.get("-MARKER_STRAT-", "depth_limit"),
        "open_boundary_buffer": open_buf,
        "gmsh_algorithm_2d": alg_code,
        "gmsh_optimize_netgen": bool(values.get("-OPTIM_NETGEN-", True)),
        "gmsh_smoothing_steps": parse_int("-GMSH_SMOOTHING-", 3),
        "enforce_min_node_degree": bool(values.get("-ENFORCE_DEGREE-", True)),
        "min_node_degree": parse_int("-MIN_NODE_DEGREE-", 3),
        "max_boundary_points": parse_int("-MAX_BND_PTS-", 800),
        "mesh_growth": parse_float("-MESH_GROWTH-", 1.2),
        "max_est_nodes": parse_int("-MAX_EST_NODES-", 500000),
        "abort_on_est_nodes": bool(values.get("-ABORT_EST-", True)),
        "export_msh": bool(values.get("-EXPORT_MSH-", True)),
        "export_plots": bool(values.get("-EXPORT_PLOTS-", True)),
    }
    return MeshConfig(**{**base_data, **gui_data})

def fill_gui_from_config(window, cfg: MeshConfig):
    global current_control_points, loaded_config
    loaded_config = cfg.model_copy(deep=True)
    window["-PROJ_NAME-"].update(cfg.project_name)
    window["-OUT_DIR-"].update(cfg.output_dir)
    window["-OVERWRITE-"].update(cfg.overwrite)
    window["-EXPORT_MSH-"].update(cfg.export_msh)
    window["-EXPORT_PLOTS-"].update(cfg.export_plots)
    window["-WORK_CRS-"].update(cfg.work_crs)
    window["-OUT_CRS-"].update(cfg.output_crs)
    window["-DOMAIN_PATH-"].update(cfg.domain_path)

    xyz_str = "; ".join(cfg.bathy_xyz_paths) if cfg.bathy_xyz_paths else (cfg.bathy_xyz_path or "")
    tif_str = "; ".join(cfg.bathy_tif_paths) if cfg.bathy_tif_paths else (cfg.bathy_tif_path or "")
    window["-XYZ_PATHS-"].update(xyz_str)
    window["-TIF_PATHS-"].update(tif_str)

    window["-IS_UTM-"].update(cfg.is_utm)
    window["-USE_ONLINE_BATHY-"].update(cfg.use_online_bathymetry)
    window["-BLEND_ONLINE_BATHY-"].update(cfg.blend_online_bathymetry)

    if cfg.subdomain_bbox and len(cfg.subdomain_bbox) == 4:
        window["-USE_ROI-"].update(True)
        window["-ROI_XMIN-"].update(str(cfg.subdomain_bbox[0]))
        window["-ROI_YMIN-"].update(str(cfg.subdomain_bbox[1]))
        window["-ROI_XMAX-"].update(str(cfg.subdomain_bbox[2]))
        window["-ROI_YMAX-"].update(str(cfg.subdomain_bbox[3]))
    else:
        window["-USE_ROI-"].update(False)

    window["-DX-"].update(str(cfg.dx))
    window["-DY-"].update(str(cfg.dy))
    window["-BUFFER-"].update(str(cfg.buffer_cells))
    window["-ENABLE_POINT_RED-"].update(cfg.enable_point_reduction)
    window["-MAX_PTS_PER_CELL-"].update(str(cfg.max_points_per_cell))
    window["-HMIN-"].update(str(cfg.hmin))
    window["-HMAX-"].update(str(cfg.hmax))
    window["-STRATEGY-"].update(cfg.strategy)
    window["-WAVE_T-"].update(str(cfg.wave_period))
    window["-NLAMBDA-"].update(str(cfg.n_lambda))
    window["-ALPHA_GRAD-"].update(str(cfg.alpha_grad))
    window["-MAX_EST_NODES-"].update(str(cfg.max_est_nodes))
    window["-ABORT_EST-"].update(cfg.abort_on_est_nodes)
    window["-Z_CONV-"].update(cfg.z_convention)
    window["-DEPTH_LIMIT-"].update(str(cfg.depth_limit))
    window["-MARKER_STRAT-"].update(cfg.marker_strategy)
    algorithm_labels = {
        1: "1: MeshAdapt",
        5: "5: Delaunay",
        6: "6: Frontal-Delaunay",
        7: "7: BAMG",
    }
    window["-GMSH_ALG-"].update(algorithm_labels.get(cfg.gmsh_algorithm_2d, str(cfg.gmsh_algorithm_2d)))
    window["-OPTIM_NETGEN-"].update(cfg.gmsh_optimize_netgen)
    window["-ENFORCE_DEGREE-"].update(cfg.enforce_min_node_degree)
    window["-GMSH_SMOOTHING-"].update(str(cfg.gmsh_smoothing_steps))
    window["-MIN_NODE_DEGREE-"].update(str(cfg.min_node_degree))
    window["-MAX_BND_PTS-"].update(str(cfg.max_boundary_points))
    window["-MESH_GROWTH-"].update(str(cfg.mesh_growth))
    window["-OPEN_BND_BUF-"].update("" if cfg.open_boundary_buffer is None else str(cfg.open_boundary_buffer))

    current_control_points = list(cfg.interest_points)
    display_vals = [f"X={ip.x}, Y={ip.y}, hmin={ip.hmin}, R={ip.radius}" for ip in current_control_points]
    window["-CPS_LIST-"].update(display_vals)

def open_lc_preview_dialog(main_window, main_values):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from scipy.stats import norm

    from swanmesh.lc_profile import (
        LcRasterResult,
        export_pos_field,
        generate_lc_raster,
        plot_lc_preview,
    )

    try:
        cfg = read_config_from_gui(main_values)
    except Exception as e:
        sg.popup_error(f"Error al leer configuración de la interfaz: {e}")
        return

    period = cfg.wave_period
    n_lambda = cfg.n_lambda
    alpha_grad = cfg.alpha_grad
    min_lc = cfg.hmin
    max_lc = cfg.hmax

    # Check CRS type for unit labels (degrees vs meters)
    crs_str = (cfg.work_crs or "").upper()
    is_geo = ("4326" in crs_str) or (not cfg.is_utm and "UTM" not in crs_str)
    unit_str = "grados (°)" if is_geo else "metros (m)"

    # Estimate domain X extent Lx = Xmax - Xmin from domain Shapefile or ROI
    domain_lx = None
    if cfg.domain_path and Path(cfg.domain_path).exists():
        try:
            from swanmesh.domain import load_domain
            dom = load_domain(cfg.domain_path, work_crs=cfg.work_crs)
            if cfg.subdomain_bbox and len(cfg.subdomain_bbox) == 4:
                dom = dom.crop_to_subdomain(cfg.subdomain_bbox)
            b = dom.get_bounds(buffer_cells=cfg.buffer_cells, dx=cfg.dx, dy=cfg.dy)
            domain_lx = float(b[2] - b[0])
        except Exception:
            pass

    if domain_lx is None or domain_lx <= 0:
        domain_lx = 2000.0 if not is_geo else (max_lc * 40.0 if max_lc > 0 else 0.05)

    h_max_val = 100.0

    layout = [
        [sg.Text(f"Parametros de Campo de Tamaño lc — Unidades ({unit_str})", font=("Helvetica", 12, "bold"))],
        [
            sg.Text(f"Ancho Dominio L_x (Xmax - Xmin) [{unit_str}]:", size=(28, 1)),
            sg.Input(str(round(domain_lx, 4)), size=(10, 1), key="-PREV_LX-"),
            sg.Text("Profundidad Max h_max (m):", size=(22, 1)),
            sg.Input(str(h_max_val), size=(10, 1), key="-PREV_HMAX_BATHY-"),
        ],
        [
            sg.Text("Estrategia / Metodología:", size=(28, 1)),
            sg.Combo(["relative_depth", "product", "dispersion_gradient", "mean", "depth_weighted", "hybrid_smooth"], default_value=cfg.strategy, key="-PREV_STRATEGY-"),
            sg.Text("Periodo T (s):", size=(22, 1)),
            sg.Input(str(period), size=(10, 1), key="-PREV_PERIOD-"),
        ],
        [
            sg.Text("N_lambda (nodos/onda):", size=(28, 1)),
            sg.Input(str(n_lambda), size=(10, 1), key="-PREV_NLAMBDA-"),
            sg.Text("Factor Gradiente alpha:", size=(22, 1)),
            sg.Input(str(alpha_grad), size=(10, 1), key="-PREV_ALPHA-"),
        ],
        [
            sg.Text(f"Min lc ({unit_str}):", size=(28, 1)),
            sg.Input(str(min_lc), size=(10, 1), key="-PREV_MIN_LC-"),
            sg.Text(f"Max lc ({unit_str}):", size=(22, 1)),
            sg.Input(str(max_lc), size=(10, 1), key="-PREV_MAX_LC-"),
        ],
        [sg.Frame("Perfil Representativo de Terreno / Batimetría", [
            [
                sg.Text("Fuente de Perfil:"),
                sg.Combo(["Sintético (Plataforma Continental)", "DEM Real (Corte Vectorial i,j)"], default_value="Sintético (Plataforma Continental)", key="-PREV_PROFILE_SRC-"),
                sg.Text("Dir i:"), sg.Input("1.0", size=(6, 1), key="-PREV_DIR_I-"),
                sg.Text("Dir j:"), sg.Input("0.0", size=(6, 1), key="-PREV_DIR_J-"),
            ]
        ])],
        [sg.Frame("Pasada 3: Punto de Control (Kernel SPH Compacto Wendland C2)", [
            [
                sg.Text(f"Posición X ({unit_str}):"), sg.Input(str(round(domain_lx * 0.5, 4)), size=(8, 1), key="-PREV_CP_X-"),
                sg.Text(f"Local min lc ({unit_str}):"), sg.Input(str(min_lc), size=(8, 1), key="-PREV_CP_HMIN-"),
                sg.Text(f"Radio R ({unit_str}):"), sg.Input(str(round(domain_lx * 0.2, 4)), size=(8, 1), key="-PREV_CP_R-"),
            ]
        ])],
        [
            sg.Button("Actualizar Vista Previa", key="-PREV_UPDATE-", button_color=("white", "blue")),
            sg.Button("Abrir Perfil Interactivo en Plotly (HTML con Zoom)", key="-PREV_OPEN_PLOTLY-", button_color=("white", "purple")),
            sg.Button("Aplicar y Guardar Parámetros en GUI Principal", key="-PREV_APPLY-", button_color=("white", "darkgreen")),
            sg.Button("Exportar Archivo .pos para Gmsh", key="-PREV_EXPORT_POS-", button_color=("white", "gray")),
            sg.Button("Cerrar", key="-PREV_CLOSE-"),
        ],
        [sg.Canvas(key="-PREV_CANVAS-", size=(850, 480))],
        [sg.Text("", key="-PREV_STATUS-", size=(95, 1), font=("Helvetica", 10, "bold"), text_color="yellow")],
    ]

    prev_win = sg.Window("swanmesh — Previsualizador y Sincronizador de Campo de Tamaño lc", layout, modal=True, finalize=True)

    fig_agg = None
    last_result = None
    cached_profile = {}

    def draw_fig(canvas_widget, figure):
        nonlocal fig_agg
        if fig_agg is not None:
            try:
                fig_agg.get_tk_widget().forget()
                fig_agg.get_tk_widget().destroy()
            except Exception:
                pass
            plt.close("all")
        fig_agg = FigureCanvasTkAgg(figure, canvas_widget)
        fig_agg.draw()
        fig_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        return fig_agg

    def safe_parse(prev_win, key, default):
        val = str(prev_win[key].get()).strip().replace(",", ".")
        try:
            return float(val)
        except Exception:
            return float(default)

    def render():
        nonlocal last_result
        try:
            l_x = safe_parse(prev_win, "-PREV_LX-", domain_lx)
            p = safe_parse(prev_win, "-PREV_PERIOD-", period)
            nl = safe_parse(prev_win, "-PREV_NLAMBDA-", n_lambda)
            ag = safe_parse(prev_win, "-PREV_ALPHA-", alpha_grad)
            mn_lc = safe_parse(prev_win, "-PREV_MIN_LC-", min_lc)
            mx_lc = safe_parse(prev_win, "-PREV_MAX_LC-", max_lc)
            h_m = safe_parse(prev_win, "-PREV_HMAX_BATHY-", h_max_val)
            strat = prev_win["-PREV_STRATEGY-"].get()

            dir_i = safe_parse(prev_win, "-PREV_DIR_I-", 1.0)
            dir_j = safe_parse(prev_win, "-PREV_DIR_J-", 0.0)
            profile_src = prev_win["-PREV_PROFILE_SRC-"].get()

            cp_x = safe_parse(prev_win, "-PREV_CP_X-", l_x * 0.5)
            cp_hmin = safe_parse(prev_win, "-PREV_CP_HMIN-", min_lc)
            cp_r = safe_parse(prev_win, "-PREV_CP_R-", 0.0)
            cp_list = [{"x": cp_x, "hmin": cp_hmin, "radius": cp_r}] if cp_r > 0 else None

            # Memoize/Cache extracted profile to prevent re-interpolating DEM on every slider/input tweak
            cache_key = (
                str(profile_src), dir_i, dir_j, round(l_x, 6), round(h_m, 2),
                tuple(cfg.bathy_xyz_paths or []), tuple(cfg.bathy_tif_paths or []), str(cfg.domain_path)
            )

            if cache_key in cached_profile:
                x, h_prof, l_x = cached_profile[cache_key]
            else:
                h_prof = None
                x = np.linspace(0, l_x, 250, dtype=np.float32)

                if "DEM Real" in str(profile_src):
                    try:
                        from swanmesh.pipeline import build_dem
                        dem = build_dem(cfg)
                        s, _, _, h_sampled = dem.sample_profile(dir_vector=(dir_i, dir_j), num_points=250)
                        if len(h_sampled) == len(x):
                            h_prof = h_sampled
                            l_x = float(s.max() - s.min())
                            x = s
                    except Exception as ex:
                        prev_win["-PREV_STATUS-"].update(f"Aviso DEM: {ex}. Usando perfil sintético.")

                if h_prof is None:
                    h_prof = h_m - (h_m * 0.95) / (1.0 + np.exp(-(x - l_x / 2.0) / (l_x / 15.0)))

                cached_profile[cache_key] = (x, h_prof, l_x)

            res = generate_lc_raster(
                x=x, h=h_prof, period=p, n_lambda=nl, alpha_grad=ag, min_lc=mn_lc, max_lc=mx_lc, control_points=cp_list,
                is_geographic=is_geo
            )

            # Recalculate based on chosen strategy if product/mean/depth_weighted
            h_span = mx_lc - mn_lc
            if strat == "product":
                max_wl = float(np.max(res.wavelength)) if np.max(res.wavelength) > 0 else 1.0
                e1 = np.clip(res.wavelength / max_wl, 0.0, 1.0)
                norm_s = (res.grad_h - np.mean(res.grad_h)) / (np.std(res.grad_h) if np.std(res.grad_h) > 1e-6 else 1.0)
                e2 = 1.0 - norm.cdf(norm_s)
                res_lc1 = mn_lc + (e1 ** 0.3) * h_span
                res_lc2 = mn_lc + (e1 ** 0.3) * (e2 ** 0.7) * h_span
                res_lc3 = res_lc2.copy()
                if cp_list:
                    for ip in cp_list:
                        dist = np.abs(res.x - ip["x"])
                        w_k = np.clip(1.0 - dist / max(1e-6, ip["radius"]), 0.0, 1.0) ** 2
                        res_lc3 = np.minimum(res_lc3, res_lc3 * (1.0 - w_k) + ip["hmin"] * w_k)
                res = LcRasterResult(x=res.x, h=res.h, grad_h=res.grad_h, wavelength=res.wavelength, lc1=res_lc1, lc2=res_lc2, lc3=res_lc3, dx=res.dx)

            elif strat == "relative_depth":
                l0 = 9.81 * p**2 / (2.0 * np.pi)
                rel = np.clip(res.h / max(l0, 1.0), 0.0, None)
                frac = np.clip((rel - 0.05) / 0.45, 0.0, 1.0)
                res_lc1 = mn_lc + frac * h_span
                s_ref = float(np.mean(res.grad_h)) if np.mean(res.grad_h) > 0 else 1e-12
                excess = np.maximum(res.grad_h - s_ref, 0.0) / max(s_ref, 1e-12)
                res_lc2 = res_lc1 / (1.0 + ag * excess)
                res_lc3 = res_lc2.copy()
                if cp_list:
                    for ip in cp_list:
                        dist = np.abs(res.x - ip["x"])
                        w_k = np.clip(1.0 - dist / max(1e-6, ip["radius"]), 0.0, 1.0) ** 2
                        res_lc3 = np.minimum(res_lc3, res_lc3 * (1.0 - w_k) + ip["hmin"] * w_k)
                res = LcRasterResult(x=res.x, h=res.h, grad_h=res.grad_h, wavelength=res.wavelength, lc1=res_lc1, lc2=res_lc2, lc3=res_lc3, dx=res.dx)

            last_result = res
            fig = plot_lc_preview(
                x=res.x, h=res.h, lc1=res.lc1, lc2=res.lc2, lc3=res.lc3, grad_h=res.grad_h,
                min_lc=mn_lc, max_lc=mx_lc, period=p,
                is_geographic=is_geo, unit_str=unit_str,
                title=f"Campo de Tamaño lc ({unit_str}) — Estrategia: {strat} | L_x={l_x:.4f}, T={p}s, N_lambda={nl}"
            )

            # Node estimation over 2D domain of extent Lx * (3.2 * Lx)
            dom_ly = l_x * 3.2
            dom_area = l_x * dom_ly
            avg_h2 = np.mean(res.lc3 ** 2)
            est_nodes_prev = int(round(dom_area / (np.sqrt(3)/2 * avg_h2))) if avg_h2 > 0 else 0

            canvas_elem = prev_win["-PREV_CANVAS-"].TKCanvas
            draw_fig(canvas_elem, fig)
            plt.close(fig)
            prev_win["-PREV_STATUS-"].update(
                f"[{strat}] Nodos Estimados: ~{est_nodes_prev:,} | Rango lc: [{res.lc3.min():.6f}, {res.lc3.max():.6f}] {unit_str}"
            )
        except Exception as err:
            prev_win["-PREV_STATUS-"].update(f"Error en vista previa: {err}")

    render()

    while True:
        ev, vals = prev_win.read()
        if ev in (sg.WIN_CLOSED, "-PREV_CLOSE-"):
            if vals is not None:
                try:
                    main_window["-STRATEGY-"].update(prev_win["-PREV_STRATEGY-"].get())
                    main_window["-WAVE_T-"].update(prev_win["-PREV_PERIOD-"].get())
                    main_window["-NLAMBDA-"].update(prev_win["-PREV_NLAMBDA-"].get())
                    main_window["-ALPHA_GRAD-"].update(prev_win["-PREV_ALPHA-"].get())
                    main_window["-HMIN-"].update(prev_win["-PREV_MIN_LC-"].get())
                    main_window["-HMAX-"].update(prev_win["-PREV_MAX_LC-"].get())
                except Exception:
                    pass
            break

        if ev == "-PREV_UPDATE-":
            render()

        elif ev == "-PREV_OPEN_PLOTLY-":
            try:
                import webbrowser

                from swanmesh.lc_profile import plot_lc_preview_plotly
                p = safe_parse(prev_win, "-PREV_PERIOD-", period)
                nl = safe_parse(prev_win, "-PREV_NLAMBDA-", n_lambda)
                ag = safe_parse(prev_win, "-PREV_ALPHA-", alpha_grad)
                mn_lc = safe_parse(prev_win, "-PREV_MIN_LC-", min_lc)
                mx_lc = safe_parse(prev_win, "-PREV_MAX_LC-", max_lc)
                h_m = safe_parse(prev_win, "-PREV_HMAX_BATHY-", h_max_val)
                cp_x = safe_parse(prev_win, "-PREV_CP_X-", domain_lx * 0.5)
                cp_hmin = safe_parse(prev_win, "-PREV_CP_HMIN-", min_lc)
                cp_r = safe_parse(prev_win, "-PREV_CP_R-", 0.0)
                cp_list = [{"x": cp_x, "hmin": cp_hmin, "radius": cp_r}] if cp_r > 0 else None

                if last_result is None:
                    prev_win["-PREV_STATUS-"].update("Primero actualice la vista previa.")
                    continue

                profile_src = prev_win["-PREV_PROFILE_SRC-"].get()
                out_html = Path(cfg.output_dir) / "swan_mesh_lc_profile.html"
                out_html.parent.mkdir(parents=True, exist_ok=True)
                plot_lc_preview_plotly(
                    x=last_result.x,
                    h=last_result.h,
                    lc1=last_result.lc1,
                    lc2=last_result.lc2,
                    lc3=last_result.lc3,
                    grad_h=last_result.grad_h,
                    min_lc=mn_lc, max_lc=mx_lc, period=p,
                    is_geographic=is_geo, unit_str=unit_str,
                    title=f"Perfil Interactivo y Vectorial de Campo de Tamaño lc — {profile_src}",
                    output_html=out_html,
                )
                webbrowser.open(f"file://{out_html.resolve()}")
                prev_win["-PREV_STATUS-"].update(f"Perfil interactivo Plotly abierto en navegador: {out_html.resolve()}")
            except Exception as err:
                prev_win["-PREV_STATUS-"].update(f"Error Plotly: {err}")

        elif ev == "-PREV_APPLY-":
            try:
                main_window["-STRATEGY-"].update(prev_win["-PREV_STRATEGY-"].get())
                main_window["-WAVE_T-"].update(prev_win["-PREV_PERIOD-"].get())
                main_window["-NLAMBDA-"].update(prev_win["-PREV_NLAMBDA-"].get())
                main_window["-ALPHA_GRAD-"].update(prev_win["-PREV_ALPHA-"].get())
                main_window["-HMIN-"].update(prev_win["-PREV_MIN_LC-"].get())
                main_window["-HMAX-"].update(prev_win["-PREV_MAX_LC-"].get())
                sg.popup("Todos los parámetros se sincronizaron y guardaron en la interfaz principal.", title="Sincronización Exitosa")
                render()
            except Exception as e:
                sg.popup_error(f"Error al actualizar la interfaz principal: {e}")

        elif ev == "-PREV_EXPORT_POS-":
            try:
                p = safe_parse(prev_win, "-PREV_PERIOD-", period)
                nl = safe_parse(prev_win, "-PREV_NLAMBDA-", n_lambda)
                ag = safe_parse(prev_win, "-PREV_ALPHA-", alpha_grad)
                mn_lc = safe_parse(prev_win, "-PREV_MIN_LC-", min_lc)
                mx_lc = safe_parse(prev_win, "-PREV_MAX_LC-", max_lc)
                h_m = safe_parse(prev_win, "-PREV_HMAX_BATHY-", h_max_val)
                cp_x = safe_parse(prev_win, "-PREV_CP_X-", domain_lx * 0.5)
                cp_hmin = safe_parse(prev_win, "-PREV_CP_HMIN-", min_lc)
                cp_r = safe_parse(prev_win, "-PREV_CP_R-", 0.0)
                cp_list = [{"x": cp_x, "hmin": cp_hmin, "radius": cp_r}] if cp_r > 0 else None

                x_pts = np.linspace(0, domain_lx, 250, dtype=np.float32)
                h_prof = h_m - (h_m * 0.95) / (1.0 + np.exp(-(x_pts - domain_lx / 2.0) / (domain_lx / 15.0)))
                res = generate_lc_raster(
                    x=x_pts, h=h_prof, period=p, n_lambda=nl, alpha_grad=ag,
                    min_lc=mn_lc, max_lc=mx_lc, control_points=cp_list, is_geographic=is_geo
                )

                out_pos = Path("./lc_field_background.pos")
                export_pos_field(res.x, res.lc3, out_pos)
                prev_win["-PREV_STATUS-"].update(f"Archivo .pos exportado exitosamente en: {out_pos.resolve()}")
            except Exception as err:
                prev_win["-PREV_STATUS-"].update(f"Error al exportar .pos: {err}")

    prev_win.close()


def update_unit_labels(window, values):
    try:
        crs_str = (values.get("-WORK_CRS-", "") or "").upper()
        is_utm = values.get("-IS_UTM-", False)
        is_geo = ("4326" in crs_str) or (not is_utm and "UTM" not in crs_str)
        if is_geo:
            dx_val = float(values.get("-DX-", "0.002") or 0.002)
            dy_val = float(values.get("-DY-", "0.002") or 0.002)
            hmin_val = float(values.get("-HMIN-", "0.001") or 0.001)
            hmax_val = float(values.get("-HMAX-", "0.05") or 0.05)
            m_per_deg = 101900.0
            window["-UNIT_DX-"].update(f"[grados] (~{dx_val * m_per_deg:.1f} m)")
            window["-UNIT_DY-"].update(f"[grados] (~{dy_val * m_per_deg:.1f} m)")
            window["-UNIT_HMIN-"].update(f"[grados] (~{hmin_val * m_per_deg:.1f} m)")
            window["-UNIT_HMAX-"].update(f"[grados] (~{hmax_val * m_per_deg:.1f} m)")
        else:
            window["-UNIT_DX-"].update("[metros]")
            window["-UNIT_DY-"].update("[metros]")
            window["-UNIT_HMIN-"].update("[metros]")
            window["-UNIT_HMAX-"].update("[metros]")
    except Exception:
        pass


def main():
    window = create_main_window()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        update_unit_labels(window, values)

        if event == "-LOAD_YAML-":
            f_path = sg.popup_get_file("Select YAML config file", file_types=(("YAML Files", "*.yaml *.yml"),))
            if f_path:
                try:
                    cfg = MeshConfig.from_yaml(f_path)
                    fill_gui_from_config(window, cfg)
                    print(f"Loaded config from {f_path}")
                except Exception as e:
                    show_copyable_error_popup("Error loading YAML config", str(e))

        elif event == "-ADD_CP-":
            try:
                from swanmesh.config import InterestPointConfig
                x_val = float(values["-CP_X-"])
                y_val = float(values["-CP_Y-"])
                h_val = float(values["-CP_HMIN-"])
                r_val = float(values["-CP_R-"])
                new_ip = InterestPointConfig(name=f"cp_{len(current_control_points)+1}", x=x_val, y=y_val, hmin=h_val, radius=r_val)
                current_control_points.append(new_ip)
                display_vals = [f"X={ip.x}, Y={ip.y}, hmin={ip.hmin}, R={ip.radius}" for ip in current_control_points]
                window["-CPS_LIST-"].update(display_vals)
                window["-CP_X-"].update("")
                window["-CP_Y-"].update("")
                window["-CP_HMIN-"].update("")
                window["-CP_R-"].update("")
            except Exception as e:
                sg.popup_error(f"Invalid Control Point values: {e}")

        elif event == "-CLEAR_CPS-":
            current_control_points.clear()
            window["-CPS_LIST-"].update([])

        elif event == "-OPEN_LC_PREVIEW-":
            open_lc_preview_dialog(window, values)

        elif event == "-HELP_SIZE_FIELD-":
            sg.popup_scrolled(
                size_field_help_text(),
                title="Ayuda del campo de tamaño y estrategias",
                size=(118, 38),
                font=("Courier", 9),
            )

        elif event == "-OPEN_PLOTLY-":
            try:
                import webbrowser

                from swanmesh.lc_profile import generate_lc_raster, plot_lc_preview_plotly
                cfg = read_config_from_gui(values)
                if not cfg.domain_path or not Path(cfg.domain_path).exists():
                    sg.popup_error("Por favor seleccione un archivo de Dominio (Shapefile) válido primero.")
                    continue
                from swanmesh.bathymetry import build_dem
                from swanmesh.domain import load_domain
                dom = load_domain(cfg.domain_path, work_crs=cfg.work_crs)
                if cfg.subdomain_bbox and len(cfg.subdomain_bbox) == 4:
                    dom = dom.crop_to_subdomain(cfg.subdomain_bbox)
                bounds = dom.get_bounds(buffer_cells=cfg.buffer_cells, dx=cfg.dx, dy=cfg.dy)
                dem = build_dem(cfg, bounds=bounds)
                s, _, _, h_sampled = dem.sample_profile(num_points=250)
                res = generate_lc_raster(
                    x=s, h=h_sampled, period=cfg.wave_period, n_lambda=cfg.n_lambda,
                    alpha_grad=cfg.alpha_grad, min_lc=cfg.hmin, max_lc=cfg.hmax,
                    control_points=cfg.interest_points
                )
                out_html = Path(cfg.output_dir) / "swan_mesh_lc_profile.html"
                out_html.parent.mkdir(parents=True, exist_ok=True)
                crs_str = (cfg.work_crs or "").upper()
                is_geo = ("4326" in crs_str) or (not cfg.is_utm and "UTM" not in crs_str)
                unit_str = "grados (°)" if is_geo else "metros (m)"
                plot_lc_preview_plotly(
                    x=res.x, h=res.h, lc1=res.lc1, lc2=res.lc2, lc3=res.lc3, grad_h=res.grad_h,
                    min_lc=cfg.hmin, max_lc=cfg.hmax, period=cfg.wave_period,
                    is_geographic=is_geo, unit_str=unit_str,
                    title=f"Perfil Interactivo y Vectorial de Malla lc — {cfg.project_name}",
                    output_html=out_html,
                )
                webbrowser.open(f"file://{out_html.resolve()}")
                print(f"Abierto perfil interactivo en Plotly: {out_html.resolve()}")
            except Exception as ex:
                show_copyable_error_popup("Error al abrir perfil Plotly", str(ex))

        elif event == "-ESTIMATE_NODES-":
            try:
                cfg = read_config_from_gui(values)
                window["-EST_TEXT-"].update("Calculando estimación...")
                window.refresh()
                import numpy as np

                from swanmesh.bathymetry import build_dem
                from swanmesh.size_field import build_size_field, estimate_mesh_nodes
                from swanmesh.slope import build_slope

                if cfg.domain_path and Path(cfg.domain_path).exists():
                    from swanmesh.domain import load_domain
                    dom = load_domain(cfg.domain_path, work_crs=cfg.work_crs)
                    if cfg.subdomain_bbox and len(cfg.subdomain_bbox) == 4:
                        dom = dom.crop_to_subdomain(cfg.subdomain_bbox)
                    bounds = dom.get_bounds(buffer_cells=cfg.buffer_cells, dx=cfg.dx, dy=cfg.dy)
                    dem = build_dem(cfg, bounds=bounds)
                else:
                    # Estimate based on synthetic domain
                    import rasterio

                    from swanmesh.bathymetry import DEMData
                    h_m = 100.0
                    w, h = 200, 200
                    lx = cfg.hmax * 40.0 if cfg.hmax > 0 else 0.05
                    ly = lx * 3.2
                    x_c = np.linspace(0, lx, w)
                    h_prof = h_m - (h_m * 0.95) / (1.0 + np.exp(-(x_c - lx / 2.0) / (lx / 15.0)))
                    synth_grid = np.tile(h_prof, (h, 1))
                    tr = rasterio.transform.from_bounds(0, 0, lx, ly, w, h)
                    dem = DEMData(grid=synth_grid, transform=tr, crs=cfg.work_crs, bounds=(0, 0, lx, ly))

                slope = build_slope(dem, slope_tif_path=cfg.slope_tif_path, is_utm=cfg.is_utm)
                sf = build_size_field(cfg, dem=dem, slope=slope)
                est = estimate_mesh_nodes(sf.grid, dx=cfg.dx, dy=cfg.dy)
                warn = ""
                if int(est["est_nodes"]) > int(cfg.max_est_nodes):
                    warn = f"  ⚠ EXCEDE max_est_nodes={cfg.max_est_nodes:,}"
                window["-EST_TEXT-"].update(
                    f"[{cfg.strategy}] ~{est['est_nodes']:,} nodos (~{est['est_elements']:,} elem) | "
                    f"lc: [{est['min_h']:.4g}, {est['max_h']:.4g}] mean={est.get('mean_h', 0):.4g}{warn}"
                )
            except Exception as e:
                window["-EST_TEXT-"].update(f"Error al estimar: {e}")

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
                warnings = _configuration_warnings(cfg)
                if warnings:
                    recommended = _recommended_n_lambda(cfg)
                    if recommended is not None and cfg.n_lambda < recommended:
                        answer = sg.popup_yes_no(
                            "El campo de tamaño puede producir una malla gruesa o sin trabajo interior:\n\n"
                            + "\n".join(warnings)
                            + f"\n\n¿Aplicar n_lambda={recommended:g} y max_est_nodes>=500000?"
                        )
                        if answer != "Yes":
                            continue

                        values = dict(values)
                        values["-NLAMBDA-"] = str(recommended)
                        values["-MAX_EST_NODES-"] = str(max(int(cfg.max_est_nodes), 500000))
                        window["-NLAMBDA-"].update(values["-NLAMBDA-"])
                        window["-MAX_EST_NODES-"].update(values["-MAX_EST_NODES-"])
                        cfg = read_config_from_gui(values)

                    remaining_warnings = _configuration_warnings(cfg)
                    if remaining_warnings:
                        answer = sg.popup_yes_no(
                            "Persisten advertencias de resolución del campo:\n\n"
                            + "\n".join(remaining_warnings)
                            + "\n\n¿Continuar de todas formas?"
                        )
                        if answer != "Yes":
                            continue

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
                qa = res.qa_report or {}
                qa_ok = qa.get("qa_passed", False)
                print(f"PIPELINE COMPLETED! Outputs in {res.output_dir}")
                print(
                    f"QA={'PASSED' if qa_ok else 'FAILED'} | "
                    f"nodes={qa.get('total_nodes')} tri={qa.get('total_triangles')} "
                    f"markers={qa.get('marker_histogram')}"
                )
                sg.popup(
                    "Pipeline Success" if qa_ok else "Pipeline Finished (QA issues)",
                    f"Mesh written to {res.output_dir}\n"
                    f"QA: {'PASSED' if qa_ok else 'FAILED'}\n"
                    f"Nodes: {qa.get('total_nodes')} | Triangles: {qa.get('total_triangles')}\n"
                    f"Markers: {qa.get('marker_histogram')}",
                )

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
