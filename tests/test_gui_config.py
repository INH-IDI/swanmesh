"""GUI configuration synchronization and geographic size-field safeguards."""

from swanmesh.config import MeshConfig
from swanmesh_gui import app


def test_gui_recommends_n_lambda_for_long_period_geographic_mesh():
    cfg = MeshConfig(
        domain_path="domain.shp",
        bathy_xyz_path="bathy.xyz",
        work_crs="EPSG:4326",
        hmin=0.001,
        hmax=0.05,
        wave_period=30.0,
        n_lambda=0.1,
    )

    assert app._recommended_n_lambda(cfg) == 0.29
    assert app._configuration_warnings(cfg)


def test_gui_does_not_recommend_n_lambda_for_product_strategy():
    cfg = MeshConfig(
        domain_path="domain.shp",
        bathy_xyz_path="bathy.xyz",
        work_crs="EPSG:4326",
        strategy="product",
        n_lambda=0.1,
    )

    assert app._recommended_n_lambda(cfg) is None
    assert not any("n_lambda" in warning for warning in app._configuration_warnings(cfg))


def test_gui_help_documents_all_size_strategies():
    help_text = app.size_field_help_text()

    for strategy in ("product", "mean", "depth_weighted", "hybrid_smooth", "dispersion_gradient", "relative_depth"):
        assert strategy in help_text
    assert "H1 =" in help_text
    assert "n_lambda" in help_text
    assert "Wendland C2" in help_text


def test_gui_preserves_hidden_yaml_fields_when_reading_values():
    original = MeshConfig(
        domain_path="domain.shp",
        bathy_xyz_path="bathy.xyz",
        input_crs="EPSG:32719",
        clamp_z_min=-5000.0,
        interp_method="nearest",
        export_sidecar_tifs=False,
    )
    app.loaded_config = original
    values = {
        "-PROJ_NAME-": "gui_mesh",
        "-OUT_DIR-": "./output",
        "-OVERWRITE-": True,
        "-WORK_CRS-": "EPSG:4326",
        "-OUT_CRS-": "EPSG:4326",
        "-DOMAIN_PATH-": "domain.shp",
        "-XYZ_PATHS-": "bathy.xyz",
        "-TIF_PATHS-": "",
        "-IS_UTM-": False,
        "-USE_ONLINE_BATHY-": False,
        "-BLEND_ONLINE_BATHY-": False,
        "-USE_ROI-": False,
        "-DX-": "0.002",
        "-DY-": "0.002",
        "-BUFFER-": "20",
        "-ENABLE_POINT_RED-": True,
        "-MAX_PTS_PER_CELL-": "5",
        "-HMIN-": "0.001",
        "-HMAX-": "0.05",
        "-WAVE_T-": "30",
        "-NLAMBDA-": "0.3",
        "-ALPHA_GRAD-": "1.0",
        "-STRATEGY-": "dispersion_gradient",
        "-Z_CONV-": "elevation_negative_down",
        "-DEPTH_LIMIT-": "-50",
        "-MARKER_STRAT-": "depth_limit",
        "-OPEN_BND_BUF-": "",
        "-GMSH_ALG-": "6: Frontal-Delaunay",
        "-OPTIM_NETGEN-": True,
        "-GMSH_SMOOTHING-": "3",
        "-ENFORCE_DEGREE-": True,
        "-MIN_NODE_DEGREE-": "3",
        "-MAX_BND_PTS-": "800",
        "-MESH_GROWTH-": "1.2",
        "-MAX_EST_NODES-": "500000",
        "-ABORT_EST-": True,
        "-EXPORT_MSH-": True,
        "-EXPORT_PLOTS-": False,
    }

    try:
        cfg = app.read_config_from_gui(values)
    finally:
        app.loaded_config = None

    assert cfg.input_crs == "EPSG:32719"
    assert cfg.clamp_z_min == -5000.0
    assert cfg.interp_method == "nearest"
    assert cfg.export_sidecar_tifs is False
    assert cfg.mesh_growth == 1.2
