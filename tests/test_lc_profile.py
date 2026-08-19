"""Unit tests for lc_profile module."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from swanmesh.lc_profile import compute_wave_length, export_pos_field, generate_lc_raster, plot_lc_preview


def test_compute_wave_length():
    # Shallow water approx: L approx T * sqrt(g * h)
    # Deep water approx: L approx g * T^2 / (2 * pi)
    L_deep = compute_wave_length(h=1000.0, period=10.0)
    assert np.isclose(L_deep, 156.13, rtol=1e-2)

    L_shallow = compute_wave_length(h=5.0, period=10.0)
    assert L_shallow < L_deep


def test_generate_lc_raster():
    x = np.linspace(0, 1000, 100, dtype=np.float32)
    # Synthetic bathymetry: slope from -50m to -5m
    h = np.linspace(50.0, 5.0, 100, dtype=np.float32)

    res = generate_lc_raster(
        x=x,
        h=h,
        period=10.0,
        n_lambda=15.0,
        alpha_grad=2.0,
        min_lc=2.0,
        max_lc=20.0,
    )

    assert len(res.x) == 100
    assert len(res.lc1) == 100
    assert len(res.lc2) == 100
    # Refined lc2 should be <= base lc1 everywhere
    assert (res.lc2 <= res.lc1 + 1e-5).all()
    # Min lc threshold respected
    assert (res.lc2 >= 2.0).all()


def test_export_pos_field(tmp_path: Path):
    x = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    lc2 = np.array([5.0, 3.0, 2.0], dtype=np.float32)
    out_pos = tmp_path / "test_field.pos"

    res_path = export_pos_field(x, lc2, out_pos)
    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert 'View "lc_field"' in content
    assert "SL(0.000000,0,0, 10.000000,0,0){5.000000, 3.000000};" in content


def test_plot_lc_preview():
    x = np.linspace(0, 500, 50, dtype=np.float32)
    h = np.linspace(30.0, 5.0, 50, dtype=np.float32)

    res = generate_lc_raster(x=x, h=h, period=12.0)
    fig = plot_lc_preview(
        x=res.x,
        h=res.h,
        lc1=res.lc1,
        lc2=res.lc2,
        grad_h=res.grad_h,
        min_lc=1.0,
        max_lc=20.0,
    )

    assert len(fig.axes) >= 3
    plt.close(fig)
