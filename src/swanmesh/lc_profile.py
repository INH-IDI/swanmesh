"""
Element size field (lc) raster generator, Gmsh .pos exporter, and GUI-integrable preview plot component.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np


class LcRasterResult(NamedTuple):
    """Encapsulates generated element size field raster data."""
    x: np.ndarray
    h: np.ndarray
    grad_h: np.ndarray
    wavelength: np.ndarray
    lc1: np.ndarray
    lc2: np.ndarray
    lc3: np.ndarray
    dx: float


def compute_wave_length(h: np.ndarray | float, period: float, g: float = 9.81) -> np.ndarray | float:
    """
    Solve linear wave dispersion relation omega^2 = g * k * tanh(k * h)
    to compute exact wavelength L = 2 * pi / k for arbitrary depth profile h.
    """
    h_arr = np.atleast_1d(np.abs(np.asanyarray(h, dtype=np.float64)))
    h_arr = np.maximum(h_arr, 0.01)  # avoid division by zero in shallow water

    omega = 2.0 * np.pi / period
    omega2 = omega**2

    # Initial guess using Fenton & McKee (1990) explicit approximation
    y = omega2 * h_arr / g
    kh = y * (1.0 / np.sqrt(np.tanh(y)))
    kh = np.maximum(kh, 1e-4)

    # Newton-Raphson iterations to refine k * h to machine precision
    for _ in range(5):
        tanh_kh = np.tanh(kh)
        sech_kh2 = 1.0 - tanh_kh**2
        f = g * (kh / h_arr) * tanh_kh - omega2
        f_prime = (g / h_arr) * (tanh_kh + kh * sech_kh2)
        kh = kh - f / f_prime

    k = kh / h_arr
    L = 2.0 * np.pi / k

    if np.ndim(h) == 0:
        return float(L[0])
    return L.astype(np.float32)


def generate_lc_raster(
    x: np.ndarray,
    h: np.ndarray,
    period: float = 30.0,
    n_lambda: float = 15.0,
    alpha_grad: float = 1.0,
    min_lc: float = 0.001,
    max_lc: float = 0.05,
    control_points: Optional[list[dict]] = None,
    dx: Optional[float] = None,
    is_geographic: bool = False,
    m_per_deg: float = 101900.0,
) -> LcRasterResult:
    """
    Generate 1D/2D element size field (lc) raster from depth profile h(x) in 3 passes:
      - Pass 1 (lc1): Base size from wave dispersion wavelength (L / N_lambda)
      - Pass 2 (lc2): Refined element size based on bathymetric slope gradient |dh/dx|
      - Pass 3 (lc3): Local SPH compact kernel (Wendland C2) radial refinement around control points
    """
    x_arr = np.asanyarray(x, dtype=np.float32)
    h_arr = np.asanyarray(h, dtype=np.float32)

    # Resample to fixed resolution dx if requested
    if dx is not None and dx > 0:
        x_grid = np.arange(x_arr.min(), x_arr.max() + dx, dx, dtype=np.float32)
        h_grid = np.interp(x_grid, x_arr, h_arr).astype(np.float32)
    else:
        x_grid = x_arr
        h_grid = h_arr
        dx = float(np.mean(np.diff(x_grid))) if len(x_grid) > 1 else 1.0

    # Metric scaling factor: meters per degree if geographic, 1.0 if projected meters
    use_geo = is_geographic or (float(np.nanmax(np.abs(x_grid))) < 360.0 and min_lc < 1.0)
    scale = float(m_per_deg) if use_geo else 1.0

    # 1. Wavelength L(x) in meters, then convert to CRS-native units for lc
    wavelength = compute_wave_length(h_grid, period=period)
    wavelength_native = wavelength / scale
    n_lam = max(float(n_lambda), 1e-6)
    lc1_raw = wavelength_native / n_lam
    lc1 = np.clip(lc1_raw, min_lc, max_lc)

    # 2. Dimensionless slope |dh/dX| with X in meters
    x_meters = x_grid * scale
    if len(x_grid) > 1:
        grad_h_meters = np.abs(np.gradient(h_grid, x_meters))
        grad_h_native = grad_h_meters
    else:
        grad_h_meters = np.zeros_like(h_grid)
        grad_h_native = np.zeros_like(h_grid)

    # 3. Refine lc2 using dimensionless slope and alpha_grad (Pass 2)
    lc2_raw = lc1 / (1.0 + alpha_grad * grad_h_native)
    lc2 = np.clip(lc2_raw, min_lc, max_lc)

    # 4. Pass 3: SPH compact kernel (Wendland C2) refinement around control points
    lc3 = lc2.copy()

    if control_points:
        for cp in control_points:
            x_cp = float(cp.get("x", 0.0))
            hmin_cp = float(cp.get("hmin", min_lc))
            r_cp = float(cp.get("radius", 0.0))
            if r_cp > 0:
                dist = np.abs(x_grid - x_cp)
                q = np.clip(dist / r_cp, 0.0, 1.0)
                # Wendland C2 SPH compact kernel
                w_kernel = ((1.0 - q) ** 4) * (1.0 + 4.0 * q)
                lc_cp = lc2 * (1.0 - w_kernel) + hmin_cp * w_kernel
                lc3 = np.minimum(lc3, lc_cp)

    lc3 = np.clip(lc3, min_lc, max_lc)

    return LcRasterResult(
        x=x_grid,
        h=h_grid,
        grad_h=grad_h_native.astype(np.float32),
        wavelength=wavelength.astype(np.float32),
        lc1=lc1.astype(np.float32),
        lc2=lc2.astype(np.float32),
        lc3=lc3.astype(np.float32),
        dx=float(dx),
    )


def export_pos_field(
    x: np.ndarray,
    lc2: np.ndarray,
    output_path: str | Path,
    view_name: str = "lc_field",
) -> Path:
    """
    Export 1D element size field as a Gmsh Background Mesh (.pos PostView format).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(f'View "{view_name}" {{\n')
        for i in range(len(x) - 1):
            x1, x2 = float(x[i]), float(x[i + 1])
            v1, v2 = float(lc2[i]), float(lc2[i + 1])
            # SL: Scalar Line element in Gmsh .pos format: SL(x1,y1,z1, x2,y2,z2){v1, v2};
            f.write(f"  SL({x1:.6f},0,0, {x2:.6f},0,0){{{v1:.6f}, {v2:.6f}}};\n")
        f.write("};\n")

    return out_p


def plot_lc_preview(
    x: np.ndarray,
    h: np.ndarray,
    lc1: np.ndarray,
    lc2: np.ndarray,
    lc3: Optional[np.ndarray] = None,
    grad_h: Optional[np.ndarray] = None,
    min_lc: Optional[float] = None,
    max_lc: Optional[float] = None,
    period: float = 30.0,
    title: str = "Mesh Size Field Preview (lc)",
    figsize: tuple[float, float] = (10, 6),
    is_geographic: bool = False,
    m_per_deg: float = 101900.0,
    unit_str: str = "m",
) -> plt.Figure:
    """
    Generate two vertically aligned subplots sharing the X-axis (dimensionless in wavelengths x / L0):
      Subplot (a): Relative Depth h(x) / h_max [primary Y] and slope gradient |dh/dx| [secondary Y].
      Subplot (b): Element size curves Pass 1 (lc1), Pass 2 (lc2), and Pass 3 SPH kernel (lc3).
    """
    x_arr = np.asanyarray(x, dtype=np.float64)
    h_arr = np.abs(np.asanyarray(h, dtype=np.float64))

    # Metric scaling factor: meters per degree if geographic
    use_geo = is_geographic or (x_arr.max() < 10.0 or (min_lc is not None and min_lc < 0.01))
    scale = float(m_per_deg) if use_geo else 1.0

    x_meters = x_arr * scale

    # Deep-water wavelength L0 = g * T^2 / (2 * pi) in meters
    g = 9.81
    l0 = float((g * (period**2)) / (2.0 * np.pi))
    if l0 <= 0:
        l0 = 1405.28

    # Dimensionless X coordinate in wavelengths
    x_wave = x_meters / l0

    # Max depth for relative depth scaling h / h_max
    h_max = float(np.max(h_arr)) if len(h_arr) > 0 and np.max(h_arr) > 0 else 1.0
    h_rel = h_arr / h_max

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Subplot (a): Real Depth h(x) [primary Y] & Slope Gradient |dh/dX| [secondary Y] ---
    color_h = "#1f77b4"
    ax_top.plot(x_wave, h_arr, color=color_h, linewidth=2, label=f"Profundidad Batimétrica h(x) (m)")
    ax_top.set_ylabel("Profundidad h (m)", color=color_h, fontsize=9, fontweight="bold")
    ax_top.tick_params(axis="y", labelcolor=color_h)
    ax_top.grid(True, linestyle="--", alpha=0.5)

    if grad_h is None and len(x_meters) > 1:
        grad_h = np.abs(np.gradient(h_arr, x_meters))

    if grad_h is not None:
        ax_grad = ax_top.twinx()
        color_g = "#d62728"
        ax_grad.plot(x_wave, grad_h, color=color_g, linestyle=":", linewidth=1.5, label="Pendiente |dh/dX|")
        ax_grad.set_ylabel("Pendiente |dh/dX| (m/m)", color=color_g, fontsize=9, fontweight="bold")
        ax_grad.tick_params(axis="y", labelcolor=color_g)

    ax_top.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax_top.set_xlim(x_wave.min(), x_wave.max())

    # Secondary X-axis on top showing physical distance in meters (m)
    ax_top_x = ax_top.twiny()
    ax_top_x.set_xlim(x_wave.min(), x_wave.max())
    x_ticks_wave = np.linspace(x_wave.min(), x_wave.max(), 6)
    ax_top.set_xticks(x_ticks_wave)
    ax_top_x.set_xticks(x_ticks_wave)
    
    top_labels = [f"{tw * l0:.0f} m" for tw in x_ticks_wave]
    ax_top_x.set_xticklabels(top_labels)
    ax_top_x.set_xlabel("Distancia Física X (m)", fontsize=9, fontweight="bold", labelpad=6)

    # --- Subplot (b): Pass 1 (lc1) -> Pass 2 (lc2 from lc1) -> Pass 3 (lc3 from lc2) ---
    ax_bot.plot(x_wave, lc1, color="#ff7f0e", linestyle="--", linewidth=1.6, label="Pasada 1: lc1 = L(h) / N_lambda")
    ax_bot.plot(x_wave, lc2, color="#2ca02c", linestyle=":", linewidth=1.8, label="Pasada 2: lc2 = lc1 / (1 + alpha * |dh/dX|)")
    
    target_lc = lc3 if lc3 is not None else lc2
    ax_bot.plot(x_wave, target_lc, color="#9467bd", linewidth=2.2, label="Pasada 3: lc3 = SPH_Kernel(lc2, CP)")
    ax_bot.fill_between(x_wave, lc1, target_lc, color="#9467bd", alpha=0.18, label="Refinamiento Acumulado (Delta)")

    if min_lc is not None:
        ax_bot.axhline(min_lc, color="red", linestyle="-.", linewidth=1.2, label=f"min_lc ({min_lc} {unit_str})")
    if max_lc is not None:
        ax_bot.axhline(max_lc, color="gray", linestyle="-.", linewidth=1.2, label=f"max_lc ({max_lc} {unit_str})")

    ax_bot.set_xlim(x_wave.min(), x_wave.max())
    ax_bot.set_xlabel(f"Distancia Adimensional X / L0  (L0 = {l0:.1f} m)", fontsize=10, fontweight="bold")
    ax_bot.set_ylabel(f"Tamaño de Malla lc ({unit_str})", fontsize=10, fontweight="bold")
    ax_bot.grid(True, linestyle="--", alpha=0.5)
    ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_lc_preview_plotly(
    x: np.ndarray,
    h: np.ndarray,
    lc1: np.ndarray,
    lc2: np.ndarray,
    lc3: Optional[np.ndarray] = None,
    grad_h: Optional[np.ndarray] = None,
    min_lc: Optional[float] = None,
    max_lc: Optional[float] = None,
    period: float = 30.0,
    title: str = "Mesh Size Field Preview (lc) — Interactive Profile",
    output_html: Optional[str | Path] = None,
    is_geographic: bool = False,
    m_per_deg: float = 101900.0,
    unit_str: str = "m",
):
    """Generate an interactive Plotly HTML figure for profile analysis with zooming and tooltips."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_arr = np.asanyarray(x, dtype=np.float64)
    h_arr = np.abs(np.asanyarray(h, dtype=np.float64))

    use_geo = is_geographic or (x_arr.max() < 10.0 or (min_lc is not None and min_lc < 0.01))
    scale = float(m_per_deg) if use_geo else 1.0

    x_meters = x_arr * scale
    g = 9.81
    l0 = float((g * (period**2)) / (2.0 * np.pi))
    if l0 <= 0:
        l0 = 1405.28

    if grad_h is None and len(x_meters) > 1:
        grad_h = np.abs(np.gradient(h_arr, x_meters))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("(a) Profundidad Batimétrica h(x) y Pendiente |dh/dX|", "(b) Tamaño de Malla lc por Pasadas"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Subplot 1: Depth h(x) [primary Y]
    fig.add_trace(
        go.Scatter(x=x_meters, y=h_arr, mode="lines", name="Profundidad h(x) (m)", line=dict(color="#1f77b4", width=2.5)),
        row=1, col=1, secondary_y=False
    )

    # Subplot 1: Slope gradient |dh/dx| [secondary Y]
    if grad_h is not None:
        fig.add_trace(
            go.Scatter(x=x_meters, y=grad_h, mode="lines", name="Pendiente |dh/dX| (m/m)", line=dict(color="#d62728", width=1.5, dash="dot")),
            row=1, col=1, secondary_y=True
        )

    # Subplot 2: Size field passes
    fig.add_trace(
        go.Scatter(x=x_meters, y=lc1, mode="lines", name="Pasada 1: lc1 = L(h)/N_lambda", line=dict(color="#ff7f0e", width=2, dash="dash")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_meters, y=lc2, mode="lines", name="Pasada 2: lc2 = lc1/(1+alpha*|dh/dx|)", line=dict(color="#2ca02c", width=2, dash="dot")),
        row=2, col=1
    )
    target_lc = lc3 if lc3 is not None else lc2
    fig.add_trace(
        go.Scatter(x=x_meters, y=target_lc, mode="lines", name="Pasada 3: lc3 (Kernel SPH/Definitivo)", line=dict(color="#9467bd", width=3)),
        row=2, col=1
    )

    if min_lc is not None:
        fig.add_hline(y=min_lc, line=dict(color="red", width=1.5, dash="dashdot"), annotation_text=f"min_lc ({min_lc} {unit_str})", row=2, col=1)
    if max_lc is not None:
        fig.add_hline(y=max_lc, line=dict(color="gray", width=1.5, dash="dashdot"), annotation_text=f"max_lc ({max_lc} {unit_str})", row=2, col=1)

    fig.update_xaxes(title_text="Distancia Física X (m)", row=2, col=1)
    fig.update_yaxes(title_text="Profundidad h (m)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Pendiente |dh/dX|", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=f"Tamaño lc ({unit_str})", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="black")),
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )

    if output_html is not None:
        out_p = Path(output_html)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_p))

    return fig
