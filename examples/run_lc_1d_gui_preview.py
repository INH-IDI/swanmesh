"""
Example script demonstrating 1D lc raster generation, Gmsh .pos export,
and embedding the preview plot inside a FreeSimpleGUI / Tkinter interface.
"""

from pathlib import Path
import FreeSimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from swanmesh.lc_profile import export_pos_field, generate_lc_raster, plot_lc_preview


def draw_figure(canvas, figure):
    """Embed Matplotlib Figure into FreeSimpleGUI Canvas."""
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def main():
    # 1. Create synthetic bathymetry profile h(x)
    x = np.linspace(0, 2000, 200, dtype=np.float32)
    # Profile with deep water offshore (60m) transitioning to shallow (3m) near coast with slope break
    h = 60.0 - 57.0 / (1.0 + np.exp(-(x - 1000.0) / 150.0))

    sg.theme("DarkTeal6")

    layout = [
        [sg.Text("Parametros del Campo de Tamaño de Elemento (lc en 3 Pasadas)", font=("Helvetica", 12, "bold"))],
        [
            sg.Text("Periodo T (s):", size=(18, 1)),
            sg.Input("10.0", size=(10, 1), key="-PERIOD-"),
            sg.Text("Profundidad Max h_max (m):", size=(22, 1)),
            sg.Input("100.0", size=(10, 1), key="-HMAX_BATHY-"),
        ],
        [
            sg.Text("N_lambda (nodos/onda):", size=(18, 1)),
            sg.Input("15.0", size=(10, 1), key="-NLAMBDA-"),
            sg.Text("Factor Gradiente alpha:", size=(18, 1)),
            sg.Input("2.0", size=(10, 1), key="-ALPHA-"),
        ],
        [
            sg.Text("Min lc Global (m):", size=(14, 1)),
            sg.Input("0.5", size=(8, 1), key="-MIN_LC-"),
            sg.Text("Max lc Global (m):", size=(14, 1)),
            sg.Input("50.0", size=(8, 1), key="-MAX_LC-"),
        ],
        [sg.Frame("Pasada 3: Refinamiento por Punto de Control (Kernel SPH Compacto Wendland C2)", [
            [
                sg.Text("Posición X (m):"), sg.Input("1000.0", size=(8, 1), key="-CP_X-"),
                sg.Text("Local min lc (m):"), sg.Input("1.0", size=(8, 1), key="-CP_HMIN-"),
                sg.Text("Radio de Influencia R (m):"), sg.Input("400.0", size=(8, 1), key="-CP_R-"),
            ]
        ])],
        [
            sg.Button("Actualizar Vista Previa", key="-UPDATE-", button_color=("white", "blue")),
            sg.Button("Exportar .pos para Gmsh", key="-EXPORT_POS-", button_color=("white", "green")),
            sg.Button("Cerrar", key="-EXIT-"),
        ],
        [sg.Canvas(key="-CANVAS-", size=(850, 520))],
        [sg.Text("", key="-STATUS-", size=(85, 1), text_color="yellow")],
    ]

    window = sg.Window("swanmesh — Previsualizador de Campo de Tamaño lc (3 Pasadas + Adimensional)", layout, finalize=True)

    fig_agg = None

    def update_plot():
        nonlocal fig_agg
        try:
            period = float(window["-PERIOD-"].get())
            h_max_val = float(window["-HMAX_BATHY-"].get())
            n_lambda = float(window["-NLAMBDA-"].get())
            alpha_grad = float(window["-ALPHA-"].get())
            min_lc = float(window["-MIN_LC-"].get())
            max_lc = float(window["-MAX_LC-"].get())

            cp_x = float(window["-CP_X-"].get())
            cp_hmin = float(window["-CP_HMIN-"].get())
            cp_r = float(window["-CP_R-"].get())

            cp_list = [{"x": cp_x, "hmin": cp_hmin, "radius": cp_r}] if cp_r > 0 else None

            # Dynamic bathymetry profile using specified h_max
            x = np.linspace(0, 2000, 200, dtype=np.float32)
            h = h_max_val - (h_max_val * 0.95) / (1.0 + np.exp(-(x - 1000.0) / 150.0))

            res = generate_lc_raster(
                x=x,
                h=h,
                period=period,
                n_lambda=n_lambda,
                alpha_grad=alpha_grad,
                min_lc=min_lc,
                max_lc=max_lc,
                control_points=cp_list,
            )

            fig = plot_lc_preview(
                x=res.x,
                h=res.h,
                lc1=res.lc1,
                lc2=res.lc2,
                lc3=res.lc3,
                grad_h=res.grad_h,
                min_lc=min_lc,
                max_lc=max_lc,
                period=period,
                title=f"Campo de Tamaño lc en 3 Pasadas (T={period}s, N_lambda={n_lambda}, alpha={alpha_grad}, CP_R={cp_r}m)",
            )

            canvas_elem = window["-CANVAS-"].TKCanvas
            fig_agg = draw_figure(canvas_elem, fig)
            plt.close(fig)
            window["-STATUS-"].update(f"Raster lc2 generado (Profundidad Relativa h/h_max, Distancia Adimensional x/L0. Min lc: {res.lc2.min():.2f}m, Max lc: {res.lc2.max():.2f}m)")
        except Exception as err:
            window["-STATUS-"].update(f"Error: {err}")

    # Draw initial plot
    update_plot()

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if event == "-UPDATE-":
            update_plot()

        elif event == "-EXPORT_POS-":
            try:
                period = float(values["-PERIOD-"])
                n_lambda = float(values["-NLAMBDA-"])
                alpha_grad = float(values["-ALPHA-"])
                min_lc = float(values["-MIN_LC-"])
                max_lc = float(values["-MAX_LC-"])

                res = generate_lc_raster(
                    x=x,
                    h=h,
                    period=period,
                    n_lambda=n_lambda,
                    alpha_grad=alpha_grad,
                    min_lc=min_lc,
                    max_lc=max_lc,
                )

                out_pos = Path("./lc_field_background.pos")
                export_pos_field(res.x, res.lc3, out_pos)
                window["-STATUS-"].update(f"Archivo background mesh Gmsh exportado exitosamente en: {out_pos.resolve()}")
            except Exception as err:
                window["-STATUS-"].update(f"Error al exportar: {err}")

    window.close()


if __name__ == "__main__":
    main()
