"""Worker threads for background pipeline execution in FreeSimpleGUI."""

import threading
import traceback

import matplotlib
matplotlib.use("Agg")

import FreeSimpleGUI as sg

from swanmesh.config import MeshConfig
from swanmesh.export.gmsh_io import convert_gmsh_to_swan
from swanmesh.pipeline import run as run_pipeline


class PipelineWorker(threading.Thread):
    """Executes swanmesh pipeline in a background thread and signals GUI thread safely."""

    def __init__(self, config: MeshConfig, window: sg.Window):
        super().__init__(daemon=True)
        self.config = config
        self.window = window

    def run(self):
        try:
            res = run_pipeline(self.config)
            self.window.write_event_value("-THREAD_PIPELINE_DONE-", (res, None))
        except Exception as e:
            err_msg = f"{e}\n\n{traceback.format_exc()}"
            self.window.write_event_value("-THREAD_PIPELINE_DONE-", (None, err_msg))

class ConvertWorker(threading.Thread):
    """Executes legacy Gmsh conversion in a background thread and signals GUI thread safely."""

    def __init__(
        self,
        msh_path: str,
        dem_path: str,
        output_dir: str,
        base_name: str,
        depth_limit: float,
        z_convention: str,
        window: sg.Window,
    ):
        super().__init__(daemon=True)
        self.msh_path = msh_path
        self.dem_path = dem_path
        self.output_dir = output_dir
        self.base_name = base_name
        self.depth_limit = depth_limit
        self.z_convention = z_convention
        self.window = window

    def run(self):
        try:
            out_path = convert_gmsh_to_swan(
                msh_path=self.msh_path,
                dem_path=self.dem_path,
                output_dir=self.output_dir,
                base_name=self.base_name,
                depth_limit=self.depth_limit,
                z_convention=self.z_convention,
            )
            self.window.write_event_value("-THREAD_CONVERT_DONE-", (out_path, None))
        except Exception as e:
            err_msg = f"{e}\n\n{traceback.format_exc()}"
            self.window.write_event_value("-THREAD_CONVERT_DONE-", (None, err_msg))
