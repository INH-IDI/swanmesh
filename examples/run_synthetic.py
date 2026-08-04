"""Example script showing programmatic usage of swanmesh Python API."""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from swanmesh.config import MeshConfig
from swanmesh.pipeline import run

def main():
    base_dir = Path("./example_output")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic domain shapefile
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    poly = Polygon(outer.exterior, [hole.exterior])

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    shp_path = base_dir / "domain.shp"
    gdf.to_file(shp_path)

    # 2. Generate synthetic bathymetry XYZ
    x = np.linspace(-1, 11, 30)
    y = np.linspace(-1, 11, 30)
    xx, yy = np.meshgrid(x, y)
    zz = -2.0 * xx - 1.0 * yy - 5.0

    df = pd.DataFrame({"X": xx.flatten(), "Y": yy.flatten(), "Z": zz.flatten()})
    xyz_path = base_dir / "bathy.xyz"
    df.to_csv(xyz_path, index=False, sep=" ")

    # 3. Create swanmesh configuration
    config = MeshConfig(
        project_name="programmatic_synthetic",
        output_dir=str(base_dir / "results"),
        domain_path=str(shp_path),
        bathy_xyz_path=str(xyz_path),
        hmin=0.2,
        hmax=0.8,
        dx=0.1,
        dy=0.1,
        strategy="hybrid_smooth",
        smooth_sigma_pixels=1.5,
    )

    # 4. Run pipeline
    print("Running swanmesh pipeline...")
    result = run(config)

    print(f"Pipeline completed successfully!")
    print(f"Output Directory: {result.output_dir}")
    print(f"QA Status: {'PASSED' if result.qa_report['qa_passed'] else 'FAILED'}")
    print(f"Generated Files: {result.files_generated}")

if __name__ == "__main__":
    main()
