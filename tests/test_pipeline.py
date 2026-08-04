"""Integration test for full swanmesh pipeline."""

from pathlib import Path

from swanmesh.config import MeshConfig
from swanmesh.pipeline import run


def test_full_pipeline_synthetic(synthetic_domain_shp: Path, synthetic_bathy_xyz: Path, tmp_path: Path):
    cfg = MeshConfig(
        project_name="pipeline_test",
        output_dir=str(tmp_path / "out"),
        domain_path=str(synthetic_domain_shp),
        bathy_xyz_path=str(synthetic_bathy_xyz),
        hmin=0.5,
        hmax=1.0,
        dx=0.2,
        dy=0.2,
        buffer_cells=5,
        strategy="product",
    )

    res = run(cfg)

    assert res.output_dir.exists()
    assert res.qa_report["qa_passed"] is True
    assert (res.output_dir / "pipeline_test.node").exists()
    assert (res.output_dir / "pipeline_test.ele").exists()
    assert (res.output_dir / "pipeline_test.bot").exists()
    assert (res.output_dir / "pipeline_test_report.json").exists()
    assert (res.output_dir / "pipeline_test_config.yaml").exists()
