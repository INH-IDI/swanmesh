"""Tests for MeshConfig validation and YAML serialization."""

from pathlib import Path

import pytest

from swanmesh.config import MeshConfig
from swanmesh.errors import ConfigurationError


def test_config_defaults():
    cfg = MeshConfig(domain_path="dummy.shp", bathy_xyz_path="dummy.xyz")
    assert cfg.project_name == "swan_mesh"
    assert cfg.strategy == "product"
    assert cfg.hmin == 0.001
    assert cfg.hmax == 0.05

def test_config_validation_hmin_greater_than_hmax():
    with pytest.raises(ConfigurationError, match="hmin .* cannot be greater than hmax"):
        MeshConfig(domain_path="dummy.shp", bathy_xyz_path="dummy.xyz", hmin=0.1, hmax=0.01)

def test_config_validation_missing_bathy():
    with pytest.raises(ConfigurationError, match="At least one of bathy_xyz_path"):
        MeshConfig(domain_path="dummy.shp")

def test_config_yaml_roundtrip(tmp_path: Path):
    cfg = MeshConfig(
        project_name="test_proj",
        domain_path="domain.shp",
        bathy_xyz_path="bathy.xyz",
        hmin=0.005,
        hmax=0.02,
    )
    yaml_file = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded = MeshConfig.from_yaml(yaml_file)
    assert loaded.project_name == "test_proj"
    assert loaded.hmin == 0.005
    assert loaded.hmax == 0.02
