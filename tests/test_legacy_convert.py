"""Tests for legacy Gmsh .msh conversion to SWAN."""

from pathlib import Path

from swanmesh.export.gmsh_io import convert_gmsh_to_swan, parse_gmsh2_ascii


def test_legacy_msh_conversion(tmp_path: Path, synthetic_dem_tif: Path):
    # Create a tiny Gmsh 2.0 ASCII mesh file fixture
    msh_content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 1.0 0.0
$EndNodes
$Elements
1
1 2 2 0 1 1 2 3
$EndElements
"""
    msh_path = tmp_path / "tiny_fixture.msh"
    msh_path.write_text(msh_content)

    nodes, elems = parse_gmsh2_ascii(msh_path)
    assert len(nodes) == 3
    assert len(elems) == 1

    out_dir = convert_gmsh_to_swan(
        msh_path=msh_path,
        dem_path=synthetic_dem_tif,
        output_dir=tmp_path / "swan_out",
        base_name="converted",
    )

    assert (out_dir / "converted.node").exists()
    assert (out_dir / "converted.ele").exists()
    assert (out_dir / "converted.bot").exists()
