import io
import pytest
import torch
import numpy as np
from plyfile import PlyData
from depthkit.stages.ply import PLYStage


def make_points(N=100):
    xyz = torch.rand(N, 3) * 2 - 1  # [-1, 1]
    rgb = torch.rand(N, 3)           # [0, 1]
    return torch.cat([xyz, rgb], dim=-1)


def test_returns_bytes():
    stage = PLYStage()
    pts = make_points(50)
    result = stage(pts)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_valid_ply_parseable():
    stage = PLYStage()
    pts = make_points(100)
    data = stage(pts)
    ply = PlyData.read(io.BytesIO(data))
    assert "vertex" in [e.name for e in ply.elements]


def test_vertex_count():
    stage = PLYStage()
    N = 42
    pts = make_points(N)
    data = stage(pts)
    ply = PlyData.read(io.BytesIO(data))
    assert len(ply["vertex"]) == N


def test_required_3dgs_fields():
    """PLY must have all fields expected by TouchDesigner 3DGS component."""
    stage = PLYStage()
    data = stage(make_points(10))
    ply = PlyData.read(io.BytesIO(data))
    names = {p.name for p in ply["vertex"].properties}
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    assert required.issubset(names), f"Missing: {required - names}"


def test_f_rest_are_zero():
    """DC-only: all f_rest_* must be 0 to avoid SH artifacts in TD."""
    stage = PLYStage()
    data = stage(make_points(10))
    ply = PlyData.read(io.BytesIO(data))
    v = ply["vertex"]
    for i in range(45):
        assert np.all(v[f"f_rest_{i}"] == 0.0), f"f_rest_{i} not zero"


def test_rotation_is_identity():
    stage = PLYStage()
    data = stage(make_points(10))
    ply = PlyData.read(io.BytesIO(data))
    v = ply["vertex"]
    assert np.allclose(v["rot_0"], 1.0)
    assert np.allclose(v["rot_1"], 0.0)
    assert np.allclose(v["rot_2"], 0.0)
    assert np.allclose(v["rot_3"], 0.0)


def test_warmup_is_noop():
    PLYStage().warmup()


def test_empty_input():
    stage = PLYStage()
    pts = torch.zeros(0, 6)
    data = stage(pts)
    ply = PlyData.read(io.BytesIO(data))
    assert len(ply["vertex"]) == 0
