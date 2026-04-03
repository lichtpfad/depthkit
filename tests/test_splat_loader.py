from __future__ import annotations

import io
import numpy as np
import pytest
from pathlib import Path
from plyfile import PlyData, PlyElement
from depthkit.stages.splat_loader import SplatData, SplatLoader


def test_splat_data_basic():
    N = 100
    data = SplatData(
        positions=np.random.randn(N, 3).astype(np.float32),
        sh_dc=np.random.randn(N, 3).astype(np.float32),
        sh_rest=np.zeros((N, 0, 3), dtype=np.float32),
        opacities_logit=np.random.randn(N).astype(np.float32),
        scales_log=np.random.randn(N, 3).astype(np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    )
    assert data.num_gaussians == N
    assert data.sh_degree == 0


def test_splat_data_sh_degree():
    N = 10
    data = SplatData(
        positions=np.zeros((N, 3), dtype=np.float32),
        sh_dc=np.zeros((N, 3), dtype=np.float32),
        sh_rest=np.zeros((N, 15, 3), dtype=np.float32),
        opacities_logit=np.zeros(N, dtype=np.float32),
        scales_log=np.zeros((N, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    )
    assert data.sh_degree == 3


def test_splat_data_colors_dc_only():
    N = 5
    C0 = 0.28209479177387814
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 1], [0, 0, 0]], dtype=np.float32)
    sh_dc = (colors - 0.5) / C0
    data = SplatData(
        positions=np.zeros((N, 3), dtype=np.float32),
        sh_dc=sh_dc,
        sh_rest=np.zeros((N, 0, 3), dtype=np.float32),
        opacities_logit=np.zeros(N, dtype=np.float32),
        scales_log=np.zeros((N, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    )
    recovered = data.colors_rgb()
    np.testing.assert_allclose(recovered, colors, atol=1e-6)


def test_splat_data_opacities_sigmoid():
    N = 3
    logits = np.array([0.0, 5.0, -5.0], dtype=np.float32)
    data = SplatData(
        positions=np.zeros((N, 3), dtype=np.float32),
        sh_dc=np.zeros((N, 3), dtype=np.float32),
        sh_rest=np.zeros((N, 0, 3), dtype=np.float32),
        opacities_logit=logits,
        scales_log=np.zeros((N, 3), dtype=np.float32),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    )
    opacities = data.opacities()
    assert abs(opacities[0] - 0.5) < 1e-5
    assert opacities[1] > 0.99
    assert opacities[2] < 0.01


def test_splat_data_scales_exp():
    N = 2
    log_scales = np.array([[0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]], dtype=np.float32)
    data = SplatData(
        positions=np.zeros((N, 3), dtype=np.float32),
        sh_dc=np.zeros((N, 3), dtype=np.float32),
        sh_rest=np.zeros((N, 0, 3), dtype=np.float32),
        opacities_logit=np.zeros(N, dtype=np.float32),
        scales_log=log_scales,
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    )
    scales = data.scales()
    np.testing.assert_allclose(scales[0], [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(scales[1], [np.exp(-2.0)] * 3, atol=1e-6)


def _make_ply_bytes(n: int, sh_rest_count: int = 0) -> bytes:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    for i in range(sh_rest_count):
        dtype.append((f"f_rest_{i}", "f4"))
    vertex = np.zeros(n, dtype=dtype)
    if n > 0:
        vertex["x"] = np.linspace(-1, 1, n)
        vertex["y"] = np.linspace(-1, 1, n)
        vertex["z"] = np.linspace(-1, 1, n)
    vertex["f_dc_0"] = 0.5
    vertex["f_dc_1"] = -0.3
    vertex["f_dc_2"] = 0.1
    vertex["opacity"] = 2.0
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -5.0
    vertex["rot_0"] = 1.0
    el = PlyElement.describe(vertex, "vertex")
    buf = io.BytesIO()
    PlyData([el]).write(buf)
    return buf.getvalue()


def test_loader_from_bytes():
    ply_bytes = _make_ply_bytes(50)
    data = SplatLoader.from_bytes(ply_bytes)
    assert data.num_gaussians == 50
    assert data.positions.shape == (50, 3)
    assert data.sh_degree == 0


def test_loader_from_file(tmp_path):
    ply_bytes = _make_ply_bytes(30)
    path = tmp_path / "test.ply"
    path.write_bytes(ply_bytes)
    data = SplatLoader.from_file(path)
    assert data.num_gaussians == 30


def test_loader_sharp_format():
    ply_bytes = _make_ply_bytes(20, sh_rest_count=0)
    data = SplatLoader.from_bytes(ply_bytes)
    assert data.sh_degree == 0
    assert data.sh_rest.shape == (20, 0, 3)


def test_loader_nerfstudio_format():
    ply_bytes = _make_ply_bytes(10, sh_rest_count=45)
    data = SplatLoader.from_bytes(ply_bytes)
    assert data.sh_degree == 3
    assert data.sh_rest.shape == (10, 15, 3)


def test_loader_preserves_values():
    ply_bytes = _make_ply_bytes(5)
    data = SplatLoader.from_bytes(ply_bytes)
    np.testing.assert_allclose(data.sh_dc[:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(data.opacities_logit, 2.0, atol=1e-6)
    np.testing.assert_allclose(data.scales_log[:, 0], -5.0, atol=1e-6)
    np.testing.assert_allclose(data.rotations[:, 0], 1.0, atol=1e-6)


def test_loader_missing_required_field():
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex = np.zeros(5, dtype=dtype)
    el = PlyElement.describe(vertex, "vertex")
    buf = io.BytesIO()
    PlyData([el]).write(buf)
    with pytest.raises(ValueError, match="opacity"):
        SplatLoader.from_bytes(buf.getvalue())


def test_loader_empty_ply():
    ply_bytes = _make_ply_bytes(0)
    data = SplatLoader.from_bytes(ply_bytes)
    assert data.num_gaussians == 0


def test_splat_loader_warmup():
    loader = SplatLoader()
    loader.warmup()
