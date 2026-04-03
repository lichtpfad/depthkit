from __future__ import annotations

import io
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from depthkit.stages.splat_loader import SplatData
from depthkit.drivers.splats_td import pack_gaussians_to_textures, SplatsTD


def _make_splat_data(n: int) -> SplatData:
    return SplatData(
        positions=np.random.randn(n, 3).astype(np.float32),
        sh_dc=np.random.randn(n, 3).astype(np.float32),
        sh_rest=np.zeros((n, 0, 3), dtype=np.float32),
        opacities_logit=np.random.randn(n).astype(np.float32),
        scales_log=np.random.randn(n, 3).astype(np.float32),
        rotations=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
    )


def test_pack_returns_dict():
    data = _make_splat_data(100)
    textures = pack_gaussians_to_textures(data, tex_width=32)
    assert "position" in textures
    assert "color" in textures
    assert "scale_opacity" in textures
    assert "rotation" in textures


def test_pack_dimensions():
    data = _make_splat_data(100)
    textures = pack_gaussians_to_textures(data, tex_width=32)
    for name, tex in textures.items():
        if name == "_meta":
            continue
        assert tex.shape == (4, 32, 4), f"{name}: {tex.shape}"
        assert tex.dtype == np.float32


def test_pack_position_values():
    data = _make_splat_data(10)
    textures = pack_gaussians_to_textures(data, tex_width=16)
    pos = textures["position"]
    for i in range(10):
        row, col = divmod(i, 16)
        np.testing.assert_allclose(pos[row, col, :3], data.positions[i])


def test_pack_color_is_rgb():
    data = _make_splat_data(5)
    textures = pack_gaussians_to_textures(data, tex_width=8)
    color = textures["color"]
    expected_rgb = data.colors_rgb()
    for i in range(5):
        row, col = divmod(i, 8)
        np.testing.assert_allclose(color[row, col, :3], expected_rgb[i], atol=1e-6)


def test_pack_scale_opacity_combined():
    data = _make_splat_data(3)
    textures = pack_gaussians_to_textures(data, tex_width=4)
    so = textures["scale_opacity"]
    expected_scales = data.scales()
    expected_opacities = data.opacities()
    for i in range(3):
        row, col = divmod(i, 4)
        np.testing.assert_allclose(so[row, col, :3], expected_scales[i], atol=1e-6)
        np.testing.assert_allclose(so[row, col, 3], expected_opacities[i], atol=1e-6)


def test_pack_large_count():
    data = _make_splat_data(1000)
    textures = pack_gaussians_to_textures(data, tex_width=256)
    assert textures["position"].shape == (4, 256, 4)


def test_pack_metadata():
    data = _make_splat_data(100)
    textures = pack_gaussians_to_textures(data, tex_width=64)
    assert textures["_meta"]["num_gaussians"] == 100
    assert textures["_meta"]["tex_width"] == 64
    assert textures["_meta"]["tex_height"] == 2


def test_splats_td_load(tmp_path):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertex = np.zeros(50, dtype=dtype)
    vertex["x"] = np.linspace(-1, 1, 50)
    vertex["rot_0"] = 1.0
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -5.0
    el = PlyElement.describe(vertex, "vertex")
    path = tmp_path / "test.ply"
    PlyData([el]).write(str(path))

    td = SplatsTD(tex_width=32)
    td.load(str(path))
    assert td.num_gaussians == 50
    assert td.position_texture.shape[0] == 2  # ceil(50/32)
    assert td.position_texture.shape[1] == 32
    assert td.color_texture.shape == td.position_texture.shape


def test_splats_td_sort_by_depth():
    data = _make_splat_data(100)
    data.positions[:, 2] = np.linspace(10, -10, 100)

    td = SplatsTD(tex_width=64)
    td._data = data
    td._repack()
    td.sort_by_depth(camera_pos=np.array([0, 0, 0], dtype=np.float32))

    pos = td.position_texture.reshape(-1, 4)[:100]
    z_values = pos[:, 2]
    assert z_values[0] > z_values[-1], "Not sorted back-to-front"
