import math

import pytest
import torch

from depthkit.stages.pointcloud import unproject_to_position_map


def test_output_shape():
    depth = torch.rand(120, 160)
    pm = unproject_to_position_map(depth)
    assert pm.shape == (120, 160, 4)
    assert pm.dtype == torch.float32


def test_channels_are_xyzd():
    """Alpha channel must equal normalised depth; Z = depth * scale."""
    depth = torch.rand(64, 64)
    scale = 3.0
    pm = unproject_to_position_map(depth, depth_scale=scale)
    # A = depth
    assert torch.allclose(pm[:, :, 3], depth, atol=1e-5)
    # Z = depth * scale
    assert torch.allclose(pm[:, :, 2], depth * scale, atol=1e-5)


def test_centre_pixel_is_zero_xy():
    """The centre pixel should have X≈0, Y≈0 (optical axis)."""
    H, W = 100, 100
    depth = torch.ones(H, W) * 0.5
    pm = unproject_to_position_map(depth, fov_deg=60.0, depth_scale=1.0)
    cy, cx = H // 2, W // 2
    assert abs(pm[cy, cx, 0].item()) < 0.05  # X near 0
    assert abs(pm[cy, cx, 1].item()) < 0.05  # Y near 0


def test_fov_affects_spread():
    """Wider FOV → larger X spread at image edges."""
    depth = torch.ones(64, 128) * 0.5
    pm_narrow = unproject_to_position_map(depth, fov_deg=30.0, depth_scale=1.0)
    pm_wide = unproject_to_position_map(depth, fov_deg=90.0, depth_scale=1.0)
    x_narrow = pm_narrow[:, -1, 0].abs().max()
    x_wide = pm_wide[:, -1, 0].abs().max()
    assert x_wide > x_narrow


def test_zero_depth_gives_zero_xyz():
    depth = torch.zeros(32, 32)
    pm = unproject_to_position_map(depth, depth_scale=5.0)
    assert torch.allclose(pm[:, :, :3], torch.zeros(32, 32, 3), atol=1e-7)


def test_device_preserved():
    depth = torch.rand(32, 32)
    pm = unproject_to_position_map(depth)
    assert pm.device == depth.device


def test_with_rgb_returns_tuple():
    depth = torch.rand(64, 80)
    rgb = torch.rand(64, 80, 3)
    result = unproject_to_position_map(depth, rgb=rgb)
    assert isinstance(result, tuple)
    pos_map, color_map = result
    assert pos_map.shape == (64, 80, 4)
    assert color_map.shape == (64, 80, 4)


def test_color_map_preserves_rgb():
    depth = torch.rand(32, 32)
    rgb = torch.rand(32, 32, 3)
    pos_map, color_map = unproject_to_position_map(depth, rgb=rgb)
    # RGB channels preserved
    assert torch.allclose(color_map[:, :, :3], rgb, atol=1e-6)
    # Alpha = 1.0
    assert torch.allclose(color_map[:, :, 3], torch.ones(32, 32), atol=1e-6)


def test_with_rgb_position_map_unchanged():
    """Position map should be identical whether rgb is passed or not."""
    depth = torch.rand(48, 64)
    rgb = torch.rand(48, 64, 3)
    pm_solo = unproject_to_position_map(depth, fov_deg=70.0, depth_scale=3.0)
    pm_with_rgb, _ = unproject_to_position_map(depth, rgb=rgb, fov_deg=70.0, depth_scale=3.0)
    assert torch.allclose(pm_solo, pm_with_rgb, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    depth = torch.rand(64, 64, device="cuda")
    pm = unproject_to_position_map(depth, fov_deg=60.0, depth_scale=5.0)
    assert pm.device.type == "cuda"
    assert pm.shape == (64, 64, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_with_rgb():
    depth = torch.rand(64, 64, device="cuda")
    rgb = torch.rand(64, 64, 3, device="cuda")
    pos_map, color_map = unproject_to_position_map(depth, rgb=rgb)
    assert pos_map.device.type == "cuda"
    assert color_map.device.type == "cuda"
