import pytest
import torch
from depthkit.stages.pointcloud import PointCloudStage


def make_inputs(H=120, W=160, device="cpu"):
    depth = torch.rand(H, W, device=device)
    rgb = torch.rand(H, W, 3, device=device)
    return depth, rgb


def test_output_shape():
    stage = PointCloudStage(max_points=500_000)
    depth, rgb = make_inputs()
    pts = stage(depth, rgb)
    assert pts.ndim == 2
    assert pts.shape[1] == 6


def test_output_bounded_by_max_points():
    stage = PointCloudStage(max_points=100)
    depth, rgb = make_inputs(H=480, W=640)
    pts = stage(depth, rgb)
    assert pts.shape[0] <= 100


def test_depth_threshold_removes_background():
    stage = PointCloudStage(depth_threshold=0.5, max_points=500_000)
    H, W = 64, 64
    depth = torch.ones(H, W) * 0.8  # all above threshold
    rgb = torch.rand(H, W, 3)
    pts = stage(depth, rgb)
    assert pts.shape[0] == 0  # all filtered


def test_xyz_range():
    """Z values should be in [0, depth_scale]."""
    stage = PointCloudStage(depth_scale=5.0, depth_threshold=1.1, max_points=500_000)
    depth, rgb = make_inputs()
    pts = stage(depth, rgb)
    z = pts[:, 2]
    assert z.min() >= 0.0
    assert z.max() <= 5.0 + 1e-5


def test_rgb_preserved():
    """RGB values in output should match input (no clamp/shift)."""
    stage = PointCloudStage(depth_threshold=1.1, max_points=500_000)
    H, W = 4, 4
    depth = torch.zeros(H, W)  # z=0 for all
    rgb = torch.rand(H, W, 3)
    pts = stage(depth, rgb)
    assert pts.shape[0] == H * W
    # RGB should be preserved
    assert torch.allclose(pts[:, 3:], rgb.reshape(-1, 3), atol=1e-5)


def test_warmup_is_noop():
    stage = PointCloudStage()
    stage.warmup()  # should not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_passthrough():
    stage = PointCloudStage(max_points=500_000)
    depth = torch.rand(64, 64, device="cuda")
    rgb = torch.rand(64, 64, 3, device="cuda")
    pts = stage(depth, rgb)
    assert pts.device.type == "cuda"
