import pytest
import torch
from depthkit.stages.depth import DepthStage


@pytest.fixture(scope="module")
def stage():
    """Shared DepthStage — loaded once for the module (slow: downloads model)."""
    s = DepthStage(model="vits", max_res=256)
    s.warmup()
    return s


def test_depth_output_shape(stage):
    H, W = 120, 160
    frame = torch.rand(H, W, 3)
    if torch.cuda.is_available():
        frame = frame.cuda()
    depth = stage(frame)
    assert depth.shape == (H, W), f"Expected ({H},{W}), got {depth.shape}"


def test_depth_output_range(stage):
    frame = torch.rand(64, 64, 3)
    if torch.cuda.is_available():
        frame = frame.cuda()
    depth = stage(frame)
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0


def test_depth_on_cpu(stage):
    """DepthStage should work on CPU frame too."""
    frame = torch.rand(64, 64, 3)  # CPU
    depth = stage(frame)
    assert depth.shape == (64, 64)


def test_invalid_model():
    with pytest.raises(ValueError):
        DepthStage(model="invalid")


def test_downscale_preserves_original_size(stage):
    """Output shape must match input even when max_res triggers downscale."""
    H, W = 480, 640  # bigger than max_res=256 -> will be downscaled
    frame = torch.rand(H, W, 3)
    if torch.cuda.is_available():
        frame = frame.cuda()
    depth = stage(frame)
    assert depth.shape == (H, W)
