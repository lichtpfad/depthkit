"""CLI driver tests — mock DepthStage to avoid model download."""
from __future__ import annotations

import io
import sys
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from depthkit.drivers.cli import (
    frame_to_tensor, process_frame, save_ply, build_parser,
)
from depthkit.stages.pointcloud import PointCloudStage
from depthkit.stages.ply import PLYStage


class FakeDepthStage:
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        H, W = frame.shape[:2]
        return torch.rand(H, W) * 0.5
    def warmup(self): pass


def make_bgr_frame(H=64, W=80):
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


def test_frame_to_tensor_shape():
    frame = make_bgr_frame(64, 80)
    t = frame_to_tensor(frame, "cpu")
    assert t.shape == (64, 80, 3)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_frame_to_tensor_rgb_conversion():
    """BGR→RGB: blue channel should map to red channel in output."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # BGR blue = 255
    t = frame_to_tensor(frame, "cpu")
    # After BGR→RGB, channel 2 should be 1.0, channels 0 and 1 should be 0
    assert torch.allclose(t[:, :, 2], torch.ones(4, 4))
    assert torch.allclose(t[:, :, 0], torch.zeros(4, 4))


def test_process_frame_returns_bytes():
    frame = make_bgr_frame()
    ply_bytes = process_frame(
        frame,
        FakeDepthStage(),
        PointCloudStage(max_points=100, depth_threshold=1.1),
        PLYStage(),
        "cpu",
    )
    assert isinstance(ply_bytes, bytes)
    assert len(ply_bytes) > 0


def test_save_ply_creates_file(tmp_path):
    out = tmp_path / "sub" / "test.ply"
    save_ply(b"PLY test data", out)
    assert out.exists()
    assert out.read_bytes() == b"PLY test data"


def test_parser_image_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "image", "--input", "photo.jpg", "--output", "out.ply",
        "--model", "vits",
    ])
    assert args.command == "image"
    assert args.input == "photo.jpg"
    assert args.model == "vits"


def test_parser_video_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "video", "--input", "vid.mp4", "--output", "frames/",
        "--step", "5",
    ])
    assert args.command == "video"
    assert args.step == 5


def test_parser_snapshot_defaults():
    parser = build_parser()
    args = parser.parse_args(["snapshot", "--output", "snap.ply"])
    assert args.command == "snapshot"
    assert args.input == "webcam:0"


def test_parser_invalid_model():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["image", "--input", "x.jpg", "--output", "x.ply",
                           "--model", "invalid"])


def test_parser_benchmark_defaults():
    parser = build_parser()
    args = parser.parse_args(["benchmark"])
    assert args.command == "benchmark"
    assert args.res == 640
    assert args.warmup == 10
    assert args.timed == 50
    assert args.model == "vitb"


def test_parser_benchmark_all_models():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--model", "all"])
    assert args.model == "all"


def test_parser_cache_flag():
    parser = build_parser()
    args = parser.parse_args(["image", "--input", "x.jpg", "--output", "x.ply",
                               "--cache", "/tmp/models"])
    assert args.cache == "/tmp/models"


def test_depth_stage_cache_dir():
    """DepthStage stores cache_dir param."""
    from depthkit.stages.depth import DepthStage
    s = DepthStage(model="vits", cache_dir="/tmp/test")
    assert s.cache_dir == "/tmp/test"
