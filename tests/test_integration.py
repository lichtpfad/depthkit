"""Integration tests for the full pipeline chain.

Uses stub stages to test Pipeline wiring without downloading models.

Note on dual-input wiring
--------------------------
Pipeline passes the output of stage N directly as input to stage N+1.
If the output is a tuple, it is unpacked via ``*result`` before the next
stage is called.  PointCloudStage requires TWO inputs (depth + rgb), so the
stage that precedes it must return a tuple ``(depth, rgb)``.  In the tests
below FakeDepthWithRGB bundles both tensors into a tuple so that Pipeline can
unpack them into PointCloudStage naturally.
"""
import io
import torch
from plyfile import PlyData
from depthkit.pipeline import Pipeline
from depthkit.stages.pointcloud import PointCloudStage
from depthkit.stages.ply import PLYStage


# ---------------------------------------------------------------------------
# Stub stages
# ---------------------------------------------------------------------------

class FakeDepthStage:
    """Returns a fixed depth map from a single RGB frame."""

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        H, W = frame.shape[:2]
        return torch.rand(H, W, device=frame.device) * 0.8

    def warmup(self) -> None:
        pass


class FakeDepthWithRGB:
    """Returns (depth, rgb) tuple so Pipeline can unpack into PointCloudStage."""

    def __call__(self, frame: torch.Tensor) -> tuple:
        H, W = frame.shape[:2]
        depth = torch.rand(H, W, device=frame.device) * 0.8
        return (depth, frame)

    def warmup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_depth_to_pointcloud_direct():
    """DepthStage → PointCloudStage wiring (called directly, not via Pipeline)."""
    frame = torch.rand(64, 64, 3)
    depth = FakeDepthStage()(frame)
    points = PointCloudStage(max_points=500_000)(depth, frame)
    assert points.shape[1] == 6


def test_depth_to_pointcloud_pipeline():
    """FakeDepthWithRGB → PointCloudStage via Pipeline tuple-unpack wiring."""
    pipe = Pipeline([FakeDepthWithRGB(), PointCloudStage(max_points=500_000)])
    frame = torch.rand(64, 64, 3)
    result = pipe(frame)
    assert result.shape[1] == 6


def test_full_pipeline_produces_valid_ply():
    """FakeDepthWithRGB → PointCloud → PLY produces a parseable 3DGS file."""
    pipe = Pipeline([
        FakeDepthWithRGB(),
        PointCloudStage(max_points=1000, depth_threshold=1.1),
        PLYStage(scale=0.002),
    ])
    H, W = 32, 32
    frame = torch.rand(H, W, 3)
    ply_bytes = pipe(frame)
    assert isinstance(ply_bytes, bytes)
    ply = PlyData.read(io.BytesIO(ply_bytes))
    assert "vertex" in [e.name for e in ply.elements]
    assert len(ply["vertex"]) <= 1000


def test_pipeline_warmup_chains_all_stages():
    warmed = []

    class Tracker:
        def __call__(self, x):
            return x

        def warmup(self):
            warmed.append(1)

    pipe = Pipeline([Tracker(), Tracker(), Tracker()])
    pipe.warmup()
    assert len(warmed) == 3


def test_stages_module_exports():
    """depthkit.stages exports DepthStage, PointCloudStage, PLYStage."""
    from depthkit.stages import DepthStage, PointCloudStage, PLYStage  # noqa: F401
