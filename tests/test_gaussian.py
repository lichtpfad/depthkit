"""Tests for GaussianStage (SHARP wrapper)."""
import pytest
import numpy as np


def has_sharp():
    try:
        import sharp  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_sharp(), reason="sharp not installed")
class TestGaussianStage:
    def test_import(self):
        from depthkit.stages.gaussian import GaussianStage
        stage = GaussianStage()
        assert stage._predictor is None  # lazy load

    def test_warmup_loads_model(self):
        from depthkit.stages.gaussian import GaussianStage
        stage = GaussianStage(device="cuda")
        stage.warmup()
        assert stage._predictor is not None

    def test_inference_returns_ply_bytes(self):
        from depthkit.stages.gaussian import GaussianStage
        stage = GaussianStage(device="cuda")

        # Create a small test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ply_bytes = stage(image)

        assert isinstance(ply_bytes, bytes)
        assert len(ply_bytes) > 1000  # PLY has header + data
        assert ply_bytes[:3] == b"ply"  # PLY magic bytes

    def test_cli_help(self):
        """Verify gaussian subcommand exists in CLI parser."""
        from depthkit.drivers.cli import build_parser
        parser = build_parser()
        # Should not raise
        args = parser.parse_args(["gaussian", "--input", "x.png", "--output", "x.ply"])
        assert args.command == "gaussian"
        assert args.focal_mm == 30.0
