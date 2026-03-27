"""3DGS stage: single image → Gaussian Splat PLY via Apple SHARP.

Wraps SHARP's predict_image() as a depthkit Stage.
Requires vendor/ml-sharp to be installed (pip install -e vendor/ml-sharp).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Default checkpoint URL (auto-downloaded on first use)
_DEFAULT_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class GaussianStage:
    """Feed-forward 3DGS prediction via SHARP.

    Input:  (H, W, 3) uint8 numpy array (RGB)
    Output: PLY bytes (standard 3DGS format, ~1.18M Gaussians)

    Model is lazy-loaded on first call or explicit warmup().
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cuda",
        focal_length_mm: float = 30.0,
    ) -> None:
        """
        Args:
            checkpoint: Path to .pt file, or None to auto-download.
            device: 'cuda', 'cpu', or 'mps'.
            focal_length_mm: Default focal length when image has no EXIF data.
        """
        self._checkpoint = checkpoint
        self._device = torch.device(device)
        self._focal_length_mm = focal_length_mm
        self._predictor = None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        from sharp.models import PredictorParams, create_predictor

        LOGGER.info("Loading SHARP model...")
        if self._checkpoint is None:
            state_dict = torch.hub.load_state_dict_from_url(
                _DEFAULT_URL, progress=True
            )
        else:
            state_dict = torch.load(
                self._checkpoint, weights_only=True, map_location=self._device
            )

        self._predictor = create_predictor(PredictorParams())
        self._predictor.load_state_dict(state_dict)
        self._predictor.eval()
        self._predictor.to(self._device)
        LOGGER.info("SHARP model loaded on %s.", self._device)

    def warmup(self) -> None:
        """Load model weights (call once before inference)."""
        self._ensure_loaded()

    def __call__(self, image: np.ndarray, focal_px: float | None = None) -> bytes:
        """Predict 3DGS from a single image.

        Args:
            image: (H, W, 3) uint8 RGB numpy array.
            focal_px: Focal length in pixels. If None, computed from
                      focal_length_mm assuming 36mm sensor width.

        Returns:
            PLY file as bytes (standard 3DGS format).
        """
        self._ensure_loaded()

        from sharp.cli.predict import predict_image
        from sharp.utils.gaussians import save_ply

        h, w = image.shape[:2]

        if focal_px is None:
            # Default: compute from mm assuming 36mm full-frame sensor
            focal_px = self._focal_length_mm * max(h, w) / 36.0

        gaussians = predict_image(
            self._predictor, image, focal_px, self._device
        )

        # Save PLY to temp file, read bytes, clean up
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            tmp = Path(f.name)
        save_ply(gaussians, focal_px, (h, w), tmp)
        ply_bytes = tmp.read_bytes()
        tmp.unlink()

        return ply_bytes
