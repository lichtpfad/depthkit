from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DepthStage:
    """Depth Anything V2 inference stage.

    Input:  (H, W, 3) float32 CUDA tensor, values in [0, 1]
    Output: (H, W)    float32 CUDA tensor, normalized depth in [0, 1]
    """

    MODEL_IDS = {
        "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, model: str = "vitb", max_res: int = 640,
                 cache_dir: str | None = None) -> None:
        if model not in self.MODEL_IDS:
            raise ValueError(f"model must be one of {list(self.MODEL_IDS)}, got {model!r}")
        self.model_key = model
        self.model_id = self.MODEL_IDS[model]
        self.max_res = max_res
        self.cache_dir = cache_dir
        self._pipe = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline as hf_pipeline
        kwargs = dict(
            task="depth-estimation",
            model=self.model_id,
            device=0 if torch.cuda.is_available() else -1,
        )
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self._pipe = hf_pipeline(**kwargs)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """Run depth estimation.

        Args:
            frame: (H, W, 3) float32 CUDA tensor, RGB values in [0, 1]

        Returns:
            depth: (H, W) float32 CUDA tensor, normalized to [0, 1]
        """
        self._ensure_loaded()

        H, W = frame.shape[:2]

        # Convert tensor -> PIL image for transformers pipeline
        frame_cpu = (frame.cpu().numpy() * 255).clip(0, 255).astype("uint8")
        pil_img = Image.fromarray(frame_cpu)

        # Resize if needed
        scale = min(self.max_res / max(H, W), 1.0)
        if scale < 1.0:
            new_w = int(W * scale)
            new_h = int(H * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        # Run inference
        result = self._pipe(pil_img)
        depth_np = result["predicted_depth"].squeeze(0).detach().cpu().numpy()  # HxW float32

        # Back to tensor on original device
        depth = torch.from_numpy(depth_np)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = torch.zeros_like(depth)

        # Resize back to original H, W if we downscaled
        if scale < 1.0:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # Move to same device as input
        return depth.to(frame.device)

    def warmup(self) -> None:
        """Load model weights and run a dummy forward pass."""
        self._ensure_loaded()
        dummy = torch.zeros(64, 64, 3)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
        self(dummy)
