from __future__ import annotations

import torch
import torch.nn.functional as F


# ImageNet normalization constants (used by DINOv2 backbone)
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _resize_to_multiple(h: int, w: int, target: int, multiple: int) -> tuple[int, int]:
    """Compute resize dimensions: fit into target, round to multiple."""
    scale = target / max(h, w)
    new_h = int(round(h * scale / multiple) * multiple)
    new_w = int(round(w * scale / multiple) * multiple)
    return max(new_h, multiple), max(new_w, multiple)


class DepthStage:
    """Depth Anything V2 inference stage (GPU-native).

    Input:  (H, W, 3) float32 CUDA tensor, values in [0, 1]
    Output: (H, W)    float32 CUDA tensor, normalized depth in [0, 1]

    All preprocessing and postprocessing runs on GPU — zero CPU roundtrips.
    """

    MODEL_IDS = {
        "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, model: str = "vitb", max_res: int = 518,
                 cache_dir: str | None = None) -> None:
        if model not in self.MODEL_IDS:
            raise ValueError(f"model must be one of {list(self.MODEL_IDS)}, got {model!r}")
        self.model_key = model
        self.model_id = self.MODEL_IDS[model]
        self.max_res = max_res
        self.cache_dir = cache_dir
        self._model = None
        self._device = None
        self._mean = None  # cached on device
        self._std = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import DepthAnythingForDepthEstimation
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self._model = DepthAnythingForDepthEstimation.from_pretrained(
            self.model_id, **kwargs
        ).to(self._device).eval()
        # Cache normalization tensors on device: shape (1, 3, 1, 1) for broadcasting
        self._mean = _MEAN.to(self._device).view(1, 3, 1, 1)
        self._std = _STD.to(self._device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """Run depth estimation entirely on GPU.

        Args:
            frame: (H, W, 3) float32 CUDA tensor, RGB values in [0, 1]

        Returns:
            depth: (H, W) float32 CUDA tensor, normalized to [0, 1]
        """
        self._ensure_loaded()

        H, W = frame.shape[:2]
        device = frame.device

        # (H, W, 3) -> (1, 3, H, W), move to model device
        x = frame.to(self._device).permute(2, 0, 1).unsqueeze(0)

        # Resize to model input (multiple of 14 for ViT patch size)
        inp_h, inp_w = _resize_to_multiple(H, W, self.max_res, 14)
        if inp_h != H or inp_w != W:
            x = F.interpolate(x, size=(inp_h, inp_w), mode="bicubic",
                              align_corners=False).clamp(0, 1)

        # ImageNet normalization (tensors auto-moved to match input device)
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)

        # Model forward pass
        out = self._model(pixel_values=x)
        depth = out.predicted_depth.squeeze(1)  # (1, H', W') -> (1, H', W')

        # Resize back to original resolution
        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        depth = depth.squeeze(0)  # (H, W)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = torch.zeros_like(depth)

        return depth.to(device)

    def warmup(self) -> None:
        """Load model weights and run a dummy forward pass."""
        self._ensure_loaded()
        dummy = torch.zeros(64, 64, 3, device=self._device)
        self(dummy)
