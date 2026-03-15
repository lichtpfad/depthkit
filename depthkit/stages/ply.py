from __future__ import annotations
import torch

class PLYStage:
    def __init__(self, scale: float = 0.002, opacity: float = 1.0) -> None:
        self.scale = scale
        self.opacity = opacity

    def __call__(self, points: torch.Tensor) -> bytes:
        raise NotImplementedError

    def warmup(self) -> None:
        pass
