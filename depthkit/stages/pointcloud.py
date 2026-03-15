from __future__ import annotations
import torch

class PointCloudStage:
    def __init__(self, fov_deg: float = 60.0, max_points: int = 200_000) -> None:
        self.fov_deg = fov_deg
        self.max_points = max_points

    def __call__(self, depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def warmup(self) -> None:
        pass  # no-op, included for interface consistency
