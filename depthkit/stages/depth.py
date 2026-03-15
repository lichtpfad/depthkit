from __future__ import annotations
import torch

class DepthStage:
    def __init__(self, model: str = "vitb", max_res: int = 640) -> None:
        self.model = model
        self.max_res = max_res

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def warmup(self) -> None:
        raise NotImplementedError
