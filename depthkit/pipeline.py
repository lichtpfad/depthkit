from __future__ import annotations
from typing import Protocol, runtime_checkable, Any
import torch


@runtime_checkable
class Stage(Protocol):
    def __call__(self, *args: Any) -> Any: ...
    def warmup(self) -> None: ...


class Pipeline:
    """Chains stages: output of stage N becomes input of stage N+1."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    def __call__(self, *args: Any) -> Any:
        result = args
        for stage in self.stages:
            if isinstance(result, tuple):
                result = stage(*result)
            else:
                result = stage(result)
        return result

    def warmup(self) -> None:
        for stage in self.stages:
            stage.warmup()
