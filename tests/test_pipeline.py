import pytest
import torch
from depthkit.pipeline import Pipeline, Stage


class DoubleStage:
    def __call__(self, x):
        return x * 2
    def warmup(self):
        pass


class AddStage:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        return x + self.n
    def warmup(self):
        pass


def test_pipeline_single_stage():
    pipe = Pipeline([DoubleStage()])
    assert pipe(3) == 6


def test_pipeline_chained():
    pipe = Pipeline([DoubleStage(), AddStage(10)])
    assert pipe(5) == 20


def test_pipeline_warmup_called():
    warmed = []
    class TrackWarmup:
        def __call__(self, x): return x
        def warmup(self): warmed.append(1)
    pipe = Pipeline([TrackWarmup(), TrackWarmup()])
    pipe.warmup()
    assert len(warmed) == 2


def test_stage_protocol():
    assert isinstance(DoubleStage(), Stage)
