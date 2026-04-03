from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SplatData:
    """Structured container for 3DGS Gaussian attributes."""

    positions: np.ndarray        # (N, 3)
    sh_dc: np.ndarray            # (N, 3)
    sh_rest: np.ndarray          # (N, K, 3)
    opacities_logit: np.ndarray  # (N,)
    scales_log: np.ndarray       # (N, 3)
    rotations: np.ndarray        # (N, 4)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def sh_degree(self) -> int:
        K = self.sh_rest.shape[1]
        deg = int(math.isqrt(K + 1)) - 1
        return max(deg, 0)

    def colors_rgb(self) -> np.ndarray:
        return np.clip(self.sh_dc * _C0 + 0.5, 0.0, 1.0)

    def opacities(self) -> np.ndarray:
        return _sigmoid(self.opacities_logit)

    def scales(self) -> np.ndarray:
        return np.exp(self.scales_log)


class SplatLoader:
    _REQUIRED_FIELDS = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }

    @classmethod
    def from_file(cls, path: str | Path) -> SplatData:
        ply = PlyData.read(str(path))
        return cls._parse(ply)

    @classmethod
    def from_bytes(cls, data: bytes) -> SplatData:
        ply = PlyData.read(io.BytesIO(data))
        return cls._parse(ply)

    @classmethod
    def _parse(cls, ply: PlyData) -> SplatData:
        v = ply["vertex"]
        names = {p.name for p in v.properties}
        missing = cls._REQUIRED_FIELDS - names
        if missing:
            raise ValueError(f"PLY missing required 3DGS fields: {sorted(missing)}")

        N = len(v)
        positions = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
        sh_dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)

        f_rest_indices = sorted(
            int(name.split("_")[-1]) for name in names if name.startswith("f_rest_")
        )
        n_rest = len(f_rest_indices)

        if n_rest > 0:
            rest_flat = np.column_stack(
                [v[f"f_rest_{i}"] for i in range(n_rest)]
            ).astype(np.float32)
            K = n_rest // 3
            sh_rest = rest_flat[:, : K * 3].reshape(N, K, 3)
        else:
            sh_rest = np.zeros((N, 0, 3), dtype=np.float32)

        opacities_logit = np.array(v["opacity"], dtype=np.float32)
        scales_log = np.column_stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]]
        ).astype(np.float32)
        rotations = np.column_stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]
        ).astype(np.float32)

        return SplatData(
            positions=positions,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            opacities_logit=opacities_logit,
            scales_log=scales_log,
            rotations=rotations,
        )

    def warmup(self) -> None:
        pass
