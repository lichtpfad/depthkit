"""TouchDesigner driver for 3DGS splat rendering.

Packs Gaussian data into 2D float32 RGBA textures suitable for
TD Script TOP → GLSL MAT instancing pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from depthkit.stages.splat_loader import SplatData, SplatLoader


def pack_gaussians_to_textures(
    data: SplatData,
    tex_width: int = 1024,
) -> dict[str, np.ndarray | dict]:
    """Pack Gaussian attributes into 2D RGBA float32 textures.

    Each Gaussian occupies one texel. Textures are 2D with a fixed
    width; height = ceil(N / width). Unused texels are zero-padded.

    Args:
        data: Parsed Gaussian splat data.
        tex_width: Texture width in pixels (default: 1024).

    Returns:
        Dict with keys:
        - "position":      (H, W, 4) — R=x, G=y, B=z, A=1.0
        - "color":         (H, W, 4) — R, G, B (activated), A=1.0
        - "scale_opacity": (H, W, 4) — R=sx, G=sy, B=sz (exp), A=opacity (sigmoid)
        - "rotation":      (H, W, 4) — R=w, G=x, B=y, A=z (quaternion)
        - "_meta":         dict with num_gaussians, tex_width, tex_height
    """
    N = data.num_gaussians
    H = max(1, math.ceil(N / tex_width))
    total = H * tex_width

    def _make_tex() -> np.ndarray:
        return np.zeros((H, tex_width, 4), dtype=np.float32)

    pos_tex = _make_tex()
    pos_flat = pos_tex.reshape(total, 4)
    pos_flat[:N, :3] = data.positions
    pos_flat[:N, 3] = 1.0

    color_tex = _make_tex()
    color_flat = color_tex.reshape(total, 4)
    color_flat[:N, :3] = data.colors_rgb()
    color_flat[:N, 3] = 1.0

    so_tex = _make_tex()
    so_flat = so_tex.reshape(total, 4)
    so_flat[:N, :3] = data.scales()
    so_flat[:N, 3] = data.opacities()

    rot_tex = _make_tex()
    rot_flat = rot_tex.reshape(total, 4)
    rot_flat[:N, :4] = data.rotations

    return {
        "position": pos_tex,
        "color": color_tex,
        "scale_opacity": so_tex,
        "rotation": rot_tex,
        "_meta": {
            "num_gaussians": N,
            "tex_width": tex_width,
            "tex_height": H,
        },
    }


class SplatsTD:
    """TouchDesigner integration for 3DGS PLY rendering.

    Loads a PLY file, packs Gaussian data into 2D textures, and
    provides numpy arrays ready for Script TOP copyNumpyArray().

    Usage in TD Script TOP::

        from depthkit.drivers.splats_td import SplatsTD
        _splats = SplatsTD()
        _splats.load(r"C:\\path\\to\\scene.ply")

        def onCook(scriptOp):
            scriptOp.copyNumpyArray(_splats.position_texture)
    """

    def __init__(self, tex_width: int = 1024) -> None:
        self._tex_width = tex_width
        self._data: SplatData | None = None
        self._textures: dict | None = None

    def load(self, path: str | Path) -> None:
        """Load a 3DGS PLY file and pack into textures."""
        self._data = SplatLoader.from_file(path)
        self._repack()

    def _repack(self) -> None:
        self._textures = pack_gaussians_to_textures(
            self._data, tex_width=self._tex_width
        )

    @property
    def num_gaussians(self) -> int:
        return self._data.num_gaussians if self._data else 0

    def _get_tex(self, key: str) -> np.ndarray | None:
        return self._textures[key] if self._textures else None

    @property
    def position_texture(self) -> np.ndarray | None:
        return self._get_tex("position")

    @property
    def color_texture(self) -> np.ndarray | None:
        return self._get_tex("color")

    @property
    def scale_opacity_texture(self) -> np.ndarray | None:
        return self._get_tex("scale_opacity")

    @property
    def rotation_texture(self) -> np.ndarray | None:
        return self._get_tex("rotation")

    def sort_by_depth(self, camera_pos: np.ndarray) -> None:
        """Sort Gaussians back-to-front relative to camera and repack."""
        if self._data is None:
            return
        diff = self._data.positions - camera_pos[np.newaxis, :]
        dist_sq = (diff * diff).sum(axis=1)
        order = np.argsort(-dist_sq)
        self._data = SplatData(
            positions=self._data.positions[order],
            sh_dc=self._data.sh_dc[order],
            sh_rest=self._data.sh_rest[order],
            opacities_logit=self._data.opacities_logit[order],
            scales_log=self._data.scales_log[order],
            rotations=self._data.rotations[order],
        )
        self._repack()
