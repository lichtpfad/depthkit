from __future__ import annotations

import io
import math

import numpy as np
import torch
from plyfile import PlyData, PlyElement


class PLYStage:
    """Converts XYZ+RGB point cloud to a valid 3DGS PLY file.

    Each point becomes a tiny isotropic Gaussian with:
    - DC-only spherical harmonics (f_rest_* = 0) to avoid SH coordinate mismatch
      in TouchDesigner (nerfstudio rotates scene ~90deg, TD shader doesn't know)
    - Identity rotation quaternion
    - Uniform small scale

    Input:  points (N, 6) float32 — X, Y, Z, R, G, B in [0, 1]
    Output: PLY file contents as bytes
    """

    def __init__(self, scale: float = 0.002, opacity: float = 1.0) -> None:
        """
        Args:
            scale: Gaussian radius in scene units (isotropic).
            opacity: Gaussian opacity [0, 1].
        """
        self.scale = scale
        self.opacity = opacity

    def __call__(self, points: torch.Tensor) -> bytes:
        """Convert point cloud to 3DGS PLY bytes.

        Args:
            points: (N, 6) float32 tensor — X, Y, Z, R, G, B

        Returns:
            PLY file as bytes, compatible with TouchDesigner Tim Gerritsen component.
        """
        pts = points.detach().cpu().numpy().astype(np.float32)
        N = pts.shape[0]

        xyz = pts[:, :3]
        rgb = pts[:, 3:]

        # RGB [0,1] → SH degree-0 coefficients
        # DC coefficient = color / (2 * sqrt(pi)) ≈ color * 0.28209
        C0 = 0.28209479177387814
        f_dc = (rgb - 0.5) / C0  # inverse of SH→RGB conversion

        # Opacity: inverse sigmoid
        opacity_val = np.clip(self.opacity, 1e-6, 1.0 - 1e-6)
        opacity_logit = math.log(opacity_val / (1.0 - opacity_val))
        opacities = np.full((N, 1), opacity_logit, dtype=np.float32)

        # Scale: log(scale), isotropic
        log_scale = math.log(self.scale)

        # Rotation: identity quaternion (w=1, x=0, y=0, z=0)
        rots = np.zeros((N, 4), dtype=np.float32)
        rots[:, 0] = 1.0  # w component

        # Build structured numpy array matching 3DGS PLY spec
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        for i in range(45):
            dtype.append((f"f_rest_{i}", "f4"))
        dtype += [
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]

        vertex = np.zeros(N, dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertex["nx"] = vertex["ny"] = vertex["nz"] = 0.0
        vertex["f_dc_0"] = f_dc[:, 0]
        vertex["f_dc_1"] = f_dc[:, 1]
        vertex["f_dc_2"] = f_dc[:, 2]
        for i in range(45):
            vertex[f"f_rest_{i}"] = 0.0
        vertex["opacity"] = opacities[:, 0]
        vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = log_scale
        vertex["rot_0"] = rots[:, 0]
        vertex["rot_1"] = rots[:, 1]
        vertex["rot_2"] = rots[:, 2]
        vertex["rot_3"] = rots[:, 3]

        el = PlyElement.describe(vertex, "vertex")

        buf = io.BytesIO()
        PlyData([el]).write(buf)
        return buf.getvalue()

    def warmup(self) -> None:
        """No-op — included for Stage protocol consistency."""
        pass
