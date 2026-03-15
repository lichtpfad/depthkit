from __future__ import annotations

import math

import torch


class PointCloudStage:
    """Projects depth map into 3D point cloud using pinhole camera model.

    Input:  depth (H, W) float32, normalized [0, 1]
            rgb   (H, W, 3) float32, values in [0, 1]
    Output: points (N, 6) float32 CUDA — columns: X, Y, Z, R, G, B
    """

    def __init__(
        self,
        fov_deg: float = 60.0,
        max_points: int = 200_000,
        depth_scale: float = 5.0,
        depth_threshold: float = 0.99,
    ) -> None:
        """
        Args:
            fov_deg: Horizontal field of view in degrees.
            max_points: Maximum output points (random downsample if exceeded).
            depth_scale: Multiplier for depth values (normalized depth → metric-ish units).
            depth_threshold: Discard points with depth > this value (remove background).
        """
        self.fov_deg = fov_deg
        self.max_points = max_points
        self.depth_scale = depth_scale
        self.depth_threshold = depth_threshold

    def __call__(self, depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Unproject depth map to 3D points.

        Args:
            depth: (H, W) float32, normalized depth in [0, 1]
            rgb:   (H, W, 3) float32, RGB in [0, 1], same device as depth

        Returns:
            points: (N, 6) float32 on same device — XYZ + RGB
        """
        H, W = depth.shape
        device = depth.device

        # Camera intrinsics from horizontal FoV
        fx = (W / 2.0) / math.tan(math.radians(self.fov_deg / 2.0))
        fy = fx
        cx = W / 2.0
        cy = H / 2.0

        # Pixel grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Scale depth
        z = depth * self.depth_scale

        # Unproject
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy

        # Stack to (H, W, 3) then flatten to (H*W, 3)
        xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        rgb_flat = rgb.reshape(-1, 3).to(device)

        # Filter by depth threshold
        mask = depth.reshape(-1) < self.depth_threshold
        xyz = xyz[mask]
        rgb_flat = rgb_flat[mask]

        # Downsample if too many points
        N = xyz.shape[0]
        if N > self.max_points:
            idx = torch.randperm(N, device=device)[: self.max_points]
            xyz = xyz[idx]
            rgb_flat = rgb_flat[idx]

        return torch.cat([xyz, rgb_flat], dim=-1)

    def warmup(self) -> None:
        """No-op — included for Stage protocol consistency."""
        pass
