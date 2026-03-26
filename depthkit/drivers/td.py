"""TouchDesigner CUDALink driver.

Two modes
---------
DepthTD
    Inline Script TOP — import depthkit inside a TD Script TOP.
    Call ``cook_numpy(top)`` each frame; feed the result to
    ``scriptOp.copyNumpyArray()``.

DepthIPCServer
    Standalone IPC server — run as a subprocess via
    ``python -m depthkit.drivers.td``.  Receives RGBA frames from
    TouchDesigner via CUDAIPCImporter, runs DepthStage, and sends the
    depth map back via CUDAIPCExporter.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from depthkit.stages.depth import DepthStage
from depthkit.stages.pointcloud import unproject_to_position_map


class DepthTD:
    """Inline Script TOP integration.

    Depth-only output (single-channel)::

        import sys; sys.path.insert(0, r"C:\\work\\depthkit")
        from depthkit.drivers.td import DepthTD
        _td = None
        def onCook(scriptOp):
            global _td
            if _td is None: _td = DepthTD()
            scriptOp.copyNumpyArray(_td.cook_numpy(scriptOp.inputs[0]))

    Position + color maps — two (H,W,4) RGBA 32-bit outputs::

        import sys; sys.path.insert(0, r"C:\\work\\depthkit")
        from depthkit.drivers.td import DepthTD
        _td = None
        _pos = _color = None
        def onCook(scriptOp):
            global _td, _pos, _color
            if _td is None: _td = DepthTD(fov_deg=58, depth_scale=5.0)
            _pos, _color = _td.cook_position_numpy(scriptOp.inputs[0])
            scriptOp.copyNumpyArray(_pos)  # R=X, G=Y, B=Z, A=depth
    """

    def __init__(self, model: str = "vitb", max_res: int = 640,
                 cache_dir: str | None = None,
                 fov_deg: float = 60.0,
                 depth_scale: float = 5.0) -> None:
        self._stage = DepthStage(model=model, max_res=max_res,
                                 cache_dir=cache_dir)
        self.fov_deg = fov_deg
        self.depth_scale = depth_scale

    def warmup(self) -> None:
        """Load weights and run a dummy pass (call once after construction)."""
        self._stage.warmup()

    def cook_tensor(self, top) -> torch.Tensor:
        """Run depth estimation on a TD TOP input.

        Args:
            top: TouchDesigner TOP operator (Script TOP input).

        Returns:
            (H, W) float32 CUDA tensor, depth normalised to [0, 1].

        Side-effect:
            Caches the input RGB tensor as ``self._last_rgb`` for use
            by ``cook_position_numpy``.
        """
        # TD returns (H, W, 4) uint8 RGBA
        arr = top.numpyArray(delayed=False)  # type: ignore[attr-defined]
        rgb = arr[:, :, :3].astype(np.float32) / 255.0  # (H, W, 3)
        frame = torch.from_numpy(rgb)
        if torch.cuda.is_available():
            frame = frame.cuda()
        self._last_rgb = frame  # cache for position map color output
        return self._stage(frame)  # (H, W) float32

    def cook_numpy(self, top) -> np.ndarray:
        """Run depth estimation and return a numpy array for copyNumpyArray.

        Returns:
            (H, W, 1) float32 array — single-channel depth in [0, 1].
            Feed directly to ``scriptOp.copyNumpyArray()``.
        """
        depth = self.cook_tensor(top).cpu().numpy()
        return depth[:, :, np.newaxis]  # (H, W, 1)

    def cook_position_numpy(self, top) -> tuple[np.ndarray, np.ndarray]:
        """Run depth estimation and return position + color maps.

        Returns:
            Tuple of two (H, W, 4) float32 arrays:
            - position_map: R=X, G=Y, B=Z, A=depth
            - color_map: R, G, B, A=1.0

            Use in TD with two Script TOPs sharing one DepthTD instance,
            or pick the map you need per Script TOP.
        """
        depth = self.cook_tensor(top)
        pos_map, color_map = unproject_to_position_map(
            depth, rgb=self._last_rgb,
            fov_deg=self.fov_deg, depth_scale=self.depth_scale,
        )
        return pos_map.cpu().numpy(), color_map.cpu().numpy()


class DepthIPCServer:
    """Standalone IPC server using cuda-link (forkni/cuda-link).

    Receives RGBA frames from TouchDesigner via CUDAIPCImporter,
    runs DepthStage, and sends the depth map back as a single-channel
    float texture via CUDAIPCExporter.

    Run::

        python -m depthkit.drivers.td --model vitb

    In TouchDesigner:
    - Add a CUDAIPCExporter CHOP to send frames → this server.
    - Add a CUDAIPCImporter CHOP to receive depth ← this server.
    """

    def __init__(self, model: str = "vitb", max_res: int = 640,
                 cache_dir: str | None = None,
                 fov_deg: float = 60.0,
                 depth_scale: float = 5.0,
                 position_map: bool = False) -> None:
        self._stage = DepthStage(model=model, max_res=max_res,
                                 cache_dir=cache_dir)
        self.fov_deg = fov_deg
        self.depth_scale = depth_scale
        self.position_map = position_map

    def run(self, max_frames: int = 0) -> None:
        """Start the IPC receive → infer → send loop.

        Args:
            max_frames: Stop after this many frames (0 = run forever).
        """
        try:
            from cuda_link import CUDAIPCImporter, CUDAIPCExporter  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "cuda-link not installed. "
                "Clone https://github.com/forkni/cuda-link and install it."
            ) from exc

        mode = "position map (XYZD)" if self.position_map else "depth only"
        print(f"[depthkit] Warming up depth model… (output: {mode})", flush=True)
        self._stage.warmup()

        importer = CUDAIPCImporter()
        exporter = CUDAIPCExporter()

        print("[depthkit] IPC server ready. Waiting for frames…", flush=True)
        frame_count = 0
        try:
            while max_frames == 0 or frame_count < max_frames:
                # Receive RGBA frame from TouchDesigner
                # cuda_link returns a CUDA tensor (H, W, 4) float32 in [0, 1]
                rgba: torch.Tensor = importer.receive()  # type: ignore[attr-defined]

                rgb = rgba[:, :, :3]  # (H, W, 3)
                depth = self._stage(rgb)  # (H, W) float32

                if self.position_map:
                    out, _color = unproject_to_position_map(
                        depth, rgb=rgb, fov_deg=self.fov_deg,
                        depth_scale=self.depth_scale,
                    )
                else:
                    H, W = depth.shape
                    out = torch.zeros(H, W, 4, dtype=torch.float32,
                                      device=depth.device)
                    out[:, :, 0] = depth

                exporter.send(out)  # type: ignore[attr-defined]
                frame_count += 1
        except KeyboardInterrupt:
            print(f"\n[depthkit] Stopped after {frame_count} frames.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="depthkit TouchDesigner IPC server"
    )
    parser.add_argument(
        "--model", choices=["vits", "vitb", "vitl"], default="vitb",
        help="Depth Anything V2 model variant (default: vitb)"
    )
    parser.add_argument(
        "--max-res", type=int, default=640,
        help="Resize longest edge to this before inference (default: 640)"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace model cache directory"
    )
    parser.add_argument(
        "--fov", type=float, default=60.0,
        help="Horizontal field of view in degrees (default: 60)"
    )
    parser.add_argument(
        "--depth-scale", type=float, default=5.0,
        help="Depth multiplier for unprojection (default: 5.0)"
    )
    parser.add_argument(
        "--position-map", action="store_true",
        help="Output RGBA position map (R=X, G=Y, B=Z, A=depth) instead of depth-only"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames, 0 = run forever (default: 0)"
    )
    args = parser.parse_args()

    server = DepthIPCServer(
        model=args.model,
        max_res=args.max_res,
        cache_dir=args.cache_dir,
        fov_deg=args.fov,
        depth_scale=args.depth_scale,
        position_map=args.position_map,
    )
    server.run(max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
