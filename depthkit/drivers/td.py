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


class DepthTD:
    """Inline Script TOP integration.

    Usage inside a TouchDesigner Script TOP::

        import sys
        sys.path.insert(0, r"C:\\work\\depthkit")
        from depthkit.drivers.td import DepthTD

        _depth_td = None

        def onSetupParameters(scriptOp):
            pass

        def onCook(scriptOp):
            global _depth_td
            if _depth_td is None:
                _depth_td = DepthTD()
            arr = _depth_td.cook_numpy(scriptOp.inputs[0])
            scriptOp.copyNumpyArray(arr)
    """

    def __init__(self, model: str = "vitb", max_res: int = 640,
                 cache_dir: str | None = None) -> None:
        self._stage = DepthStage(model=model, max_res=max_res,
                                 cache_dir=cache_dir)

    def warmup(self) -> None:
        """Load weights and run a dummy pass (call once after construction)."""
        self._stage.warmup()

    def cook_tensor(self, top) -> torch.Tensor:
        """Run depth estimation on a TD TOP input.

        Args:
            top: TouchDesigner TOP operator (Script TOP input).

        Returns:
            (H, W) float32 CUDA tensor, depth normalised to [0, 1].
        """
        # TD returns (H, W, 4) uint8 RGBA
        arr = top.numpyArray(delayed=False)  # type: ignore[attr-defined]
        rgb = arr[:, :, :3].astype(np.float32) / 255.0  # (H, W, 3)
        frame = torch.from_numpy(rgb)
        if torch.cuda.is_available():
            frame = frame.cuda()
        return self._stage(frame)  # (H, W) float32

    def cook_numpy(self, top) -> np.ndarray:
        """Run depth estimation and return a numpy array for copyNumpyArray.

        Returns:
            (H, W, 1) float32 array — single-channel depth in [0, 1].
            Feed directly to ``scriptOp.copyNumpyArray()``.
        """
        depth = self.cook_tensor(top).cpu().numpy()
        return depth[:, :, np.newaxis]  # (H, W, 1)


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
                 cache_dir: str | None = None) -> None:
        self._stage = DepthStage(model=model, max_res=max_res,
                                 cache_dir=cache_dir)

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

        print("[depthkit] Warming up depth model…", flush=True)
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

                # Pack depth into a single-channel RGBA tensor for export:
                # R = depth, G = B = A = 0
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
        "--max-frames", type=int, default=0,
        help="Stop after N frames, 0 = run forever (default: 0)"
    )
    args = parser.parse_args()

    server = DepthIPCServer(
        model=args.model,
        max_res=args.max_res,
        cache_dir=args.cache_dir,
    )
    server.run(max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
