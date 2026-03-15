"""CLI driver: image / video / webcam → PLY file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from depthkit.stages.depth import DepthStage
from depthkit.stages.pointcloud import PointCloudStage
from depthkit.stages.ply import PLYStage


def frame_to_tensor(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    """Convert BGR uint8 frame from OpenCV to float32 RGB tensor on device."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0
    return t.to(device)


def process_frame(frame_bgr: np.ndarray, depth_stage: DepthStage,
                  pc_stage: PointCloudStage, ply_stage: PLYStage,
                  device: str) -> bytes:
    """Run full pipeline on a single frame, return PLY bytes."""
    frame = frame_to_tensor(frame_bgr, device)
    depth = depth_stage(frame)
    points = pc_stage(depth, frame)
    return ply_stage(points)


def save_ply(ply_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ply_bytes)


def build_stages(model: str, max_res: int, fov: float,
                 max_points: int, scale: float) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_stage = DepthStage(model=model, max_res=max_res)
    pc_stage = PointCloudStage(fov_deg=fov, max_points=max_points)
    ply_stage = PLYStage(scale=scale)
    return depth_stage, pc_stage, ply_stage, device


def cmd_image(args: argparse.Namespace) -> None:
    """Process a single image file."""
    src = Path(args.input)
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    depth_s, pc_s, ply_s, device = build_stages(
        args.model, args.max_res, args.fov, args.max_points, args.scale)

    print(f"Loading model ({args.model})...")
    depth_s.warmup()

    frame = cv2.imread(str(src))
    if frame is None:
        print(f"Error: could not read {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {src.name} ({frame.shape[1]}x{frame.shape[0]})...")
    ply_bytes = process_frame(frame, depth_s, pc_s, ply_s, device)

    out = Path(args.output)
    save_ply(ply_bytes, out)
    print(f"Saved: {out}")


def cmd_video(args: argparse.Namespace) -> None:
    """Process a video file — one PLY per N frames, or a single merged cloud."""
    src = Path(args.input)
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    depth_s, pc_s, ply_s, device = build_stages(
        args.model, args.max_res, args.fov, args.max_points, args.scale)

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {src.name} — {total} frames @ {fps:.1f}fps")
    print(f"Loading model ({args.model})...")
    depth_s.warmup()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    step = max(1, args.step)

    with tqdm(total=total, unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                ply_bytes = process_frame(frame, depth_s, pc_s, ply_s, device)
                out_path = out_dir / f"frame_{saved:06d}.ply"
                save_ply(ply_bytes, out_path)
                saved += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Saved {saved} PLY files to {out_dir}")


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Capture a single frame from webcam and save as PLY."""
    cam_id = int(args.input.replace("webcam:", "").replace("webcam", "0"))

    depth_s, pc_s, ply_s, device = build_stages(
        args.model, args.max_res, args.fov, args.max_points, args.scale)

    print(f"Opening webcam {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: could not open webcam {cam_id}", file=sys.stderr)
        sys.exit(1)

    # Discard first few frames (camera warmup)
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("Error: could not capture frame", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model ({args.model})...")
    depth_s.warmup()
    print(f"Processing frame ({frame.shape[1]}x{frame.shape[0]})...")

    ply_bytes = process_frame(frame, depth_s, pc_s, ply_s, device)
    out = Path(args.output)
    save_ply(ply_bytes, out)
    print(f"Saved: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthkit",
        description="Depth estimation → point cloud → 3DGS PLY",
    )
    # Common options
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", choices=["vits", "vitb", "vitl"], default="vitb",
                        help="Depth Anything V2 model variant (default: vitb)")
    common.add_argument("--max-res", type=int, default=640,
                        help="Max resolution for depth inference (default: 640)")
    common.add_argument("--fov", type=float, default=60.0,
                        help="Horizontal field of view in degrees (default: 60)")
    common.add_argument("--max-points", type=int, default=200_000,
                        help="Max points in output cloud (default: 200000)")
    common.add_argument("--scale", type=float, default=0.002,
                        help="Gaussian scale in scene units (default: 0.002)")

    sub = parser.add_subparsers(dest="command", required=True)

    # image subcommand
    p_img = sub.add_parser("image", parents=[common], help="Process a single image")
    p_img.add_argument("--input", required=True, help="Input image path")
    p_img.add_argument("--output", required=True, help="Output .ply path")

    # video subcommand
    p_vid = sub.add_parser("video", parents=[common], help="Process a video file")
    p_vid.add_argument("--input", required=True, help="Input video path")
    p_vid.add_argument("--output", required=True, help="Output directory for PLY files")
    p_vid.add_argument("--step", type=int, default=1,
                       help="Process every Nth frame (default: 1)")

    # snapshot subcommand
    p_snap = sub.add_parser("snapshot", parents=[common], help="Capture from webcam")
    p_snap.add_argument("--input", default="webcam:0",
                        help="Webcam source: 'webcam:0' (default: webcam:0)")
    p_snap.add_argument("--output", required=True, help="Output .ply path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "image":
        cmd_image(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "snapshot":
        cmd_snapshot(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
