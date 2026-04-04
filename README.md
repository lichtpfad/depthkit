# depthkit

> Modular depth estimation + 3D Gaussian Splatting pipeline for TouchDesigner

**Status:** alpha (v0.1.0) &bull; **Python:** 3.11+ &bull; **GPU:** CUDA 12.4

## What

depthkit turns a single image into depth maps, point clouds, and 3D Gaussian Splats — rendered in real-time inside TouchDesigner. GPU-native pipeline with no CPU roundtrips.

## Features

- **Depth estimation** — monocular depth from any image, GPU-native (2x FPS vs naive implementations)
- **3D Gaussian Splatting** — single image → 3DGS PLY via SHARP model (1M+ Gaussians)
- **TouchDesigner integration** — CUDALink driver, texture packing, GLSL billboard rendering
- **Modular pipeline** — Stage protocol, chain any combination of stages
- **CLI** — `depthkit depth`, `depthkit gaussian`, `depthkit pointcloud`

## Quick Start

```bash
# Install (requires Python 3.11, CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Generate depth map
depthkit depth input.png -o depth.png

# Generate 3DGS from single image
depthkit gaussian input.png -o output.ply

# Run in TouchDesigner — see docs/td-setup.md
```

## Pipeline Architecture

```
Image
  ├── DepthStage ──→ PointCloudStage ──→ PLYStage ──→ .ply
  └── GaussianStage ──→ 3DGS .ply ──→ SplatLoader ──→ SplatsTD ──→ TD textures
                                                                      ↓
                                                              GLSL billboard
                                                              rendering (>30 FPS)
```

Each stage implements the `Stage` protocol — independently testable, composable.

## TouchDesigner

Two integration paths:

1. **CUDALink driver** (`depthkit.drivers.td`) — GPU memory sharing, real-time depth
2. **Splat renderer** (`depthkit.drivers.splats_td`) — load 3DGS PLY, pack to textures, GLSL rendering

Setup guides: [docs/td-setup.md](docs/td-setup.md), [docs/td-splats-setup.md](docs/td-splats-setup.md)

## Status

| Area | Status |
|------|--------|
| Depth pipeline (CPU→GPU) | Done |
| Point cloud export | Done |
| 3DGS generation (SHARP) | Done |
| 3DGS TD renderer | Done |
| .tox component | In progress (#19) |
| CLI | Done |
| Tests | 10 files, pytest |

## Requirements

- Python 3.11+
- CUDA 12.4 + PyTorch
- TouchDesigner 2024+ (for TD integration)
- ~4GB VRAM for depth, ~8GB for 3DGS

## License

TBD
