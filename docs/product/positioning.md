# depthkit — Positioning

## One-Liner

Modular depth estimation + 3DGS rendering pipeline for TouchDesigner.

## For Whom

- TouchDesigner artists creating installations with depth/3D effects
- AV performers needing real-time depth-to-pointcloud visuals
- Hou2Touch School students learning computational depth
- Technical artists exploring Gaussian Splatting in real-time contexts

## Problem

Getting depth estimation and 3D Gaussian Splatting into TouchDesigner requires stitching together ML models, Python environments, CUDA drivers, and custom GLSL — a multi-day integration nightmare. Most TD artists give up or use pre-baked static depth maps. No existing .tox provides a clean camera-to-3D pipeline.

## Solution

depthkit is a modular Python package with a Stage protocol pipeline:

- **DepthStage** — GPU-native monocular depth estimation (no CPU roundtrips, 2x FPS vs naive)
- **PointCloudStage** — unproject depth to XYZ point clouds
- **PLYStage** — export standard PLY files
- **GaussianStage** — single image → 3DGS PLY via SHARP model (1.18M+ Gaussians)
- **SplatLoader + SplatsTD** — pack Gaussians into TD textures, render with custom GLSL billboarding
- **CUDALink driver** — direct GPU memory sharing with TouchDesigner

Ships as Python package (pip install) + .toe project + planned .tox component.

## Why Now

- 3DGS went from paper (2023) to production-ready models in 18 months — SHARP, gsplat, nerfstudio
- TouchDesigner 2024+ has CUDALink and Script TOP improvements enabling GPU-native ML integration
- No one has packaged this for TD artists yet — it's all notebook code and CLI tools
- First h2t-factory product candidate — validates the scan→build→package→promote pipeline

## Competitive Landscape

### Depth in TD

| Tool | Approach | Weakness |
|------|----------|----------|
| Torin Blankensmith DepthAnything .tox | Monolithic .tox, Patreon | No pipeline, no CLI, no 3DGS |
| olegchomp TDDepthAnything | TensorRT-accelerated | NVIDIA-only, no splat integration |
| TD 2025 experimental Depth Anything v2 | Native tutorial | Not a reusable component |

### 3DGS in TD

| Tool | Approach | Weakness |
|------|----------|----------|
| yeataro TD-Gaussian-Splatting | CUDA rasterizer | NVIDIA-only, viewer only (no generation) |
| Lake Heckaman TDGS 1.1 | Patreon, SPZ+PLY, particles | Most feature-rich viewer, but no depth→splat pipeline |
| Tim Gerritsen renderTOP | Free PLY renderer | Minimal, viewer only |

### Adjacent (Unity/Unreal/Web)

- **Unity:** Aras P. viewer (147 FPS / 6.1M splats), gsplat-unity
- **Unreal:** NanoGS (March 2026, Nanite-style LOD, free)
- **Web:** Three.js / Babylon.js 8.0 WebGPU viewers

### depthkit differentiators

1. **Only tool connecting depth estimation → 3DGS generation → rendering** in one pipeline. All competitors treat depth and splatting as separate workflows.
2. **Modular Stage protocol** — composable pipeline vs monolithic .tox files.
3. **Python package + CLI** — batch processing, automation, headless. All TD competitors are GUI-only.
4. **GLSL splat rendering** — no CUDA dependency for rendering (works on AMD too).

## Key Metrics

1. **PLY generation time** — seconds from single image to 3DGS PLY (target: <60s on RTX 3080+)
2. **TD render FPS** — real-time splat rendering frame rate (target: >30 FPS at 1080p)
3. **Sales/downloads** — Patreon/Gumroad (target: 50 in first month as first factory product)

## MVP Scope

### In

- Python package installable via pip (already done: v0.1.0)
- CLI driver: `depthkit depth`, `depthkit gaussian` (done)
- TD driver: CUDALink integration (done)
- 3DGS: SplatLoader + SplatsTD + GLSL renderer (done)
- .tox component with clean custom parameters (issue #19 — key deliverable)
- README.md with install + usage + GIF demo
- Landing page for Patreon/Gumroad listing

### Out (for now)

- Video input (webcam, NDI) — single image only for MVP
- Multi-view reconstruction
- Training custom 3DGS models (uses pre-trained SHARP)
- Web viewer

## Definition of Done

- [ ] .tox drops into TD, loads PLY, renders 3DGS at >30 FPS
- [ ] `pip install depthkit` works on clean Python 3.11 + CUDA 12.4
- [ ] CLI: `depthkit gaussian image.png -o output.ply` produces valid 3DGS
- [ ] README.md with GIF demo, install instructions, feature list
- [ ] Landing page published
- [ ] Listed on Patreon or Gumroad with price point
- [ ] LICENSE file added (code: MIT, check SHARP model separately)

## Factory Context

First product candidate for h2t-factory pipeline. Validates:
- factory-scan → factory-pick → factory-research → factory-build → factory-package → factory-promote → factory-publish

Success here = template for all future factory products (assets, plugins, tools).
