# depthkit — MVP Scope

## One-Liner

Drag-drop .tox for depth estimation + 3DGS rendering in TouchDesigner.

## In Scope (MVP)

- .tox component: load PLY → render 3DGS in TD (issue #19)
- Python package: `pip install depthkit` (already working v0.1.0)
- CLI: `depthkit depth`, `depthkit gaussian` commands
- TD CUDALink driver for GPU-native depth
- 3DGS pipeline: SplatLoader → texture packing → GLSL rendering
- README.md with GIF demo and install guide
- Landing page for Patreon/Gumroad
- LICENSE (MIT for code)

## Out of Scope (post-MVP)

- Video/webcam/NDI input (single image only)
- Multi-view 3DGS reconstruction
- Training custom models
- Web-based 3DGS viewer
- Second output as SOP geometry (issue #15)

## Definition of Done

- [ ] .tox loads in TD 2024+, user sets PLY path, renders at >30 FPS
- [ ] `pip install depthkit` on clean Python 3.11 + CUDA 12.4 succeeds
- [ ] `depthkit gaussian image.png -o out.ply` produces valid 3DGS PLY
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] README.md with GIF, install, features, architecture diagram
- [ ] Landing page ready
- [ ] Patreon/Gumroad listing live with price
- [ ] LICENSE committed

## Dependencies

- **Blocks:** h2t-factory publish pipeline, Patreon asset pack
- **Blocked by:** issue #19 (.tox packaging), SHARP model license check

## Horizon

**Target:** 1 month (as declared in projects.yaml)
**Effort:** ~20 hours — .tox packaging (~8h), README+landing (~4h), factory-package (~4h), testing+polish (~4h)
