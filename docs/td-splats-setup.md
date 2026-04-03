# depthkit-splats: TouchDesigner Setup

## Network Architecture

```
Base COMP "depthkit_splats"
├── Script TOP "position"      — (H, W, 4) float32 position texture
├── Script TOP "color"         — (H, W, 4) float32 color texture
├── Script TOP "scale_opacity" — (H, W, 4) float32 scale + opacity
├── Script TOP "rotation"      — (H, W, 4) float32 quaternion
├── Grid SOP "quad"            — 1×1 quad (Rows 2, Cols 2)
├── GLSL MAT "splat_mat"       — Gaussian splat shader
├── Geometry COMP "renderer"
│   ├── In SOP → quad
│   └── Render TOP "render"
└── Out TOP                    — final output
```

## Custom Parameters (on Base COMP)

| Parameter | Type   | Default | Description                          |
|-----------|--------|---------|--------------------------------------|
| Plyfile   | File   | ""      | Path to .ply file                    |
| Texwidth  | Int    | 1024    | Texture packing width                |
| Splatscale| Float  | 1.0     | Global scale multiplier              |

## Script TOP Code

All four Script TOPs share one `SplatsTD` instance. Create a shared
Text DAT "splats_shared" with:

```python
import sys
sys.path.insert(0, r"C:\work\depthkit")
from depthkit.drivers.splats_td import SplatsTD

splats = None

def ensure_loaded(base):
    global splats
    path = base.par.Plyfile.eval()
    if not path:
        return None
    if splats is None or getattr(splats, '_path', None) != path:
        splats = SplatsTD(tex_width=int(base.par.Texwidth))
        splats.load(path)
        splats._path = path
    return splats
```

Then each Script TOP uses:

```python
# position Script TOP
def onCook(scriptOp):
    shared = op('splats_shared').module
    s = shared.ensure_loaded(scriptOp.parent())
    if s:
        scriptOp.copyNumpyArray(s.position_texture)
```

Replace `position_texture` with `color_texture`, `scale_opacity_texture`,
or `rotation_texture` for the other three Script TOPs.

## GLSL MAT Setup

1. **Vertex shader:** paste contents of `depthkit/glsl/splat_vertex.glsl`
2. **Pixel shader:** paste contents of `depthkit/glsl/splat_pixel.glsl`
3. **Uniforms → Samplers:**
   - `uPositionTex` → position Script TOP
   - `uColorTex` → color Script TOP
   - `uScaleOpacity` → scale_opacity Script TOP
   - `uRotationTex` → rotation Script TOP
4. **Uniforms → Integers:**
   - `uTexWidth`: 1024 (match Texwidth parameter)
   - `uNumGaussians`: from `_meta["num_gaussians"]`
5. **Uniforms → Floats:**
   - `uSplatScale`: from Splatscale parameter (default 1.0)

## Geometry COMP + Render TOP

1. **Grid SOP:** Rows 2, Cols 2 (minimal quad, 2 triangles)
2. **GLSL MAT:** set `Num Instances` = num_gaussians
3. **Geometry COMP:** apply splat_mat as Material
4. **Render TOP:** connect Camera COMP with orbit controls

## Render Settings

- **Depth test:** ON (sort handles ordering)
- **Blending:** Premultiplied Alpha (fragColor outputs pre-multiplied RGB)
- **Background:** Black or transparent

## Performance Notes

- 1M Gaussians @ 1024 width → 4 textures × 1024×1024 × RGBA32F ≈ 64 MB VRAM
- Sort once on load for static scenes
- For camera motion: call `sort_by_depth()` periodically (not every frame for >100K)
- Discard threshold (0.004) eliminates ~30% of fragment work
