# depthkit

Modular depth estimation pipeline: camera → point cloud → TouchDesigner / PLY.

## Environment

Uses **UV** for dependency management (global package cache — torch not re-downloaded per project).

UV executable: `C:\Users\stani\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe`

### Setup

```bash
UV="C:/Users/stani/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe/uv.exe"

# Create venv
"$UV" venv .venv --python 3.11

# Install PyTorch (CUDA 12.4) — uses global cache, fast if already downloaded
"$UV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install depthkit + dev deps
"$UV" pip install -e ".[dev]"
```

### Run tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

### Python

Always use `.venv/Scripts/python.exe` in this repo — NOT C:\work\nerf\venv.

## Worktrees

Stored in `.worktrees/` (gitignored).
