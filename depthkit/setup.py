"""Self-contained setup for depthkit TD asset.

Detects environment, installs dependencies, configures sys.path.
Designed to run inside TouchDesigner's Python or standalone.

Usage from TD Script DAT::

    exec(open('C:/path/to/depthkit/depthkit/setup.py').read())
    setup = DepthkitSetup()
    setup.diagnose()           # → dict with environment info
    setup.install(callback=print)  # → installs everything, logs to callback
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Known UV locations on Windows (winget install)
_UV_WINGET = Path(os.environ.get("LOCALAPPDATA", "")) / \
    "Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe/uv.exe"

# PyTorch index URLs by CUDA version
_TORCH_INDEX = {
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


class DepthkitSetup:
    """Environment setup and dependency installer for depthkit."""

    def __init__(self, venv_path: str | Path | None = None,
                 depthkit_root: str | Path | None = None):
        """
        Args:
            venv_path: Where to create/find the venv. Default: <depthkit_root>/.venv
            depthkit_root: Root of the depthkit repo. Default: auto-detect from this file.
        """
        if depthkit_root is None:
            # This file is at depthkit/setup.py → parent.parent = repo root
            depthkit_root = Path(__file__).resolve().parent.parent

        self.root = Path(depthkit_root)
        self.venv = Path(venv_path) if venv_path else self.root / ".venv"

    @property
    def python(self) -> Path:
        """Path to the venv Python executable."""
        if platform.system() == "Windows":
            return self.venv / "Scripts" / "python.exe"
        return self.venv / "bin" / "python"

    @property
    def site_packages(self) -> Path:
        """Path to the venv site-packages directory."""
        if platform.system() == "Windows":
            return self.venv / "Lib" / "site-packages"
        # Linux/Mac: lib/python3.XX/site-packages
        for p in (self.venv / "lib").glob("python3.*/site-packages"):
            return p
        return self.venv / "lib" / "site-packages"

    def _find_uv(self) -> Path | None:
        """Find UV package manager, or None if not available."""
        # Check PATH first
        uv = shutil.which("uv")
        if uv:
            return Path(uv)
        # Check known Windows location
        if _UV_WINGET.exists():
            return _UV_WINGET
        return None

    def diagnose(self) -> dict:
        """Detect environment state. Returns dict with all findings."""
        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "depthkit_root": str(self.root),
            "venv_path": str(self.venv),
            "venv_exists": self.venv.exists(),
            "uv_available": self._find_uv() is not None,
            "uv_path": str(self._find_uv()) if self._find_uv() else None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "depthkit_installed": False,
            "sharp_installed": False,
            "sharp_vendor_exists": (self.root / "vendor" / "ml-sharp").exists(),
        }

        # Check CUDA via nvidia-smi (works without torch)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,name",
                 "--format=csv,noheader"],
                text=True, timeout=5
            ).strip()
            if out:
                parts = out.split(", ")
                info["cuda_available"] = True
                info["gpu_name"] = parts[1] if len(parts) > 1 else "unknown"
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        # Detect CUDA toolkit version
        try:
            out = subprocess.check_output(
                ["nvidia-smi"], text=True, timeout=5
            )
            for line in out.split("\n"):
                if "CUDA Version:" in line:
                    ver = line.split("CUDA Version:")[1].strip().split()[0]
                    info["cuda_version"] = ver
                    break
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        # Check if depthkit is installed in venv
        if self.venv.exists():
            try:
                subprocess.check_call(
                    [str(self.python), "-c", "import depthkit"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
                )
                info["depthkit_installed"] = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            try:
                subprocess.check_call(
                    [str(self.python), "-c", "import sharp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
                )
                info["sharp_installed"] = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        return info

    def install(self, cuda_version: str = "auto",
                log=None) -> bool:
        """Install depthkit + dependencies into venv.

        Args:
            cuda_version: "auto" (detect), "12.4", "12.1", "11.8", or "cpu".
            log: Callable for progress messages (e.g. print). Default: print.

        Returns:
            True if install succeeded.
        """
        if log is None:
            log = print

        uv = self._find_uv()
        if uv is None:
            log("[depthkit] ERROR: UV not found. Install: winget install astral-sh.uv")
            return False

        # Detect CUDA
        if cuda_version == "auto":
            info = self.diagnose()
            raw = info.get("cuda_version") or ""
            # Map to closest supported version
            if raw.startswith("13.") or raw.startswith("12.4") or raw.startswith("12.5") or raw.startswith("12.6"):
                cuda_version = "12.4"
            elif raw.startswith("12."):
                cuda_version = "12.1"
            elif raw.startswith("11."):
                cuda_version = "11.8"
            else:
                cuda_version = "cpu"
            log(f"[depthkit] Detected CUDA: {raw} → using torch index: cu{cuda_version.replace('.', '')}")

        torch_index = _TORCH_INDEX.get(cuda_version, _TORCH_INDEX["12.4"])

        def run(cmd, desc):
            log(f"[depthkit] {desc}...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                log(f"[depthkit] ERROR: {desc} failed:\n{result.stderr[-500:]}")
                return False
            return True

        # Step 1: Create venv
        if not self.venv.exists():
            if not run([str(uv), "venv", str(self.venv), "--python", "3.11"],
                       f"Creating venv at {self.venv}"):
                return False
        else:
            log(f"[depthkit] Venv exists at {self.venv}")

        # Step 2: Install PyTorch
        if not run([str(uv), "pip", "install",
                    "torch", "torchvision",
                    "--index-url", torch_index,
                    "--python", str(self.python)],
                   f"Installing PyTorch ({cuda_version})"):
            return False

        # Step 3: Install depthkit
        if not run([str(uv), "pip", "install", "-e", f"{self.root}[dev]",
                    "--python", str(self.python)],
                   "Installing depthkit"):
            return False

        # Step 4: Install SHARP (if vendor exists)
        sharp_path = self.root / "vendor" / "ml-sharp"
        if sharp_path.exists():
            if not run([str(uv), "pip", "install", "-e", str(sharp_path),
                        "--python", str(self.python)],
                       "Installing SHARP (3DGS)"):
                return False

        # Step 5: Fix Windows SSL
        if platform.system() == "Windows":
            run([str(uv), "pip", "install", "pip-system-certs",
                 "--python", str(self.python)],
                "Installing SSL fix for Windows")

        log("[depthkit] Setup complete.")
        return True

    def configure_sys_path(self) -> None:
        """Add venv site-packages and depthkit root to sys.path.

        Call this inside TouchDesigner before importing depthkit.
        Also cleans up stale worktree paths.
        """
        # Remove stale paths
        sys.path[:] = [p for p in sys.path if ".worktrees" not in p]

        sp = str(self.site_packages)
        root = str(self.root)

        if sp not in sys.path:
            sys.path.insert(0, sp)
        if root not in sys.path:
            sys.path.insert(0, root)

        # Purge cached modules to ensure fresh imports
        for mod in list(sys.modules.keys()):
            if "depthkit" in mod or "sharp" in mod or "transformers" in mod:
                del sys.modules[mod]
