#!/usr/bin/env python3
"""Report optional PyGo accelerator toolchains available on this machine."""
from __future__ import annotations
import shutil
import subprocess


def version(cmd: str) -> str | None:
    path = shutil.which(cmd)
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
        text = (out.stdout or out.stderr).splitlines()
        return text[0].strip() if text else "available"
    except Exception:
        return "available"


def main() -> int:
    nvcc = version("nvcc")
    hipcc = version("hipcc")
    print(f"CUDA/nvcc: {nvcc or 'not found'}")
    print(f"ROCm/hipcc: {hipcc or 'not found'}")
    print("TPU/XLA: API hooks only in PyGo 0.1.0; no bundled TPU runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
