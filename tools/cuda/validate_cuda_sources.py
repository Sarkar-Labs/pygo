#!/usr/bin/env python3
"""Validate and optionally compile every PyGo CUDA translation unit with nvcc.

Usage:
  python tools/cuda/validate_cuda_sources.py
  python tools/cuda/validate_cuda_sources.py --nvcc /usr/local/cuda/bin/nvcc
  python tools/cuda/validate_cuda_sources.py --keep-objects

Without nvcc the script still performs strict source/header validation and exits
with a distinct status (2) so CPU-only CI can distinguish "toolkit unavailable"
from "CUDA source validation failed".
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cpp" / "cuda"


def function_signatures(header: str):
    return re.findall(r"\b(?:void|Tensor|bool|float|double|int|int64_t)\s+(\w+)\s*\(([^;]*)\);", header, re.S)


def definition_params(source: str, name: str):
    m = re.search(r"\b(?:void|Tensor|bool|float|double|int|int64_t)\s+" + re.escape(name) + r"\s*\(([^)]*)\)", source, re.S)
    return [] if not m else [p for p in m.group(1).split(",") if p.strip()]


def static_validate(files):
    errors = []
    for cu in files:
        source = cu.read_text(errors="replace")
        if "[cite:" in source or "```" in source:
            errors.append(f"{cu}: contains documentation/citation artifact")
        includes = re.findall(r'#include\s+"([^"]+\.cuh)"', source)
        for inc in includes:
            header_path = CUDA_DIR / inc
            if not header_path.exists():
                errors.append(f"{cu}: missing local header {inc}")
                continue
            header = header_path.read_text(errors="replace")
            for name, params in function_signatures(header):
                hp = [p for p in params.split(",") if p.strip()]
                dp = definition_params(source, name)
                if not dp:
                    errors.append(f"{cu}: declaration {name} has no definition")
                elif len(hp) != len(dp):
                    errors.append(f"{cu}: {name}: header has {len(hp)} parameters, definition has {len(dp)}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvcc", default=None, help="Path to nvcc; defaults to PATH")
    ap.add_argument("--keep-objects", action="store_true")
    args = ap.parse_args()
    files = sorted(CUDA_DIR.glob("*.cu"))
    if not files:
        print("ERROR: no CUDA translation units found")
        return 1

    static_errors = static_validate(files)
    print(f"PyGo CUDA translation units: {len(files)}")
    if static_errors:
        for e in static_errors:
            print("ERROR:", e)
        return 1
    print("Static CUDA source/header validation: PASS")

    nvcc = args.nvcc or shutil.which("nvcc")
    if not nvcc:
        print("nvcc: NOT FOUND (CUDA compilation was not attempted)")
        return 2

    outdir = Path(tempfile.mkdtemp(prefix="pygo-nvcc-"))
    failures = []
    print(f"nvcc: {nvcc}")
    print(f"Object directory: {outdir}")
    for cu in files:
        obj = outdir / (cu.stem + ".o")
        cmd = [nvcc, "-std=c++17", "--expt-relaxed-constexpr", "-c", str(cu),
               "-o", str(obj), "-I", str(ROOT / "cpp" / "include"),
               "-I", str(ROOT / "cpp"), "-I", str(CUDA_DIR)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            failures.append((cu, result.stdout))
            print(f"FAIL {cu.name}")
            print(result.stdout[-4000:])
        else:
            print(f"PASS {cu.name}")

    if failures:
        print(f"CUDA nvcc compilation: FAIL ({len(failures)}/{len(files)} files)")
        return 1
    print(f"CUDA nvcc compilation: PASS ({len(files)}/{len(files)} files)")
    if not args.keep_objects:
        shutil.rmtree(outdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
