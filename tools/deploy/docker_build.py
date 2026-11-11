"""Command-line utility implementation for ``docker_build``."""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

def inspect_path(path: str) -> dict[str, object]:
    """Return deterministic metadata for a project artifact."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_file():
        data = p.read_bytes()
        return {"path": str(p), "kind": "file", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    files = [x for x in p.rglob("*") if x.is_file()]
    total = sum(x.stat().st_size for x in files)
    return {"path": str(p), "kind": "directory", "files": len(files), "bytes": total}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PyGo docker_build utility")
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args(argv)
    print(inspect_path(args.path))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
