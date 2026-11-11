"""Small ctypes loader for the stable PyGo C ABI.

The loader is intentionally dependency-free apart from the Python standard
library.  Applications can set PYGO_LIBRARY to an explicit shared-library
path, or use a CMake installation containing libpygo.
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path

class NativeError(RuntimeError):
    pass

def _candidates():
    explicit = os.environ.get("PYGO_LIBRARY")
    if explicit:
        yield Path(explicit)
    root = Path(__file__).resolve().parent
    for p in (root / "libpygo.so", root / "libpygo.dylib", root / "pygo.dll", root.parent / "lib" / "libpygo.so", root.parent / "lib" / "libpygo.dylib"):
        yield p

def load_native():
    for path in _candidates():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.pygo_abi_version.restype = ctypes.c_uint32
            if lib.pygo_abi_version() != 1:
                raise NativeError("unsupported PyGo native ABI")
            lib.pygo_last_error.restype = ctypes.c_char_p
            return lib
    raise NativeError("PyGo native library not found; build/install PyGo or set PYGO_LIBRARY")
