"""In-process CPython bridge for the Go-first PyGo runtime.

This module is intentionally small and dependency-light.  The Go executable
embeds CPython, initializes this package, and (when PyGo was built with
pybind11) exposes ``_pygo_native`` from the same process.  Tensor capsules are
borrowed views over native storage; this module never starts a Python
subprocess and never copies tensor storage merely to cross the language
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # The extension exists only in an embedded/native build with pybind11.
    import _pygo_native as _native
except ImportError:  # Source-tree/Python-only environments remain usable.
    _native = None


@dataclass(frozen=True)
class EmbeddedRuntimeInfo:
    """Immutable description of the native runtime visible to Python."""

    embedded: bool
    pybind11: bool
    zero_copy: bool
    architecture: str


def native_available() -> bool:
    """Return whether the statically linked native extension is available."""
    return _native is not None


def runtime_info() -> EmbeddedRuntimeInfo:
    """Return runtime capabilities without starting another process."""
    if _native is None:
        return EmbeddedRuntimeInfo(
            embedded=False,
            pybind11=False,
            zero_copy=False,
            architecture="Go -> CGo -> CPython -> Python-only runtime",
        )
    return EmbeddedRuntimeInfo(
        embedded=True,
        pybind11=True,
        zero_copy=bool(_native.has_zero_copy()),
        architecture=str(_native.architecture()),
    )


def tensor_view(capsule: Any) -> Any:
    """Create a metadata view over a Go/CGo ``PyGoTensor`` capsule.

    The returned object contains the original native data address, shape,
    strides, dtype and device.  No tensor bytes are copied.
    """
    if _native is None:
        raise RuntimeError(
            "PyGo native extension is unavailable; build libpygo with "
            "embedded CPython and pybind11 support"
        )
    return _native.tensor_view(capsule)


__all__ = ["EmbeddedRuntimeInfo", "native_available", "runtime_info", "tensor_view"]
