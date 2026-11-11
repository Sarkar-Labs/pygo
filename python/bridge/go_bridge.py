"""
PyGo Python Bridge — Go CGo → Python → C++ (pybind11)

Go calls Python functions via embedded CPython interpreter.
Python calls C++ kernels via pybind11 bindings.
"""
import ctypes
import sys
from typing import Any
try:
    import pygo_cpp  # optional pybind11 extension
except ImportError:
    pygo_cpp = None

def _require_cpp_bridge():
    if pygo_cpp is None:
        raise RuntimeError("pygo_cpp extension is not installed; build the optional C++ Python extension before using GoBridge")

class GoBridge:
    """Receives calls from Go via CGo/CPython and dispatches to C++ kernels."""

    def tensor_new(self, shape, dtype, device):
        _require_cpp_bridge()
        return pygo_cpp.tensor_new(shape, dtype, device)

    def tensor_backward(self, tensor_ptr):
        _require_cpp_bridge()
        return pygo_cpp.autograd_backward(tensor_ptr)

    def tensor_matmul(self, a_ptr, b_ptr):
        _require_cpp_bridge()
        return pygo_cpp.kernels.matmul(a_ptr, b_ptr)

    def nn_forward(self, module_ptr, input_ptr):
        _require_cpp_bridge()
        return pygo_cpp.nn_forward(module_ptr, input_ptr)

    def optim_step(self, optim_ptr):
        _require_cpp_bridge()
        return pygo_cpp.optim_step(optim_ptr)

# Global bridge instance — called by CGo
_bridge = GoBridge()

# C-callable exports for CGo
import ctypes
@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int)
def pygo_tensor_new(shape_ptr, ndim, dtype, device):
    shape = [shape_ptr[i] for i in range(ndim)]
    return _bridge.tensor_new(shape, dtype, device)
