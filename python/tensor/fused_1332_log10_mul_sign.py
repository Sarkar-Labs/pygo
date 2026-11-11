"""Executable fused PyGo tensor operation: log10_mul_sign."""
from __future__ import annotations
from typing import Any

OP_NAME = "log10_mul_sign"
PRIMARY = "log10"
BINARY = "mul"
SECONDARY = 'sign'

def _binary(x: Any, y: Any):
    if BINARY == "add": return x + y
    if BINARY == "sub": return x - y
    if BINARY == "mul": return x * y
    if BINARY == "div": return x / y
    return x.matmul(y)

def _unary(value: Any, op: str):
    method = getattr(value, op, None)
    if method is None: raise AttributeError(f"{type(value).__name__} lacks {op} for {OP_NAME}")
    return method()

def fused_1332_log10_mul_sign(x: Any, y: Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires two tensors")
    result = _unary(_binary(x,y), PRIMARY)
    if SECONDARY: result = _unary(result, SECONDARY)
    return result

def fused_1332_log10_mul_sign_contiguous(x: Any, y: Any):
    return fused_1332_log10_mul_sign(x.contiguous(), y.contiguous())

def fused_1332_log10_mul_sign_cpu(x: Any, y: Any):
    return fused_1332_log10_mul_sign(x.cpu(), y.cpu())

def fused_1332_log10_mul_sign_clone(x: Any, y: Any):
    return fused_1332_log10_mul_sign(x.clone(), y.clone())

def fused_1332_log10_mul_sign_detached(x: Any, y: Any):
    return fused_1332_log10_mul_sign(x.detach(), y.detach())

def fused_1332_log10_mul_sign_same_shape(x: Any, y: Any) -> bool:
    return tuple(x.shape()) == tuple(y.shape()) if callable(getattr(x,'shape',None)) else tuple(x.shape)==tuple(y.shape)

def fused_1332_log10_mul_sign_metadata(x: Any) -> dict[str, Any]:
    shape = x.shape() if callable(getattr(x,'shape',None)) else x.shape
    return {"name": OP_NAME, "shape": tuple(shape), "ndim": len(shape), "numel": x.numel()}
def fused_1332_log10_mul_sign_validate(x: Any, y: Any) -> None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: operands must be defined")
    if hasattr(x, "device") and hasattr(y, "device") and x.device != y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1332_log10_mul_sign_reference(x: Any, y: Any):
    """Reference eager path used by differential tests."""
    return fused_1332_log10_mul_sign(x, y)
