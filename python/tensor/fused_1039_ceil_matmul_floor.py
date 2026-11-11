"""PyGo fused operation ceil_matmul_floor; canonical eager semantics."""
from typing import Any
OP_NAME="ceil_matmul_floor"

def fused_1039_ceil_matmul_floor(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "ceil" in {"sum","mean","max","min","argmax"}: r=getattr(x,"ceil")(0)
    elif "ceil"=="flatten": r=x.flatten()
    elif "ceil"=="squeeze": r=x.squeeze()
    elif "ceil"=="unsqueeze": r=x.unsqueeze(0)
    elif "ceil"=="clone": r=x.clone()
    elif "ceil"=="detach": r=x.detach()
    elif "ceil"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"ceil")()
        if 'floor': r=getattr(r,"floor")()
    return r

def fused_1039_ceil_matmul_floor_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1039_ceil_matmul_floor_reference(x:Any,y:Any):
    fused_1039_ceil_matmul_floor_validate(x,y); return fused_1039_ceil_matmul_floor(x,y)

def fused_1039_ceil_matmul_floor_shape(x:Any,y:Any):
    r=fused_1039_ceil_matmul_floor(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1039_ceil_matmul_floor_numel(x:Any,y:Any):
    r=fused_1039_ceil_matmul_floor(x,y); return r.numel()
