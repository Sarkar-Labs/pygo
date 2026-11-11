"""PyGo fused operation sqrt_mul_ceil; canonical eager semantics."""
from typing import Any
OP_NAME="sqrt_mul_ceil"

def fused_0502_sqrt_mul_ceil(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sqrt" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sqrt")(0)
    elif "sqrt"=="flatten": r=x.flatten()
    elif "sqrt"=="squeeze": r=x.squeeze()
    elif "sqrt"=="unsqueeze": r=x.unsqueeze(0)
    elif "sqrt"=="clone": r=x.clone()
    elif "sqrt"=="detach": r=x.detach()
    elif "sqrt"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sqrt")()
        if 'ceil': r=getattr(r,"ceil")()
    return r

def fused_0502_sqrt_mul_ceil_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0502_sqrt_mul_ceil_reference(x:Any,y:Any):
    fused_0502_sqrt_mul_ceil_validate(x,y); return fused_0502_sqrt_mul_ceil(x,y)

def fused_0502_sqrt_mul_ceil_shape(x:Any,y:Any):
    r=fused_0502_sqrt_mul_ceil(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0502_sqrt_mul_ceil_numel(x:Any,y:Any):
    r=fused_0502_sqrt_mul_ceil(x,y); return r.numel()
