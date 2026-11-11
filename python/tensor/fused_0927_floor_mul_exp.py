"""PyGo fused operation floor_mul_exp; canonical eager semantics."""
from typing import Any
OP_NAME="floor_mul_exp"

def fused_0927_floor_mul_exp(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "floor" in {"sum","mean","max","min","argmax"}: r=getattr(x,"floor")(0)
    elif "floor"=="flatten": r=x.flatten()
    elif "floor"=="squeeze": r=x.squeeze()
    elif "floor"=="unsqueeze": r=x.unsqueeze(0)
    elif "floor"=="clone": r=x.clone()
    elif "floor"=="detach": r=x.detach()
    elif "floor"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"floor")()
        if 'exp': r=getattr(r,"exp")()
    return r

def fused_0927_floor_mul_exp_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0927_floor_mul_exp_reference(x:Any,y:Any):
    fused_0927_floor_mul_exp_validate(x,y); return fused_0927_floor_mul_exp(x,y)

def fused_0927_floor_mul_exp_shape(x:Any,y:Any):
    r=fused_0927_floor_mul_exp(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0927_floor_mul_exp_numel(x:Any,y:Any):
    r=fused_0927_floor_mul_exp(x,y); return r.numel()
