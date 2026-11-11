"""PyGo fused operation cos_add_floor; canonical eager semantics."""
from typing import Any
OP_NAME="cos_add_floor"

def fused_0855_cos_add_floor(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "cos" in {"sum","mean","max","min","argmax"}: r=getattr(x,"cos")(0)
    elif "cos"=="flatten": r=x.flatten()
    elif "cos"=="squeeze": r=x.squeeze()
    elif "cos"=="unsqueeze": r=x.unsqueeze(0)
    elif "cos"=="clone": r=x.clone()
    elif "cos"=="detach": r=x.detach()
    elif "cos"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"cos")()
        if 'floor': r=getattr(r,"floor")()
    return r

def fused_0855_cos_add_floor_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0855_cos_add_floor_reference(x:Any,y:Any):
    fused_0855_cos_add_floor_validate(x,y); return fused_0855_cos_add_floor(x,y)

def fused_0855_cos_add_floor_shape(x:Any,y:Any):
    r=fused_0855_cos_add_floor(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0855_cos_add_floor_numel(x:Any,y:Any):
    r=fused_0855_cos_add_floor(x,y); return r.numel()
