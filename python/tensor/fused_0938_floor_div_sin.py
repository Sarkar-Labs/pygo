"""PyGo fused operation floor_div_sin; canonical eager semantics."""
from typing import Any
OP_NAME="floor_div_sin"

def fused_0938_floor_div_sin(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "floor" in {"sum","mean","max","min","argmax"}: r=getattr(x,"floor")(0)
    elif "floor"=="flatten": r=x.flatten()
    elif "floor"=="squeeze": r=x.squeeze()
    elif "floor"=="unsqueeze": r=x.unsqueeze(0)
    elif "floor"=="clone": r=x.clone()
    elif "floor"=="detach": r=x.detach()
    elif "floor"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"floor")()
        if 'sin': r=getattr(r,"sin")()
    return r

def fused_0938_floor_div_sin_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0938_floor_div_sin_reference(x:Any,y:Any):
    fused_0938_floor_div_sin_validate(x,y); return fused_0938_floor_div_sin(x,y)

def fused_0938_floor_div_sin_shape(x:Any,y:Any):
    r=fused_0938_floor_div_sin(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0938_floor_div_sin_numel(x:Any,y:Any):
    r=fused_0938_floor_div_sin(x,y); return r.numel()
