"""PyGo fused operation max_unsqueeze; canonical eager semantics."""
from typing import Any
OP_NAME="max_unsqueeze"

def fused_1724_max_unsqueeze(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "max" in {"sum","mean","max","min","argmax"}: r=getattr(x,"max")(0)
    elif "max"=="flatten": r=x.flatten()
    elif "max"=="squeeze": r=x.squeeze()
    elif "max"=="unsqueeze": r=x.unsqueeze(0)
    elif "max"=="clone": r=x.clone()
    elif "max"=="detach": r=x.detach()
    elif "max"=="contiguous": r=x.contiguous()
    else:
        if "unsqueeze"=="add": r=x+y
        elif "unsqueeze"=="sub": r=x-y
        elif "unsqueeze"=="mul": r=x*y
        elif "unsqueeze"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"max")()
        if None: r=getattr(r,"None")()
    return r

def fused_1724_max_unsqueeze_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1724_max_unsqueeze_reference(x:Any,y:Any):
    fused_1724_max_unsqueeze_validate(x,y); return fused_1724_max_unsqueeze(x,y)

def fused_1724_max_unsqueeze_shape(x:Any,y:Any):
    r=fused_1724_max_unsqueeze(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1724_max_unsqueeze_numel(x:Any,y:Any):
    r=fused_1724_max_unsqueeze(x,y); return r.numel()
