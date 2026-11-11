"""PyGo fused operation min_unsqueeze; canonical eager semantics."""
from typing import Any
OP_NAME="min_unsqueeze"

def fused_1736_min_unsqueeze(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "min" in {"sum","mean","max","min","argmax"}: r=getattr(x,"min")(0)
    elif "min"=="flatten": r=x.flatten()
    elif "min"=="squeeze": r=x.squeeze()
    elif "min"=="unsqueeze": r=x.unsqueeze(0)
    elif "min"=="clone": r=x.clone()
    elif "min"=="detach": r=x.detach()
    elif "min"=="contiguous": r=x.contiguous()
    else:
        if "unsqueeze"=="add": r=x+y
        elif "unsqueeze"=="sub": r=x-y
        elif "unsqueeze"=="mul": r=x*y
        elif "unsqueeze"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"min")()
        if None: r=getattr(r,"None")()
    return r

def fused_1736_min_unsqueeze_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1736_min_unsqueeze_reference(x:Any,y:Any):
    fused_1736_min_unsqueeze_validate(x,y); return fused_1736_min_unsqueeze(x,y)

def fused_1736_min_unsqueeze_shape(x:Any,y:Any):
    r=fused_1736_min_unsqueeze(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1736_min_unsqueeze_numel(x:Any,y:Any):
    r=fused_1736_min_unsqueeze(x,y); return r.numel()
