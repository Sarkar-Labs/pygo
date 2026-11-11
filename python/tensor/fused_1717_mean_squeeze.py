"""PyGo fused operation mean_squeeze; canonical eager semantics."""
from typing import Any
OP_NAME="mean_squeeze"

def fused_1717_mean_squeeze(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "mean" in {"sum","mean","max","min","argmax"}: r=getattr(x,"mean")(0)
    elif "mean"=="flatten": r=x.flatten()
    elif "mean"=="squeeze": r=x.squeeze()
    elif "mean"=="unsqueeze": r=x.unsqueeze(0)
    elif "mean"=="clone": r=x.clone()
    elif "mean"=="detach": r=x.detach()
    elif "mean"=="contiguous": r=x.contiguous()
    else:
        if "squeeze"=="add": r=x+y
        elif "squeeze"=="sub": r=x-y
        elif "squeeze"=="mul": r=x*y
        elif "squeeze"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"mean")()
        if None: r=getattr(r,"None")()
    return r

def fused_1717_mean_squeeze_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1717_mean_squeeze_reference(x:Any,y:Any):
    fused_1717_mean_squeeze_validate(x,y); return fused_1717_mean_squeeze(x,y)

def fused_1717_mean_squeeze_shape(x:Any,y:Any):
    r=fused_1717_mean_squeeze(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1717_mean_squeeze_numel(x:Any,y:Any):
    r=fused_1717_mean_squeeze(x,y); return r.numel()
