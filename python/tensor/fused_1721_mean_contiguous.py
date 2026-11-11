"""PyGo fused operation mean_contiguous; canonical eager semantics."""
from typing import Any
OP_NAME="mean_contiguous"

def fused_1721_mean_contiguous(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "mean" in {"sum","mean","max","min","argmax"}: r=getattr(x,"mean")(0)
    elif "mean"=="flatten": r=x.flatten()
    elif "mean"=="squeeze": r=x.squeeze()
    elif "mean"=="unsqueeze": r=x.unsqueeze(0)
    elif "mean"=="clone": r=x.clone()
    elif "mean"=="detach": r=x.detach()
    elif "mean"=="contiguous": r=x.contiguous()
    else:
        if "contiguous"=="add": r=x+y
        elif "contiguous"=="sub": r=x-y
        elif "contiguous"=="mul": r=x*y
        elif "contiguous"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"mean")()
        if None: r=getattr(r,"None")()
    return r

def fused_1721_mean_contiguous_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1721_mean_contiguous_reference(x:Any,y:Any):
    fused_1721_mean_contiguous_validate(x,y); return fused_1721_mean_contiguous(x,y)

def fused_1721_mean_contiguous_shape(x:Any,y:Any):
    r=fused_1721_mean_contiguous(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1721_mean_contiguous_numel(x:Any,y:Any):
    r=fused_1721_mean_contiguous(x,y); return r.numel()
