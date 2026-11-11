"""PyGo fused operation log10_div_log2; canonical eager semantics."""
from typing import Any
OP_NAME="log10_div_log2"

def fused_1703_log10_div_log2(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "log10" in {"sum","mean","max","min","argmax"}: r=getattr(x,"log10")(0)
    elif "log10"=="flatten": r=x.flatten()
    elif "log10"=="squeeze": r=x.squeeze()
    elif "log10"=="unsqueeze": r=x.unsqueeze(0)
    elif "log10"=="clone": r=x.clone()
    elif "log10"=="detach": r=x.detach()
    elif "log10"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"log10")()
        if 'log2': r=getattr(r,"log2")()
    return r

def fused_1703_log10_div_log2_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1703_log10_div_log2_reference(x:Any,y:Any):
    fused_1703_log10_div_log2_validate(x,y); return fused_1703_log10_div_log2(x,y)

def fused_1703_log10_div_log2_shape(x:Any,y:Any):
    r=fused_1703_log10_div_log2(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1703_log10_div_log2_numel(x:Any,y:Any):
    r=fused_1703_log10_div_log2(x,y); return r.numel()
