"""PyGo fused operation log10_sub_cos; canonical eager semantics."""
from typing import Any
OP_NAME="log10_sub_cos"

def fused_1661_log10_sub_cos(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "log10" in {"sum","mean","max","min","argmax"}: r=getattr(x,"log10")(0)
    elif "log10"=="flatten": r=x.flatten()
    elif "log10"=="squeeze": r=x.squeeze()
    elif "log10"=="unsqueeze": r=x.unsqueeze(0)
    elif "log10"=="clone": r=x.clone()
    elif "log10"=="detach": r=x.detach()
    elif "log10"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"log10")()
        if 'cos': r=getattr(r,"cos")()
    return r

def fused_1661_log10_sub_cos_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1661_log10_sub_cos_reference(x:Any,y:Any):
    fused_1661_log10_sub_cos_validate(x,y); return fused_1661_log10_sub_cos(x,y)

def fused_1661_log10_sub_cos_shape(x:Any,y:Any):
    r=fused_1661_log10_sub_cos(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1661_log10_sub_cos_numel(x:Any,y:Any):
    r=fused_1661_log10_sub_cos(x,y); return r.numel()
