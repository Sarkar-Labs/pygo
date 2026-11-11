"""PyGo fused operation log10_add_sign; canonical eager semantics."""
from typing import Any
OP_NAME="log10_add_sign"

def fused_1680_log10_add_sign(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "log10" in {"sum","mean","max","min","argmax"}: r=getattr(x,"log10")(0)
    elif "log10"=="flatten": r=x.flatten()
    elif "log10"=="squeeze": r=x.squeeze()
    elif "log10"=="unsqueeze": r=x.unsqueeze(0)
    elif "log10"=="clone": r=x.clone()
    elif "log10"=="detach": r=x.detach()
    elif "log10"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"log10")()
        if 'sign': r=getattr(r,"sign")()
    return r

def fused_1680_log10_add_sign_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1680_log10_add_sign_reference(x:Any,y:Any):
    fused_1680_log10_add_sign_validate(x,y); return fused_1680_log10_add_sign(x,y)

def fused_1680_log10_add_sign_shape(x:Any,y:Any):
    r=fused_1680_log10_add_sign(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1680_log10_add_sign_numel(x:Any,y:Any):
    r=fused_1680_log10_add_sign(x,y); return r.numel()
