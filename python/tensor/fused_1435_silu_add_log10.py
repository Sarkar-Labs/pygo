"""PyGo fused operation silu_add_log10; canonical eager semantics."""
from typing import Any
OP_NAME="silu_add_log10"

def fused_1435_silu_add_log10(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "silu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"silu")(0)
    elif "silu"=="flatten": r=x.flatten()
    elif "silu"=="squeeze": r=x.squeeze()
    elif "silu"=="unsqueeze": r=x.unsqueeze(0)
    elif "silu"=="clone": r=x.clone()
    elif "silu"=="detach": r=x.detach()
    elif "silu"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"silu")()
        if 'log10': r=getattr(r,"log10")()
    return r

def fused_1435_silu_add_log10_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1435_silu_add_log10_reference(x:Any,y:Any):
    fused_1435_silu_add_log10_validate(x,y); return fused_1435_silu_add_log10(x,y)

def fused_1435_silu_add_log10_shape(x:Any,y:Any):
    r=fused_1435_silu_add_log10(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1435_silu_add_log10_numel(x:Any,y:Any):
    r=fused_1435_silu_add_log10(x,y); return r.numel()
