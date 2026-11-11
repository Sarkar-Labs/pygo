"""PyGo fused operation sum_detach; canonical eager semantics."""
from typing import Any
OP_NAME="sum_detach"

def fused_1714_sum_detach(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sum" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sum")(0)
    elif "sum"=="flatten": r=x.flatten()
    elif "sum"=="squeeze": r=x.squeeze()
    elif "sum"=="unsqueeze": r=x.unsqueeze(0)
    elif "sum"=="clone": r=x.clone()
    elif "sum"=="detach": r=x.detach()
    elif "sum"=="contiguous": r=x.contiguous()
    else:
        if "detach"=="add": r=x+y
        elif "detach"=="sub": r=x-y
        elif "detach"=="mul": r=x*y
        elif "detach"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sum")()
        if None: r=getattr(r,"None")()
    return r

def fused_1714_sum_detach_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1714_sum_detach_reference(x:Any,y:Any):
    fused_1714_sum_detach_validate(x,y); return fused_1714_sum_detach(x,y)

def fused_1714_sum_detach_shape(x:Any,y:Any):
    r=fused_1714_sum_detach(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1714_sum_detach_numel(x:Any,y:Any):
    r=fused_1714_sum_detach(x,y); return r.numel()
