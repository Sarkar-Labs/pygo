"""PyGo fused operation sqrt_sub_silu; canonical eager semantics."""
from typing import Any
OP_NAME="sqrt_sub_silu"

def fused_0521_sqrt_sub_silu(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sqrt" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sqrt")(0)
    elif "sqrt"=="flatten": r=x.flatten()
    elif "sqrt"=="squeeze": r=x.squeeze()
    elif "sqrt"=="unsqueeze": r=x.unsqueeze(0)
    elif "sqrt"=="clone": r=x.clone()
    elif "sqrt"=="detach": r=x.detach()
    elif "sqrt"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sqrt")()
        if 'silu': r=getattr(r,"silu")()
    return r

def fused_0521_sqrt_sub_silu_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0521_sqrt_sub_silu_reference(x:Any,y:Any):
    fused_0521_sqrt_sub_silu_validate(x,y); return fused_0521_sqrt_sub_silu(x,y)

def fused_0521_sqrt_sub_silu_shape(x:Any,y:Any):
    r=fused_0521_sqrt_sub_silu(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0521_sqrt_sub_silu_numel(x:Any,y:Any):
    r=fused_0521_sqrt_sub_silu(x,y); return r.numel()
