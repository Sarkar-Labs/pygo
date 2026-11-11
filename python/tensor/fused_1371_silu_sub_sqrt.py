"""PyGo fused operation silu_sub_sqrt; canonical eager semantics."""
from typing import Any
OP_NAME="silu_sub_sqrt"

def fused_1371_silu_sub_sqrt(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "silu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"silu")(0)
    elif "silu"=="flatten": r=x.flatten()
    elif "silu"=="squeeze": r=x.squeeze()
    elif "silu"=="unsqueeze": r=x.unsqueeze(0)
    elif "silu"=="clone": r=x.clone()
    elif "silu"=="detach": r=x.detach()
    elif "silu"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"silu")()
        if 'sqrt': r=getattr(r,"sqrt")()
    return r

def fused_1371_silu_sub_sqrt_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1371_silu_sub_sqrt_reference(x:Any,y:Any):
    fused_1371_silu_sub_sqrt_validate(x,y); return fused_1371_silu_sub_sqrt(x,y)

def fused_1371_silu_sub_sqrt_shape(x:Any,y:Any):
    r=fused_1371_silu_sub_sqrt(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1371_silu_sub_sqrt_numel(x:Any,y:Any):
    r=fused_1371_silu_sub_sqrt(x,y); return r.numel()
