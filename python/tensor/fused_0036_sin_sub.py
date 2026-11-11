"""PyGo fused operation sin_sub; canonical eager semantics."""
from typing import Any
OP_NAME="sin_sub"

def fused_0036_sin_sub(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sin" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sin")(0)
    elif "sin"=="flatten": r=x.flatten()
    elif "sin"=="squeeze": r=x.squeeze()
    elif "sin"=="unsqueeze": r=x.unsqueeze(0)
    elif "sin"=="clone": r=x.clone()
    elif "sin"=="detach": r=x.detach()
    elif "sin"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sin")()
        if None: r=getattr(r,"None")()
    return r

def fused_0036_sin_sub_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0036_sin_sub_reference(x:Any,y:Any):
    fused_0036_sin_sub_validate(x,y); return fused_0036_sin_sub(x,y)

def fused_0036_sin_sub_shape(x:Any,y:Any):
    r=fused_0036_sin_sub(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0036_sin_sub_numel(x:Any,y:Any):
    r=fused_0036_sin_sub(x,y); return r.numel()
