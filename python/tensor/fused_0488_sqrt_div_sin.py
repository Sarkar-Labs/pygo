"""PyGo fused operation sqrt_div_sin; canonical eager semantics."""
from typing import Any
OP_NAME="sqrt_div_sin"

def fused_0488_sqrt_div_sin(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sqrt" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sqrt")(0)
    elif "sqrt"=="flatten": r=x.flatten()
    elif "sqrt"=="squeeze": r=x.squeeze()
    elif "sqrt"=="unsqueeze": r=x.unsqueeze(0)
    elif "sqrt"=="clone": r=x.clone()
    elif "sqrt"=="detach": r=x.detach()
    elif "sqrt"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sqrt")()
        if 'sin': r=getattr(r,"sin")()
    return r

def fused_0488_sqrt_div_sin_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0488_sqrt_div_sin_reference(x:Any,y:Any):
    fused_0488_sqrt_div_sin_validate(x,y); return fused_0488_sqrt_div_sin(x,y)

def fused_0488_sqrt_div_sin_shape(x:Any,y:Any):
    r=fused_0488_sqrt_div_sin(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0488_sqrt_div_sin_numel(x:Any,y:Any):
    r=fused_0488_sqrt_div_sin(x,y); return r.numel()
