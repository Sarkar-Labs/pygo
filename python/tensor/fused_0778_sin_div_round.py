"""PyGo fused operation sin_div_round; canonical eager semantics."""
from typing import Any
OP_NAME="sin_div_round"

def fused_0778_sin_div_round(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sin" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sin")(0)
    elif "sin"=="flatten": r=x.flatten()
    elif "sin"=="squeeze": r=x.squeeze()
    elif "sin"=="unsqueeze": r=x.unsqueeze(0)
    elif "sin"=="clone": r=x.clone()
    elif "sin"=="detach": r=x.detach()
    elif "sin"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sin")()
        if 'round': r=getattr(r,"round")()
    return r

def fused_0778_sin_div_round_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0778_sin_div_round_reference(x:Any,y:Any):
    fused_0778_sin_div_round_validate(x,y); return fused_0778_sin_div_round(x,y)

def fused_0778_sin_div_round_shape(x:Any,y:Any):
    r=fused_0778_sin_div_round(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0778_sin_div_round_numel(x:Any,y:Any):
    r=fused_0778_sin_div_round(x,y); return r.numel()
