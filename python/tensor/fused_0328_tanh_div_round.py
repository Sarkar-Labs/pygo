"""PyGo fused operation tanh_div_round; canonical eager semantics."""
from typing import Any
OP_NAME="tanh_div_round"

def fused_0328_tanh_div_round(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "tanh" in {"sum","mean","max","min","argmax"}: r=getattr(x,"tanh")(0)
    elif "tanh"=="flatten": r=x.flatten()
    elif "tanh"=="squeeze": r=x.squeeze()
    elif "tanh"=="unsqueeze": r=x.unsqueeze(0)
    elif "tanh"=="clone": r=x.clone()
    elif "tanh"=="detach": r=x.detach()
    elif "tanh"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"tanh")()
        if 'round': r=getattr(r,"round")()
    return r

def fused_0328_tanh_div_round_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0328_tanh_div_round_reference(x:Any,y:Any):
    fused_0328_tanh_div_round_validate(x,y); return fused_0328_tanh_div_round(x,y)

def fused_0328_tanh_div_round_shape(x:Any,y:Any):
    r=fused_0328_tanh_div_round(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0328_tanh_div_round_numel(x:Any,y:Any):
    r=fused_0328_tanh_div_round(x,y); return r.numel()
