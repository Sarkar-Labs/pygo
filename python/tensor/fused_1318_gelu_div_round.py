"""PyGo fused operation gelu_div_round; canonical eager semantics."""
from typing import Any
OP_NAME="gelu_div_round"

def fused_1318_gelu_div_round(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "gelu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"gelu")(0)
    elif "gelu"=="flatten": r=x.flatten()
    elif "gelu"=="squeeze": r=x.squeeze()
    elif "gelu"=="unsqueeze": r=x.unsqueeze(0)
    elif "gelu"=="clone": r=x.clone()
    elif "gelu"=="detach": r=x.detach()
    elif "gelu"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"gelu")()
        if 'round': r=getattr(r,"round")()
    return r

def fused_1318_gelu_div_round_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1318_gelu_div_round_reference(x:Any,y:Any):
    fused_1318_gelu_div_round_validate(x,y); return fused_1318_gelu_div_round(x,y)

def fused_1318_gelu_div_round_shape(x:Any,y:Any):
    r=fused_1318_gelu_div_round(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1318_gelu_div_round_numel(x:Any,y:Any):
    r=fused_1318_gelu_div_round(x,y); return r.numel()
