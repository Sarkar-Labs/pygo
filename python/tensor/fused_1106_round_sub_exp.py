"""PyGo fused operation round_sub_exp; canonical eager semantics."""
from typing import Any
OP_NAME="round_sub_exp"

def fused_1106_round_sub_exp(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "round" in {"sum","mean","max","min","argmax"}: r=getattr(x,"round")(0)
    elif "round"=="flatten": r=x.flatten()
    elif "round"=="squeeze": r=x.squeeze()
    elif "round"=="unsqueeze": r=x.unsqueeze(0)
    elif "round"=="clone": r=x.clone()
    elif "round"=="detach": r=x.detach()
    elif "round"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"round")()
        if 'exp': r=getattr(r,"exp")()
    return r

def fused_1106_round_sub_exp_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1106_round_sub_exp_reference(x:Any,y:Any):
    fused_1106_round_sub_exp_validate(x,y); return fused_1106_round_sub_exp(x,y)

def fused_1106_round_sub_exp_shape(x:Any,y:Any):
    r=fused_1106_round_sub_exp(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1106_round_sub_exp_numel(x:Any,y:Any):
    r=fused_1106_round_sub_exp(x,y); return r.numel()
