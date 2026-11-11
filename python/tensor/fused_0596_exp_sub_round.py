"""PyGo fused operation exp_sub_round; canonical eager semantics."""
from typing import Any
OP_NAME="exp_sub_round"

def fused_0596_exp_sub_round(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "exp" in {"sum","mean","max","min","argmax"}: r=getattr(x,"exp")(0)
    elif "exp"=="flatten": r=x.flatten()
    elif "exp"=="squeeze": r=x.squeeze()
    elif "exp"=="unsqueeze": r=x.unsqueeze(0)
    elif "exp"=="clone": r=x.clone()
    elif "exp"=="detach": r=x.detach()
    elif "exp"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"exp")()
        if 'round': r=getattr(r,"round")()
    return r

def fused_0596_exp_sub_round_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0596_exp_sub_round_reference(x:Any,y:Any):
    fused_0596_exp_sub_round_validate(x,y); return fused_0596_exp_sub_round(x,y)

def fused_0596_exp_sub_round_shape(x:Any,y:Any):
    r=fused_0596_exp_sub_round(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0596_exp_sub_round_numel(x:Any,y:Any):
    r=fused_0596_exp_sub_round(x,y); return r.numel()
