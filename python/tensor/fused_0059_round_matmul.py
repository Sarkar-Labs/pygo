"""PyGo fused operation round_matmul; canonical eager semantics."""
from typing import Any
OP_NAME="round_matmul"

def fused_0059_round_matmul(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "round" in {"sum","mean","max","min","argmax"}: r=getattr(x,"round")(0)
    elif "round"=="flatten": r=x.flatten()
    elif "round"=="squeeze": r=x.squeeze()
    elif "round"=="unsqueeze": r=x.unsqueeze(0)
    elif "round"=="clone": r=x.clone()
    elif "round"=="detach": r=x.detach()
    elif "round"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"round")()
        if None: r=getattr(r,"None")()
    return r

def fused_0059_round_matmul_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0059_round_matmul_reference(x:Any,y:Any):
    fused_0059_round_matmul_validate(x,y); return fused_0059_round_matmul(x,y)

def fused_0059_round_matmul_shape(x:Any,y:Any):
    r=fused_0059_round_matmul(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0059_round_matmul_numel(x:Any,y:Any):
    r=fused_0059_round_matmul(x,y); return r.numel()
