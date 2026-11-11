"""PyGo fused operation round_matmul_round; canonical eager semantics."""
from typing import Any
OP_NAME="round_matmul_round"

def fused_1139_round_matmul_round(x:Any,y:Any):
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
        if 'round': r=getattr(r,"round")()
    return r

def fused_1139_round_matmul_round_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1139_round_matmul_round_reference(x:Any,y:Any):
    fused_1139_round_matmul_round_validate(x,y); return fused_1139_round_matmul_round(x,y)

def fused_1139_round_matmul_round_shape(x:Any,y:Any):
    r=fused_1139_round_matmul_round(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1139_round_matmul_round_numel(x:Any,y:Any):
    r=fused_1139_round_matmul_round(x,y); return r.numel()
