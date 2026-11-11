"""PyGo fused operation abs_div_exp; canonical eager semantics."""
from typing import Any
OP_NAME="abs_div_exp"

def fused_0388_abs_div_exp(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "abs" in {"sum","mean","max","min","argmax"}: r=getattr(x,"abs")(0)
    elif "abs"=="flatten": r=x.flatten()
    elif "abs"=="squeeze": r=x.squeeze()
    elif "abs"=="unsqueeze": r=x.unsqueeze(0)
    elif "abs"=="clone": r=x.clone()
    elif "abs"=="detach": r=x.detach()
    elif "abs"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"abs")()
        if 'exp': r=getattr(r,"exp")()
    return r

def fused_0388_abs_div_exp_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0388_abs_div_exp_reference(x:Any,y:Any):
    fused_0388_abs_div_exp_validate(x,y); return fused_0388_abs_div_exp(x,y)

def fused_0388_abs_div_exp_shape(x:Any,y:Any):
    r=fused_0388_abs_div_exp(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0388_abs_div_exp_numel(x:Any,y:Any):
    r=fused_0388_abs_div_exp(x,y); return r.numel()
