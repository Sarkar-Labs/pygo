"""PyGo fused operation abs_div_neg; canonical eager semantics."""
from typing import Any
OP_NAME="abs_div_neg"

def fused_0438_abs_div_neg(x:Any,y:Any):
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
        if 'neg': r=getattr(r,"neg")()
    return r

def fused_0438_abs_div_neg_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0438_abs_div_neg_reference(x:Any,y:Any):
    fused_0438_abs_div_neg_validate(x,y); return fused_0438_abs_div_neg(x,y)

def fused_0438_abs_div_neg_shape(x:Any,y:Any):
    r=fused_0438_abs_div_neg(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0438_abs_div_neg_numel(x:Any,y:Any):
    r=fused_0438_abs_div_neg(x,y); return r.numel()
