"""PyGo fused operation log_div_log; canonical eager semantics."""
from typing import Any
OP_NAME="log_div_log"

def fused_0663_log_div_log(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "log" in {"sum","mean","max","min","argmax"}: r=getattr(x,"log")(0)
    elif "log"=="flatten": r=x.flatten()
    elif "log"=="squeeze": r=x.squeeze()
    elif "log"=="unsqueeze": r=x.unsqueeze(0)
    elif "log"=="clone": r=x.clone()
    elif "log"=="detach": r=x.detach()
    elif "log"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"log")()
        if 'log': r=getattr(r,"log")()
    return r

def fused_0663_log_div_log_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0663_log_div_log_reference(x:Any,y:Any):
    fused_0663_log_div_log_validate(x,y); return fused_0663_log_div_log(x,y)

def fused_0663_log_div_log_shape(x:Any,y:Any):
    r=fused_0663_log_div_log(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0663_log_div_log_numel(x:Any,y:Any):
    r=fused_0663_log_div_log(x,y); return r.numel()
