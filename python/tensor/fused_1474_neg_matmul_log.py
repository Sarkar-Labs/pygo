"""PyGo fused operation neg_matmul_log; canonical eager semantics."""
from typing import Any
OP_NAME="neg_matmul_log"

def fused_1474_neg_matmul_log(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "neg" in {"sum","mean","max","min","argmax"}: r=getattr(x,"neg")(0)
    elif "neg"=="flatten": r=x.flatten()
    elif "neg"=="squeeze": r=x.squeeze()
    elif "neg"=="unsqueeze": r=x.unsqueeze(0)
    elif "neg"=="clone": r=x.clone()
    elif "neg"=="detach": r=x.detach()
    elif "neg"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"neg")()
        if 'log': r=getattr(r,"log")()
    return r

def fused_1474_neg_matmul_log_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1474_neg_matmul_log_reference(x:Any,y:Any):
    fused_1474_neg_matmul_log_validate(x,y); return fused_1474_neg_matmul_log(x,y)

def fused_1474_neg_matmul_log_shape(x:Any,y:Any):
    r=fused_1474_neg_matmul_log(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1474_neg_matmul_log_numel(x:Any,y:Any):
    r=fused_1474_neg_matmul_log(x,y); return r.numel()
