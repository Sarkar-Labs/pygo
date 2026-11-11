"""PyGo fused operation neg_mul; canonical eager semantics."""
from typing import Any
OP_NAME="neg_mul"

def fused_0077_neg_mul(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "neg" in {"sum","mean","max","min","argmax"}: r=getattr(x,"neg")(0)
    elif "neg"=="flatten": r=x.flatten()
    elif "neg"=="squeeze": r=x.squeeze()
    elif "neg"=="unsqueeze": r=x.unsqueeze(0)
    elif "neg"=="clone": r=x.clone()
    elif "neg"=="detach": r=x.detach()
    elif "neg"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"neg")()
        if None: r=getattr(r,"None")()
    return r

def fused_0077_neg_mul_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0077_neg_mul_reference(x:Any,y:Any):
    fused_0077_neg_mul_validate(x,y); return fused_0077_neg_mul(x,y)

def fused_0077_neg_mul_shape(x:Any,y:Any):
    r=fused_0077_neg_mul(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0077_neg_mul_numel(x:Any,y:Any):
    r=fused_0077_neg_mul(x,y); return r.numel()
