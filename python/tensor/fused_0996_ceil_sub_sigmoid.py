"""PyGo fused operation ceil_sub_sigmoid; canonical eager semantics."""
from typing import Any
OP_NAME="ceil_sub_sigmoid"

def fused_0996_ceil_sub_sigmoid(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "ceil" in {"sum","mean","max","min","argmax"}: r=getattr(x,"ceil")(0)
    elif "ceil"=="flatten": r=x.flatten()
    elif "ceil"=="squeeze": r=x.squeeze()
    elif "ceil"=="unsqueeze": r=x.unsqueeze(0)
    elif "ceil"=="clone": r=x.clone()
    elif "ceil"=="detach": r=x.detach()
    elif "ceil"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"ceil")()
        if 'sigmoid': r=getattr(r,"sigmoid")()
    return r

def fused_0996_ceil_sub_sigmoid_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0996_ceil_sub_sigmoid_reference(x:Any,y:Any):
    fused_0996_ceil_sub_sigmoid_validate(x,y); return fused_0996_ceil_sub_sigmoid(x,y)

def fused_0996_ceil_sub_sigmoid_shape(x:Any,y:Any):
    r=fused_0996_ceil_sub_sigmoid(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0996_ceil_sub_sigmoid_numel(x:Any,y:Any):
    r=fused_0996_ceil_sub_sigmoid(x,y); return r.numel()
