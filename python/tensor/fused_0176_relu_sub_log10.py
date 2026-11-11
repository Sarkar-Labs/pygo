"""PyGo fused operation relu_sub_log10; canonical eager semantics."""
from typing import Any
OP_NAME="relu_sub_log10"

def fused_0176_relu_sub_log10(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "relu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"relu")(0)
    elif "relu"=="flatten": r=x.flatten()
    elif "relu"=="squeeze": r=x.squeeze()
    elif "relu"=="unsqueeze": r=x.unsqueeze(0)
    elif "relu"=="clone": r=x.clone()
    elif "relu"=="detach": r=x.detach()
    elif "relu"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"relu")()
        if 'log10': r=getattr(r,"log10")()
    return r

def fused_0176_relu_sub_log10_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0176_relu_sub_log10_reference(x:Any,y:Any):
    fused_0176_relu_sub_log10_validate(x,y); return fused_0176_relu_sub_log10(x,y)

def fused_0176_relu_sub_log10_shape(x:Any,y:Any):
    r=fused_0176_relu_sub_log10(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0176_relu_sub_log10_numel(x:Any,y:Any):
    r=fused_0176_relu_sub_log10(x,y); return r.numel()
