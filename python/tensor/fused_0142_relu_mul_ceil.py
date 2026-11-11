"""PyGo fused operation relu_mul_ceil; canonical eager semantics."""
from typing import Any
OP_NAME="relu_mul_ceil"

def fused_0142_relu_mul_ceil(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "relu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"relu")(0)
    elif "relu"=="flatten": r=x.flatten()
    elif "relu"=="squeeze": r=x.squeeze()
    elif "relu"=="unsqueeze": r=x.unsqueeze(0)
    elif "relu"=="clone": r=x.clone()
    elif "relu"=="detach": r=x.detach()
    elif "relu"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"relu")()
        if 'ceil': r=getattr(r,"ceil")()
    return r

def fused_0142_relu_mul_ceil_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0142_relu_mul_ceil_reference(x:Any,y:Any):
    fused_0142_relu_mul_ceil_validate(x,y); return fused_0142_relu_mul_ceil(x,y)

def fused_0142_relu_mul_ceil_shape(x:Any,y:Any):
    r=fused_0142_relu_mul_ceil(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0142_relu_mul_ceil_numel(x:Any,y:Any):
    r=fused_0142_relu_mul_ceil(x,y); return r.numel()
