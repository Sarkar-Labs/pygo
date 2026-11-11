"""PyGo fused operation sigmoid_matmul; canonical eager semantics."""
from typing import Any
OP_NAME="sigmoid_matmul"

def fused_0009_sigmoid_matmul(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sigmoid" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sigmoid")(0)
    elif "sigmoid"=="flatten": r=x.flatten()
    elif "sigmoid"=="squeeze": r=x.squeeze()
    elif "sigmoid"=="unsqueeze": r=x.unsqueeze(0)
    elif "sigmoid"=="clone": r=x.clone()
    elif "sigmoid"=="detach": r=x.detach()
    elif "sigmoid"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sigmoid")()
        if None: r=getattr(r,"None")()
    return r

def fused_0009_sigmoid_matmul_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0009_sigmoid_matmul_reference(x:Any,y:Any):
    fused_0009_sigmoid_matmul_validate(x,y); return fused_0009_sigmoid_matmul(x,y)

def fused_0009_sigmoid_matmul_shape(x:Any,y:Any):
    r=fused_0009_sigmoid_matmul(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0009_sigmoid_matmul_numel(x:Any,y:Any):
    r=fused_0009_sigmoid_matmul(x,y); return r.numel()
