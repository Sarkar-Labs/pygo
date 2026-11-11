"""PyGo fused operation gelu_add_relu; canonical eager semantics."""
from typing import Any
OP_NAME="gelu_add_relu"

def fused_1260_gelu_add_relu(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "gelu" in {"sum","mean","max","min","argmax"}: r=getattr(x,"gelu")(0)
    elif "gelu"=="flatten": r=x.flatten()
    elif "gelu"=="squeeze": r=x.squeeze()
    elif "gelu"=="unsqueeze": r=x.unsqueeze(0)
    elif "gelu"=="clone": r=x.clone()
    elif "gelu"=="detach": r=x.detach()
    elif "gelu"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"gelu")()
        if 'relu': r=getattr(r,"relu")()
    return r

def fused_1260_gelu_add_relu_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1260_gelu_add_relu_reference(x:Any,y:Any):
    fused_1260_gelu_add_relu_validate(x,y); return fused_1260_gelu_add_relu(x,y)

def fused_1260_gelu_add_relu_shape(x:Any,y:Any):
    r=fused_1260_gelu_add_relu(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1260_gelu_add_relu_numel(x:Any,y:Any):
    r=fused_1260_gelu_add_relu(x,y); return r.numel()
