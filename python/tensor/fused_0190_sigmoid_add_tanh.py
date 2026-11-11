"""PyGo fused operation sigmoid_add_tanh; canonical eager semantics."""
from typing import Any
OP_NAME="sigmoid_add_tanh"

def fused_0190_sigmoid_add_tanh(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sigmoid" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sigmoid")(0)
    elif "sigmoid"=="flatten": r=x.flatten()
    elif "sigmoid"=="squeeze": r=x.squeeze()
    elif "sigmoid"=="unsqueeze": r=x.unsqueeze(0)
    elif "sigmoid"=="clone": r=x.clone()
    elif "sigmoid"=="detach": r=x.detach()
    elif "sigmoid"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sigmoid")()
        if 'tanh': r=getattr(r,"tanh")()
    return r

def fused_0190_sigmoid_add_tanh_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0190_sigmoid_add_tanh_reference(x:Any,y:Any):
    fused_0190_sigmoid_add_tanh_validate(x,y); return fused_0190_sigmoid_add_tanh(x,y)

def fused_0190_sigmoid_add_tanh_shape(x:Any,y:Any):
    r=fused_0190_sigmoid_add_tanh(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0190_sigmoid_add_tanh_numel(x:Any,y:Any):
    r=fused_0190_sigmoid_add_tanh(x,y); return r.numel()
