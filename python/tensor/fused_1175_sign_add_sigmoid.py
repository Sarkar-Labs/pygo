"""PyGo fused operation sign_add_sigmoid; canonical eager semantics."""
from typing import Any
OP_NAME="sign_add_sigmoid"

def fused_1175_sign_add_sigmoid(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sign" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sign")(0)
    elif "sign"=="flatten": r=x.flatten()
    elif "sign"=="squeeze": r=x.squeeze()
    elif "sign"=="unsqueeze": r=x.unsqueeze(0)
    elif "sign"=="clone": r=x.clone()
    elif "sign"=="detach": r=x.detach()
    elif "sign"=="contiguous": r=x.contiguous()
    else:
        if "add"=="add": r=x+y
        elif "add"=="sub": r=x-y
        elif "add"=="mul": r=x*y
        elif "add"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sign")()
        if 'sigmoid': r=getattr(r,"sigmoid")()
    return r

def fused_1175_sign_add_sigmoid_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1175_sign_add_sigmoid_reference(x:Any,y:Any):
    fused_1175_sign_add_sigmoid_validate(x,y); return fused_1175_sign_add_sigmoid(x,y)

def fused_1175_sign_add_sigmoid_shape(x:Any,y:Any):
    r=fused_1175_sign_add_sigmoid(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1175_sign_add_sigmoid_numel(x:Any,y:Any):
    r=fused_1175_sign_add_sigmoid(x,y); return r.numel()
