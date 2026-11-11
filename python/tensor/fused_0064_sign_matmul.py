"""PyGo fused operation sign_matmul; canonical eager semantics."""
from typing import Any
OP_NAME="sign_matmul"

def fused_0064_sign_matmul(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sign" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sign")(0)
    elif "sign"=="flatten": r=x.flatten()
    elif "sign"=="squeeze": r=x.squeeze()
    elif "sign"=="unsqueeze": r=x.unsqueeze(0)
    elif "sign"=="clone": r=x.clone()
    elif "sign"=="detach": r=x.detach()
    elif "sign"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sign")()
        if None: r=getattr(r,"None")()
    return r

def fused_0064_sign_matmul_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0064_sign_matmul_reference(x:Any,y:Any):
    fused_0064_sign_matmul_validate(x,y); return fused_0064_sign_matmul(x,y)

def fused_0064_sign_matmul_shape(x:Any,y:Any):
    r=fused_0064_sign_matmul(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0064_sign_matmul_numel(x:Any,y:Any):
    r=fused_0064_sign_matmul(x,y); return r.numel()
