"""PyGo fused operation sign_sub_log10; canonical eager semantics."""
from typing import Any
OP_NAME="sign_sub_log10"

def fused_1256_sign_sub_log10(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sign" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sign")(0)
    elif "sign"=="flatten": r=x.flatten()
    elif "sign"=="squeeze": r=x.squeeze()
    elif "sign"=="unsqueeze": r=x.unsqueeze(0)
    elif "sign"=="clone": r=x.clone()
    elif "sign"=="detach": r=x.detach()
    elif "sign"=="contiguous": r=x.contiguous()
    else:
        if "sub"=="add": r=x+y
        elif "sub"=="sub": r=x-y
        elif "sub"=="mul": r=x*y
        elif "sub"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sign")()
        if 'log10': r=getattr(r,"log10")()
    return r

def fused_1256_sign_sub_log10_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1256_sign_sub_log10_reference(x:Any,y:Any):
    fused_1256_sign_sub_log10_validate(x,y); return fused_1256_sign_sub_log10(x,y)

def fused_1256_sign_sub_log10_shape(x:Any,y:Any):
    r=fused_1256_sign_sub_log10(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1256_sign_sub_log10_numel(x:Any,y:Any):
    r=fused_1256_sign_sub_log10(x,y); return r.numel()
