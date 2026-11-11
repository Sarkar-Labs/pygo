"""PyGo fused operation sign_matmul_exp; canonical eager semantics."""
from typing import Any
OP_NAME="sign_matmul_exp"

def fused_1199_sign_matmul_exp(x:Any,y:Any):
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
        if 'exp': r=getattr(r,"exp")()
    return r

def fused_1199_sign_matmul_exp_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1199_sign_matmul_exp_reference(x:Any,y:Any):
    fused_1199_sign_matmul_exp_validate(x,y); return fused_1199_sign_matmul_exp(x,y)

def fused_1199_sign_matmul_exp_shape(x:Any,y:Any):
    r=fused_1199_sign_matmul_exp(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1199_sign_matmul_exp_numel(x:Any,y:Any):
    r=fused_1199_sign_matmul_exp(x,y); return r.numel()
