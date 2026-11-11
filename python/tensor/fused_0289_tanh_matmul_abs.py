"""PyGo fused operation tanh_matmul_abs; canonical eager semantics."""
from typing import Any
OP_NAME="tanh_matmul_abs"

def fused_0289_tanh_matmul_abs(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "tanh" in {"sum","mean","max","min","argmax"}: r=getattr(x,"tanh")(0)
    elif "tanh"=="flatten": r=x.flatten()
    elif "tanh"=="squeeze": r=x.squeeze()
    elif "tanh"=="unsqueeze": r=x.unsqueeze(0)
    elif "tanh"=="clone": r=x.clone()
    elif "tanh"=="detach": r=x.detach()
    elif "tanh"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"tanh")()
        if 'abs': r=getattr(r,"abs")()
    return r

def fused_0289_tanh_matmul_abs_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0289_tanh_matmul_abs_reference(x:Any,y:Any):
    fused_0289_tanh_matmul_abs_validate(x,y); return fused_0289_tanh_matmul_abs(x,y)

def fused_0289_tanh_matmul_abs_shape(x:Any,y:Any):
    r=fused_0289_tanh_matmul_abs(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0289_tanh_matmul_abs_numel(x:Any,y:Any):
    r=fused_0289_tanh_matmul_abs(x,y); return r.numel()
