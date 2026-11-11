"""PyGo fused operation abs_matmul_relu; canonical eager semantics."""
from typing import Any
OP_NAME="abs_matmul_relu"

def fused_0364_abs_matmul_relu(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "abs" in {"sum","mean","max","min","argmax"}: r=getattr(x,"abs")(0)
    elif "abs"=="flatten": r=x.flatten()
    elif "abs"=="squeeze": r=x.squeeze()
    elif "abs"=="unsqueeze": r=x.unsqueeze(0)
    elif "abs"=="clone": r=x.clone()
    elif "abs"=="detach": r=x.detach()
    elif "abs"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"abs")()
        if 'relu': r=getattr(r,"relu")()
    return r

def fused_0364_abs_matmul_relu_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0364_abs_matmul_relu_reference(x:Any,y:Any):
    fused_0364_abs_matmul_relu_validate(x,y); return fused_0364_abs_matmul_relu(x,y)

def fused_0364_abs_matmul_relu_shape(x:Any,y:Any):
    r=fused_0364_abs_matmul_relu(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0364_abs_matmul_relu_numel(x:Any,y:Any):
    r=fused_0364_abs_matmul_relu(x,y); return r.numel()
