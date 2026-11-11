"""PyGo fused operation log2_matmul_cos; canonical eager semantics."""
from typing import Any
OP_NAME="log2_matmul_cos"

def fused_1574_log2_matmul_cos(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "log2" in {"sum","mean","max","min","argmax"}: r=getattr(x,"log2")(0)
    elif "log2"=="flatten": r=x.flatten()
    elif "log2"=="squeeze": r=x.squeeze()
    elif "log2"=="unsqueeze": r=x.unsqueeze(0)
    elif "log2"=="clone": r=x.clone()
    elif "log2"=="detach": r=x.detach()
    elif "log2"=="contiguous": r=x.contiguous()
    else:
        if "matmul"=="add": r=x+y
        elif "matmul"=="sub": r=x-y
        elif "matmul"=="mul": r=x*y
        elif "matmul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"log2")()
        if 'cos': r=getattr(r,"cos")()
    return r

def fused_1574_log2_matmul_cos_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_1574_log2_matmul_cos_reference(x:Any,y:Any):
    fused_1574_log2_matmul_cos_validate(x,y); return fused_1574_log2_matmul_cos(x,y)

def fused_1574_log2_matmul_cos_shape(x:Any,y:Any):
    r=fused_1574_log2_matmul_cos(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_1574_log2_matmul_cos_numel(x:Any,y:Any):
    r=fused_1574_log2_matmul_cos(x,y); return r.numel()
