"""PyGo fused operation cos_div_tanh; canonical eager semantics."""
from typing import Any
OP_NAME="cos_div_tanh"

def fused_0823_cos_div_tanh(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "cos" in {"sum","mean","max","min","argmax"}: r=getattr(x,"cos")(0)
    elif "cos"=="flatten": r=x.flatten()
    elif "cos"=="squeeze": r=x.squeeze()
    elif "cos"=="unsqueeze": r=x.unsqueeze(0)
    elif "cos"=="clone": r=x.clone()
    elif "cos"=="detach": r=x.detach()
    elif "cos"=="contiguous": r=x.contiguous()
    else:
        if "div"=="add": r=x+y
        elif "div"=="sub": r=x-y
        elif "div"=="mul": r=x*y
        elif "div"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"cos")()
        if 'tanh': r=getattr(r,"tanh")()
    return r

def fused_0823_cos_div_tanh_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0823_cos_div_tanh_reference(x:Any,y:Any):
    fused_0823_cos_div_tanh_validate(x,y); return fused_0823_cos_div_tanh(x,y)

def fused_0823_cos_div_tanh_shape(x:Any,y:Any):
    r=fused_0823_cos_div_tanh(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0823_cos_div_tanh_numel(x:Any,y:Any):
    r=fused_0823_cos_div_tanh(x,y); return r.numel()
