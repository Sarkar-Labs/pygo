"""PyGo fused operation sigmoid_mul_cos; canonical eager semantics."""
from typing import Any
OP_NAME="sigmoid_mul_cos"

def fused_0222_sigmoid_mul_cos(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "sigmoid" in {"sum","mean","max","min","argmax"}: r=getattr(x,"sigmoid")(0)
    elif "sigmoid"=="flatten": r=x.flatten()
    elif "sigmoid"=="squeeze": r=x.squeeze()
    elif "sigmoid"=="unsqueeze": r=x.unsqueeze(0)
    elif "sigmoid"=="clone": r=x.clone()
    elif "sigmoid"=="detach": r=x.detach()
    elif "sigmoid"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"sigmoid")()
        if 'cos': r=getattr(r,"cos")()
    return r

def fused_0222_sigmoid_mul_cos_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0222_sigmoid_mul_cos_reference(x:Any,y:Any):
    fused_0222_sigmoid_mul_cos_validate(x,y); return fused_0222_sigmoid_mul_cos(x,y)

def fused_0222_sigmoid_mul_cos_shape(x:Any,y:Any):
    r=fused_0222_sigmoid_mul_cos(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0222_sigmoid_mul_cos_numel(x:Any,y:Any):
    r=fused_0222_sigmoid_mul_cos(x,y); return r.numel()
