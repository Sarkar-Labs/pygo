"""PyGo fused operation abs_mul_sigmoid; canonical eager semantics."""
from typing import Any
OP_NAME="abs_mul_sigmoid"

def fused_0367_abs_mul_sigmoid(x:Any,y:Any):
    if x is None or y is None: raise ValueError(f"{OP_NAME} requires defined operands")
    if "abs" in {"sum","mean","max","min","argmax"}: r=getattr(x,"abs")(0)
    elif "abs"=="flatten": r=x.flatten()
    elif "abs"=="squeeze": r=x.squeeze()
    elif "abs"=="unsqueeze": r=x.unsqueeze(0)
    elif "abs"=="clone": r=x.clone()
    elif "abs"=="detach": r=x.detach()
    elif "abs"=="contiguous": r=x.contiguous()
    else:
        if "mul"=="add": r=x+y
        elif "mul"=="sub": r=x-y
        elif "mul"=="mul": r=x*y
        elif "mul"=="div": r=x/y
        else: r=x.matmul(y)
        r=getattr(r,"abs")()
        if 'sigmoid': r=getattr(r,"sigmoid")()
    return r

def fused_0367_abs_mul_sigmoid_validate(x:Any,y:Any)->None:
    if x is None or y is None: raise ValueError(f"{OP_NAME}: null operand")
    if hasattr(x,"device") and hasattr(y,"device") and x.device!=y.device: raise ValueError(f"{OP_NAME}: device mismatch")

def fused_0367_abs_mul_sigmoid_reference(x:Any,y:Any):
    fused_0367_abs_mul_sigmoid_validate(x,y); return fused_0367_abs_mul_sigmoid(x,y)

def fused_0367_abs_mul_sigmoid_shape(x:Any,y:Any):
    r=fused_0367_abs_mul_sigmoid(x,y); return tuple(r.shape() if callable(getattr(r,"shape",None)) else r.shape)

def fused_0367_abs_mul_sigmoid_numel(x:Any,y:Any):
    r=fused_0367_abs_mul_sigmoid(x,y); return r.numel()
