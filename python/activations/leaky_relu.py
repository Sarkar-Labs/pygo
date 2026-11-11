"""
PyGo LeakyReLU Module — Leaky Rectified Linear Unit Activation.

This module provides a production-grade, differentiable LeakyReLU implementation
built on top of the PyGo Autograd Engine and Tensor core API.
"""

from __future__ import annotations
from typing import Optional, Tuple
from pygo.autograd.engine import Function
from pygo.tensor import Tensor


class LeakyReLUFunction(Function):
    """
    Autograd Function for element-wise LeakyReLU activation.
    
    Forward: f(x) = max(0, x) + negative_slope * min(0, x)
    Backward: f'(x) = 1 if x > 0 else negative_slope
    """

    @staticmethod
    def forward(ctx, input_tensor: Tensor, negative_slope: float = 0.01) -> Tensor:
        ctx.negative_slope = negative_slope
        
        # Create boolean mask for negative values
        mask = input_tensor <= 0.0
        ctx.save_for_backward(mask)

        # Apply leaky relu element-wise
        output = input_tensor.clone()
        output[mask] = output[mask] * negative_slope
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None]:
        mask, = ctx.saved_tensors
        negative_slope = ctx.negative_slope

        # Compute gradient: grad_input = grad_output * (1 if x > 0 else negative_slope)
        grad_input = grad_output.clone()
        grad_input[mask] = grad_input[mask] * negative_slope

        return grad_input, None


def leaky_relu(input_tensor: Tensor, negative_slope: float = 0.01) -> Tensor:
    """
    Applies the LeakyReLU function element-wise.

    Args:
        input_tensor (Tensor): Input tensor.
        negative_slope (float): Controls the angle of the negative slope (default: 0.01).

    Returns:
        Tensor: Transformed tensor.
    """
    return LeakyReLUFunction.apply(input_tensor, negative_slope)


class LeakyReLU:
    """
    Neural Network Layer Wrapper for LeakyReLU.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def __call__(self, input_tensor: Tensor) -> Tensor:
        return self.forward(input_tensor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return leaky_relu(input_tensor, negative_slope=self.negative_slope)

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"