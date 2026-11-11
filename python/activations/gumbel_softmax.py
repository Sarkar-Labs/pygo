"""
PyGo Gumbel-Softmax Module — Categorical Reparameterization Trick.

This module provides a production-ready, differentiable Gumbel-Softmax layer
built on top of the PyGo Autograd Engine and Tensor core API.
"""

from __future__ import annotations
import math
from typing import Optional, Union, Tuple
from pygo.autograd.engine import Function, is_grad_enabled
from pygo.tensor import Tensor


class GumbelSoftmaxFunction(Function):
    """
    Autograd Function implementing the Gumbel-Softmax reparameterization trick.
    
    Forward: Calculates continuous relaxed categorical probability samples.
    Backward: Computes analytic gradients through relaxed softmax or STE.
    """

    @staticmethod
    def forward(ctx, logits: Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-20) -> Tensor:
        if tau <= 0:
            raise ValueError(f"Temperature `tau` must be positive (> 0), got {tau}.")

        # Generate Gumbel noise: G = -log(-log(U + eps) + eps)
        uniform_noise = Tensor.rand_like(logits)
        gumbel_noise = -((-((uniform_noise + eps).log()) + eps).log())

        # Perturb logits and apply temperature scaling
        perturbed_logits = (logits + gumbel_noise) / tau
        y_soft = perturbed_logits.softmax(dim=-1)

        ctx.save_for_backward(y_soft)
        ctx.tau = tau
        ctx.hard = hard

        if not hard:
            return y_soft

        # Straight-Through Estimator (STE)
        y_hard = y_soft.argmax(dim=-1, keepdim=True)
        one_hot_hard = Tensor.zeros_like(y_soft).scatter_(dim=-1, index=y_hard, src=1.0)
        
        # Save hard selection for reference
        ctx.saved_hard = one_hot_hard
        return one_hot_hard

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None, None]:
        y_soft, = ctx.saved_tensors
        tau = ctx.tau

        # Softmax Jacobian-Vector Product (JVP) gradient computation
        sum_grad_y = (grad_output * y_soft).sum(dim=-1, keepdim=True)
        grad_logits = (grad_output - sum_grad_y) * y_soft / tau

        return grad_logits, None, None, None


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-20
) -> Tensor:
    """
    Samples from the Gumbel-Softmax distribution.

    Args:
        logits (Tensor): Input unnormalized log probabilities (*, num_classes).
        tau (float): Temperature parameter (> 0).
        hard (bool): If True, returns discrete one-hot output with ST gradients.
        eps (float): Small float constant for numerical stability.

    Returns:
        Tensor: Sampled tensor from Gumbel-Softmax distribution.
    """
    return GumbelSoftmaxFunction.apply(logits, tau, hard, eps)


class GumbelSoftmax:
    """
    Neural Network Layer Wrapper for Gumbel-Softmax.
    """

    def __init__(self, tau: float = 1.0, hard: bool = False, eps: float = 1e-20) -> None:
        if tau <= 0:
            raise ValueError(f"Temperature `tau` must be positive (> 0), got {tau}.")
        self.tau = tau
        self.hard = hard
        self.eps = eps

    def __call__(self, logits: Tensor) -> Tensor:
        return self.forward(logits)

    def forward(self, logits: Tensor) -> Tensor:
        return gumbel_softmax(logits, tau=self.tau, hard=self.hard, eps=self.eps)

    def set_temperature(self, tau: float) -> None:
        """Updates annealing temperature."""
        if tau <= 0:
            raise ValueError(f"Temperature `tau` must be positive (> 0), got {tau}.")
        self.tau = tau