import torch
from torch import nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
from collections.abc import Iterable
import einx
from typing import cast
import math

def softmax(x: Tensor, dim: int=-1) -> Tensor:
    safe_x = x - torch.max(x)
    safe_x_exp = torch.exp(safe_x)
    return safe_x_exp / torch.sum(safe_x_exp, dim=dim, keepdim=True)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size, vocab_size = inputs.shape
    # log_softmax = torch.log(softmax(inputs))

    # Log-softmax for numerical stability (log(softmax(inputs)))
    log_softmax = inputs - inputs.logsumexp(dim=1, keepdim=True)
    loss = log_softmax[range(batch_size), targets]
    return -torch.mean(loss)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: Float=1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    all_grads = torch.cat([grad.flatten() for grad in grads])
    l2_norm = torch.norm(all_grads, 2)
    if l2_norm > max_l2_norm:
        clip_coeff = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad.mul_(clip_coeff)

