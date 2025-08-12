import torch
from torch import nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
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

