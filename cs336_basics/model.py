import torch
from torch import nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device|None=None, dtype: torch.dtype|None=None):
        """
        Linear Transformation: y = Wx
        """
        super().__init__()
        # weight
        self.weight: Float[Tensor, ' dout d_in'] = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        
        # weight initialization
        std = 2 / (d_in + d_out) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: Float[Tensor, ' ... d_in']) -> Float[Tensor, ' ... d_out']:
        """Apply the linear transformation to the input."""
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')