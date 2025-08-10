import torch
from torch import nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einx
from typing import cast
import math

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device|None=None, dtype: torch.dtype|None=None):
        """
        Linear Transformation: y = Wx
        """
        super().__init__()
        # weight
        self.weight: Float[Tensor, ' dout d_in'] = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype), requires_grad=True)
        
        # weight initialization
        std = 2 / (d_in + d_out) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: Float[Tensor, ' ... d_in']) -> Float[Tensor, ' ... d_out']:
        """Apply the linear transformation to the input."""
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
    
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        """
        Construct an embedding module.
        """
        super().__init__()

        std = 1.0

        self.weight: Float[Tensor, ' vocab_size d_model'] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model,device=device, dtype=dtype), 
                mean=0, std=std, 
                a=-3, b=3),
            requires_grad=True
        )

    def forward(self, token_ids: Int[Tensor, ' ...']):
        """Lookup the embedding vectors for the given token IDs."""
        return self.weight[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_type)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: Float[Tensor, ' ... d_model']) -> Float[Tensor, ' ... d_model']:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        assert d_k % 2 == 0

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len)
        angles = einsum(positions, freqs, 'seq, half_d -> seq half_d')

        self.register_buffer('cos_cache', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cache', torch.sin(angles), persistent=False)

    def forward(self, x: Float[Tensor, " ... seq d"],
    token_positions: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d half) -> half ... half_d', half=2)

        # just to make pylance happy
        cos_cache = cast(Tensor, self.cos_cache)
        sin_cache = cast(Tensor, self.sin_cache)

        cos: Tensor = einx.get_at('[pos] half_d, ... -> ... half_d', cos_cache, token_positions)
        sin: Tensor = einx.get_at('[pos] half_d, ... -> ... half_d', sin_cache, token_positions)

        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2

        result: Tensor = cast(Tensor, einx.rearrange('... half_d, ... half_d -> ... (half_d (1+1))', x1_rot, x2_rot))

        return result




    
def silu(x):
    return x * torch.sigmoid(x)

def softmax(x: Tensor, dim: int=-1) -> Tensor:
    safe_x = x - torch.max(x)
    safe_x_exp = torch.exp(safe_x)
    return safe_x_exp / torch.sum(safe_x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = K.shape[-1]
    attn_scores = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / math.sqrt(d_k)
    if mask is not None:
        attn_scores = torch.where(mask, attn_scores, -torch.inf) 

    probs = softmax(attn_scores, dim=-1)
    return einsum(probs, V, '... queries keys, ... keys d_v -> ... queries d_v')