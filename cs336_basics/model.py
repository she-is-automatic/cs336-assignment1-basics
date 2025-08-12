import torch
from torch import nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einx
from typing import cast
import math
from .nn_utils import softmax

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


class CausalMHSA(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 position_embedding: RoPE|None=None,
                 device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v, device, dtype)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model, device, dtype)

        self.position_embedding = position_embedding

    def forward(
                    self, 
                    x: Float[Tensor, " ... sequence_length d_in"],
                    token_positions: Int[Tensor, " ... seq"] | None = None
                ) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        Given the key, query, and value projection weights of a naive unbatched
        implementation of multi-head attention, return the output of an optimized batched
        implementation. This implementation should handle the key, query, and value projections
        for all heads in a single matrix multiply.
        This function should not use RoPE.
        See section 3.2.2 of Vaswani et al., 2017.

        Args:
            x (Float[Tensor, "... sequence_length d_in"]): input Tensor.
            token_positions (Int[Tensor, "... sequence_length"]): position indices


        Returns:
            Casual Multi-Head Self-Attention output.
        """
        *b, seq_len, d_model = x.shape
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split head
        Q, K, V = (
            rearrange(X, '... seq (head d) -> ... head seq d', head=self.num_heads)
            for X in (Q, K, V)
        )

        if self.position_embedding is not None:
            # apply RoPE
            if token_positions is None:
                token_positions = cast(Tensor, einx.rearrange('seq -> b... seq', torch.arange(seq_len), b=[1]*len(b)))

            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            Q = self.position_embedding(Q, token_positions)
            K = self.position_embedding(K, token_positions)

        # casual Mask
        seq = torch.arange(seq_len, device=x.device)
        qi = cast(Tensor, einx.rearrange('q -> b... 1 q 1', seq, b=[1]*len(b)))
        kj = cast(Tensor,einx.rearrange('k -> b... 1 1 k', seq, b=[1]*len(b)))
        causal_mask = qi >= kj

        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        attn_output = rearrange(attn_output, '... h seq d -> ... seq (h d)')

        output = self.output_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 position_embedding: RoPE|None=None,
                 device=None,
                 dtype=None):
        super().__init__()
        self.attn = CausalMHSA(
            d_model=d_model,
            num_heads=num_heads,
            position_embedding=position_embedding,
            device=device,
            dtype=dtype
            )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        x_attn = self.attn(self.ln1(x))
        attn_output = x + x_attn
        x_ff = self.ffn(self.ln2(attn_output))
        ff_output = x_ff + attn_output
        return ff_output


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None):
        super().__init__()

        # save config for further save and re-create the model
        # such as: model = TransformerLM(**loaded_config)
        self.config = {
            k: v for k, v in locals().items() 
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }

        # input embedding
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        # position embedding
        d_head = d_model // num_heads
        self.position_embedding = RoPE(rope_theta, d_head, context_length, device)
        
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    position_embedding=self.position_embedding,
                    device=device,
                    dtype=dtype
                )
                for i in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)








    
def silu(x):
    return x * torch.sigmoid(x)



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