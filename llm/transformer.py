import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor

from llm.layers import Embedding, RMSNorm, SwiGLU, CausalMHSARoPE


class TransformerBlock(nn.Module):
    """
    d_model (int): The dimensionality of the Transformer block input.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_ff (int): Dimensionality of the feed-forward inner layer.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    theta (float): RoPE parameter.
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = CausalMHSARoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        """
        pre-norm Transformer block

        in_features: Input tensor of shape (batch_size, sequence_length, d_model).
        Returns:
            Output tensor of the same shape as in_features.
        """
        out = in_features + self.attn(self.ln1(in_features), token_positions)
        out = out + self.ffn(self.ln2(out))
        return out
