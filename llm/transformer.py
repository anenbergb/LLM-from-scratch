import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor

from llm.layers import Linear, Embedding, RMSNorm, SwiGLU, CausalMHSARoPE, RotaryPositionalEmbedding


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
        RoPE: RotaryPositionalEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = CausalMHSARoPE(d_model, num_heads, RoPE=RoPE, device=device, dtype=dtype)
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


class TransformerLM(nn.Module):
    """
    Transformer model with RoPE.

    vocab_size (int): Size of the vocabulary.
    context_length (int): The maximum number of tokens to process at once.
    num_layers (int): The number of Transformer layers to use.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_model (int): The dimensionality of the model embeddings and sublayer outputs.
    d_ff (int): Dimensionality of the feed-forward inner layer
    rope_theta (float): The RoPE $\Theta$ parameter.
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.RoPE = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, RoPE=self.RoPE, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch sequence_length"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length vocab_size"]:
        """
        in_indices: Input tensor of shape (batch_size, sequence_length).
        Returns:
            Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
