import torch
from torch import nn
from einops import einsum
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor


class Linear(nn.Module):
    """
    Linear transformation module
    This function should accept the following parameters:
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device:  Device to store the parameters on
    dtype:  Data type of the parameters
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        variance = 2 / (in_features + out_features)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer
        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        """
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    Embedding layer
    This function should accept the following parameters:
    num_embeddings: int number of embeddings. Size of vocabulary
    embedding_dim: int dimension of each embedding vector, i.e. d_model
    device:  Device to store the parameters on
    dtype:  Data type of the parameters
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # W is (num_embeddings, d_model)
        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer. Look up the embedding vectors for the given token IDs.
        :param token_ids: Input tensor of shape (batch_size, sequence_length)
        :return: Output tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        assert token_ids.dtype == torch.long, "token_ids must be of type long"
        return self.W[token_ids]


class RMSNorm(nn.Module):
    """
    RMSNorm layer
    This function should accept the following parameters:
    d_model: int hidden dimension of the model
    eps: float small value to avoid division by zero. numerical stability
    device:  Device to store the parameters on
    dtype:  Data type of the parameters
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        # W is the learnable gain
        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer
        :param x: Input tensor of shape (batch_size, sequence_length, d_model)
        :return: Output tensor of shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean + self.eps)
        x /= rms  # normalize
        x *= self.W  # apply learnable gain
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU layer
    This function should accept the following parameters:
    d_model: int hidden dimension of the model
    d_ff: int hidden dimension of the feedforward layer
    device:  Device to store the parameters on
    dtype:  Data type of the parameters

    SiLU = x * sigmoid(x)
    GLU = sigmoid(W1 * x) * (W2 * x)
    SwiGLU(x,W1,W2,W3) = W2 * (SiLU(W1*x) * W3*x)

    x (d_model,)
    W1 (d_ff, d_model)
    W3 (d_ff, d_model)
    W2 (d_model, d_ff)

    d_ff = 8/3 * d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % 64 == 0, "d_model (model hidden dim) must be a multiple of 64"
        self.d_model = d_model
        if d_ff is None:
            d_ff = math.floor((8 / 3) * d_model)
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, d_ff)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], **kwargs) -> None:
        """
        Load state dict for the SwiGLU layer
        :param state_dict: State dict containing the weights
        :param strict: Whether to enforce that the keys in state_dict match the keys returned by this module's state_dict()
        """
        for Wnum in ["W1", "W2", "W3"]:
            assert Wnum in state_dict, f"{Wnum} not found in state_dict"
            getattr(self, Wnum).load_state_dict({"W": state_dict[Wnum]}, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer
        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        SwiGLU(x,W1,W2,W3) = W2 * (SiLU(W1*x) * W3*x)
        """
        w1_out = self.W1(x)
        w3_out = self.W3(x)
        silu_out = w1_out * torch.sigmoid(w1_out)
        out = self.W2(silu_out * w3_out)
        return out


class RotaryPositionalEmbedding(nn.Module):
    """
    theta: float theta value for the RoPE
    d_k: int dimension of query and key vectors
    max_seq_len: int, maximum sequence length that will be inputted
    device: device to store the buffer on

    We precompute the cosine and sine values for every possible token position from
    0 to max_seq_len - 1 because we need to apply a position-dependent rotation
    to each query/key vector at each position in the sequence.
    * for every position (0, 1, 2, ..., max_seq_len-1),
    * for every dimension (in pairs of 2, because rotations are 2D),
    sin_tensor or cos_tensor shape (max_seq_len, d_k)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0
        dk2 = d_k // 2
        k = torch.arange(dk2, device=device, dtype=torch.float32)
        theta_scale = 1.0 / (theta ** (2 * k / d_k))
        seq_indices = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # theta_ik of shape (max_seq_len, d_k //2)
        theta_ik = seq_indices.view(len(seq_indices), 1) * theta_scale.view(1, len(theta_scale))
        cos_tensor = torch.cos(theta_ik)
        sin_tensor = torch.sin(theta_ik)
        self.register_buffer("cos", cos_tensor, persistent=False)
        self.register_buffer("sin", sin_tensor, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """
        cos = self.cos[token_positions]  # (...,seq_len, d_k // 2)
        sin = self.sin[token_positions]

        # interleave the final dimension, so it's [cos_0, cos_0, cos_1, cos_1, ..., cos_d/2, cos_d/2]
        cos_expanded = torch.repeat_interleave(cos, 2, dim=-1)  # (..., seq_len, d_k)
        sin_expanded = torch.repeat_interleave(sin, 2, dim=-1)  # (..., seq_len, d_k)

        # transform [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] to [-1, 0, -3, 2, -5, 4, -7, 6, -9, 8]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # (..., seq_len, d_k/2) stacked to (...,seq_len, d_k/2, 2)
        # reshape will interleave the -x2 and x1.
        x_neg_shift = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        # Equivalent to
        # x_neg_shift = torch.empty_like(x)
        # x_neg_shift[..., 1::2] = x[..., ::2]
        # x_neg_shift[..., ::2] = -x[..., 1::2]
        return x * cos_expanded + x_neg_shift * sin_expanded


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_along_dim = torch.max(in_features, dim=dim, keepdim=True).values
    in_features_submax = in_features - max_along_dim
    exp = torch.exp(in_features_submax)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
            This should be a boolean tensor where
                - `True` indicates that this token should be attended to (not masked)
                  The attention probabilities of positions with a mask value of True should collectively sum to 1.
                - `False` indicates that this token should not be attended to (masked)
                  The attention probabilities of positions with a mask value of False should be set to 0.
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")  # Q @ K^T
    scores /= math.sqrt(d_k)
    if mask is not None:
        scores += ~mask * -1e9
    attention_weights = softmax(scores, dim=-1)
    return einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
