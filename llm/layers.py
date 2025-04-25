import torch
from torch import nn
from einops import einsum
import math


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
        # Remove W3 from state_dict if it exists
        assert "W1" in state_dict, "W1 not found in state_dict"
        assert "W2" in state_dict, "W2 not found in state_dict"
        assert "W3" in state_dict, "W2 not found in state_dict"
        self.W1.load_state_dict({"W": state_dict["W1"]}, **kwargs)
        self.W2.load_state_dict({"W": state_dict["W2"]}, **kwargs)
        self.W3.load_state_dict({"W": state_dict["W3"]}, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer
        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        """
        w1_out = self.W1(x)
        w3_out = self.W3(x)
        silu_out = w1_out * torch.sigmoid(w1_out)
        out = self.W2(silu_out * w3_out)
        return out
