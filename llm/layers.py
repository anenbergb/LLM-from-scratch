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
