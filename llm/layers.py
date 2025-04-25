import torch
from torch import nn
from einops import einsum


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer
        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        """
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
