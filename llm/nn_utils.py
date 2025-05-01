import torch
from einops import einsum, rearrange
import einx
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
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


def log_softmax(in_features: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of log_softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to log_softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply log_softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        log_softmax normalizing the specified `dim`.
    """
    max_along_dim = torch.max(in_features, dim=dim, keepdim=True).values
    in_features_submax = in_features - max_along_dim
    exp = torch.exp(in_features_submax)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    return in_features_submax - log_sum_exp


def cross_entropy(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]) -> Float[Tensor, ""]:
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
    # (batch_size, seq_len, vocab_size)
    neg_log_prob = -log_softmax(inputs, dim=-1)
    # targets index the corret class for each example
    # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, 1)
    gathered = torch.gather(neg_log_prob, -1, targets.unsqueeze(-1))
    return torch.mean(gathered)
