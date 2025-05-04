import torch
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Iterable


def softmax(in_features: Float[Tensor, " ..."], dim: int = -1, temperature: int = 1.0) -> Float[Tensor, " ..."]:
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
    assert temperature > 0
    if temperature != 1:
        in_features = in_features / temperature

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


def perplexity(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]) -> Float[Tensor, ""]:
    return torch.exp(cross_entropy(inputs, targets))


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> float:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum((g**2).sum() for g in grads)).item()

    clip_coef = min(1, max_l2_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef

    return norm
