from collections.abc import Callable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with weight decay.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square.
            Typically the default is (0.9, 0.999), but for LLMs like LLaMA and GPT-3, the
            default is (0.9, 0.95).
        eps (float, optional): Term added to the denominator to improve numerical stability.
            Default: 1e-8
        weight_decay (float, optional): Weight decay coefficient. Default: 0.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]  # Get the beta parameters.
            eps = group["eps"]  # Get the epsilon parameter.
            weight_decay = group["weight_decay"]  # Get the weight decay parameter.
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data  # Get the gradient of loss with respect to p.
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                # Get the first moment estimate from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                # Get the second moment estimate from the state, or initial value.
                v = state.get("v", torch.zeros_like(p.data))

                # Apply weight decay.
                p.data -= lr * weight_decay * p.data

                # Update the first moment estimate.
                m = beta1 * m + (1 - beta1) * grad
                # Update the second moment estimate.
                v = beta2 * v + (1 - beta2) * grad * grad
                # Compute the adjusted learning rate for iteration t
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # Update weight tensor in-place.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Cosine learning rate decay schedule with linear warmup.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.
            T_c should be greater than T_w.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    else:
        return min_learning_rate


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr=1)
    opt = AdamW([weights], lr=1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
