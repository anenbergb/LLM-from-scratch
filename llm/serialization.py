import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # Move model parameters to CPU before saving
    model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    optimizer_cpu_state = {
        k: {ik: iv.cpu() if torch.is_tensor(iv) else iv for ik, iv in v.items()} if isinstance(v, dict) else v
        for k, v in optimizer.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": model_cpu_state,
            "optimizer_state_dict": optimizer_cpu_state,
            "iteration": iteration,
        },
        out,
    )
    torch.cuda.empty_cache()  # Free unused cached GPU memory


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
