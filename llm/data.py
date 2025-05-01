import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Randomly sample starting indices for each example in the batch
    starting_indices = np.random.randint(low=0, high=len(dataset) - context_length, size=batch_size)

    # Sample input sequences and labels based on the starting indices
    x = np.array([dataset[i : i + context_length] for i in starting_indices])
    y = np.array([dataset[i + 1 : i + context_length + 1] for i in starting_indices])

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y
