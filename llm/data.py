from typing import Iterator
import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import IterableDataset


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


def random_training_iterator(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    max_iter: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yields `max_iter` random training batches using get_batch.
    """
    for _ in range(max_iter):
        yield get_batch(dataset, batch_size, context_length, device)


class SequentialValidationDataset(IterableDataset):
    def __init__(
        self,
        dataset: npt.NDArray,
        context_length: int,
        batch_size: int,
        device: str = "cpu",
    ):
        """
        IterableDataset that yields sequential non-overlapping batches from a 1D tokenized dataset.

        Args:
            dataset: 1D numpy array of token IDs.
            context_length: Length of each sequence.
            batch_size: Number of sequences per batch.
            device: Device to move tensors to.
        """
        self.dataset = dataset
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

        self.total_tokens = len(dataset)
        self.num_sequences = (self.total_tokens - 1) // context_length

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Generate all sequences
        all_x = []
        all_y = []
        for i in range(self.num_sequences):
            start = i * self.context_length
            end = start + self.context_length + 1
            if end > self.total_tokens:
                break
            seq = self.dataset[start:end]
            all_x.append(seq[:-1])
            all_y.append(seq[1:])

        all_x = np.stack(all_x)
        all_y = np.stack(all_y)

        # Yield in batches
        for i in range(0, len(all_x), self.batch_size):
            xb = torch.LongTensor(all_x[i : i + self.batch_size])
            yb = torch.LongTensor(all_y[i : i + self.batch_size])

            if "cuda" in self.device:
                xb = xb.pin_memory().to(self.device, non_blocking=True)
                yb = yb.pin_memory().to(self.device, non_blocking=True)
            else:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

            yield xb, yb
