import argparse
import os
import sys
from loguru import logger
import torch
import timeit
from typing import Callable
import numpy as np
from collections import defaultdict

from llm.nn_utils import cross_entropy
from llm.optimizer import AdamW
from llm.tools.train_llm import set_all_seeds, load_model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run the LLM pre-training.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Precision to use for training.",
    )
    # data loading
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")

    parser.add_argument(
        "--num-warmups",
        type=int,
        default=5,
        help="Number of warmup steps for benchmarking",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of trials for benchmarking",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size. This is the size of the BPE vocabulary.",
    )

    # Model parameters
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Dimensionality of the model",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=1344,
        help="Dimensionality of the feed-forward layer. This is roughly (8/3)*d_model while beineg a multiple of 64.",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter",
    )
    parser.add_argument(
        "--weight-sharing",
        action="store_true",
        default=False,
        help="Use weight sharing between token embeddings and output layer. Uses the Linear layer weight initialization.",
    )

    return parser.parse_args()


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = timeit.default_timer()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)
    mean_time = float(np.mean(times))
    return mean_time


def benchmark_llm(args):
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    # allows PyTorch to use TF32 instead of true FP32 when doing matrix multiplications
    torch.set_float32_matmul_precision("high")
    if args.precision == "bf16":
        precision = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            logger.error("bf16 is not supported on this device. Please use fp16 or fp32.")
            sys.exit(1)
    else:
        precision = torch.float32
    logger.info(f"Using device: {device} and precision: {precision}")

    model = load_model(args, args.vocab_size, device)
    random_batch = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )  # torch.int64

    optimizer = AdamW(model.parameters())

    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    with torch.autocast(device_type=device, dtype=precision):
        for _ in range(args.num_warmups):
            model(random_batch)
    torch.cuda.synchronize()

    times = defaultdict(list)
    for trial in range(args.num_trials):  # Do it multiple times to capture variance
        start_time = timeit.default_timer()
        with torch.autocast(device_type=device, dtype=precision):
            logits = model(random_batch)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times["forward"].append((end_time - start_time) * 1000)
        with torch.autocast(device_type=device, dtype=precision):
            loss = cross_entropy(logits, random_batch)

        start_time = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times["backward"].append((end_time - start_time) * 1000)

        start_time = timeit.default_timer()
        optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times["optimizer"].append((end_time - start_time) * 1000)
        optimizer.zero_grad()

    for key, time_list in times.items():
        mean_time = float(np.mean(time_list))
        std_time = float(np.std(time_list))
        logger.info(f"{key} time: {mean_time:.3f} ± {std_time:.3f} ms over {len(time_list)} runs")
        # logger.info(f"{key} time per token: {mean_time / (args.batch_size * args.context_length):.2f} ms")


if __name__ == "__main__":
    args = get_args()
    sys.exit(benchmark_llm(args))
