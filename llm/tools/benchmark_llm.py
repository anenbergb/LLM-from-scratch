import argparse
import os
import sys
from loguru import logger
import torch
import timeit
from typing import Callable
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from torch.profiler import profile

from llm.nn_utils import cross_entropy
from llm.optimizer import AdamW
from llm.tools.train_llm import set_all_seeds, load_model

import torch.cuda.nvtx as nvtx


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Benchmark the LLM model by running a forward and backward pass.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-memory-profile",
        type=str,
        default=None,
        help="Save memory profile to this file. If not set, no memory profile will be saved.",
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
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Precision to use for training.",
    )
    # data loading
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--context-length", type=int, default=32, help="Context length")

    parser.add_argument(
        "--num-warmups",
        type=int,
        default=10,
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
    elif args.precision == "fp16":
        precision = torch.float16
    else:
        precision = torch.float32
    logger.info(f"Using device: {device} and precision: {precision}")

    model = load_model(args, args.vocab_size, device, log_flops_params=False)
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

    if args.save_memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times = defaultdict(list)
    for trial in range(args.num_trials):  # Do it multiple times to capture variance
        start_time = timeit.default_timer()
        with torch.autocast(device_type=device, dtype=precision), nvtx.range("forward"):
            logits = model(random_batch)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times["forward"].append((end_time - start_time) * 1000)
        with torch.autocast(device_type=device, dtype=precision):
            loss = cross_entropy(logits, random_batch)

        try:
            start_time = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times["backward"].append((end_time - start_time) * 1000)
        except:
            times["backward"].append(np.inf)

        try:
            start_time = timeit.default_timer()
            with nvtx.range("optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times["optimizer"].append((end_time - start_time) * 1000)
        except:
            times["optimizer"].append(np.inf)
        optimizer.zero_grad()

    if args.save_memory_profile:
        logger.info(f"Memory profile saved to {args.save_memory_profile}")
        torch.cuda.memory._dump_snapshot(args.save_memory_profile)
        # stop recording memory history
        torch.cuda.memory._record_memory_history(enabled=None)

    data = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    # Add timing statistics
    for key in ["forward", "backward", "optimizer"]:
        data[f"{key}_mean"] = f"{float(np.mean(times[key])):.3f}"
        data[f"{key}_std"] = f"{float(np.std(times[key])):.3f}"

    print(
        tabulate(
            [data.values()],
            headers=data.keys(),
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    args = get_args()
    sys.exit(benchmark_llm(args))
