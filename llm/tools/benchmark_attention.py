import argparse
import torch
import timeit
import numpy as np
from tabulate import tabulate
from loguru import logger
from llm.layers import scaled_dot_product_attention


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark scaled dot product attention at various sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--d-models", nargs="+", type=int, default=[16, 32, 64, 128], help="List of embedding dimensions."
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384], help="List of sequence lengths."
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="Precision to use."
    )
    parser.add_argument("--num-warmups", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of benchmark trials.")

    parser.add_argument(
        "--save-memory-profile",
        type=str,
        default=None,
        help="If set, save CUDA memory snapshot to this path.",
    )

    return parser.parse_args()


def benchmark_attention(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.precision == "bf16":
        dtype = torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            logger.error("bf16 is not supported on this device.")
            exit(1)
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info(f"Using device: {device}, precision: {args.precision}")

    results = []

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            logger.info(f"Benchmarking d_model={d_model}, seq_len={seq_len}")
            Q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            K = torch.randn_like(Q)
            V = torch.randn_like(Q)

            try:
                # Warmup
                for _ in range(args.num_warmups):
                    out = scaled_dot_product_attention(Q, K, V)
                    _ = out.sum()
                torch.cuda.synchronize()

                forward_times = []
                for _ in range(args.num_trials):
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    out = scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    forward_times.append((end - start) * 1000)

                if args.save_memory_profile:
                    torch.cuda.memory._record_memory_history(max_entries=1000000)

                # Memory profiling and backward timing
                torch.cuda.reset_peak_memory_stats()
                out = scaled_dot_product_attention(Q, K, V)
                loss = out.sum()
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

            except:
                logger.error(f"Likely ran out of memory during forward pass for d_model={d_model}, seq_len={seq_len}.")
                forward_times = [np.inf]
                peak_mem = np.inf

            try:
                backward_times = []
                for _ in range(args.num_trials):
                    out = scaled_dot_product_attention(Q, K, V)
                    loss = out.sum()
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    backward_times.append((end - start) * 1000)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
            except:
                logger.error(f"Likely ran out of memory during backward pass for d_model={d_model}, seq_len={seq_len}.")
                backward_times = [np.inf]

            if args.save_memory_profile:
                logger.info(f"Memory profile saved to {args.save_memory_profile}")
                torch.cuda.memory._dump_snapshot(args.save_memory_profile)
                torch.cuda.memory._record_memory_history(enabled=None)

            results.append(
                {
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_mean_ms": f"{np.mean(forward_times):.2f}",
                    "forward_std_ms": f"{np.std(forward_times):.2f}",
                    "backward_mean_ms": f"{np.mean(backward_times):.2f}",
                    "backward_std_ms": f"{np.std(backward_times):.2f}",
                    "peak_mem_MB": f"{peak_mem:.1f}",
                }
            )

    print(tabulate([r.values() for r in results], headers=results[0].keys(), tablefmt="github"))


if __name__ == "__main__":
    args = get_args()
    benchmark_attention(args)
