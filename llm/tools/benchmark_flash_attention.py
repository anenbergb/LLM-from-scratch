import argparse
import os
import torch
import triton
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import product
from llm.flash_attention import FlashAttention, attention_pytorch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention vs PyTorch attention.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--n-heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"], help="List of dtypes: float32 bfloat16")
    parser.add_argument(
        "--seq-lengths", nargs="+", type=int, default=[2**i for i in range(7, 17)], help="Sequence lengths to test."
    )
    parser.add_argument(
        "--d-heads", nargs="+", type=int, default=[2**i for i in range(4, 8)], help="Head dimensions to test."
    )
    parser.add_argument("--not-causal", action="store_true", default=False, help="Don't use causal attention.")
    parser.add_argument("--num-warmups", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of benchmark trials.")
    parser.add_argument("--output-dir", type=str, default="flash_bench_outputs", help="Output directory for plots.")
    parser.add_argument("--plot-seq-len", type=int, default=16384, help="Sequence length for plotting bar chart.")
    parser.add_argument("--plot-d-head", type=int, default=64, help="Head dim for plotting bar chart.")
    return parser.parse_args()


def benchmark_flash_attention(args):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # Assumes you're using GPU 0

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_seq_len not in args.seq_lengths:
        raise ValueError(f"--plot-seq-len={args.plot_seq_len} is not in --seq-lengths {args.seq_lengths}")
    if args.plot_d_head not in args.d_heads:
        raise ValueError(f"--plot-d-head={args.plot_d_head} is not in --d-heads {args.d_heads}")

    causal = not args.not_causal

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype_to_string = {v: k for k, v in dtype_map.items()}
    dtypes = [dtype_map[d] for d in args.dtypes]

    compiled_attention = torch.compile(attention_pytorch, mode="max-autotune")
    results = []
    latency_plot_data = {torch.float32: [], torch.bfloat16: []}

    for dtype in dtypes:
        for seq_len, d_head in product(args.seq_lengths, args.d_heads):
            print(f"Running: seq={seq_len}, d_head={d_head}, dtype={dtype}")
            qkv = torch.randn((3, args.n_heads, seq_len, d_head), dtype=dtype, device="cuda", requires_grad=True)
            q, k, v = qkv[0], qkv[1], qkv[2]

            def benchmark(fn):
                def fwd_bwd():
                    out = fn(q, k, v, causal)
                    loss = out.sum()
                    loss.backward()

                torch.cuda.empty_cache()
                fwd_bwd()  # one warmup to compile and initialize
                torch.cuda.synchronize()

                mem_info_before = nvmlDeviceGetMemoryInfo(handle).used

                time = triton.testing.do_bench(fwd_bwd, rep=args.num_trials, warmup=args.num_warmups)

                torch.cuda.synchronize()
                mem_info_after = nvmlDeviceGetMemoryInfo(handle).used
                peak_device_mem_gb = max(mem_info_before, mem_info_after) / (1000 * 1024**2)
                return time, peak_device_mem_gb

            time_pyt = time_compile = time_triton = None
            mem_pyt = mem_compile = mem_triton = None

            try:
                time_pyt, mem_pyt = benchmark(attention_pytorch)
                q.grad = k.grad = v.grad = None
            except Exception as e:
                print(f"⚠️ Skipped benchmark(attention_pytorch) ({seq_len}, {d_head}, {dtype}): {e}")
            try:
                time_compile, mem_compile = benchmark(compiled_attention)
                q.grad = k.grad = v.grad = None
            except Exception as e:
                print(f"⚠️ Skipped benchmark(compiled_attention) ({seq_len}, {d_head}, {dtype}): {e}")
            try:
                time_triton, mem_triton = benchmark(FlashAttention.apply)
                q.grad = k.grad = v.grad = None
            except Exception as e:
                print(f"⚠️ Skipped benchmark(FlashAttention.apply) ({seq_len}, {d_head}, {dtype}): {e}")

            results.append(
                [
                    seq_len,
                    d_head,
                    dtype_to_string.get(dtype, f"{dtype}"),
                    "N/A" if time_pyt is None else f"{time_pyt:.2f}",
                    "N/A" if time_compile is None else f"{time_compile:.2f}",
                    "N/A" if time_triton is None else f"{time_triton:.2f}",
                    "N/A" if mem_pyt is None else f"{mem_pyt:.2f}",
                    "N/A" if mem_compile is None else f"{mem_compile:.2f}",
                    "N/A" if mem_triton is None else f"{mem_triton:.2f}",
                ]
            )

            if seq_len == args.plot_seq_len and d_head == args.plot_d_head:
                latency_plot_data[dtype].append(("PyTorch", time_pyt))
                latency_plot_data[dtype].append(("Compiled", time_compile))
                latency_plot_data[dtype].append(("Triton", time_triton))

    print("\n== Benchmark Results ==")
    print(
        tabulate(
            results,
            headers=[
                "Seq Len",
                "D Head",
                "Dtype",
                "PyTorch (ms)",
                "Compiled (ms)",
                "Triton (ms)",
                "PyTorch (GB)",
                "Compiled (GB)",
                "Triton (GB)",
            ],
            tablefmt="github",
        )
    )

    # Plot latency comparison
    for dtype in dtypes:
        data = latency_plot_data[dtype]
        if not data:
            continue
        labels = [name for name, _ in data]
        values = [(0 if time is None else time) for _, time in data]

        plt.figure()
        plt.bar(labels, values)
        plt.title(f"Latency @ seq={args.plot_seq_len}, d_head={args.plot_d_head} ({dtype})")
        plt.ylabel("Latency (ms)")
        plt.grid(axis="y")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"latency_{dtype}.png")
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    args = get_args()
    benchmark_flash_attention(args)
