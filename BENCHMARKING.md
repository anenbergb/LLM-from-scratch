# End-to-End Benchmarking of the Transformer LLM
All benchmarking is performed on an NVIDIA RTX 5090 GPU with 32 Gb of vRAM

Benchmarking the forward, backward, and optimizer update for the TransformerLM at float32
- batch-size: 4
- context-length: 128
- vocabulary-size: 10,000
- 5 warmup steps, 10 measurement steps

|   Size |   d_model |     d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
|----------:|----------:|---------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| tiny      |       512 |   1344 |            4 |          16 |         22.704 |         5.313 |          25.165 |         17.579 |           13.211 |           2.184 |
| small     |       768 |   3072 |           12 |          12 |         57.358 |        15.602 |          64.199 |         31.564 |           38.943 |           8.455 |
| medium    |      1024 |   4096 |           24 |          16 |         88.199 |        30.364 |         104.871 |         43.512 |           60.916 |          31.087 |
| large     |      1280 |   5120 |           36 |          20 |        124.505 |        78.886 |         122.454 |         23.102 |            93.57 |          33.799 |
| xl   |      1600 |   6400 |           48 |          25 |        129.189 |        48.891 |             inf |            nan |              inf |             nan |
| 2.7B |      2560 |  10240 |           32 |          32 |        137.095 |        56.249 |             inf |            nan |              inf |             nan |

- the backward pass consistently takes at least 1x longer than the forward pass
- the standard deviation for the measurements is fairly high

### Nsight Systems Profiler

```
    --python-backtrace=cuda
```
* Capture the full Python call stack (backtrace) each time a CUDA API function is called.
* Can be used to trace CUDA execution back to your Python code, including the file name and line number.
* You’ll see the call stack in Nsight Systems GUI, attached to CUDA events like:
- cudaLaunchKernel
- cudaMemcpyAsync
- cuStreamSynchronize, etc.

```
    --trace=cuda,osrt,nvtx
```
These trace options are used to get a full performance profile of the CPU logic, GPU execution, and annotated code regions

- **`cuda`** – Captures GPU activity (kernels, memory transfers, streams).
- **`osrt`** – Captures CPU-side OS events (threads, scheduling, sync primitives).
- **`nvtx`** – Captures custom code annotations via NVTX (e.g., `nvtx.range_push()`).

Profiling with Nsight Systems adds latency the pytorch execution. For example, the latency for the `medium` model size by the following amounts when the python benchmark
script was wrapped with `nsys profile -o output_file --trace=cuda,osrt,nvtx` and with `nvtx` annotations added to the scaled dot product attention.
| nsys profile |   d_model |   d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
|----------- |-----------|--------|--------------|-------------|----------------|---------------|-----------------|----------------|------------------|-----------------|
| NO |      1024 |   4096 |           24 |          16 |         88.799 |        36.441 |         103.395 |         25.744 |           49.827 |          18.464 |
| YES |      1024 |   4096 |           24 |          16 |        101.934 |        34.614 |         143.293 |         32.401 |           89.952 |          21.156 |

The following Nsight Systems application view shows that the NVTX Range Summaries for values such as `forward`, `backward` and `optimizer` agree with the python `timeit.default_timer()` timing. For example, the average time reported in Nsight Systems for the `forward` pass is `101.758 ms` which is very close to the `101.934 ms` measured in python.
![image](https://github.com/user-attachments/assets/a4cb18bd-da35-47b0-94d0-a14c21a6193c)

- The CUDA kernels that take the most GPU time during the forward pass are `void cutlass::Kernel2`. These kernels are invoked 72 and 24 times.
- A slightly different kernel, namely the `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nt_align4>(T1::Params)` takes the most time in the backward pass.
- Besides the matrix multiplications, the PyTorch CUDA kernel for copying float data (`direct_copy_kernel_cuda`) takes up a non-trivial amount of runtime.
```
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
42.1%	3.672 ms	72	51.000 μs	51.232 μs	50.208 μs	51.552 μs	434 ns	void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)
15.0%	1.308 ms	24	54.500 μs	54.480 μs	54.240 μs	54.912 μs	206 ns	void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align4>(T1::Params)
7.6%	659.137 μs	288	2.288 μs	1.984 μs	1.281 μs	6.336 μs	1.263 μs	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
```

The average runtime of `scaled_dot_product_attention` is **1.168 ms**, broken down as follows:

- **Attention score (QKᵀ)**: 223 µs — compute-intensive, high FLOPs. Larger matrix.
- **Softmax**: 195 µs — low FLOPs but memory-bound and latency-sensitive. Despite fewer FLOPs, softmax involves nonlinear ops + memory-bound behavior (reads/writes, reduction across a dimension), leading to relatively high runtime per FLOP.
- **Final matmul (AV)**: 50 µs — moderate FLOPs, runs efficiently. Smaller matrix.

Runtime differences do not directly reflect FLOP counts due to variations in memory access patterns, kernel efficiency, and operation type. Softmax, for example, has low computational cost but relatively high runtime due to memory-bound behavior.

### Floating Point Precision
#### float32
<img src="https://github.com/user-attachments/assets/765f48c9-157b-4414-a0b7-49dd79ec79f1" width="500"/>

```
x = torch.zeros(4, 8)
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```
One matrix in the feedforward layer of GPT-3:
```
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
```
#### float16
<img src="https://github.com/user-attachments/assets/bc3f9e10-2c89-4867-b764-a7ac5f5c21c0" width="500"/>

```
x = torch.zeros(4, 8, dtype=torch.float16)
assert x.element_size() == 2
However, the dynamic range (especially for small numbers) isn't great.
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0  # Underflow!
If this happens when you train, you can get instability.
```

#### bfloat16
<img src="https://github.com/user-attachments/assets/f6a843b1-bba1-4d2d-973f-bf62428b57b5" width="500"/>

- Google Brain developed bfloat (brain floating point) in 2018 to address this issue.
- bfloat16 uses the same memory as float16 but has the same dynamic range as float32!
- The only catch is that the resolution is worse, but this matters less for deep learning.

```
 x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0  # No underflow!
```

#### fp8
<img src="https://github.com/user-attachments/assets/5c886196-15d6-4c82-a63d-78e3313922b6" width="500"/>

- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).

Implications on training:
- Training with float32 works, but requires lots of memory.
- Training with fp8, float16 and even bfloat16 is risky, and you can get instability.
- Solution: use mixed precision training

#### Mixed Precision Training
- certain operations (e.g., matrix multiplies) are performed in lower-precision datatypes, while other operations that require the full dynamic range of FP32 (e.g., accumulations and reductions) are kept as-is.
- it is generally a good idea to keep accumulations in higher precision even if the tensors themselves being accumulated have been downcasted.

#### Mixed Precision: LayerNorm vs Feed-Forward Layers

In FP16 mixed precision, Layer Normalization is treated differently due to **numerical instability** in computing mean and variance. These operations are sensitive to:
- Small differences between large values (risk of cancellation)
- Precision loss in squaring, summing, and normalization

As a result, FP16 autocasting often runs LayerNorm in **FP32** to maintain stability.

With **BF16**, this precaution is typically unnecessary:
- BF16 has the same exponent range as FP32, reducing overflow/underflow risk
- LayerNorm can usually run in BF16 without significant loss of accuracy

> 🧠 LayerNorm is precision-sensitive due to statistics computation. BF16 handles this better than FP16 due to its wider dynamic range.


Benchmarking the forward, backward, and optimizer update for the TransformerLM at bfloat16
- batch-size: 4
- context-length: 128
- vocabulary-size: 10,000
- 5 warmup steps, 10 measurement steps

|   Size |   d_model |     d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
|----------:|----------:|---------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| tiny    | 512 |   1344 |            4 |          16 |          6.795 |         0.465 |           15.31 |         20.901 |            3.819 |           0.493 |
| small     |  768 |   3072 |           12 |          12 |         56.518 |        16.073 |         107.934 |          53.93 |           30.995 |          13.005 |
| medium |      1024 |   4096 |           24 |          16 |         63.071 |         0.867 |          96.169 |         24.091 |           40.205 |           6.164 |
| large |      1280 |   5120 |           36 |          20 |         114.05 |        59.272 |         144.909 |         31.679 |           60.535 |          10.149 |
| xl |      1600 |   6400 |           48 |          25 |         123.09 |        49.886 |             inf |            nan |              inf |             nan |
| 2.7B |      2560 |  10240 |           32 |          32 |         73.141 |        44.009 |         133.101 |         69.393 |              inf |             nan |

### Memory Profiling

```
... # warm-up phase in your benchmarking script

# Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
... # what you want to profile in your benchmarking script
# Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)
```
- saves memory_snapshot.pickle that you can load into the following online tool https://pytorch.org/memory_viz to visualize the overall memory usage timeline as well
as each individual allocation that was made, with its size and a stack trace leading to the code where it
originates.

The active memory timeline for only the forward pass is
![image](https://github.com/user-attachments/assets/3ce6a2c1-06a2-4f75-a0b9-c3ca7be71c54)
and the timeline for the full training step (foward, backward, optimizer step)
![image](https://github.com/user-attachments/assets/4266a3f2-922f-46d7-8637-9e2c45298835)

# Benchmarking Scaled Dot Product Attention

The following script was run to benchmark the scaled dot product attention
```
python llm/tools/benchmark_attention.py \
--batch-size 8 \
--d-models 16 32 64 128 \
--seq-lens 256 1024 4096 8192 16384 \
--precision bf16 \
--num-warmups 10 \
--num-trials 100
```
|   d_model |   seq_len |   forward_mean_ms |   forward_std_ms |   backward_mean_ms |   backward_std_ms |   peak_mem_MB |
|-----------|-----------|-------------------|------------------|--------------------|-------------------|---------------|
|        16 |       256 |              0.13 |             0.01 |               0.71 |              3.87 |          13.4 |
|        16 |      1024 |              0.14 |             0.07 |               0.32 |              0.07 |          97.5 |
|        16 |      4096 |              2.29 |             0.02 |               6.99 |              0.17 |        1301.2 |
|        16 |      8192 |              8.87 |             0.42 |              29.26 |              2.22 |        5146.6 |
|        16 |     16384 |             33.92 |             0.95 |             105.11 |              4.26 |       20517   |
|        32 |       256 |              0.34 |             0.24 |               0.63 |              0.08 |          25.8 |
|        32 |      1024 |              0.35 |             0.32 |               0.71 |              0.16 |          98.5 |
|        32 |      4096 |              2.35 |             0.14 |               7.1  |              0.2  |        1305.4 |
|        32 |      8192 |              8.84 |             0.16 |              27.18 |              1.44 |        5155.6 |
|        32 |     16384 |             35.09 |             2.01 |             105.13 |              3.58 |       20535   |
|        64 |       256 |              0.15 |             0.12 |               1.51 |              1.43 |          30.3 |
|        64 |      1024 |              0.36 |             0.95 |               1.01 |              1.12 |         100.7 |
|        64 |      4096 |              2.92 |             1.23 |               7.16 |              0.25 |        1313.9 |
|        64 |      8192 |              9.18 |             0.13 |              27.31 |              1.38 |        5173.6 |
|        64 |     16384 |             37.04 |             2.38 |             108.13 |              5.13 |       20571   |
|       128 |       256 |              0.31 |             0.12 |               0.64 |              0.06 |          39.3 |
|       128 |      1024 |              0.31 |             0.09 |               0.64 |              0.04 |         104.9 |
|       128 |      4096 |              3.11 |             1.1  |               7.38 |              0.05 |        1330.9 |
|       128 |      8192 |              9.87 |             0.02 |              28.55 |              1.86 |        5209.6 |
|       128 |     16384 |             42.05 |             2.43 |             113.17 |              5.27 |       20643   |


#### Benchmarking JIT-Compiled Attention
`torch.compile` will automatically generate fused Triton kernels by dynamically analyzing the computation graph.


The runtimes for the `torch.compile` version of the scaled dot product attention show faster runtime and lower peak memory usage.

|   d_model |   seq_len |   forward_mean_ms |   forward_std_ms |   backward_mean_ms |   backward_std_ms |   peak_mem_MB |
|-----------|-----------|-------------------|------------------|--------------------|-------------------|---------------|
|        16 |       256 |              1.81 |             1.82 |               3.72 |             28.87 |          12.4 |
|        16 |      1024 |              0.76 |             0.54 |              36.35 |            340.99 |          73.3 |
|        16 |      4096 |              1.72 |             0.24 |               3.97 |             19.48 |        1037   |
|        16 |      8192 |              5.67 |             1.34 |               8.87 |             18.19 |        4114.4 |
|        16 |     16384 |             17.6  |             0.48 |              74.18 |            440.8  |       16412.6 |
|        32 |       256 |              0.68 |             0.14 |               5.86 |             45.87 |          16.7 |
|        32 |      1024 |              1.05 |             1.36 |               3.28 |             21.27 |          74.4 |
|        32 |      4096 |              1.75 |             0.18 |               4.03 |             20.42 |        1041.3 |
|        32 |      8192 |              5.84 |             0.34 |               9.5  |             21.06 |        4123.4 |
|        32 |     16384 |             16.96 |             1.36 |              28.5  |             24.06 |       16430.6 |
|        64 |       256 |              0.7  |             0.16 |               3.15 |             27.44 |          21.2 |
|        64 |      1024 |              0.7  |             0.18 |               2.42 |             20.84 |          76.5 |
|        64 |      4096 |              1.91 |             0.2  |               4.31 |             21.48 |        1049.8 |
|        64 |      8192 |              5.67 |             0.48 |               9.67 |             21.4  |        4141.4 |
|        64 |     16384 |             18.83 |             1.06 |              29.85 |             33.49 |       16466.6 |
|       128 |       256 |              0.84 |             0.12 |               4.51 |             33.62 |          30.2 |
|       128 |      1024 |              1.03 |             0.84 |               3.83 |             33.37 |          80.8 |
|       128 |      4096 |              2.17 |             0.17 |               5.92 |             33.33 |        1066.8 |
|       128 |      8192 |              6.16 |             0.26 |              11.19 |             33.54 |        4177.4 |
|       128 |     16384 |             20.66 |             0.41 |              32.89 |             33.31 |       16538.6 |


Benchmarking the forward, backward, and optimizer update for the TransformerLM at bfloat16
- batch-size: 4
- context-length: 128
- vocabulary-size: 10,000
- 5 warmup steps, 100 measurement steps

Applying `torch.compile` to the full TransformerLM results in faster runtimes


|   Size |   d_model |     d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
|----------:|----------:|---------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| tiny      |       512 |   1344 |            4 |          16 |          0.935 |          0.02 |           1.827 |          0.024 |             3.29 |           0.041 |
| small    |       768 |   3072 |           12 |          12 |          2.828 |         0.032 |           6.583 |          0.051 |            9.621 |           0.066 |
| medium    |      1024 |   4096 |           24 |          16 |          6.283 |         0.041 |          15.915 |          0.072 |           20.138 |           0.033 |
| large     |      1280 |   5120 |           36 |          20 |         11.438 |         0.615 |          32.058 |            1.1 |           50.003 |           1.087 |
| xl   |      1600 |   6400 |           48 |          25 |         24.952 |         2.642 |             inf |            nan |              inf |             nan |
| 2.7B |      2560 |  10240 |           32 |          32 |        132.185 |       144.635 |             inf |            nan |           12.053 |           7.158 |
