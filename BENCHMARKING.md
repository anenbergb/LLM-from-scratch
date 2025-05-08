# End-to-End Benchmarking of the Transformer LLM
All benchmarking is performed on an NVIDIA RTX 5090 GPU with 32 Gb of vRAM

Benchmarking the forward, backward, and optimizer update for the TransformerLM
- batch-size: 4
- context-length: 128
- vocabulary-size: 10,000
- 10 warmup steps, 100 measurement steps

|   d_model |   d_model |     d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
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
