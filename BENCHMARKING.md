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