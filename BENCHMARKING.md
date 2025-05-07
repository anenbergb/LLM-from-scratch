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
