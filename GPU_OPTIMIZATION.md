# GPU Architecture and Optimizations

## CPU vs GPU Architecture Summary

- **CPU Architecture:**
  - Optimized for minimizing latency within individual threads.
  - Requires large caches and complex control units to reduce data access time.
  - Context switching between threads is expensive; thus, only a few threads run per core.
  - Prioritizes fast execution per thread using low-latency memory access.

- **GPU Architecture:**
  - Focuses on hiding instruction and memory latency through massive parallelism.
  - Threads are lightweight and can be switched with no performance cost.
  - Frequently switches between threads every clock cycle to keep computation going.
  - Requires thousands of concurrent threads to maintain high efficiency.
  - Uses available threads to continue processing while others wait for data.

- **Key Difference:**
  - CPUs aim to reduce latency per thread using complex hardware (each thread finishes quickly).
  - GPUs manage latency by context-switching across many active threads. GPUs optimize for throughput (total processed data).

https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/
## GPU Architecture Diagram
![image](https://github.com/user-attachments/assets/553ceab5-9455-41fb-a1ef-70189f4e9471)
- GPUs have many SM (streaming multiprocessors) that independently execute ‘blocks’ (jobs).
- Each SM further contains many SPs (streaming processor) that can execute ‘threads’ in parallel
- The closer the memory to the SM, the faster it is
  - L1 and shared memory is inside the SM.
  - L2 cache is on die
  - global memory are the memory chips next to the GPU. global memory (VRAM) is accessed by the HBM2 connectors (or GDDR6+ for consumer GPUs)

### The Memory Accesses Latencies

| Memory Type        | CPI (cycles) |
|--------------------|--------------|
| Global memory      | 290          |
| L2 cache           | 200          |
| L1 cache           | 33           |
| Shared Memory (ld/st) | (23/19)     |

## Execution model of a GPU
<img src="https://github.com/user-attachments/assets/77803d57-ab2e-47f4-8ff4-de043a71ef93" width="800"/>

- Threads: Threads ‘do the work’ in parallel – all threads execute the same instructions but with different inputs (SIMT).
- Blocks: Blocks are groups of threads. Each block runs on a SM with its own shared memory.
- Warp: Threads always execute in a ‘warp’ of 32 consecutively numbered threads each.

## Memory model of a GPU
<img src="https://github.com/user-attachments/assets/f7d3a550-f376-4b3a-9b8e-c1d46cca3197" width="500"/>

- Each thread can access its own register, and shared memory within the block.
- Information that goes across blocks need to be read/written to global memory (slow)

## GPU Summary
- GPUs are massively parallel – same instructions applied across many workers.
  - Easy to scale up by adding more SMs (more workers)
  - Easy to program
- Threads are ‘lightweight’ and can be stopped and started
- Compute (and esp matmuls) have scaled faster than memory
- The memory hierarchy (SM, L1, L2, global memory) must be taken into account to speed up operations

## How to speed up computation on GPU
### (1) Control Divergence
<img src="https://github.com/user-attachments/assets/cfb7a8c9-e54b-49db-a9b7-3c798c56ed58" width="500"/>

- minimize conditionals that lead to significant overhead from the execution model

### (2) Low Precision Computation (optimization: trade memory for compute/accuracy)
- lower bit representations require less data movement
- modern GPUs accelerate low / mixed precision operations

https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf

### (3) Operator Fusion  (optimization: reduces memory accesses)

<img src="https://github.com/user-attachments/assets/8f5857bb-9611-4c49-ba2e-97957da9fee3" width="300"/>
<img src="https://github.com/user-attachments/assets/9c867b59-4367-482d-b210-8c6ac44d55dd" width="300"/>

- combine consecutive kernels into a single fused kernel to avoid moving memory movement
- Do as much of the computation in one place (in memory) before shipping it back to storage
- TensorRT performs layer fusion https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/best-practices/index.html

### (4) Recomputation (optimization: trade memory for compute/accuracy)
- throw away the intermediate activations computed in the forward pass, and recompute them in the backwards pass. This eliminates the slow memory transfers caused by saving and reloading the activations to memory.

### (5) Memory Coalescing and DRAM (optimization: reduces memory accesses)
<img src="https://github.com/user-attachments/assets/f374051e-fbcf-4b09-987e-8a26d26281c1" width="300"/>

- DRAM (global memory) is read in ‘burst mode’ – each read gives you a chunk of bytes
- Each address space is partitioned into burst sections. Whenever a location is accessed, all other locations in the same section are also delivered to the processor
- When all threads of a warp execute a load instruction, if all accessed locations fall into the same burst section, only one DRAM request will be made and the access is fully coalesced.

<img src="https://github.com/user-attachments/assets/ceaedb9b-c51d-4f7b-8f83-08e549f7bfda" width="300"/>
<img src="https://github.com/user-attachments/assets/1d778936-2957-4a6d-80a2-84946cf15a79" width="300"/>

- Memory traversal order matters
- For row-major matrices – threads that move along rows are not coalesced

### (6) Tiling (optimization: move memory to shared memory)
<img src="https://github.com/user-attachments/assets/1e3db433-8671-41d1-80a3-82d3e562c137" width="500"/>

- Tiling is the idea of grouping and ordering threads to minimize global memory access.
- Cut up the matrix into smaller ‘tiles’, and load this into shared memory

- Non-tiled matrix multiply: each input is read 𝑁 times from global memory
- Tiled matrix multiply: each input is read N/T timesfrom global memory, and 𝑇 times within each tile. This is a factor of 𝑇 reduction in global memory access.

- Advantages: repeated reads now access shared, not global memory and memory access can be coalesced

Factors affecting tile sizes
- Coalesced memory access
- Shared memory size
- Divisibility of the matrix dim. If the tile size doesn't divide the matrix size, then this will result in low utilization

<img src="https://github.com/user-attachments/assets/bbbdd34e-14cf-45e4-8d12-b2c78d12cc50" width="500"/>
- would like to align the tiles with the memory burst pattern (memory coalescing) to minimize memory reads to compute each tile.
- padding might be necessary to ensure that matrices are aligned

https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

# Flash Attention
- applies tiling and recomputation to overcome the technical challenge of computing exact attention in sub-quadratic HBM access.

<img src="https://github.com/user-attachments/assets/dcc5a0ed-7275-4620-9e8e-4b9fdb7b5e5f" width="150"/>
<img src="https://github.com/user-attachments/assets/1180e902-6288-4a15-a2be-1b8088ac817d" width="500"/>

<img src="https://github.com/user-attachments/assets/c83beab5-aa41-450e-933c-6d4036d7f712" width="500"/>

- tiling for a KQV matrix multiply
- Tile-wise computation of the inner products
- Tile-wise computation of the softmax via the online, telescoping sum trick
  - incrementally updating the max and telescoping sum to compute softmax tile-by-tile
- Fusion of the exponential operator

<img src="https://github.com/user-attachments/assets/a5892ebc-4c10-41d6-bd34-e6960a63e348" width="800"/>

# Triton
https://openai.com/index/triton/

| Feature                                  | CUDA   | Triton    |
|------------------------------------------|--------|-----------|
| Memory coalescing (transfer from DRAM)   | manual | automatic |
| Shared memory management                 | manual | automatic |
| Scheduling within SMs                    | manual | automatic |
| Scheduling across SMs                    | manual | manual    |

- Think about thread blocks rather than threads
- Triton operates on blocks, CUDA operates on threads.
- Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening).
- In Triton, a program instance is a block of threads all running the same program, and these thread blocks can be run in parallel on the GPU. Instead of taking tensors as arguments, we take pointers to their first elements, as well as strides for each tensor that tell us how to move along axes.

References:
- https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_06.json


# High-Performance Flash Attention Implementation

I implemented the following optimizations on top of the baseline Flash Attention v2 algorithm
- Added `@triton.autotune` to tune the tile sizes (Q_TILE_SIZE and K_TILE_SIZE) per kernel
- Optimized the backward pass to perform two passes over the input, one for dQ and another for dK and dV to avoid atomics or synchronization between blocks.
- Skip tiles that are fully masked (causal masking) to eliminate unnecessary computation 
- Avoid per-element masking for tiles that are guarenteed to be fully valid -- e.g. those squarely beneath the lower diagonal
- Only apply causal masking to the diagonal tiles

I benchmarked my FlashAttention v2 implementation and observed a **\~7× speedup** in inference performance compared to the naive PyTorch version. This was measured using Query, Key, and Value tensors with batch size 1, sequence length 13,312, embedding dimension 64, causal masking enabled, and the bfloat16 data type. All experiments were conducted on an NVIDIA RTX 5090 GPU.

[flash attention implementation](llm/flash_attention.py)

[benchmarking script](llm/tools/benchmark_flash_attention.py) `python llm/tools/benchmark_flash_attention.py`

![latency_torch bfloat16](https://github.com/user-attachments/assets/930519db-25ef-4b37-9cbb-e74b26683669)

|   Seq Len |   D Head | Dtype    | PyTorch (ms)   | Compiled (ms)   | Triton (ms)   | PyTorch (GB)   | Compiled (GB)   | Triton (GB)   |                                                           
|-----------|----------|----------|----------------|-----------------|---------------|----------------|-----------------|---------------|                                                           
|       128 |       64 | bfloat16 | 0.22           | 0.08            | 0.07          | 2.32           | 2.47            | 3.84          |                                                           
|       128 |      128 | bfloat16 | 0.20           | 0.14            | 0.15          | 4.11           | 4.11            | 5.49          |                                                           
|       512 |       64 | bfloat16 | 0.35           | 0.16            | 0.14          | 5.50           | 5.58            | 5.59          |                                                           
|       512 |      128 | bfloat16 | 0.26           | 0.20            | 0.29          | 5.58           | 5.58            | 5.58          |                                                           
|      1024 |       64 | bfloat16 | 0.70           | 0.33            | 0.28          | 5.58           | 5.59            | 5.59          |                                                           
|      1024 |      128 | bfloat16 | 1.03           | 0.50            | 0.72          | 5.59           | 5.59            | 5.59          |                                                           
|      8192 |       64 | bfloat16 | 52.46          | 28.18           | 7.70          | 15.57          | 9.68            | 9.73          |
|      8192 |      128 | bfloat16 | 56.00          | 35.44           | 26.79         | 19.70          | 13.52           | 13.52         |
|     13312 |       64 | bfloat16 |         **136.71** |           **47.99** |         **19.57** |          29.55 |           24.64 |         26.19 |
|     16384 |       64 | bfloat16 | N/A            | 73.39           | 28.29         | N/A            | 28.25           | 28.41         |
|     16384 |      128 | bfloat16 | N/A            | 89.22           | 101.21        | N/A            | 28.63           | 28.89         |
|     32768 |       64 | bfloat16 | N/A            | N/A             | 108.88        | N/A            | N/A             | 28.89         |
|     32768 |      128 | bfloat16 | N/A            | N/A             | 397.12        | N/A            | N/A             | 29.98         |
|     65536 |       64 | bfloat16 | N/A            | N/A             | 426.74        | N/A            | N/A             | 29.98         |
|     65536 |      128 | bfloat16 | N/A            | N/A             | 1574.34       | N/A            | N/A             | 32.15         |
|       128 |       64 | float32  | 0.21           | N/A             | 0.46          | 27.82          | N/A             | 27.83         |
|       128 |      128 | float32  | 0.69           | N/A             | 0.24          | 27.83          | N/A             | 27.82         |
|       512 |       64 | float32  | 0.39           | N/A             | 0.34          | 27.88          | N/A             | 27.84         |
|       512 |      128 | float32  | 0.68           | N/A             | 0.48          | 27.90          | N/A             | 28.09         |
|      1024 |       64 | float32  | 1.28           | N/A             | 0.33          | 28.18          | N/A             | 28.12         |
|      1024 |      128 | float32  | 1.70           | N/A             | 0.91          | 28.25          | N/A             | 28.13         |
|      8192 |       64 | float32  | N/A            | N/A             | 9.26          | N/A            | N/A             | 28.47         |
|      8192 |      128 | float32  | N/A            | N/A             | 34.79         | N/A            | N/A             | 28.95         |
|     16384 |       64 | float32  | N/A            | N/A             | 33.23         | N/A            | N/A             | 28.95         |
|     16384 |      128 | float32  | N/A            | N/A             | 132.00        | N/A            | N/A             | 30.05         |
|     32768 |       64 | float32  | N/A            | N/A             | 126.61        | N/A            | N/A             | 30.04         |
|     32768 |      128 | float32  | N/A            | N/A             | 513.56        | N/A            | N/A             | 32.22         |
|     65536 |       64 | float32  | N/A            | N/A             | 496.87        | N/A            | N/A             | 32.24         |
|     65536 |      128 | float32  | N/A            | N/A             | N/A           | N/A            | N/A             | N/A           |


TMA (Tensor Memory Accelerator) is a hardware feature to accelerate blockwise asynchronous memory transfers, particularly in support of Tensor Cores and high-efficiency shared memory usage.
- https://research.colfax-intl.com/tutorial-hopper-tma
- https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

TMA allows:
- Asynchronous, hardware-accelerated copy of rectangular (tensor-shaped) tiles from global memory into shared memory.
- It replaces the need for custom load loops that read global memory tile-by-tile.
- It minimizes address calculation overhead, coalescing inefficiencies, and memory bottlenecks.

In Triton terms, this could replace a tl.load(...) pattern with a single instruction that copies an entire tile efficiently to shared memory, suitable for use in tl.dot(...) or matmul kernels.

The output from running `python llm/tools/09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128`

```
215.919 257794.850 ROOT
├─ 419.013 3280.062 cublas [M=8192, N=8192, K=1024]                                                                                                                                                 
│  └─ nan 3280.062 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 247.661 693.684 cublas [M=8192, N=8192, K=128]                                                                                                                                                   
│  └─ nan 693.684 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 341.896 1004.976 cublas [M=8192, N=8192, K=256]                                                                                                                                                  
│  └─ nan 1004.976 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 332.291 1551.037 cublas [M=8192, N=8192, K=384]                                                                                                                                                  
│  └─ nan 1551.037 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 351.456 1955.280 cublas [M=8192, N=8192, K=512]                                                                                                                                                  
│  └─ nan 1955.280 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 361.021 2379.347 cublas [M=8192, N=8192, K=640]                                                                                                                                                  
│  └─ nan 2379.347 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 369.327 2791.001 cublas [M=8192, N=8192, K=768]                                                                                                                                                  
│  └─ nan 2791.001 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 376.035 3198.084 cublas [M=8192, N=8192, K=896]                                                                                                                                                  
│  └─ nan 3198.084 sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f16_execute_kernel__5x_cublas
├─ 232.209 5918.773 matmul_kernel [M=8192, N=8192, K=1024]                                                                                                                                          
├─ 203.670 843.516 matmul_kernel [M=8192, N=8192, K=128]                                                                                                                                            
├─ 220.541 1557.978 matmul_kernel [M=8192, N=8192, K=256]                                                                                                                                           
├─ 224.497 2295.784 matmul_kernel [M=8192, N=8192, K=384]                                                                                                                                           
├─ 228.916 3001.952 matmul_kernel [M=8192, N=8192, K=512]                                                                                                                                           
├─ 229.983 3735.036 matmul_kernel [M=8192, N=8192, K=640]                                                                                                                                           
├─ 230.118 4479.399 matmul_kernel [M=8192, N=8192, K=768]                                                                                                                                           
├─ 231.251 5200.370 matmul_kernel [M=8192, N=8192, K=896]                                                                                                                                           
├─ 207.247 6631.662 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=1024]                                                                                                                    
├─ 174.468 984.703 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=128]                                                                                                                      
├─ 183.898 1868.415 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=256]                                                                                                                     
├─ 198.820 2592.279 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=384]                                                                                                                     
├─ 202.012 3401.759 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=512]                                                                                                                     
├─ 203.839 4214.087 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=640]                                                                                                                     
├─ 205.520 5015.540 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=768]                                                                                                                     
├─ 206.514 5823.296 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=896]                                                                                                                     
├─ 202.212 6796.786 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=1024]                                                                                                                 
├─ 166.063 1034.540 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=128]                                                                                                                  
├─ 186.589 1841.470 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=256]                                                                                                                  
├─ 193.280 2666.573 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=384]                                                                                                                  
├─ 196.729 3493.098 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=512]                                                                                                                  
├─ 198.757 4321.833 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=640]                                                                                                                  
├─ 200.328 5145.522 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=768]                                                                                                                  
├─ 200.644 5993.647 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=896] 
├─ 218.517 6289.614 matmul_kernel_persistent [M=8192, N=8192, K=1024]                                                                                                                               
├─ 179.225 958.564 matmul_kernel_persistent [M=8192, N=8192, K=128]                                                                                                                                 
├─ 202.975 1692.807 matmul_kernel_persistent [M=8192, N=8192, K=256]                                                                                                                                
├─ 208.800 2468.377 matmul_kernel_persistent [M=8192, N=8192, K=384]                                                                                                                                
├─ 213.013 3226.074 matmul_kernel_persistent [M=8192, N=8192, K=512]                              
├─ 214.985 3995.590 matmul_kernel_persistent [M=8192, N=8192, K=640]                              
├─ 216.442 4762.443 matmul_kernel_persistent [M=8192, N=8192, K=768]                              
├─ 217.662 5525.030 matmul_kernel_persistent [M=8192, N=8192, K=896]                              
├─ 227.562 6039.631 matmul_kernel_tma [M=8192, N=8192, K=1024]                                    
├─ 199.443 861.392 matmul_kernel_tma [M=8192, N=8192, K=128]                                      
├─ 203.310 1690.016 matmul_kernel_tma [M=8192, N=8192, K=256]                                     
├─ 221.793 2323.768 matmul_kernel_tma [M=8192, N=8192, K=384]                                     
├─ 222.727 3085.370 matmul_kernel_tma [M=8192, N=8192, K=512]                                     
├─ 225.611 3807.410 matmul_kernel_tma [M=8192, N=8192, K=640]                                     
├─ 225.973 4561.573 matmul_kernel_tma [M=8192, N=8192, K=768]                                     
├─ 227.212 5292.818 matmul_kernel_tma [M=8192, N=8192, K=896]                                     
├─ 207.528 6622.674 matmul_kernel_tma_persistent [M=8192, N=8192, K=1024]                         
├─ 167.850 1023.526 matmul_kernel_tma_persistent [M=8192, N=8192, K=128]                          
├─ 190.237 1806.156 matmul_kernel_tma_persistent [M=8192, N=8192, K=256]                          
├─ 199.451 2584.071 matmul_kernel_tma_persistent [M=8192, N=8192, K=384]                          
├─ 202.596 3391.954 matmul_kernel_tma_persistent [M=8192, N=8192, K=512]                          
├─ 204.326 4204.024 matmul_kernel_tma_persistent [M=8192, N=8192, K=640]                          
├─ 205.789 5008.986 matmul_kernel_tma_persistent [M=8192, N=8192, K=768]                          
├─ 206.662 5819.117 matmul_kernel_tma_persistent [M=8192, N=8192, K=896]                          
├─ 202.506 6786.894 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=1024]                                                                                                                        
├─ 172.481 996.043 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=128]                        
├─ 178.462 1925.328 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=256]                                                                                                                         
├─ 191.341 2693.602 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=384]                                                                                                                         
├─ 197.525 3479.027 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=512]                                                                                                                         
├─ 199.419 4307.491 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=640]                                                                                                                         
├─ 200.859 5131.922 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=768]                                                                                                                         
├─ 201.821 5958.687 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=896]                                                                                                                         
├─ 194.799 7055.420 matmul_kernel_tma_ws [M=8192, N=8192, K=1024]                                 
├─ 123.441 1391.747 matmul_kernel_tma_ws [M=8192, N=8192, K=128]                                  
├─ 152.096 2259.077 matmul_kernel_tma_ws [M=8192, N=8192, K=256]                                  
├─ 172.785 2982.877 matmul_kernel_tma_ws [M=8192, N=8192, K=384]                                  
├─ 181.091 3794.752 matmul_kernel_tma_ws [M=8192, N=8192, K=512]                                  
├─ 186.284 4611.216 matmul_kernel_tma_ws [M=8192, N=8192, K=640]                                  
├─ 189.960 5426.361 matmul_kernel_tma_ws [M=8192, N=8192, K=768]                                  
└─ 192.663 6241.934 matmul_kernel_tma_ws [M=8192, N=8192, K=896] 
```
