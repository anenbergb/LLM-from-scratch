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

