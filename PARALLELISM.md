# Parallelism
## Part 1: Networking Basics for LLMs

**Limits of GPU-based Scaling**
- **Compute**: Even the fastest supercomputers must parallelize workloads.
- **Memory**: Single GPUs can't hold massive models.

**Solution: Multi-GPU/Multi-node Parallelism**
- Intra-node: High-speed interconnects within a machine. NVLink connects GPUs directly, bypass CPU
- Inter-node: High-speed networking across machines. NVSwitch connects GPUs directly, bypass Ethernet


<img width="400" src="https://github.com/user-attachments/assets/c75cf7b7-d6e6-4fde-bb2e-feb0447ed918" />

**Generalized hierarchy (from small/fast to big/slow):**
- Single node, single GPU: L1 cache / shared memory
- Single node, single GPU: HBM
- Single node, multi-GPU: NVLink
- Multi-node, multi-GPU: NVSwitch

**Collective Communication Primitives**
- Reduce: performs some associative/commutative operation (sum, min, max) 
- Broadcast/scatter is inverse of gather
- All: means destination is all devices

**Broadcast**

<img width="400" src="https://github.com/user-attachments/assets/d089dd46-2005-4d38-8e94-2134c09ed7e5" />

**Scatter**
- tensor is split up and sent to different GPUs

<img width="400" src="https://github.com/user-attachments/assets/440fd41f-cb0d-41c6-a749-32af5c2e144f" />

**Gather**

<img width="400" src="https://github.com/user-attachments/assets/efdcc2f2-6b84-4e93-ab27-5dbc37a700a7" />

**Reduce**

<img width="400" src="https://github.com/user-attachments/assets/71741252-ac01-4ba7-a2be-19f72e9a8e41" />

**All-gather**

<img width="400" src="https://github.com/user-attachments/assets/ef2ce99b-8f3c-47fa-a4b1-c14b2e0c4127" />

**Reduce-scatter**

<img width="400" src="https://github.com/user-attachments/assets/92c01411-957b-4fe8-a8a7-31e6c1a40abf" />

**All-reduce = reduce-scatter + all-gather**

<img width="400" src="https://github.com/user-attachments/assets/c459fb80-c643-4a8b-947a-fe70cd11e925" />

### NVIDIA Collective Communication Library (NCCL)
- NCCL translates collective operations into low-level packets that are sent between GPUs
- Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)
- Optimizes the path between GPUs
- Launches CUDA kernels to send/receive data

### PyTorch distributed library (torch.distributed)
- Provides clean interface for collective operations (e.g., all_gather_into_tensor)
- Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)
- Also supports higher-level algorithms (e.g., FullyShardedDataParallel) 
## Part 2: Parallel LLM Training Forms

### Data parallelism
Sharding strategy: each rank gets a slice of the data

<img width="200" src="https://github.com/user-attachments/assets/867723b8-5b0e-4a47-87fa-e4efe03e34e2" />

**Steps**
- batch of data is split across GPUs
- each GPU comutes gradients for their own batch
- `dist.all_reduce(op=AVG)` to average the gradients per param across GPUs, then `optimizer.step()`

**Summary**
- Compute scaling – each GPU gets B/M examples.
- Communication overhead – transmits 2x # params every batch. OK if batches are big
- Memory scaling – none. Every GPU needs # params at least

**Issues**:
- High memory overhead; 5x model size in memory (5 copies of weights and 16 bytes per param)

**ZeRO Optimizations**:
- **Stage 1**: Shard optimizer state
- **Stage 2**: Shard gradients too
- **Stage 3 (FSDP)**: Shard everything (params, grads, states)

#### ZeRO: Solve memory overhead issue of DP
Split up the expensive parts (state) and use the reduce-scatter equivalent

<img width="500"  src="https://github.com/user-attachments/assets/a1dce9a3-5aa2-4d47-a4bb-a848da5ce6bf" />

**Stage 1:**
- Shard optimizer state (first + second moments) across GPUs
- all GPUs have parameters + gradients
- Each worker is responsible for updating a subset of params (corresponding to their slice)
1. All GPUs compute full gradient on their subset of the batch
2. ReduceScatter the gradients - incur #params communication cost
3. Each machine updates their params using their gradient + state
4. All Gather the parameters - incur #params communication cost

**Stage 2:**
- Shard gradients too
- we can never instantiate a full gradient vector, but each worker must compute a full gradient (since we’re data parallel)
1. Everyone incrementally goes backward on the computation graph
-  After computing a layer’s gradients, immediately reduce to send this to the
right worker
- Once gradients are not needed in the backward graph, immediately free it.
2. Each machine updates their param using their gradient + state.
3. All Gather the parameters.

**Stage 3: Fully-Sharded Data Parallelism (FSDP)**
- Shard everything (params, grads, states)
- Each GPU device needs to gather the weight shards from all other GPUs before we can perform our forward or backward pass.
- Send and request parameters on demand while stepping through the compute graph.

Incremental computation / communication
- Parameters / gradients are requested / sent and then immediately freed

Overlapping communication and computation
- The all-gathers happen all at once while forward happens, masking the comm cost.
  
<img width="800" src="https://github.com/user-attachments/assets/665a3771-6db2-4d3a-8b71-9f5fc20a384d" />

#### ZeRO Trade-offs
| Stage       | Comm Cost       | Memory Usage                |
|-------------|------------------|-----------------------------|
| Naive DDP   | 2 * #params       | High                        |
| ZeRO Stage 1| 2 * #params      | Reduced (sharded state)     |
| ZeRO Stage 2| 2 * #params      | Further reduced             |
| ZeRO Stage 3| 3 * #params       | Lowest (fully sharded)      |

It doesn't require any knowledge of the model architecture!

#### ZeRO in Practice: Will It Fit?

It's possible to train larger LLM models when using higher degrees of ZeRO data parallelism. 

The following results were performed using bfloat16 with 8× A100 80GB GPUs

| Strategy       | Max Size (Params) | Formula for Bytes per Param |
|----------------|-------------------|------------------------------|
| **Baseline**   | 6.66B             | 12                           |
| **ZeRO Stage 1** | 16B              | 5                            |
| **ZeRO Stage 2** | 24.62B           | 2 (param) + (10 grad+state)/8 |
| **ZeRO Stage 3** | 53.33B           | 12 / 8                       |


- https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

## Part 3: Scaling Models Further

There's diminishing returns to training on larger batch size.
So you effectively have a fixed maximum batch size and you can spend it in different ways.

### Model Parallelism
- Scaling up in memory (without changing batch size) with model parallelism. Enables training with larger models by reducing the activation memory.
- Split up model parameters across GPUs (like ZeRO3)
- But communicate activations (while ZeRO3 sends params).

### 1. Pipeline Parallelism (PP)
Sharding strategy: each rank gets subset of layers, transfer all data/activations
- Layer-wise: The model is split layerwise into multiple stages, where each stage is run on a different device. This will result in poor GPU utilization as each GPU waits for gradients of previous layer. 

<img width="200" src="https://github.com/user-attachments/assets/134e6b9f-0b16-4cab-838d-06ccda3517e2" />

**Steps:**
- Split layers across GPUs
- Split data batch into **micro-batches** to reduce idle time. Requires a sufficiently large overall batch size to hide the "bubble"
- Each GPU will wait for previous rank to pass it the activations, then it will run `.forward()` on it's subset of layers, and `dist.send(tensor=x, dst=rank+1)` the activations to the next GPU
- Each GPU will start computing on the next microbatch

**Benefits:**
- Good for memory savings (compared to data parallel), especially across nodes
- Pipeline has good communication properties (compared to FSDP) - it only depends only on activations (𝑏 × 𝑠 × ℎ) and is point to point
- Pipelines should be used on slower network links (i.e. inter-node) as a way to get better memory-wise scaling.

<img width="400"  src="https://github.com/user-attachments/assets/a66209e2-3b3a-4c3a-92ef-f1de815530e6" />

**Zero bubble pipelining**
- can be very complicated

<img width="800" src="https://github.com/user-attachments/assets/01a92fb4-427c-49b4-bdeb-6ac5e7767108" />

### 2. Tensor Parallelism (TP)
Sharding strategy: each rank gets part of each layer, transfer all data/activations. Cut the model along the hidden dimension.
Each GPU gets every layer, but only a slice of the hidden dimension of each layer.

<img width="200" src="https://github.com/user-attachments/assets/91071728-4cae-4a10-95ca-a8a8a2177336" />

**Steps**
- Split model along hidden dim
- Each GPU has a slice of the activations
- `dist.all_gather` to the activations so every GPU has activations of the whole model

<img width="289" src="https://github.com/user-attachments/assets/dba035be-43b8-4cae-ac57-16b0b1c5ac0d" />

- Can be used for any matrix multiply by breaking into submatrices
- Split matrices across GPUs (columns/rows). Split matrices along width (hidden) dimension.
- All-reduce required in forward/backward passes!
- Ideal within a node (fast interconnects), e.g. the 8 GPUs on a single node.
- Tensor Parallelism can be used effectively together with FSDP if we shard the weights and the activations along corresponding dimensions.

#### Tensor vs Pipeline Parallelism

✅ Pros of Tensor Parallelism
- **No idle time**: No bubbles if network is fast — no waiting for other devices.
- **Low complexity**: Easy to implement without major infrastructure changes.
- **Works with small batches**: Doesn't require large batch sizes.

❌ Cons of Tensor Parallelism
- **High communication overhead**, especially compared to pipeline parallelism.
  - **Pipeline**: `bsh` (batch × sequence × hidden) point-to-point communication per microbatch.
  - **Tensor**: `8 × bsh × ((n_devices - 1) / n_devices)` per layer with all-reduce operations.

> Use tensor parallelism when you have low-latency, high-bandwidth interconnects.


### Memory Challenges
- **Activation memory** is dynamic and often exceeds parameter memory
- **Recomputation** and **sequence parallelism** can reduce memory use

### Sequence Parallelism (Activation)

<img width="600" src="https://github.com/user-attachments/assets/dc207e3a-9d7b-4046-b7ca-4f8513c23e78" />

- Split activations across the **sequence** axis
- Enables linear memory scaling for activations

Activation Memory Per Transformer Layer

| Configuration                                                                 | Activation Memory Expression                        |
|------------------------------------------------------------------------------|------------------------------------------------------|
| No parallelism                                                               | *sbh* (34 + 5 *a<sup>s</sup>*/h)                     |
| Tensor parallel (baseline)                                                   | *sbh* (10 + 24/t + 5 *a<sup>s</sup>*/(h·t))          |
| Tensor + Sequence parallel                                                   | *sbh* (34/t + 5 *a<sup>s</sup>*/(h·t))               |
| Tensor parallel + Selective activation recomputation                         | *sbh* (10 + 24/t)                                   |
| Tensor + Sequence parallel + Selective activation recomputation              | *sbh* (34/t)                                        |

### Context Parallelism / Ring Attention
- Split attention heads or contexts across GPUs
- Each machine is responsible for a different query, and then keys/values travel between machines in a ring-like fashion.

#### Expert Parallelism (EP)
- Split model "experts" across GPUs (used in MoE models)
- each device computes the output results for their own expert
- the networking is more complicated since one expert could be overloaded

## LLM Parallelism Comparison Table

| Strategy               | Sync Overhead              | Memory       | Bandwidth                                 | Batch Size | Easy to Use? |
|------------------------|----------------------------|--------------|-------------------------------------------|------------|--------------|
| DDP / ZeRO1            | Per-batch                  | **No scaling** | 2 × #params                              | **Linear** | Very         |
| FSDP (ZeRO3)           | **3× Per-FSDP block**       | Linear       | 3 × #params                               | **Linear** | Very         |
| Pipeline               | Per-pipeline               | Linear       | Activations                               | **Linear** | NO           |
| Tensor + Sequence      | **2× transformer block**    | Linear       | **8 × activations per-layer all-reduce** | No impact  | No           |

Have to balance limited resource – memory, bandwidth, batch size
- if batch size is too small, then there's no way to be efficient
- as batch size increases, you can mix FSDP and Tensor Parallel you can get to a point where you're compute bound
- if batch size is very big, then you can get away with just pure FSDP

<img width="500" src="https://github.com/user-attachments/assets/abde96b7-0ae5-468d-8c23-90264f37a092" />

## 3D Parallelism
- Combine:
  - **Tensor parallel (within node)**
  - **Pipeline parallel (across nodes)**
  - **Data parallel (across batches)**

#### Strategy:
1. Fit model into memory with tensor/pipeline parallelism
- Tensor parallel up to GPUs / machine
- Pipeline parallel across machines
- Or use ZeRO-3 depending on bandwidth
2. Scale training (across all GPUs) with data parallelism

- if your batch size is small, use gradient accumulation to trade higher batch sizes for better communication efficiency since you're synchronizing less often across machines.

For example,
- Tensor parallel first up to 8, then caps out at 8.
- Pipeline parallel goes up to make the model fit.
- data parallel gradually decreases with scale, with the largest model having DP=6

## Real-World Examples

- **Dolma 7B**: FSDP (ZeRO Stage 3)
- **DeepSeek**: ZeRO1 + Tensor + Pipeline + Sequence
- **Yi**: ZeRO1 + Tensor + Pipeline; Yi-Lightning uses Expert parallel
- **LLaMA3 405B**: Mixed strategies per training stage. https://arxiv.org/abs/2407.21783
- **Gemma 2 (2B, 9B, 27B)**: ZeRO-3, Model Parallel (TP+SP), Data Parallel

need fault tolerant architectures because the GPUs will fail during training at scale
 
## Final Recap

- **Scaling requires** multi-GPU/multi-node setups.
- **No single parallelism method suffices**—combine strategies.
- **Rules of thumb** help guide efficient parallelization design:
  - Tensor parallel up to 8 GPUs
  - Pipeline across machines
  - Data parallel for final scaling
- Activation memory is a key bottleneck—must be managed explicitly.


## 📚 Further Reading

- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO Paper](https://arxiv.org/pdf/1910.02054.pdf)
- [Scaling Laws Paper](https://arxiv.org/abs/2001.08361)

