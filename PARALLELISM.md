# Parallelism
## Part 1: Networking Basics for LLMs

### Limits of GPU-based Scaling
- **Compute**: Even the fastest supercomputers must parallelize workloads.
- **Memory**: Single GPUs can't hold massive models.

### Solution: Multi-GPU/Multi-node Parallelism
- Intra-node: High-speed interconnects within a machine.
- Inter-node: High-speed networking across machines.

### Collective Communication Primitives
- **All Reduce**, **Broadcast**, **Reduce**, **All Gather**, **Reduce Scatter**
- `Reduce = Reduce-Scatter + All-Gather`

<img width="600" src="https://github.com/user-attachments/assets/5dec072e-52ab-470a-b1da-ad9a7f2b6faa" />

<img width="400" src="https://github.com/user-attachments/assets/41d08b56-3405-400f-afee-d8a8762acce2" />

## Part 2: Parallel LLM Training Forms

### Data parallelism

Batches of data are split across multiple devices, and each device computes
gradients for their own batch. These gradients must somehow be averaged across devices.

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

#### 1. Pipeline Parallelism (PP)

Layer-wise: The model is split layerwise into multiple stages, where each stage is run on a different device. This will result in poor GPU utilization as each GPU waits for gradients of previous layer. 

Pipeline parallel:
- Split layers across GPUs
- Use **micro-batches** to reduce idle time. Send off a microbatch, then start computing next microbatch. Requires a sufficiently large overall batch size to hide the "bubble"
- Good for memory savings (compared to data parallel), especially across nodes
- Pipeline has good communication properties (compared to FSDP) - it only depends only on activations (𝑏 × 𝑠 × ℎ) and is point to point

Pipelines should be used on slower network links (i.e. inter-node) as a way to get
better memory-wise scaling.

<img width="400"  src="https://github.com/user-attachments/assets/a66209e2-3b3a-4c3a-92ef-f1de815530e6" />

**Zero bubble pipelining**
- can be very complicated

<img width="800" src="https://github.com/user-attachments/assets/01a92fb4-427c-49b4-bdeb-6ac5e7767108" />

#### 2. Tensor Parallelism (TP)
Activations are sharded across a new dimension, and each device
computes the output results for their own shard. With Tensor Parallel we can either shard along
the inputs or the outputs the operation we are sharding. Tensor Parallelism can be used effectively
together with FSDP if we shard the weights and the activations along corresponding dimensions.

<img width="289" src="https://github.com/user-attachments/assets/dba035be-43b8-4cae-ac57-16b0b1c5ac0d" />

- Can be used for any matrix multiply by breaking into submatrices
- Split matrices across GPUs (columns/rows). Split matrices along width dimension.
- All-reduce required in forward/backward passes!
- Ideal within a node (fast interconnects), e.g. the 8 GPUs on a single node.

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

## 🧪 Real-World Examples

- **Dolma 7B**: FSDP (ZeRO Stage 3)
- **DeepSeek**: ZeRO1 + Tensor + Pipeline + Sequence
- **Yi**: ZeRO1 + Tensor + Pipeline; Yi-Lightning uses Expert parallel
- **LLaMA3 405B**: Mixed strategies per training stage
- **Gemma 2 (2B, 9B, 27B)**: ZeRO-3, Model Parallel (TP+SP), Data Parallel

## 🧾 Final Recap

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

