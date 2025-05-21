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

#### Fully-Sharded Data Parallelism (FSDP)
Optimizer states, gradients, and weights are split
across devices. If we’re using only DP and FSDP, every device needs to gather the weight shards from
all other devices before we can perform our forward or backward pass.


#### ZeRO: Solve memory overhead issue of DP
Split up the expensive parts (state) and use the reduce-scatter equivalent

<img width="500"  src="https://github.com/user-attachments/assets/a1dce9a3-5aa2-4d47-a4bb-a848da5ce6bf" />

**Stage 1:**
- Shard optimizer state (first + second moments) across GPUs
- all GPUs have parameters + gradients
- Each worker is responsible for updating a subset of params (corresponding to their slice)
1. all GPUs compute full gradient on their subset of the batch
2. ReduceScatter the gradients - incur #params communication cost
3. Each machine updates their params using their gradient + state
4. All Gather the parameters - incur #params communication cost

**Stage 2:**
- Shard gradients too
- we can never instantiate a full gradient vector, but each worker must compute a full gradient (since we’re data parallel)
1. Everyone incrementally goes backward on the computation graph

#### ZeRO Trade-offs
| Stage       | Comm Cost       | Memory Usage                |
|-------------|------------------|-----------------------------|
| Naive DDP   | 2 * #params       | High                        |
| ZeRO Stage 1| 2 * #params       | Reduced (sharded state)     |
| ZeRO Stage 2| 2 * #params       | Further reduced             |
| ZeRO Stage 3| 3 * #params       | Lowest (fully sharded)      |


### Model Parallelism

#### Tensor Parallelism (TP)
Activations are sharded across a new dimension, and each device
computes the output results for their own shard. With Tensor Parallel we can either shard along
the inputs or the outputs the operation we are sharding. Tensor Parallelism can be used effectively
together with FSDP if we shard the weights and the activations along corresponding dimensions.
#### Pipeline Parallelism (PP)
The model is split layerwise into multiple stages, where each stage is
run on a different device.

### Activation Parallelism
- sequence parallel

#### Expert Parallelism (EP)
We separate experts (in Mixture-of-Experts models) onto different
devices, and each device computes the output results for their own expert.



---

## 🧩 Part 3: Scaling Models Further

### Model Parallelism
#### 1. Pipeline Parallelism
- Split layers across GPUs
- Use **micro-batches** to reduce idle time
- Good for memory savings, especially across nodes

#### 2. Tensor Parallelism
- Split matrices across GPUs (columns/rows)
- All-reduce required in forward/backward passes
- Ideal within a node (fast interconnects)

### Sequence Parallelism
- Split activations across the **sequence** axis
- Enables linear memory scaling for activations

### Memory Challenges
- **Activation memory** is dynamic and often exceeds parameter memory
- **Recomputation** and **sequence parallelism** can reduce memory use

---

## 🛠️ Advanced & Hybrid Strategies

### Expert Parallelism
- Split model "experts" across GPUs (used in MoE models)

### Context Parallelism / Ring Attention
- Split attention heads or contexts across GPUs

### 3D Parallelism
- Combine:
  - **Tensor parallel (within node)**
  - **Pipeline parallel (across nodes)**
  - **Data parallel (across batches)**

#### Strategy:
1. Fit model with tensor/pipeline parallelism
2. Scale training with data parallelism

---

## 🧪 Real-World Examples

- **Dolma 7B**: FSDP (ZeRO Stage 3)
- **DeepSeek**: ZeRO1 + Tensor + Pipeline + Sequence
- **Yi**: ZeRO1 + Tensor + Pipeline; Yi-Lightning uses Expert parallel
- **LLaMA3 405B**: Mixed strategies per training stage
- **Gemma 2 (2B, 9B, 27B)**: ZeRO-3, Model Parallel (TP+SP), Data Parallel

---

## 🧾 Final Recap

- **Scaling requires** multi-GPU/multi-node setups.
- **No single parallelism method suffices**—combine strategies.
- **Rules of thumb** help guide efficient parallelization design:
  - Tensor parallel up to 8 GPUs
  - Pipeline across machines
  - Data parallel for final scaling
- Activation memory is a key bottleneck—must be managed explicitly.

---

## 📚 Further Reading

- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO Paper](https://arxiv.org/pdf/1910.02054.pdf)
- [Scaling Laws Paper](https://arxiv.org/abs/2001.08361)

