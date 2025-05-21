# Inference
"It's all about the memory for speed"

 **arithmetic intensity:** is how much compute we do per byte transferred (want to be high)
 
 ```
 intensity = flops / bytes_transferred`
 accelerator_intensity = flops_per_second / memory_bandwidth
 ```
- If computation intensity > accelerator intensity, compute-limited (good)
- If computation intensity < accelerator intensity, memory-limited (bad)

The batch size for LLM generation is 1, so arithmetic intensity is 1.
- Memory-limited (read D x F matrix, perform only 2DF FLOPs)


#### Naive Sampling
- Naive inference: to generate each token, feed history into Transformer
- Complexity: generating T tokens requires $O(T^3)$ FLOPs (one feedforward pass is $O(T^2)$)
- Observation: a lot of the work can be shared across prefixes
![image](https://github.com/user-attachments/assets/1a21b1cc-fe51-419f-af4a-19fba95d29fd)

https://jax-ml.github.io/scaling-book/inference/
## KV cache
- KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector
- Fill up the KV cache with either tokens you've prefilled or generated so far

![image](https://github.com/user-attachments/assets/964c0da2-8bde-4a8f-95bd-f75bb513acaf)

Two stages of inference:
1. Prefill: given a prompt, encode into vectors (parallelizable like in training)
2. Generation: generate new response tokens (sequential)

For MLP layer
1. Prefill: easy to make compute-limited (good) by making B T large enough
2. Generation:
- Generating one token at a time (T=1)
- B is number of concurrent requests, hard to make large enough!

For Attention layer
1. Prefill: T=S
2. Generation: T=1 

Unlike MLPs, attention layer has no dependence on B, so batching doesn't help!
- In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)
- In attention layers, every sequence has its own vectors KV cache (Q, K, V all depend on B)

Summary
- Prefill is compute-limited, generation is memory-limited
- MLP intensity is B (requires concurrent requests), attention intensity is 1 (impossible to improve)

The memory, throughput, and latency depends on the shape of the Transformer.   
- parameter_size = num_params * 2  # 2 for bf16
- kv_cache_size = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

Total memory usage:

`memory = B * kv_cache_size + parameter_size`

Latency is determined by memory IO (read all parameters and KV cache for each step)

`latency = memory / memory_bandwidth`
 
 Throughput is the inverse of latency, but we're generating B tokens in parallel
 
 `throughput = B / latency`

**Tradeoff between latency and throughput:**
- Smaller batch sizes yields better latency but worse throughput
- Larger batch sizes yields better throughput but worse latency

**Easy parallelism:** if you launch M copies of the model, latency is the same, throughput increases by M!

**Harder parallelism:** shard the model and the KV cache

Note: time-to-first-token (TTFT) is essentially a function of prefill. Use smaller batch sizes during prefill for faster TTFT. Use larger batch sizes during generation to improve throughput

# Architecture Optimizations to speed up inference
- memory is the bottleneck for inference
- goal: reduce the size of the KV cache, while not losing accuracy

### Grouped-query Attention (GQA)
- Idea: N query heads, but only K key and value heads, each interacting with N/K query heads
- Multi-headed attention (MHA): K=N
- Multi-query attention (MQA): K=1
- Group-query attention (GQA): K is somewhere in between

https://arxiv.org/pdf/2305.13245
    
<img width="600" src="https://github.com/user-attachments/assets/68bc9454-8e68-44f7-b6c3-4b45d05f791b" />
<img width="400" src="https://github.com/user-attachments/assets/c1fe54e5-539b-471e-8f6c-f301c264c278" />

Latency/throughput improvement:
- Much faster up to # groups = 8
- by reducing the number of K/V pairs, the memory of K/V cache reduces -> throughput latency will improve
- increasing the batch size will also increase throughput

### Multi-head latent attention (MLA)
- Idea: project down each key and value vector from N*H dimensions to C dimensions
  - DeepSeek v2: reduce N*H = 16384 to C = 512
  - Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions

https://arxiv.org/abs/2405.04434

![image](https://github.com/user-attachments/assets/3b4139fa-626c-46e9-bef0-a5d82e2de1ac)
Latency/throughput improvement due to reduced KV cache size

### Cross-layer attention (CLA)
- Idea: share KVs across layers (just as GQA shares KVs across heads)
- Empirically improves the pareto frontier of accuracy and KV cache size (latency and throughput)
    
https://arxiv.org/abs/2405.12981

<img width="400" src="https://github.com/user-attachments/assets/291ac41e-d913-439a-abe9-9697ffbe4111" />

### Local Attention
- Idea: just look at the local context, which is most relevant for modeling
- Effective context scales linearly with the number of layers
- KV cache is independent of sequence length!


    
[Longformer](https://arxiv.org/pdf/2004.05150)
[OpenAI](https://arxiv.org/pdf/1904.10509)
[Mistral 7B](https://arxiv.org/pdf/2310.06825)

![image](https://github.com/user-attachments/assets/9c4da714-2f4e-4c2f-a4d4-3ec0ec70563b)

- Problem: this can still hurt accuracy
- Solution: interleave local attention with global attention (hybrid layers)
  - Example: character.ai uses 1 global layer every 6 layers (in addition to CLA to share KV cache across layers)

<img width="400" src="https://github.com/user-attachments/assets/fa065130-cf41-4b99-980b-78770a0b0a30" />


**Summary:**
- Goal: reduce the KV cache size (since inference is memory-limited) without hurting accuracy
- Lower-dimensional KV cache (GQA, MLA, shared KV cache)
- Local attention on some of the layers

# Alternatives to the Transformer
Attention + autoregression is fundamentally memory-limited (Transformers were not designed with inference in mind).

### State-space models
- Idea: from signal processing to model long-context sequences in a sub-quadratic time

**S4:** based on classic state space models, good at synthetic long-context tasks
  - https://arxiv.org/abs/2111.00396
  - https://docs.google.com/presentation/d/1wrQO4uzwWr73SGj7aFxeVR9Cz0PY-mzJipn12enM39k/edit#slide=id.p

<img width="800" src="https://github.com/user-attachments/assets/7d5ca420-825b-4969-994a-d5872f9bf2df" />

Weaknesses:
- bad at solving associative recall tasks important for language (where Transformers do well)
- good for signal processing tasks, but bad at isolating single key/value pair to get the answer


<img width="400" src="https://github.com/user-attachments/assets/c27df063-aa2a-4676-8060-c44ceb33b82b" />

**Mamba**: allow SSM parameters to be input-dependent, match Transformers at 1B scale
- https://arxiv.org/abs/2312.00752
 
**Jamba:** interleave Transformer-Mamba layers (1:7 ratio) with a 52B MoE
- speedup by only using Tranformer every 8 layers
- https://arxiv.org/abs/2403.19887

<img width="500" src="https://github.com/user-attachments/assets/0c9a7fa6-ea00-4577-82e3-e09574d50e47" />

**BASED:** use linear attention + local attention
- https://arxiv.org/abs/2402.18668

<img width="500" src="https://github.com/user-attachments/assets/e04feba0-2185-4b9e-9325-9fe4fcc78a71" />

**MiniMax-01:** use linear attention + full attention (456B parameter MoE)
- https://arxiv.org/pdf/2501.08313


**Takeaways:**
- Linear + local attention (still need some full attention) yield serious SOTA models
- Replace O(T) KV cache with O(1) state => much more efficient for inference

### Diffusion Models
- Popular for image generation, but harder to get working for text generation
- https://arxiv.org/abs/2205.14217

<img width="600" src="https://github.com/user-attachments/assets/bf41799d-f466-4d78-9446-e6fa73d3f4f6" />
   
- Idea: generate each token in parallel (not autoregressively), refine multiple time steps
- Start with random noise (over entire sequence), iteratively refine it

<img width="600" src="https://github.com/user-attachments/assets/9c917a3e-6cf2-4ebb-be56-0623adccdd82" />

- significant gains in inference

# Quantization
- Key idea: reduce the precision of numbers
- Less memory means lower latency/ higher throughput (since inference is memory-limited).
- **Quantization-aware training (QAT):** train with quantization, but doesn't scale up
- **Post-training quantization (PTQ):** run on sample data to determine scale and zero point for each layer or tensor

<img width="400" src="https://github.com/user-attachments/assets/9d9a4cb9-8d90-4399-99b0-973fa5feddf4" />

-  fp32 (4 bytes): needed for parameters and optimizer states during training
-  bf16 (2 bytes): default for inference
-  fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare
-  int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for inference only
- int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate

### LLM.int8()
- Standard quantization (scale by max of absolute values)
- https://huggingface.co/blog/hf-bitsandbytes-integration
- The motivation is storage (fiting big model into memory). Inference speed is 15-23% slower than fp16

<img width="600" src="https://github.com/user-attachments/assets/3bcb4470-8385-421d-835c-d15fdd53579e" />

- Problem: outliers (which appear in larger networks) screw everything up
- Solution: extract outliers and process them in fp16

### Activation-aware Quantization
- https://arxiv.org/abs/2306.00978
- Idea: select which weights (0.1-1%) to keep in high precision based on activations
- fp16 -> int3 produces 4x lower memory, 3.2x speedup

<img width="800" src="https://github.com/user-attachments/assets/9eaafbbc-951f-4917-a94c-8af3bb260296" />

### Model Pruning
- Key idea: just rip out parts of an expensive model to make it cheaper
- The reduced model is worse than the original, but the accuracy can be restored by distillation from the original model.
- https://arxiv.org/abs/2407.14679

** Algorithm:**
1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples)
2. Remove unimportant layers to get a smaller model
3. Distill the original model into pruned model

<img width="800" src="https://github.com/user-attachments/assets/1e9c1860-1350-4357-8db0-413a20e0a25f" />

### Speculative Sampling (decoding)
two stages of KV cache inference
- Prefill: given a sequence, encode tokens in parallel (compute-limited) [note: also gives you probabilities]
- Generation: generate one token at a time (memory-limited)

Idea: checking is faster than generation
- Use a cheaper draft model p to guess a few tokens (e.g., 4)
- Evaluate with target model q (process tokens in parallel), and accept if it looks good\

[speculative sampling video](https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4)

<img width="600" src="https://github.com/user-attachments/assets/530a9515-0604-45f4-bee3-be550b83690a" />

- This is modified rejection sampling with proposal p and target q
- Modification: always generate at least one candidate (rejection sampling will keep looping)
- Key property: guaranteed to be an exact sample from the target model!

In practice:
- Target model has 70B parameters, draft model has 8B parameters
- Target model has 8B parameters, draft model has 1B parameters
- Try to make draft model as close to target (distillation)

**Medusa:** draft model generates multiple tokens in parallel
- https://arxiv.org/abs/2401.10774
    
**EAGLE:** draft model takes high-level features from target model
- https://arxiv.org/pdf/2401.15077

<img width="600" src="https://github.com/user-attachments/assets/df919fca-85e4-4509-bc07-7de23c9443dd" />


**Summary:**
- Exact sampling from target model (thanks to math)!
- Exploits asymmetry between checking and generation
- Lots of room for innovation on the draft model (involves training)

# Handling Dynamic workloads
Batching over sequences in live traffic is tricky because:
- Requests arrive at different times (waiting for batch is bad for early requests)
- Sequences have shared prefixes (e.g., system prompts, generating multiple samples)
- Sequences have different lengths (padding is inefficient)

### Continuous Batching
- [Orca: distributed server system for transformer based generative models](https://www.usenix.org/system/files/osdi22-yu.pdf)
- https://www.youtube.com/watch?v=Ob9PPLxETYU

<img width="600" src="https://github.com/user-attachments/assets/bb3bcd56-23c1-4923-8fd7-483a3c5a123e" />

**Problem:**
- Training: get a dense block of tokens (batch size x sequence length)
- Inference: requests arrive and finish at different times, so you have a ragged array

**Solution: iteration-level scheduling**
- Decode step by step
- Add new requests to the batch as they arrive (so don't have to wait until generation completes)

**Problem:**
- Batching only works when all sequences have the same dimensionality (right?)
- But each request might have a different length

**Solution: selective batching**
- Training: when all sequences of the same length, operate on a B x S x H tensor
- But we might have different lengths: [3, H], [9, H], [5, H], etc.
- Attention computation: process each sequence separately
- Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H]

### Paged Attention
- Paper that introduced vLLM in addition to PagedAttention https://arxiv.org/pdf/2309.06180

**Previous status quo:**
- Request comes in
- Allocate section of KV cache for prompt and response (up to a max length)

![image](https://github.com/user-attachments/assets/9ea9242a-d317-4437-b55d-bb2289969266)

**Problem: fragmentation** (what happens to your hard drive)
- But this is wasteful since we might generate much fewer tokens (internal fragmentation)!
- Might be extra unused space between sections (external fragmentation)!

**Solution: PagedAttention** (remember operating systems)
- Divide the KV cache of a sequence into non-contiguous blocks

<img width="600" src="https://github.com/user-attachments/assets/e050bf95-5808-453b-97f2-bacab4c65e79" />

Two requests share the KV caches:

<img width="600" src="https://github.com/user-attachments/assets/097deb70-b634-4ed3-a04a-6f693784ea52" />

In general, multiples types of sharing KV caches across sequences:

<img width="600" src="https://github.com/user-attachments/assets/635c73fd-bc62-4b1d-bf6e-18c9ee6aedfe" />

- Sharing the system prompt
- Sampling multiple responses per prompt (e.g., for program synthesis)

**Solution:** share prefixes, copy-on-write at the block level


<img width="600" src="https://github.com/user-attachments/assets/937d3735-080e-4f8f-b58d-cb552ddca873" />

Other vLLM optimizations:
- Kernel to fuse block read and attention (reduce kernel launch overhead)
- Use latest kernels (FlashAttention, FlashDecoding)
- Use CUDA graphs to avoid kernel launch overhead


**Summary:** use ideas from operating systems (paging) to make use of memory for dynamic workloads

    
## Summary
- Inference is important (actual use, evaluation, reinforcement learning)
- Different characteristics compared to training (memory-limited, dynamic)
- Techniques: new architectures, quantization, pruning/distillation, speculative decoding
- Ideas from systems (speculative execution, paging)
- New architectures have huge potential for improvement

**Taking shortcuts (lossy)**
- reduce kv cache size
- alternatives to the transformer
- quantization
- pruning

Summary: reduce inference complexity without hurting accuracy

**From scratch recipe:**
1. Define faster model architecture
2. Train faster model

**Distillation recipe:**
1. Define faster model architecture
2. Initialize weights using original model (which has a different architecture)
3. Repair faster model (distillation)

**Taking shortcuts but double check (lossless)**
- speculative sampling


## References:
- https://jax-ml.github.io/scaling-book/
- https://stanford-cs336.github.io/spring2025/
- 
