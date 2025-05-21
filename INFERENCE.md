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


## References:
- https://jax-ml.github.io/scaling-book/
- https://stanford-cs336.github.io/spring2025/
- 
