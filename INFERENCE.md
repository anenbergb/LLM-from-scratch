# Inference

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



## References:
 - https://jax-ml.github.io/scaling-book/
