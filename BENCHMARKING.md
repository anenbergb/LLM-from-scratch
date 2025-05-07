# End-to-End Benchmarking of the Transformer LLM
All benchmarking is performed on an NVIDIA RTX 5090 GPU with 32 Gb of vRAM

Benchmarking the forward, backward, and optimizer update for the TransformerLM
- batch-size: 4
- context-length: 32
- vocabulary-size: 10,000
- 5 warmup steps, 10 measurement steps

|   d_model |   d_model |     d_ff |   num_layers |   num_heads |   forward_mean |   forward_std |   backward_mean |   backward_std |   optimizer_mean |   optimizer_std |
|----------:|----------:|---------:|-------------:|------------:|---------------:|--------------:|----------------:|---------------:|-----------------:|----------------:|
| tiny      |       512 |   1344 |            4 |          16 |         11.046 |         3.993 |          35.077 |         56.929 |            6.824 |           0.586 |
| small     |       768 |   3072 |           12 |          12 |         28.367 |         0.489 |          47.132 |         24.288 |           20.101 |           2.172 |
| medium    |       1024 |   4096 |           24 |          16 |         30.735 |          0.52 |          48.224 |         23.389 |           21.873 |           3.185 |
| large     |  1280 |   5120 |           36 |          20 |        120.652 |        73.313 |         154.943 |         50.076 |           83.401 |          35.291 |
| xl   |      1600 |   6400 |           48 |          25 |        182.555 |        69.583 |         312.771 |         76.066 |              inf |             nan |
| 2.7B |      2560 |  10240 |           32 |          32 |        176.731 |        61.485 |             inf |            nan |              inf |             nan |

- the backward pass consistently takes at least 1x longer than the forward pass
- the standard deviation for the measurements is fairly high