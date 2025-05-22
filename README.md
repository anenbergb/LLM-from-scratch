# A Modern Large Language Model Implementation

This repository features a from-scratch implementation of a modern large language model (LLM), developed while following the [Stanford CS336 course on LLMs (Spring 2025)](https://stanford-cs336.github.io/spring2025/). It includes most of the course assignments from the [official GitHub repo](https://github.com/stanford-cs336/), along with summarized lecture notes.

## Key highlights
- Byte Pair Encoding (BPE) tokenizer  
- Autoregressive causal Transformer with:
  - RMSNorm  
  - SwiGLU activation  
  - Rotary Positional Embeddings (RoPE)  
- Flash Attention 2 implementation in Triton with extensive GPU benchmarking


### Installation
```
pip install -e .
```
### Triton support
It's necessary to install `triton` from source in order to enable `triton` support on the latest NVIDIA Blackwell GPUs (e.g. RTX 5090) with compute capability 120.


When Triton builds its C++/CUDA backends, it links against the libstdc++.so.6 from the GCC that did the build — typically the system's /usr/lib/...

If you're working within a Conda environment, you should first install and use a Conda-managed GCC
```
conda install -c conda-forge gxx_linux-64 gcc_linux-64
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
```
and then follow the installation instructions https://github.com/triton-lang/triton
```
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```
And then re-install PyTorch nightly build to match with the latest Triton
```
pip install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
``` 


### Run unit tests

```sh
pytest tests
```

### Run Formatting
```
ruff format llm
```
### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

# Byte-level BPE Tokenizer

### BPE Tokenizer Training

This process trains a byte-level BPE tokenizer on the TinyStories or OWL corpus:

1. **Pre-tokenization (Parallelized):**  
   Text is split using a GPT-2-style regex (see [`pretokenization.py`](llm/pretokenization.py)) into "pre-tokens". This is roughly similar to splitting the text on whitespace. Token frequencies are counted. Special tokens like `<|endoftext|>` are excluded.

2. **Initialize Vocabulary:**  
   Start with 256 byte-level tokens (one per byte value). Add `<|endoftext|>` as token 257.

3. **Compute BPE Merges:**  
   Represent pre-tokens as byte sequences. Repeatedly merge the most frequent adjacent byte pairs to form new tokens until the target vocabulary size (e.g., 10,000) is reached.

**Output:**  
An ordered list of merge operations and the final vocabulary mapping.
```
python llm/tools/train_bpe.py \
--file-path owt_train.txt \
--vocab-size 32000 \
--num-processes 32 \
--save-path bpe_32k_owt_train.pkl
```
### Encode Corpus with BPE

Use the trained BPE tokenizer to convert the text corpus into token IDs. The resulting token ID array is saved as a NumPy file for use in LLM training.
```
python llm/tools/bpe_encode_document.py \
--chunk-size 10000 \
--tokenized-dataset-pickle bpe_32k_owt_train.pkl \
--file-path owt_train.txt  \
--save-path owt_train.npy
```

# Transformer Language Model
![image](https://github.com/user-attachments/assets/e2ec2635-30d5-4c73-b5f6-0719e38a557f)

We implement an auto-regressive transformer language model following modern best practices, including:

- Pre-Norm transformer blocks  
- RMS Layer Normalization  
- SwiGLU feed-forward layers (SiLU activation + Gated Linear Unit)  
- Relative Positional Embeddings (RoPE)  
- Causal multi-head self-attention

We also implement the following components from scratch:

- Cross-entropy loss  
- AdamW optimizer  
- Cosine annealing scheduler with warm-up  
- Gradient clipping

# Experiments training the LLM


### Experiments tuning the learning rate and batch size on the TinyStories dataset

The [TinyStories dataset](https://huggingface.co/datasets/TinyStories) consists of short stories, totaling **540.8M tokens** for training and **546k tokens** for testing after BPE tokenization with a 10k vocabulary.

A few short experiments were conducted to explore the effects of learning rate, batch size, and training duration on model performance:

- **Learning Rate**:
  - `max_lr = 1e-3` achieved the lowest training loss within a fixed schedule of 10k iterations.
  - `max_lr = 1e-2` caused training to diverge (exploding loss and gradient norms).
  - Lower learning rates (`2e-4`, `5e-4`) resulted in higher validation loss.

- **Batch Size**:
  - Increasing batch size from **128** to **256** reduced validation loss from **1.384** to **1.332**.
  - Training time increased from **11 minutes** to **30 minutes**.

- **Training Schedule**:
  - Extending training from **10k** to **50k** iterations (with batch size 256) further reduced validation loss to **1.224**.
  - Training time increased to **2h 26min**.

These quick experiments help establish baseline hyperparameters for further training and tuning.

## Experiments on OpenWebText dataset
The OpenWebText dataset is a publicly available alternative to OpenAI's proprietary WebText corpus. It is significantly larger than the TinyStories dataset, containing **2.72 trillion tokens** for training and **66 million tokens** for testing after BPE tokenization with a 32k vocabulary.

### Weight Tying Experiment
Performed experiment training the "tiny" LLM model but varying the weight initialization scheme.
- model 1 (orange): uses separate weights for the token embeddings and for the final fully connected layer
- model 2 (blue): ties the weights of the token embeddings and final fully connected layer, but uses the final fully connected layer weight initialization
- model 3 (red):  ties the weights of the token embeddings and final fully connected layer, but uses the token embedding weight initialization.

final fully connected layer weight initialization
```
variance = 2 / (in_features + out_features)
std = math.sqrt(variance)
nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
```
token embedding weight initialization
```
nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
```
Training loss and Gradient Norm

<img width="300" alt="image" src="https://github.com/user-attachments/assets/7d57cd89-e2ec-401a-9703-e4c5adeb4744" />
<img width="300" alt="image" src="https://github.com/user-attachments/assets/034d7101-2ec4-4f5a-82ed-b1ab373123a4" />

- Blue curve (weight tying + FC layer init) has the lowest final training loss (~3.75).
- Orange curve (no weight tying) finishes higher (~3.85).
- Red curve (weight tying + embedding init) performs worst (~4.0), plateauing early.

- Blue curve shows a healthy increase in gradient norm after ~20k steps — a sign that the model continues learning and doesn’t saturate.
- Red curve shows flat and suppressed gradients, indicating undertraining or poor signal flow — likely due to bad initialization.
- Orange curve shows moderate gradient growth but not as strong as the blue curve.

Conclusion: apply weight tying, but use the fully connected layer weight initialization

### Model Size Experiment

I trained four different model sizes for **100k iterations** to study the impact of scale on performance. The *tiny* and *small* models used a maximum learning rate of `1e-3`, while the *medium* and *large* models required lower learning rates (`3e-4` and `2e-4` respectively) to prevent training divergence. All models were trained with a context length of 256 tokens.

Due to GPU memory constraints (single RTX 5090 with 32 GB vRAM), the *medium* and *large* models were also trained with smaller batch sizes.

| Size   | d_model | d_ff | num_layers | num_heads | Learning Rate | Batch Size |
|--------|---------|------|------------|-----------|----------------|-------------|
| tiny   | 512     | 1344 | 4          | 16        | 1e-3           | 64          |
| small  | 768     | 3072 | 12         | 12        | 1e-3           | 64          |
| medium | 1024    | 4096 | 24         | 16        | 3e-4           | 32          |
| large  | 1280    | 5120 | 36         | 20        | 2e-4           | 8           |

- **Red** curve: Tiny model  
- **Blue** curve: Small model  
- **Green** curve: Medium model  
- **Gray** curve: Large model  

#### Results

As shown in the training and validation loss plots below, the **small model (blue curve)** achieved the lowest final validation loss of **3.355**. The *medium* and *large* models underperformed, likely due to their **lower learning rates** and **smaller batch sizes**, which limited training efficiency within the 100k iteration budget.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/cdf671a6-50e3-4738-93c5-30d775e9cc4f"/>
<img width="600" alt="image" src="https://github.com/user-attachments/assets/bf076847-e993-48ff-a613-ccbe9555fca2"/>

Here’s a rephrased version in Markdown:

### 500k Iteration Experiment

I trained the medium-sized model for **500k iterations** using a maximum learning rate of `5e-4`, resulting in a final validation loss of **3.202**. This is an improvement over the **3.421** validation loss achieved with **100k iterations** at a lower learning rate of `3e-4`.

## Generating Text
Given the trained LLM, we can generate text using the [generateLLM](llm/generation.py) function which accepts arguments such as
  - `max_new_tokens`: Maximum number of tokens to generate.
  - `temperature`: Controls randomness. Set to `0` for greedy decoding, `>1` for more diverse output.
  - `top_k`: Limits sampling to the top-k most probable tokens.
  - `top_p`: Enables nucleus sampling by keeping tokens within cumulative probability `top_p`.
  - `eos_token_id`: Optional token ID to stop generation early.
  - `seed`: Optional random seed for reproducibility.

The method handles input padding or truncation based on the model’s context length and iteratively generates one token at a time using the selected sampling strategy.

Below is a sample generation from the medium-sized model trained for 500k iterations on the OpenWebText dataset. While the output is somewhat coherent, it lacks the fluency and consistency desired. This is likely due to several factors: despite the extended training (500k iterations with a batch size of 32 and context length of 256), the model only completed a fraction of a full epoch over the dataset. Additionally, the model's capacity may have been insufficient for effectively modeling a dataset of this scale.

```
PROMPT:
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. One of the key algorithms in supervised learning is
GENERATED:
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. One of the key algorithms in supervised learning is the "prediction function", in order to get the best from the machine learning model for a given state, which is what we've got in the case of this article. For instance, the algorithm would be able to learn that if it is the case that there are no real-world data points that it's not the case, then it will be a good idea to run it through a few different tests, one of which will be a bit more complicated.

I am a big fan of the new algorithm. But the way it works has a lot to do with how you define and test it. The first step is to determine what the output of your algorithm is, the number of variables, the time it takes, etc. The best thing about this is that you can have your models run in realtime, which means it's much easier to test them than it is to test them in realtime.
```

For comparison, here is a sample generation from a tiny-sized model trained for just 50k iterations on the much smaller TinyStories dataset. Despite its significantly smaller size and shorter training time, the generated text appears noticeably more coherent, with a story that makes clearer sense.

```
PROMPT:
Once upon a time, there was a pretty girl named Lily. She loved to eat
GENERATED:
Once upon a time, there was a pretty girl named Lily. She loved to eat pizza while she played games and shared her toys with her friends. One day, she decided to make a special menu for her friends.
Lily called her friends Tom and Lily and said, "Come see the menu! We have pizza!" Tom and Lily came and saw the menu. They all wanted to eat cheese for their lunch. Lily's mom gave them some pizza, and they all sat down to eat.
After they were done eating, Lily said, "I will record the yummy food on my plate. I am sure my friends will love it." She held her plate and started to record. They all smiled and said, "Thank you for the yummy food!" Lily was happy to share her yummy food with her friends.
<|endoftext|>
```
