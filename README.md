# A Modern Large Language Model Implementation
This project implements a modern LLM from scratch.
The project was inspired by [assignment #1 of Stanford CS336](https://github.com/stanford-cs336/assignment1-basics).


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

I trained four different model sizes for **100k iterations** to study the impact of scale on performance. The *tiny* and *small* models used a maximum learning rate of `1e-3`, while the *medium* and *large* models required lower learning rates (`3e-4` and `2e-4` respectively) to prevent training divergence.

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
