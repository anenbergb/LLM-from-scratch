# A Modern Large Language Model Implementation
This project implements a modern LLM from scratch.
The project was inspired by [assignment #1 of Stanford CS336](https://github.com/stanford-cs336/assignment1-basics).


### Installation
```
pip install -e .
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

# Experiments on TinyStories dataset
Given a fixed training schedule of 10k iterations, the max-lr = 1e-3 model outperforms the other models because it minimizes the loss the fastest. The max-lr = 1e-2 model diverges -- the grad_norm diverges, and the training loss explodes.

Increasing the batch size from 128 to 256 decreases the final val loss from 1.384 to 1.332. The training time increases from 11m to 30 min.
Extending the training schedule from 10k iterations to 50k iterations (with 256 batch size) further decreases the val loss to 1.224 with the increase in training time to 2h 26min.
Training with the lower max-lr options (2e-4 and 5e-4) resulted in higher val loss