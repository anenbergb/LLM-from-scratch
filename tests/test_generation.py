import pytest
import torch
from llm.transformer import TransformerLM
from llm.tokenization import Tokenizer
from llm.generation import generateLLM


PROMPT = "Once upon a time there was a little boy named Ben. Ben loved to"


def set_all_seeds(seed=42):
    import random
    import numpy as np

    # Python built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def test_generation():
    seed = 42
    set_all_seeds(seed)

    tokenized_dataset_pkl = "/media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_10k_tinystories.pkl"
    eos_token = "<|endoftext|>"
    tokenizer = Tokenizer.from_pickle(tokenized_dataset_pkl, special_tokens=[eos_token])
    vocab_size = len(tokenizer.vocab)
    device = "cuda:0"

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1334,
        rope_theta=10000,
        device=device,
    ).to(device)
    model.eval()

    generated_text = generateLLM(
        model,
        tokenizer,
        PROMPT,
        max_new_tokens=100,
        temperature=1.0,
        top_k=10,
        top_p=0,
        eos_token=eos_token,
        seed=seed,
    )
    print(f"\nPROMPT:\n{PROMPT}\nGENERATED:\n{PROMPT}{generated_text}")
