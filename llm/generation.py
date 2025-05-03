from contextlib import contextmanager
import torch
from llm.transformer import TransformerLM
from llm.tokenization import Tokenizer


@contextmanager
def temporary_eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def generateLLM(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    eos_token: str = "<|endoftext|>",
    seed: int | None = None,
    device: str | None = None,
) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_token_ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.int64, device=device)
    eos_token_id = tokenizer.encode(eos_token)

    with temporary_eval_mode(model), torch.inference_mode():
        generated_token_ids = model.generate(
            prompt_token_ids_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seed=seed,
        )

    generated_text = tokenizer.decode(generated_token_ids.cpu().tolist())
    return generated_text
