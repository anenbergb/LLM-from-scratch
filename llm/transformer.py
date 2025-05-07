import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from loguru import logger

from llm.layers import Linear, Embedding, RMSNorm, SwiGLU, CausalMHSARoPE, RotaryPositionalEmbedding
from llm.nn_utils import softmax


class TransformerBlock(nn.Module):
    """
    d_model (int): The dimensionality of the Transformer block input.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_ff (int): Dimensionality of the feed-forward inner layer.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    theta (float): RoPE parameter.
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        RoPE: RotaryPositionalEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = CausalMHSARoPE(d_model, num_heads, RoPE=RoPE, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        """
        pre-norm Transformer block

        in_features: Input tensor of shape (batch_size, sequence_length, d_model).
        Returns:
            Output tensor of the same shape as in_features.
        """
        out1 = in_features + self.attn(self.ln1(in_features), token_positions=token_positions)
        out2 = out1 + self.ffn(self.ln2(out1))
        return out2


class TransformerLM(nn.Module):
    """
    Transformer model with RoPE.

    vocab_size (int): Size of the vocabulary.
    context_length (int): The maximum number of tokens to process at once.
    num_layers (int): The number of Transformer layers to use.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_model (int): The dimensionality of the model embeddings and sublayer outputs.
    d_ff (int): Dimensionality of the feed-forward inner layer
    rope_theta (float): The RoPE Theta parameter.
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        rope_theta: float,
        weight_sharing: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.RoPE = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, RoPE=self.RoPE, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # weight sharing / weight tying
        if weight_sharing:
            self.token_embeddings.weight = self.lm_head.weight

    def forward(
        self,
        in_indices: Int[Tensor, " batch sequence_length"],
    ) -> Float[Tensor, " batch sequence_length vocab_size"]:
        """
        in_indices: Input token ids (batch_size, sequence_length).
        Returns:
            Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        assert in_indices.shape[-1] <= self.context_length
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        in_indices: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        eos_token_id: int | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            in_indices: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
                Temperature > 1 will increase randomness of the distribution, and encourage
                    generation of less probable tokens
                Temperature in range [0, 1] will reduce the randomness, increasing the
                    probability of more likely tokens and avoiding predictions that might
                    be too unexpected.
                Temperature = 0 will be greedy decoding.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
                The probabilities are re-normalized before sampling the next token.
            top_p: float
                Top-p sampling (nucleus sampling)
                Rather than sampling from the K words with the highest probability,
                use all the most likely words whose cumulative probability exceeds `top_p` value.
                If we use top_p=0.95, we will first filter out to keep the most likely words
                that cumulatively have probability 0.94 or higher.
                We then redistribute the probability and do regular sampling.
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.

        """
        assert top_k >= 0
        assert top_p >= 0

        if in_indices.dim() == 1:
            in_indices = in_indices.unsqueeze(0)
        else:
            assert in_indices.dim() == 2
            assert in_indices.shape[0] == 1

        seq_len = in_indices.shape[-1]
        if seq_len < self.context_length:
            x = torch.zeros((1, self.context_length), dtype=in_indices.dtype, device=in_indices.device)
            x[:, -seq_len:] = in_indices
        elif seq_len > self.context_length:
            # ignore the prefix if the provided sequence is too long
            x = in_indices[:, -self.context_length :]
        else:
            x = in_indices

        generator = torch.Generator(device=in_indices.device)
        if seed is not None:
            generator.manual_seed(seed)

        generated_token_ids = []
        while len(generated_token_ids) < max_new_tokens:
            logits = self.forward(x)  # (1, context_length, vocab_size)
            next_token_logits = logits[0, -1]  # (vocab_size,)
            if top_k > 0 and top_k < self.vocab_size:
                top_k_tensor = torch.topk(next_token_logits, top_k)
                prob = softmax(top_k_tensor.values, temperature=temperature)
                vocab_indices = top_k_tensor.indices
            else:
                prob = softmax(next_token_logits, dim=-1, temperature=temperature)
                vocab_indices = torch.arange(len(next_token_logits))

            if top_p > 0:
                prob_sort = torch.sort(prob, descending=True)
                prob_cumsum = torch.cumsum(prob_sort.values, 0)
                threshold_indices = torch.where(prob_cumsum <= top_p)[0]
                threshold_index = 0 if len(threshold_indices) == 0 else threshold_indices[-1]

                prob = prob_sort.values[: threshold_index + 1]
                top_p_indices = prob_sort.indices[: threshold_index + 1]
                vocab_indices = vocab_indices[top_p_indices]

            # it isn't necessary to re-normalize the probabilities
            sampled_index = torch.multinomial(prob, 1, generator=generator).item()
            vocab_index = vocab_indices[sampled_index]
            generated_token_ids.append(vocab_index)
            if eos_token_id is not None and vocab_index == eos_token_id:
                break
            x_new = x.clone()
            x_new[0, :-1] = x[0, 1:]
            x_new[0, -1] = vocab_index
            x = x_new

        return torch.tensor(generated_token_ids, dtype=torch.int64, device=in_indices.device)
