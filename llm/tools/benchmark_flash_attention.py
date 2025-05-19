import torch
import triton
from llm.flash_attention import FlashAttention, attention_pytorch


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    sequence_length = 512
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    flash = FlashAttention.apply
    # flash = attention_pytorch

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    torch.cuda.synchronize()
    results = triton.testing.do_bench(flash_forward_backward, rep=200, warmup=10)  #  return_mode="all")
    print(results)


if __name__ == "__main__":
    test_timing_flash_forward_backward()
