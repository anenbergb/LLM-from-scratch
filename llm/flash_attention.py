import torch
from einops import einsum, rearrange
import math

import triton
import triton.language as tl


def attention_pytorch(Q, K, V, is_casual=False):
    d_k = K.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")  # Q @ K^T
    scores /= math.sqrt(d_k)
    if is_casual:
        mask = torch.tril(torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool))
        # might be a better way to expand the mask
        mask = mask[None, :, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attention_weights = torch.softmax(scores, dim=-1)
    return einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        """
        Forward pass for FlashAttention.
        Args:
            ctx: Context object to save information for backward pass.
            Q: Query tensor of shape (batch_size, num_queries, head_dim).
            K: Key tensor of shape (batch_size, num_keys, head_dim).
            V: Value tensor of shape (batch_size, num_keys, head_dim).
            is_casual: If True, use causal attention mask.
        Returns:
            Output tensor of shape (batch_size, num_queries, head_dim).

        - save L, Q, K, V, O for the backward pass
        - returns O
        """
        assert Q.is_contiguous(), "Our pointer arithmetic will assume contiguous Q"
        assert K.is_contiguous(), "Our pointer arithmetic will assume contiguous K"
        assert V.is_contiguous(), "Our pointer arithmetic will assume contiguous V"
        assert Q.dtype == K.dtype == V.dtype, "Q, K, and V must have the same dtype"
        assert Q.device == K.device == V.device, "Q, K, and V must be on the same device"
        D = Q.shape[-1]
        num_queries = Q.shape[-2]  # Nq
        num_keys = K.shape[-2]  # Nk
        batch_size = Q.shape[0]
        assert Q.shape[-1] == D == D, "Q, K, and V must have the same last dimension"
        assert V.shape[-2] == num_keys, "V must have the same number of keys as K"
        assert K.shape[0] == V.shape[0] == batch_size, "Q, K, and V must have the same batch size"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)  # 16 // Roughly 16 loops through the embedding dimension
        Bq = ctx.QUERY_TILE_SIZE = 16  # Bq, Each thread processes 16 batch elements at a time
        Bk = ctx.KEY_TILE_SIZE = 16  # Bk

        # split Q into Tq = ceil(Nq / Bq) tiles of size Bq x D
        # split K into Tk = ceil(Nk / Bk) tiles of size Bk x D
        # split V into Tk = ceil(Nk / Bk) tiles of size Bk x D

        output = torch.empty((batch_size, num_queries, D), device=Q.device, dtype=Q.dtype)
        logsumexp = torch.empty((batch_size, num_queries), device=Q.device, dtype=Q.dtype)

        for i in range(0, num_queries, Bq):
            Q_tile = Q[:, i : i + Bq, :]  # maybe add .contiguous()
            O_tile = torch.zeros((batch_size, Bq, D), device=Q.device, dtype=Q.dtype)
            # softmax denominator
            l_tile = torch.zeros((batch_size, Bq), device=Q.device, dtype=Q.dtype)
            # running max initialized to -inf
            # softmax will be computed over the key dimension for the score matrix (batch_size, num_queries, num_keys)
            # so the max should be across the key dimension
            m_tile = torch.full((batch_size, Bq), torch.finfo(Q.dtype).min, device=Q.device, dtype=Q.dtype)

            for j in range(0, num_keys, Bk):
                # Compute the attention scores
                K_tile = K[:, j : j + Bk, :]  # .contiguous()
                V_tile = V[:, j : j + Bk, :]  # .contiguous()

                scores = einsum(Q_tile, K_tile, "... queries d_k, ... keys d_k -> ... queries keys")  # Q @ K^T
                scores /= math.sqrt(D)  # (batch_size, Bq, Bk)

                row_max = torch.max(scores, dim=-1, keepdim=False).values  # (batch_size, Bq)
                new_max = torch.maximum(m_tile, row_max)  # elementwise max

                # unnormalized softmax values (numerator)
                P_j = torch.exp(scores - new_max[..., None])  # (batch_size, Bq, Bk)
                row_sum_P_j = torch.sum(P_j, dim=-1, keepdim=False)  # (batch_size, Bq)
                # update the softmax denominator (which is sum of exponential scores) by subtracting slightly more
                # 'max' value given the updated new_max and add in the newly computed row_sum_P_j
                exp_m_diff = torch.exp(m_tile - new_max)  # (batch_size, Bq)
                l_tile = exp_m_diff * l_tile + row_sum_P_j  # (batch_size, Bq)

                # diag(exp_m_diff) * O_tile
                # P_j (bs, Bq, Bk) * V_tile (bs, Bk, D) = (bs, Bq, D)
                O_tile = exp_m_diff[..., None] * O_tile + einsum(P_j, V_tile, "... Bq Bk, ... Bk D -> ... Bq D")
                m_tile = new_max  # update the max for the next iteration

            # diag(l_tile)^-1 * O_tile
            output[:, i : i + Bq, :] = O_tile / l_tile[..., None]
            logsumexp[:, i : i + Bq] = m_tile + torch.log(l_tile)

        ctx.save_for_backward(logsumexp, Q, K, V, output)
        ctx.is_casual = is_casual
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for FlashAttention.
        Args:
            ctx: Context object containing saved information from forward pass.
            grad_output: Gradient of the output tensor.
        Returns:
            Gradients for Q, K, V, and dropout mask.
        """
        # Implement the backward pass using FlashAttention
        raise NotImplementedError("Backward pass not implemented yet.")


if __name__ == "__main__":
    # Example usage
    Q = torch.randn(4, 128, 64, requires_grad=True)
    K = torch.randn(4, 128, 64, requires_grad=True)
    V = torch.randn(4, 128, 64, requires_grad=True)

    Q_ref = Q.clone().detach().requires_grad_(True)
    K_ref = K.clone().detach().requires_grad_(True)
    V_ref = V.clone().detach().requires_grad_(True)

    pytorch_out = attention_pytorch(Q_ref, K_ref, V_ref)

    flash_pytorch_out = FlashAttentionPytorch.apply(Q, K, V)

    assert torch.allclose(pytorch_out, flash_pytorch_out, atol=1e-2), "Outputs do not match!"
    print("✅ Outputs match!")
