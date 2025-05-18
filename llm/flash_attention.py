import torch
from einops import einsum, rearrange
import math

import triton
import triton.language as tl

"""
References:
- https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
"""


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
    """
    The forward method combines forward() and setup_context() because
    it's necessary to store the the intermediate logsumexp tensor
    for the backward pass.
    """

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

        Bq = ctx.QUERY_TILE_SIZE = 16  # Bq, Each thread processes 16 batch elements at a time
        Bk = ctx.KEY_TILE_SIZE = 16  # Bk
        ctx.is_casual = is_casual

        # split Q into Tq = ceil(Nq / Bq) tiles of size Bq x D
        # split K into Tk = ceil(Nk / Bk) tiles of size Bk x D
        # split V into Tk = ceil(Nk / Bk) tiles of size Bk x D

        output = torch.empty((batch_size, num_queries, D), device=Q.device, dtype=Q.dtype)
        logsumexp = torch.empty((batch_size, num_queries), device=Q.device, dtype=Q.dtype)

        for i in range(0, num_queries, Bq):
            Q_tile = Q[:, i : i + Bq, :]
            O_tile = torch.zeros((batch_size, Bq, D), device=Q.device, dtype=Q.dtype)
            # softmax denominator
            l_tile = torch.zeros((batch_size, Bq), device=Q.device, dtype=Q.dtype)
            # running max initialized to -inf
            # softmax will be computed over the key dimension for the score matrix (batch_size, num_queries, num_keys)
            # so the max should be across the key dimension
            m_tile = torch.full((batch_size, Bq), torch.finfo(Q.dtype).min, device=Q.device, dtype=Q.dtype)

            for j in range(0, num_keys, Bk):
                # Compute the attention scores
                K_tile = K[:, j : j + Bk, :]
                V_tile = V[:, j : j + Bk, :]

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

        ctx.save_for_backward(Q, K, V, logsumexp, output)
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

        Q: Query tensor of shape (batch_size, num_queries, head_dim).
        K: Key tensor of shape (batch_size, num_keys, head_dim).
        V: Value tensor of shape (batch_size, num_keys, head_dim).
        is_casual: If True, use causal attention mask.
        logsumexp: of shape (batch_size, num_queries)
        output: of shape (batch_size, num_queries, head_dim)

        grad_output: of shape (batch_size, num_queries, head_dim)

        """
        # Implement the backward pass using FlashAttention
        Q, K, V, logsumexp, output = ctx.saved_tensors
        is_casual = ctx.is_casual
        Bq = ctx.QUERY_TILE_SIZE
        Bk = ctx.KEY_TILE_SIZE
        batch_size, num_queries, head_dim = Q.shape

        # D = rowsum(O * dO) = rowsum(P * dP) = diag(P*dP^T)
        D = torch.sum(output * grad_output, dim=-1, keepdim=False)  # (bs, num_queries)

        # scores Q @ K^T / sqrt(d_k)
        S = einsum(Q, K, "... queries d, ... keys d -> ... queries keys") / math.sqrt(head_dim)
        # don't need to compute softmax because we have logsumexp. P = softmax(S, dim=-1)
        # logsumexp is size (bs, num_queries)
        P = torch.exp(S - logsumexp[..., None])  # (bs, num_queries, num_keys)

        dV = einsum(P, grad_output, "... queries keys, ... queries d -> ... keys d")
        dP = einsum(grad_output, V, "... queries d, ... keys d -> ... queries keys")
        # (bs, queries, keys) * ( (bs, queries, keys) - (bs, queries) )
        dS = P * (dP - D[..., None])  # (bs, queries, keys)
        dQ = einsum(dS, K, "... queries keys, ... keys d -> ... queries d") / math.sqrt(head_dim)
        dK = einsum(dS, Q, "... queries keys, ... queries d -> ... keys d") / math.sqrt(head_dim)

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_casual: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dtype = Q_block_ptr.type.element_ty

    # offset parameter specifies the starting point (in terms of indices)
    # of the block block within the larger tensor. It is used to determine
    # where the block begins in the memory layout of the tensor.
    # K_block_ptr block will start at first row and first column of the K tensor
    # because we will be accumulating intermediate results for the given Q_tile

    # The order parameter defines the memory access pattern for the block.
    # It specifies the order in which dimensions are traversed when accessing
    # elements in the block.
    # order=(1, 0) means that the second dimension (columns) is traversed first,
    # followed by the first dimension (rows). This is a column-major access pattern.

    # K of shape (bs, Nk, D) will be split into blocks of size (Bk, D)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V of shape (bs, Nk, D) will be split into blocks of size (Bk, D)
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # offset within the output tensor of size (bs, Nq, D)
    # will only write to a single output tile of size (Q_TILE_SIZE, D)
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # offset within the logsumexp tensor of size (bs, Nq)
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bq, D)
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Bq, D)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Bq,) log(sum(exp(Scores_ij))
    m_tile = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Bq,)

    # easier compiler to unroll and optimize the loop
    for tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_offset = tile_idx * K_TILE_SIZE
        # Adjust K and V block pointers for this tile
        K_block_ptr_k = K_block_ptr.advance((k_offset, 0))
        V_block_ptr_k = V_block_ptr.advance((k_offset, 0))

        # inserts a 0.0 for any out-of-bounds elements to avoid reading past N_KEY
        K_tile = tl.load(K_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)
        V_tile = tl.load(V_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)

        # mask to prevent reading past N_KEY
        mask = ((tl.arange(0, K_TILE_SIZE) + k_offset) < N_KEYS)[None, :]  # shape: (1, Bk)

        if is_casual:
            # causal mask should be of shape (Bq, Bk)
            query_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            key_indices = tl.arange(0, K_TILE_SIZE) + k_offset
            mask = tl.where(query_indices[:, None] >= key_indices[None, :], mask, False)

        # Compute the attention scores, ensuring that invalid elements are masked
        # (Bq, D) @ (Bk, D)^T = (Bq, Bk)
        scores = tl.where(mask, tl.dot(Q_tile, K_tile.T).to(tl.float32) * scale, float("-inf"))

        row_max = tl.max(scores, axis=-1, keep_dims=False)  # (Bq,)
        new_max = tl.maximum(m_tile, row_max)  # elementwise max

        # unnormalized softmax values (numerator)
        P_j = tl.exp(scores - new_max[:, None])  # (Bq, Bk)
        row_sum_P_j = tl.sum(P_j, axis=-1, keep_dims=False)  # (Bq,)
        exp_m_diff = tl.exp(m_tile - new_max)  # (Bq,)
        l_tile = exp_m_diff * l_tile + row_sum_P_j  # (Bq,)

        # diag(exp_m_diff) * O_tile
        # P_j (Bq, Bk) * V_tile (Bk, D) = (Bq, D)
        O_tile = exp_m_diff[:, None] * O_tile + tl.dot(P_j, V_tile.to(tl.float32))

        m_tile = new_max  # update the max for the next iteration

    # diag(l_tile)^-1 * O_tile
    O_tile = O_tile / l_tile[:, None]
    L_tile = m_tile + tl.log(l_tile)  # logsumexp

    O_tile = O_tile.to(dtype)
    L_tile = L_tile.to(dtype)

    tl.store(O_block_ptr, O_tile, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_tile, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):
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
        assert Q.is_cuda, "FlashAttention requires CUDA"
        assert Q.device == K.device == V.device, "Q, K, and V must be on the same device"
        D = Q.shape[-1]
        num_queries = Q.shape[-2]  # Nq
        num_keys = K.shape[-2]  # Nk
        batch_size = Q.shape[0]
        assert Q.shape[-1] == D == D, "Q, K, and V must have the same last dimension"
        assert V.shape[-2] == num_keys, "V must have the same number of keys as K"
        assert K.shape[0] == V.shape[0] == batch_size, "Q, K, and V must have the same batch size"

        Bq = ctx.QUERY_TILE_SIZE = 16  # Bq, Each thread processes 16 batch elements at a time
        Bk = ctx.KEY_TILE_SIZE = 16  # Bk
        ctx.is_casual = is_casual

        # split Q into Tq = ceil(Nq / Bq) tiles of size Bq x D
        # split K into Tk = ceil(Nk / Bk) tiles of size Bk x D
        # split V into Tk = ceil(Nk / Bk) tiles of size Bk x D
        Tq = triton.cdiv(num_queries, Bq)

        output = torch.empty((batch_size, num_queries, D), device=Q.device, dtype=Q.dtype)
        logsumexp = torch.empty((batch_size, num_queries), device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(D)
        # Triton program instance will only load elements from a single batch index
        # and only read/write to a single query tile of Q, O, L
        flash_fwd_kernel[(Tq, batch_size)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=output,
            L_ptr=logsumexp,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=output.stride(0),
            stride_oq=output.stride(1),
            stride_od=output.stride(2),
            stride_lb=logsumexp.stride(0),
            stride_lq=logsumexp.stride(1),
            N_QUERIES=num_queries,
            N_KEYS=num_keys,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_casual=is_casual,
        )
        ctx.save_for_backward(Q, K, V, logsumexp, output)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(4, 128, 64, requires_grad=True, device=device)
    K = torch.randn(4, 128, 64, requires_grad=True, device=device)
    V = torch.randn(4, 128, 64, requires_grad=True, device=device)

    Q_pt = Q.clone().detach().requires_grad_(True)
    K_pt = K.clone().detach().requires_grad_(True)
    V_pt = V.clone().detach().requires_grad_(True)

    Q_triton = Q.clone().detach().requires_grad_(True)
    K_triton = K.clone().detach().requires_grad_(True)
    V_triton = V.clone().detach().requires_grad_(True)

    pytorch_out = attention_pytorch(Q_pt, K_pt, V_pt)

    flash_pytorch_out = FlashAttentionPytorch.apply(Q, K, V)

    flash_triton_out = FlashAttention.apply(Q_triton, K_triton, V_triton)

    assert torch.allclose(pytorch_out, flash_pytorch_out, atol=1e-2), (
        "Pytorch & FlashAttentionPytorch outputs do not match!"
    )
    print("✅ Pytorch & FlashAttentionPytorch outputs match!")
    assert torch.allclose(pytorch_out, flash_triton_out, atol=1e-2), "Pytorch & Triton outputs do not match!"
    print("✅ Pytorch & Triton outputs match!")
