import torch
from einops import einsum
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


@triton.autotune(
    configs=[
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
    ],
    key=["N_QUERIES", "N_KEYS", "head_dim"],
)
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
    head_dim: tl.constexpr,
    is_casual: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    num_warps determines how many independent threads the kernel uses to execute a tile.
    warp = 32 threads
        num_warps = 4 -> 128 threads
        num_warps = 8 -> 256 threads
    As tile size increases from 32x32 to 128x128
    - There's more work per tile: more rows x more dot products x more memory
    - You need more threads to parallelize that work effectively
    - Larger tiles also increase shared memory and register pressure -> distribute across more threads
    Rule of thumb:
        Tile_size (16-32) -> num_warps (2-4)
        Tile_size (64) -> num_warps (4)
        Tile_size (128) -> num_warps (8)

    head_dim is included in the autotune because head_dim affects
    - dot product sizes
    - memory stride
    - num loads/stores
    """

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_offset = query_tile_index * Q_TILE_SIZE
    if q_offset >= N_QUERIES:
        return

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, head_dim),
        strides=(stride_qq, stride_qd),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
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
        shape=(N_KEYS, head_dim),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )

    # V of shape (bs, Nk, D) will be split into blocks of size (Bk, D)
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, head_dim),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )

    # offset within the output tensor of size (bs, Nq, D)
    # will only write to a single output tile of size (Q_TILE_SIZE, D)
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    # offset within the logsumexp tensor of size (bs, Nq)
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_offset,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bq, D)
    O_tile = tl.zeros((Q_TILE_SIZE, head_dim), dtype=tl.float32)  # (Bq, D)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Bq,) log(sum(exp(Scores_ij))
    m_tile = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Bq,)

    # easier compiler to unroll and optimize the loop
    for tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_offset = tile_idx * K_TILE_SIZE

        fully_masked = False
        apply_mask = False
        if is_casual:
            # Case 1: Fully masked – skip
            if k_offset >= q_offset + Q_TILE_SIZE:
                fully_masked = True
            # Case 2: Fully valid – no mask needed
            # Just perform 1D bounds check is cheap
            elif k_offset + K_TILE_SIZE <= q_offset:
                apply_mask = False
            # Case 3: Diagonal – apply per-element mask
            # Need to perform the full 2D mask
            else:
                apply_mask = True

        if not fully_masked:
            # Adjust K and V block pointers for this tile
            K_block_ptr_k = K_block_ptr.advance((k_offset, 0))
            V_block_ptr_k = V_block_ptr.advance((k_offset, 0))

            # inserts a 0.0 for any out-of-bounds elements to avoid reading past N_KEY
            K_tile = tl.load(K_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)
            V_tile = tl.load(V_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)

            # Compute the attention scores, ensuring that invalid elements are masked
            # (Bq, D) @ (Bk, D)^T = (Bq, Bk)
            scores = tl.dot(Q_tile, K_tile.T).to(tl.float32) * scale
            key_indices = tl.arange(0, K_TILE_SIZE) + k_offset
            if is_casual and apply_mask:
                query_indices = tl.arange(0, Q_TILE_SIZE) + q_offset
                qk_mask = query_indices[:, None] >= key_indices[None, :]  # (Bq, Bk)

                # mask to prevent reading past N_KEY
                key_mask = key_indices < N_KEYS  # shape: (Bk,)
                mask = qk_mask & key_mask[None, :]
                scores = tl.where(mask, scores, float("-inf"))
            else:
                # mask to prevent reading past N_KEY
                key_mask = key_indices < N_KEYS  # shape: (Bk,)
                scores = tl.where(key_mask[None, :], scores, float("-inf"))

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

    tl.store(O_block_ptr, O_tile.to(dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_tile.to(dtype), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
    ],
    key=["N_QUERIES", "N_KEYS", "head_dim"],
)
@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    dO_ptr,
    dK_ptr,
    dV_ptr,
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
    head_dim: tl.constexpr,
    is_casual: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Backward pass kernel for FlashAttention.
    This kernel computes the gradients for K, and V tensors.

    Q: Query tensor of shape (batch_size, num_queries, head_dim).
    K: Key tensor of shape (batch_size, num_keys, head_dim).
    V: Value tensor of shape (batch_size, num_keys, head_dim).
    is_casual: If True, use causal attention mask.
    logsumexp: of shape (batch_size, num_queries)
    output: of shape (batch_size, num_queries, head_dim)

    grad_output: of shape (batch_size, num_queries, head_dim)
    """
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    k_offset = key_tile_index * K_TILE_SIZE

    if k_offset >= N_KEYS:
        return

    fully_masked = is_casual and (k_offset > N_QUERIES - 1)
    if fully_masked:
        return  # All queries are before this key block, causal mask zeroes everything

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # K of shape (bs, Nk, d) will be split into blocks of size (Bk, d)

    # Key/value tile pointers (fixed for this kernel)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, head_dim),
        strides=(stride_kk, stride_kd),
        offsets=(k_offset, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dtype = K_block_ptr.type.element_ty

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, head_dim),
        strides=(stride_vk, stride_vd),
        offsets=(k_offset, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, head_dim),
        strides=(stride_kk, stride_kd),
        offsets=(k_offset, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, head_dim),
        strides=(stride_vk, stride_vd),
        offsets=(k_offset, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )

    # Load K/V tiles (static for this kernel)
    K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dK_tile = tl.zeros((K_TILE_SIZE, head_dim), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, head_dim), dtype=tl.float32)

    # Query-tile-related pointers (advance in the loop)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, head_dim),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_offset = query_tile_index * Q_TILE_SIZE
        query_fully_masked = is_casual and (q_offset + Q_TILE_SIZE - 1 < k_offset)

        fully_masked = False
        apply_mask = False
        if is_casual:
            # Case 1: Fully masked – skip
            if q_offset + Q_TILE_SIZE - 1 < k_offset:
                fully_masked = True
            # Case 2: Fully valid – no mask needed
            elif q_offset >= k_offset + K_TILE_SIZE:
                apply_mask = False  # full valid
            # Case 3: Diagonal – apply per-element mask
            else:
                apply_mask = True  # diagonal

        if not query_fully_masked:
            Q_tile = tl.load(Q_block_ptr.advance((q_offset, 0)), boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            O_tile = tl.load(O_block_ptr.advance((q_offset, 0)), boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            dO_tile = tl.load(dO_block_ptr.advance((q_offset, 0)), boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            L_tile = tl.load(L_block_ptr.advance((q_offset,)), boundary_check=(0,), padding_option="zero").to(
                tl.float32
            )
            # ensure that accumulation is done in float32
            D_tile = tl.sum(O_tile * dO_tile, axis=1)

            S = tl.dot(Q_tile, K_tile.T) * scale  # (Bq, Bk)
            # don't need to compute softmax because we have logsumexp. P = softmax(S, dim=-1)
            P = tl.exp(S - L_tile[:, None])  # (Bq, Bk)

            query_indices = tl.arange(0, Q_TILE_SIZE) + q_offset
            if is_casual and apply_mask:
                key_indices = tl.arange(0, K_TILE_SIZE) + k_offset
                qk_mask = query_indices[:, None] >= key_indices[None, :]  # (Bq, Bk)
                query_mask = query_indices < N_QUERIES
                qk_mask &= query_mask[:, None]
                P = tl.where(qk_mask, P, 0.0)
            else:
                query_mask = query_indices < N_QUERIES
                P = tl.where(query_mask[:, None], P, 0.0)

            dV_tile += tl.dot(P.T, dO_tile)  # (Bk,Bq)*(Bq,d)->(Bk, d)
            dP = tl.dot(dO_tile, V_tile.T)  # (Bq, d)*(d, Bk)->(Bq, Bk)
            dS = P * (dP - D_tile[:, None]) * scale  # (Bq, Bk)
            dK_tile += tl.dot(dS.T, Q_tile)  # (Bk, Bq)*(Bq, d)->(Bk, d)

    tl.store(dK_block_ptr, dK_tile.to(dtype), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile.to(dtype), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
    ],
    key=["N_QUERIES", "N_KEYS", "head_dim"],
)
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    head_dim: tl.constexpr,
    is_casual: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Backward pass kernel for FlashAttention.
    This kernel computes the gradients for Q tensor.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_offset = query_tile_index * Q_TILE_SIZE
    if q_offset >= N_QUERIES:
        return

    # Q, O, dO, L, dQ pointers
    # Query-side tiles (fixed per kernel instance)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, head_dim),
        strides=(stride_qq, stride_qd),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dtype = Q_block_ptr.type.element_ty
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, head_dim),
        strides=(stride_qq, stride_qd),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_offset,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load query-side tiles, cast to float32
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    O_tile = tl.load(O_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    L_tile = tl.load(L_block_ptr, boundary_check=(0,)).to(tl.float32)

    D_tile = tl.sum(O_tile * dO_tile, axis=1)  # (Bq,)
    dQ_tile = tl.zeros((Q_TILE_SIZE, head_dim), dtype=tl.float32)

    # Key-side pointers
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, head_dim),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_kb,
        shape=(N_KEYS, head_dim),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, head_dim),
        order=(1, 0),
    )

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_offset = key_tile_index * K_TILE_SIZE

        fully_masked = False
        apply_mask = False
        if is_casual:
            # Tile is fully masked
            if k_offset > q_offset + Q_TILE_SIZE - 1:
                fully_masked = True
            elif k_offset + K_TILE_SIZE - 1 <= q_offset:
                apply_mask = False  # full valid
            else:
                apply_mask = True  # diagonal

        if not fully_masked:
            K_tile = tl.load(K_block_ptr.advance((k_offset, 0)), boundary_check=(0, 1)).to(tl.float32)
            V_tile = tl.load(V_block_ptr.advance((k_offset, 0)), boundary_check=(0, 1)).to(tl.float32)
            S = tl.dot(Q_tile, K_tile.T) * scale
            P = tl.exp(S - L_tile[:, None])  # softmax weights

            key_indices = tl.arange(0, K_TILE_SIZE) + k_offset
            if is_casual and apply_mask:
                # causal mask should be of shape (Bq, Bk)
                query_indices = tl.arange(0, Q_TILE_SIZE) + q_offset
                qk_mask = query_indices[:, None] >= key_indices[None, :]  # (Bq, Bk)
                # mask to prevent reading past N_KEY
                key_mask = key_indices < N_KEYS  # shape: (Bk,)
                qk_mask &= key_mask[None, :]
                P = tl.where(qk_mask, P, 0.0)
            else:
                key_mask = key_indices < N_KEYS  # shape: (Bk,)
                P = tl.where(key_mask[None, :], P, 0.0)

            dP = tl.dot(dO_tile, V_tile.T)
            dS = P * (dP - D_tile[:, None]) * scale

            dQ_tile += tl.dot(dS, K_tile)

    tl.store(dQ_block_ptr, dQ_tile.to(dtype), boundary_check=(0, 1))


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
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        assert Q.dtype == K.dtype == V.dtype, "Q, K, and V must have the same dtype"
        assert Q.is_cuda, "FlashAttention requires CUDA"
        assert Q.device == K.device == V.device, "Q, K, and V must be on the same device"
        head_dim = Q.shape[-1]
        num_queries = Q.shape[-2]  # Nq
        num_keys = K.shape[-2]  # Nk
        batch_size = Q.shape[0]
        assert Q.shape[-1] == head_dim == head_dim, "Q, K, and V must have the same last dimension"
        assert V.shape[-2] == num_keys, "V must have the same number of keys as K"
        assert K.shape[0] == V.shape[0] == batch_size, "Q, K, and V must have the same batch size"

        ctx.is_casual = is_casual

        # Bq = QUERY_TILE_SIZE, Bk = KEY_TILE_SIZE
        # Bq and Bk will be set by triton.autotune, so we have to just set a dummy value for now
        # split Q into Tq = ceil(Nq / Bq) tiles of size Bq x D
        # split K into Tk = ceil(Nk / Bk) tiles of size Bk x D
        # split V into Tk = ceil(Nk / Bk) tiles of size Bk x D
        Tq = triton.cdiv(num_queries, 1)

        output = torch.empty((batch_size, num_queries, head_dim), device=Q.device, dtype=Q.dtype)
        logsumexp = torch.empty((batch_size, num_queries), device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(head_dim)
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
            head_dim=head_dim,
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

        Q: Query tensor of shape (batch_size, num_queries, head_dim).
        K: Key tensor of shape (batch_size, num_keys, head_dim).
        V: Value tensor of shape (batch_size, num_keys, head_dim).
        is_casual: If True, use causal attention mask.
        logsumexp: of shape (batch_size, num_queries)
        output: of shape (batch_size, num_queries, head_dim)

        grad_output: of shape (batch_size, num_queries, head_dim)

        """
        Q, K, V, logsumexp, output = ctx.saved_tensors
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        logsumexp = logsumexp.contiguous()
        output = output.contiguous()
        grad_output = grad_output.contiguous()

        assert Q.dtype == logsumexp.dtype == output.dtype == grad_output.dtype, "All tensors must have the same dtype"
        assert Q.is_cuda, "FlashAttention requires CUDA"
        assert Q.device == logsumexp.device == output.device == grad_output.device, (
            "All tensors must be on the same device"
        )

        is_casual = ctx.is_casual
        batch_size, num_queries, head_dim = Q.shape
        num_keys = K.shape[-2]

        Tq = triton.cdiv(num_queries, 1)  # number of query tiles
        Tk = triton.cdiv(num_keys, 1)  # number of key tiles

        scale = 1.0 / math.sqrt(head_dim)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        flash_bwd_dkv_kernel[(Tk, batch_size)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=output,
            L_ptr=logsumexp,
            dO_ptr=grad_output,
            dK_ptr=dK,
            dV_ptr=dV,
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
            head_dim=head_dim,
            is_casual=is_casual,
        )
        flash_bwd_dq_kernel[(Tq, batch_size)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=output,
            L_ptr=logsumexp,
            dO_ptr=grad_output,
            dQ_ptr=dQ,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_ob=output.stride(0),
            stride_oq=output.stride(1),
            stride_od=output.stride(2),
            stride_lb=logsumexp.stride(0),
            stride_lq=logsumexp.stride(1),
            N_QUERIES=num_queries,
            N_KEYS=num_keys,
            scale=scale,
            head_dim=head_dim,
            is_casual=is_casual,
        )
        return dQ, dK, dV, None


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
