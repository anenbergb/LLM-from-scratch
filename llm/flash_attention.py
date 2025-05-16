import torch
from einops import einsum, rearrange
import math
from typing import Tuple

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


@torch.library.custom_op("llm::flash_attention_pytorch_with_logsumexp", mutates_args=())
def flash_attention_pytorch_with_logsumexp(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_casual: bool) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Bq = 16  # Bq, QUERY_TILE_SIZE Each thread processes 16 batch elements at a time
    Bk = 16  # Bk, KEY_TILE_SIZE

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

    return output, logsumexp

@torch.library.custom_op("llm::flash_attention_pytorch", mutates_args=())
def flash_attention_pytorch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_casual: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    output, logsumexp = flash_attention_pytorch_with_logsumexp(Q, K, V, is_casual)
    return output

def flash_attention_pytorch_backward(
    ctx: torch.autograd.Function, grad_output: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for FlashAttention.
    Args:
        ctx: Context object containing saved information from forward pass.
        grad_output: Gradient of the output tensor.
    Returns:
        Gradients for Q, K, V
    """
    # Implement the backward pass using FlashAttention
    raise NotImplementedError("Backward pass not implemented yet.")


def flash_attention_pytorch_setup_context(
    ctx: torch.autograd.Function, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool], output: torch.Tensor
) -> None:
    Q, K, V, is_causal = inputs
    # ctx.is_casual = keyword_only_inputs.get("is_casual", False)
    ctx.is_causal = is_causal
    ctx.QUERY_TILE_SIZE = 16  # Bq, Each thread processes 16 batch elements at a time
    ctx.KEY_TILE_SIZE = 16  # Bk

    # also should save the logsumexp tensor
    ctx.save_for_backward(Q, K, V, output)


torch.library.register_autograd(
    "llm::flash_attention_pytorch",
    flash_attention_pytorch_backward,
    setup_context=flash_attention_pytorch_setup_context,
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
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
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

    # output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    # logsumexp = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bq, D)
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Bq, D)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Bq,)
    m_tile = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Bq,)

    # for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    for k_offset in range(0, N_KEYS, K_TILE_SIZE):
        # Adjust K and V block pointers for this tile
        K_block_ptr_k = K_block_ptr.advance((k_offset, 0))
        V_block_ptr_k = V_block_ptr.advance((k_offset, 0))

        # mask to prevent reading past N_KEYS
        # This is optional if padding_option="zero" is enough, but helps with precision
        # mask = (tl.arange(0, K_TILE_SIZE) + k_offset) < N_KEYS  # shape (Bk,)

        K_tile = tl.load(K_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)
        V_tile = tl.load(V_block_ptr_k, boundary_check=(0, 1), padding_option="zero")  # (Bk, D)

        # Compute the attention scores
        # (Bq, D) @ (Bk, D)^T = (Bq, Bk)
        # Triton doesn't yset support compute capability 12.x
        # scores = tl.dot(Q_tile, K_tile.T) * scale

        # Expand Q_tile to shape (Bq, 1, D)
        # Expand K_tile to shape (1, Bk, D)
        # Their product will be shape (Bq, Bk, D)
        # Then reduce over the last dimension to get (Bq, Bk)
        scores = tl.sum(Q_tile[:, None, :] * K_tile[None, :, :], axis=2) * scale

        row_max = tl.max(scores, axis=-1, keep_dims=False)  # (Bq,)
        new_max = tl.maximum(m_tile, row_max)  # elementwise max

        # unnormalized softmax values (numerator)
        P_j = tl.exp(scores - new_max[:, None])  # (Bq, Bk)
        row_sum_P_j = tl.sum(P_j, axis=-1, keep_dims=False)  # (Bq,)
        exp_m_diff = tl.exp(m_tile - new_max)  # (Bq,)
        l_tile = exp_m_diff * l_tile + row_sum_P_j  # (Bq,)

        # diag(exp_m_diff) * O_tile
        # P_j (Bq, Bk) * V_tile (Bk, D) = (Bq, D)
        # O_tile = exp_m_diff[:, None] * O_tile + tl.dot(P_j, V_tile.to(tl.float32))
        O_tile = exp_m_diff[:, None] * O_tile + tl.sum(P_j[:, :, None] * V_tile.to(tl.float32)[None, :, :], axis=1)

        m_tile = new_max  # update the max for the next iteration

    # diag(l_tile)^-1 * O_tile
    O_tile = O_tile / l_tile[:, None]
    L_tile = m_tile + tl.log(l_tile)

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

        # double check if the dtype should be hardcoded to float32
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
        )
        ctx.save_for_backward(logsumexp, Q, K, V, output)
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

    pytorch_out = attention_pytorch(Q_pt, K_pt, V_pt, False)

    flash_pytorch_out = torch.ops.llm.flash_attention_pytorch(Q, K, V, False)

    # flash_triton_out = FlashAttention.apply(Q_triton, K_triton, V_triton)

    assert torch.allclose(pytorch_out, flash_pytorch_out, atol=1e-2), (
        "Pytorch & FlashAttentionPytorch outputs do not match!"
    )
    print("✅ Pytorch & FlashAttentionPytorch outputs match!")
    # assert torch.allclose(pytorch_out, flash_triton_out, atol=1e-2), "Pytorch & Triton outputs do not match!"
    # print("✅ Pytorch & Triton outputs match!")
