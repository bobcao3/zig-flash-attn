import math

import torch

FULLY_MASKED_LSE = float("-inf")


def visible_keys(lengths_q, lengths_k, max_seqlen_q, max_seqlen_k, window_size):
    rows = torch.arange(max_seqlen_q, device=lengths_q.device)[None, :, None]
    columns = torch.arange(max_seqlen_k, device=lengths_q.device)[None, None, :]
    diagonal = (lengths_k - lengths_q)[:, None, None]
    visible = (rows < lengths_q[:, None, None]) & (columns < lengths_k[:, None, None])
    if window_size[1] >= 0:
        visible &= columns <= rows + diagonal + window_size[1]
    if window_size[0] >= 0:
        visible &= columns >= rows + diagonal - window_size[0]
    return visible


def masked_attention(q, k, v, visible, upcast, softmax_scale):
    out_dtype = torch.float32 if upcast else q.dtype
    q, k, v = (x.float() if upcast else x.clone(memory_format=torch.contiguous_format) for x in (q, k, v))
    batch, seqlen_q, heads, head_dim = q.shape
    seqlen_k, groups = k.shape[1], heads // k.shape[2]
    bias = torch.zeros(visible.shape[0], seqlen_q, -(-seqlen_k // 16) * 16, dtype=q.dtype, device=q.device)
    bias[..., :seqlen_k].masked_fill_(~visible, float("-inf"))
    bias = bias[..., :seqlen_k][:, None].expand(batch, heads, seqlen_q, seqlen_k)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q.transpose(1, 2),
        k.transpose(1, 2).repeat_interleave(groups, dim=1),
        v.transpose(1, 2).repeat_interleave(groups, dim=1),
        bias,
        True,
        0.0,
        False,
        scale=softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim),
    )
    fully_masked = ~visible.any(dim=-1)
    lse = lse[..., :seqlen_q].masked_fill(fully_masked[:, None], FULLY_MASKED_LSE)
    out = out.transpose(1, 2).masked_fill(fully_masked[:, :, None, None], 0.0)
    return out.to(out_dtype), lse


def attention_ref(q, k, v, *, causal=False, window_size=(-1, -1), upcast=True, softmax_scale=None):
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    lengths_q = torch.full((1,), seqlen_q, device=q.device)
    lengths_k = torch.full((1,), seqlen_k, device=q.device)
    visible = visible_keys(lengths_q, lengths_k, seqlen_q, seqlen_k, window_size)
    return masked_attention(q, k, v, visible, upcast, softmax_scale)


def on_device(tensor, device):
    return tensor.pin_memory().to(device, non_blocking=True)


def sequence_rows(starts, lengths):
    offsets = starts - torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    return torch.repeat_interleave(offsets, lengths) + torch.arange(int(lengths.sum()))


def varlen_attention_ref(
    q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False, window_size=(-1, -1), upcast=True, softmax_scale=None
):
    if causal:
        window_size = (window_size[0], 0)
    cu_q, cu_k = cu_seqlens_q.cpu().long(), cu_seqlens_k.cpu().long()
    lengths_q, lengths_k = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
    count, heads = len(lengths_q), q.shape[1]
    max_seqlen_q, max_seqlen_k = max(int(lengths_q.max()), 1), max(int(lengths_k.max()), 1)
    rows_q = on_device(sequence_rows(torch.arange(count) * max_seqlen_q, lengths_q), q.device)
    rows_k = on_device(sequence_rows(torch.arange(count) * max_seqlen_k, lengths_k), q.device)

    def padded(x, rows, longest):
        flat = x.new_zeros(count * longest, *x.shape[1:]).index_copy(0, rows, x[: rows.shape[0]])
        return flat.view(count, longest, *x.shape[1:])

    visible = visible_keys(
        on_device(lengths_q, q.device), on_device(lengths_k, q.device), max_seqlen_q, max_seqlen_k, window_size
    )
    out, lse = masked_attention(
        padded(q, rows_q, max_seqlen_q), padded(k, rows_k, max_seqlen_k), padded(v, rows_k, max_seqlen_k), visible, upcast,
        softmax_scale,
    )
    out = out.reshape(count * max_seqlen_q, *q.shape[1:]).index_select(0, rows_q)
    return out, lse.transpose(0, 1).reshape(heads, count * max_seqlen_q).index_select(1, rows_q)


def reference_gradients(function, q, k, v, dout, *arguments, **options):
    inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    out, _ = function(*inputs, *arguments, **options)
    return torch.autograd.grad(out, inputs, dout.to(out.dtype).clone(memory_format=torch.contiguous_format))
