import math
from typing import NamedTuple

import torch

FULLY_MASKED_LSE = float("-inf")
longest_keys = 0


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
    global longest_keys
    longest_keys = visible.shape[-1]
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


def aligned(tensor):
    if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, window, scale):
        out, lse, seed, offset, _ = torch.ops.aten._flash_attention_forward(
            q, k, v, cu_q, cu_k, max_q, max_k, 0.0, False, False,
            scale=scale, window_size_left=window[0], window_size_right=window[1],
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, seed, offset)
        ctx.arguments = (max_q, max_k, window, scale)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _):
        q, k, v, out, lse, cu_q, cu_k, seed, offset = ctx.saved_tensors
        max_q, max_k, window, scale = ctx.arguments
        gradients = torch.ops.aten._flash_attention_backward(
            aligned(dout), q, k, v, out, lse, cu_q, cu_k, max_q, max_k, 0.0, False, seed, offset,
            scale=scale, window_size_left=window[0], window_size_right=window[1],
        )
        return (*gradients, None, None, None, None, None, None)


def flash_expresses(window, max_q, max_k):
    # FlashAttention-2 replaces the right bound of a window with a left bound by max_k when the right bound is open or
    # at least max_k. Bottom-right alignment then hides keys from the first rows of a query longer than max_k + 1.
    left, right = window
    return left < 0 or 0 <= right < max_k or max_q <= max_k + 1


def flash_attention(q, k, v, cu_q, cu_k, max_q, max_k, window, scale):
    global longest_keys
    longest_keys = max_k
    out, lse = FlashAttention.apply(*map(aligned, (q, k, v)), cu_q, cu_k, max_q, max_k, window, scale)
    # FlashAttention-2 gives a row without visible keys a logsumexp of +inf.
    return out, lse.masked_fill(lse == float("inf"), FULLY_MASKED_LSE)


def attention_ref(q, k, v, *, causal=False, window_size=(-1, -1), upcast=True, softmax_scale=None):
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    if not upcast and flash_expresses(window_size, seqlen_q, seqlen_k):
        return flash_attention(q, k, v, None, None, seqlen_q, seqlen_k, window_size, softmax_scale)
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
    if not upcast and flash_expresses(window_size, max_seqlen_q, max_seqlen_k):
        cu_device = [on_device(cu.int(), q.device) for cu in (cu_q, cu_k)]
        return flash_attention(q, k, v, *cu_device, max_seqlen_q, max_seqlen_k, window_size, softmax_scale)
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


def inter_sequence_visible(cu_seqlens_q, cu_seqlens_k, visible, windows):
    cu_q, cu_k = cu_seqlens_q.cpu().long(), cu_seqlens_k.cpu().long()
    mask = torch.zeros(int(cu_q[-1]), int(cu_k[-1]), dtype=torch.bool)
    for query, key in visible.nonzero().tolist():
        length_q, length_k = int(cu_q[query + 1] - cu_q[query]), int(cu_k[key + 1] - cu_k[key])
        block = torch.ones(length_q, length_k, dtype=torch.bool)
        if query == key:
            left, right = windows[query]
            rows = torch.arange(length_q)[:, None] + length_k - length_q
            columns = torch.arange(length_k)[None, :]
            if left >= 0:
                block &= columns >= rows - left
            if right >= 0:
                block &= columns <= rows + right
        mask[cu_q[query] : cu_q[query + 1], cu_k[key] : cu_k[key + 1]] = block
    return mask


def inter_sequence_attention_ref(q, k, v, mask, upcast=True, softmax_scale=None):
    out, lse = masked_attention(q[None], k[None], v[None], on_device(mask[None], q.device), upcast, softmax_scale)
    return out[0], lse[0]


class StableAttention(NamedTuple):
    out: torch.Tensor
    lse: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    top: torch.Tensor
    margin: torch.Tensor
    spread: torch.Tensor
    visible: torch.Tensor
    pivot: torch.Tensor
    out_scale: torch.Tensor
    dq_scale: torch.Tensor
    dk_scale: torch.Tensor
    dv_scale: torch.Tensor
    corrected_keys: torch.Tensor


PIVOT_PROBABILITY = 0.75


def stable_attention_f64(q, k, v, dout, visible, softmax_scale=None):
    """Attention in float64 with the log-sum-exp `top + log1p(rest)`, where `rest` sums the exponentials of
    the keys other than the top one (eq. 5 of Barron et al., "A Fix for Softmax and Cross-Entropy"). The
    score gradients are `dS_j = P_j ((dP_j - dP_top) - E)` with `E = sum_i P_i (dP_i - dP_top)`, so the top
    key's term `-P_top E` equals the negated sum of the others (their eq. 7) without cancellation.
    `dq_scale` and `dk_scale` sum the error weights of the terms of each gradient row. The weight of `dS_j`
    is `|dS_j| + P_j sum(|out * dout|)`; to first order, 2**-8 times the weight bounds the error that
    rounding `dS_j` and `out` to bfloat16 causes in `dS_j`. The top key of a row with a pivot carries the
    sum of the weights of the other keys of the row."""
    batch, seqlen_q, heads, head_dim = q.shape
    groups = heads // k.shape[2]
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    q64, dout64 = (x.double().transpose(1, 2) for x in (q, dout))
    k64, v64 = (x.double().transpose(1, 2).repeat_interleave(groups, dim=1) for x in (k, v))
    mask = visible[:, None].to(q.device)
    z = (scale * q64 @ k64.transpose(-1, -2)).masked_fill(~mask, float("-inf"))
    live = mask.any(-1)
    top, top_key = z.max(-1, keepdim=True)
    top = top.masked_fill(~live[..., None], 0.0)
    others = (z - top).exp().scatter(-1, top_key, 0.0)
    rest = others.sum(-1, keepdim=True)
    p = (others.scatter(-1, top_key, 1.0) / (1.0 + rest)).masked_fill(~live[..., None], 0.0)
    dp = dout64 @ v64.transpose(-1, -2)
    shifted = dp - dp.gather(-1, top_key)
    ds = p * (shifted - (p * shifted).sum(-1, keepdim=True))
    pivot = live & (p.amax(-1) > PIVOT_PROBABILITY)
    out = p @ v64
    weight = ds.abs() + p * (out.abs() * dout64.abs()).sum(-1, keepdim=True)
    pivot_weight = weight.scatter(-1, top_key, 0.0).sum(-1, keepdim=True)
    weight = torch.where(pivot[..., None], weight.scatter(-1, top_key, pivot_weight), weight)
    spread = torch.where(mask, others * z.abs(), 0.0).sum(-1) / rest[..., 0]
    pivot_counts = torch.zeros_like(p[..., 0, :]).scatter_add(-1, top_key[..., 0], pivot.double())

    def per_kv_head(x):
        return x.unflatten(1, (-1, groups)).sum(2).transpose(1, 2)

    def magnitude(weights, vectors):
        return (weights.abs() @ vectors.norm(dim=-1, keepdim=True))[..., 0]

    return StableAttention(
        out=out.transpose(1, 2),
        lse=torch.where(live, top[..., 0] + rest[..., 0].log1p(), float("-inf")),
        dq=(scale * ds @ k64).transpose(1, 2),
        dk=per_kv_head(scale * ds.transpose(-1, -2) @ q64),
        dv=per_kv_head(p.transpose(-1, -2) @ dout64),
        top=top[..., 0],
        margin=torch.where(live, -rest[..., 0].log(), float("nan")),
        spread=spread,
        visible=mask.sum(-1).expand(batch, heads, seqlen_q),
        pivot=pivot,
        out_scale=magnitude(p, v64).transpose(1, 2),
        dq_scale=scale * magnitude(weight, k64).transpose(1, 2),
        dk_scale=per_kv_head(scale * magnitude(weight.transpose(-1, -2), q64)[..., None])[..., 0],
        dv_scale=per_kv_head(magnitude(p.transpose(-1, -2), dout64)[..., None])[..., 0],
        corrected_keys=per_kv_head(pivot_counts[..., None])[..., 0] > 0,
    )


def varlen_visible(cu_seqlens_q, cu_seqlens_k, window_size=(-1, -1)):
    cu_q, cu_k = cu_seqlens_q.cpu().long(), cu_seqlens_k.cpu().long()
    lengths_q, lengths_k = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
    windows = visible_keys(
        lengths_q, lengths_k, max(int(lengths_q.max()), 1), max(int(lengths_k.max()), 1), window_size
    )
    mask = torch.zeros(int(cu_q[-1]), int(cu_k[-1]), dtype=torch.bool)
    for sequence, (length_q, length_k) in enumerate(zip(lengths_q.tolist(), lengths_k.tolist())):
        rows, columns = slice(cu_q[sequence], cu_q[sequence + 1]), slice(cu_k[sequence], cu_k[sequence + 1])
        mask[rows, columns] = windows[sequence, :length_q, :length_k]
    return mask
