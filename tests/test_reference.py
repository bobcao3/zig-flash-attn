import pytest
import torch

from reference import attention_ref, reference_gradients, varlen_attention_ref
from shapes import HEAD_DIM, HEAD_LAYOUTS, WINDOW_NAMES, cu_seqlens_of, cuda_seed, near_maximum_lengths, window_for

TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def explicit_attention(q, k, v, causal, window_size):
    seqlen_q, num_head = q.shape[1], q.shape[2]
    seqlen_k, num_head_kv = k.shape[1], k.shape[2]
    q, k, v = (tensor.double() for tensor in (q, k, v))
    k, v = (tensor.repeat_interleave(num_head // num_head_kv, dim=2) for tensor in (k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * q.shape[-1] ** -0.5
    row = torch.arange(seqlen_q, device=q.device)[:, None]
    column = torch.arange(seqlen_k, device=q.device)[None, :]
    offset = column - row - (seqlen_k - seqlen_q)
    left, right = window_size
    right = 0 if causal else right
    visible = torch.ones_like(offset, dtype=torch.bool)
    if right >= 0:
        visible &= offset <= right
    if left >= 0:
        visible &= offset >= -left
    scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.exp(scores - lse[..., None]).masked_fill(torch.isneginf(lse)[..., None], 0.0)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v), lse


def explicit_varlen_attention(q, k, v, sizes_q, sizes_k, causal, window_size):
    outputs, lses = zip(
        *(
            explicit_attention(q_i[None], k_i[None], v_i[None], causal, window_size)
            for q_i, k_i, v_i in zip(q.split(sizes_q), k.split(sizes_k), v.split(sizes_k))
        )
    )
    return torch.cat([out[0] for out in outputs]), torch.cat([lse[0] for lse in lses], dim=1)


def explicit_gradients(function, q, k, v, dout, *arguments):
    inputs = [tensor.detach().double().requires_grad_() for tensor in (q, k, v)]
    out, _ = function(*inputs, *arguments)
    return torch.autograd.grad(out, inputs, dout.double())


def assert_gradients_close(gradients, expected_gradients):
    for gradient, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient.double(), expected, **TOLERANCE)


@pytest.mark.parametrize("layout", HEAD_LAYOUTS)
@pytest.mark.parametrize("window_name", WINDOW_NAMES)
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(3, 3), (40, 17), (17, 40), (256, 256), (1024, 1023)])
def test_attention_ref_matches_explicit_softmax(layout, window_name, seqlen_q, seqlen_k):
    cuda_seed(seqlen_q + seqlen_k)
    num_head, num_head_kv = HEAD_LAYOUTS[layout]
    q = torch.randn(2, seqlen_q, num_head, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, seqlen_k, num_head_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, seqlen_k, num_head_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    window, causal = window_for(window_name, seqlen_k)
    out, lse = attention_ref(q, k, v, causal=causal, window_size=window)
    expected, expected_lse = explicit_attention(q, k, v, causal, window)
    torch.testing.assert_close(out.double(), expected, **TOLERANCE)
    torch.testing.assert_close(lse.double(), expected_lse, **TOLERANCE)
    assert_gradients_close(
        reference_gradients(attention_ref, q, k, v, dout, causal=causal, window_size=window),
        explicit_gradients(explicit_attention, q, k, v, dout, causal, window),
    )


def test_attention_ref_with_zero_window_attends_only_the_diagonal_key():
    cuda_seed(5)
    q, k, v = (torch.randn(2, 33, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out, lse = attention_ref(q, k, v, window_size=(0, 0))
    diagonal_scores = (q.float() * k.float()).sum(dim=-1) * HEAD_DIM**-0.5
    torch.testing.assert_close(out, v.float(), **TOLERANCE)
    torch.testing.assert_close(lse, diagonal_scores.transpose(1, 2), **TOLERANCE)


@pytest.mark.parametrize("layout", HEAD_LAYOUTS)
@pytest.mark.parametrize("window_name", ["dense", "causal", "local"])
def test_varlen_attention_ref_matches_explicit_softmax(layout, window_name):
    cuda_seed(1)
    generator = torch.Generator().manual_seed(1)
    lengths_q = near_maximum_lengths(6, 40, generator)
    lengths_q[:2] = 0
    lengths_k = near_maximum_lengths(6, 40, generator, zero_lengths=True)
    num_head, num_head_kv = HEAD_LAYOUTS[layout]
    q = torch.randn(int(lengths_q.sum()), num_head, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(int(lengths_k.sum()), num_head_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(int(lengths_k.sum()), num_head_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    window, causal = window_for(window_name, 40)
    cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
    sizes = lengths_q.tolist(), lengths_k.tolist()
    out, lse = varlen_attention_ref(q, k, v, cu_q, cu_k, causal, window)
    expected, expected_lse = explicit_varlen_attention(q, k, v, *sizes, causal, window)
    torch.testing.assert_close(out.double(), expected, **TOLERANCE)
    torch.testing.assert_close(lse.double(), expected_lse, **TOLERANCE)
    assert_gradients_close(
        reference_gradients(varlen_attention_ref, q, k, v, dout, cu_q, cu_k, causal, window),
        explicit_gradients(explicit_varlen_attention, q, k, v, dout, *sizes, causal, window),
    )
