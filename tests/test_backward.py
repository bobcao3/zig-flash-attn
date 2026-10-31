import itertools

import pytest
import torch

from zigfa._impl import backward as backward_impl, forward as forward_impl, specialization
from checks import case, queue_attention_checks, queue_requirement
from reference import attention_ref, varlen_attention_ref
from shapes import (
    BATCHED_WINDOW_SEQLENS,
    GQA_QHEAD_PER_KVHEAD_SHAPES,
    HEAD_DIMS,
    HEAD_LAYOUTS,
    LONG_SEQLENS,
    NARROW_WINDOWS,
    PACKED_GQA_QHEAD_PER_KVHEAD,
    PADDING_ROWS,
    SHORT_QUERY_SEQLENS,
    SMALL_VARLEN_HEAD_COUNTS,
    SMALL_VARLEN_SHAPES,
    VARLEN_SEQLENS,
    ZERO_LENGTH_QUERY_PADDING_ROWS,
    ZERO_LENGTH_QUERY_SEQLENS,
    cu_seqlens_of,
    cuda_seed,
    near_maximum_lengths,
    uniform_lengths,
    window_for,
)


def hint_variants(lengths_q, lengths_k):
    return [{}, dict(max_seqlen_q=int(lengths_q.max()), max_seqlen_k=int(lengths_k.max()))]


def launch_forward_backward(q, k, v, dout, lse_shape, window, **varlen):
    window_arguments = dict(
        window_size_left=window[0],
        window_size_right=window[1],
        stream=torch.cuda.current_stream().cuda_stream,
    )
    out = torch.full_like(q, float("nan"))
    lse = torch.full(lse_shape, float("nan"), device="cuda", dtype=torch.float32)
    cu_seqlens = varlen.get("cu_seqlens_q")
    cu_seqlens_shape = None if cu_seqlens is None else tuple(cu_seqlens.shape)
    hints = {name: varlen[name] for name in ("max_seqlen_q", "max_seqlen_k") if name in varlen}
    forward_workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in forward_impl.compute_scratch_shapes(
            tuple(q.shape), k_shape=tuple(k.shape), cu_seqlens_shape=cu_seqlens_shape, **hints
        )
    )
    forward_impl.forward(q, k, v, out, lse, forward_workspace, **window_arguments, **varlen)
    dq, dk, dv = (torch.full_like(tensor, float("nan")) for tensor in (q, k, v))
    workspace = tuple(
        torch.empty(shape, device="cuda", dtype=torch.float32)
        for shape in backward_impl.compute_scratch_shapes(
            tuple(q.shape), tuple(lse.shape), tuple(k.shape), *window, cu_seqlens_shape=cu_seqlens_shape, **hints
        )
    )
    backward_impl.backward(q, k, v, out, dout, lse, dq, dk, dv, workspace, **window_arguments, **varlen)
    return out, lse, dq, dk, dv


def batched_inputs(batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim):
    q = torch.randn(batch, seqlen_q, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v, torch.randn_like(q)


def varlen_inputs(total_q, total_k, num_head, num_head_kv, head_dim):
    q = torch.randn(total_q, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_k, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total_k, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v, torch.randn_like(q)


def check_batched(q, k, v, dout, window, causal=False):
    batch, seqlen_q, num_head, _ = q.shape
    out, lse, *gradients = launch_forward_backward(q, k, v, dout, (batch, num_head, seqlen_q), window)
    queue_attention_checks([(out, lse)], [gradients], attention_ref, q, k, v, dout, causal=causal, window_size=window)


def check_varlen(q, k, v, dout, cu_q, cu_k, causal, window, variants, total_q=None):
    total_q = q.shape[0] if total_q is None else total_q
    valid = slice(0, total_q)
    forward_results, gradient_results = [], []
    for hints in variants:
        with case(variant=len(forward_results)):
            out, lse, dq, dk, dv = launch_forward_backward(
                q, k, v, dout, (q.shape[1], q.shape[0]), window,
                cu_seqlens_q=cu_q.cuda(non_blocking=True), cu_seqlens_k=cu_k.cuda(non_blocking=True), **hints,
            )
            forward_results.append((out[valid], lse[:, valid]))
            gradient_results.append((dq[valid], dk, dv))
            if total_q < q.shape[0]:
                queue_requirement("out rows past the last boundary", torch.isnan(out[total_q:]).all())
                queue_requirement("lse rows past the last boundary", torch.isnan(lse[:, total_q:]).all())
                queue_requirement("dq rows past the last boundary", torch.isnan(dq[total_q:]).all())
    queue_attention_checks(
        forward_results, gradient_results, varlen_attention_ref, q[valid], k, v, dout[valid], cu_q, cu_k, causal, window
    )


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_batched_forward_and_backward(head_dim):
    for layout, (window_name, seqlen_q, seqlen_k) in itertools.product(HEAD_LAYOUTS, BATCHED_WINDOW_SEQLENS):
        with case(layout=layout, window=window_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k):
            cuda_seed(0)
            batch = 9 if seqlen_k <= 2048 else 2
            q, k, v, dout = batched_inputs(batch, seqlen_q, seqlen_k, *HEAD_LAYOUTS[layout], head_dim)
            check_batched(q, k, v, dout, *window_for(window_name, seqlen_k))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_causal_large_query_groups(head_dim):
    head_counts = [(16, 1), (32, 1), (32, 2)]
    shapes = [(8, 256), (4, 1024), (2, 2048)]
    for (num_head, num_head_kv), (batch, seqlen) in itertools.product(head_counts, shapes):
        with case(num_head=num_head, num_head_kv=num_head_kv, batch=batch, seqlen=seqlen):
            cuda_seed(0)
            q, k, v, dout = batched_inputs(batch, seqlen, seqlen, num_head, num_head_kv, head_dim)
            check_batched(q, k, v, dout, *window_for("causal", seqlen))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_gqa_group_shapes(head_dim):
    for window_name, (batch, num_head, num_head_kv, seqlen) in itertools.product(
        ["dense", "causal"], GQA_QHEAD_PER_KVHEAD_SHAPES
    ):
        with case(window=window_name, batch=batch, num_head=num_head, num_head_kv=num_head_kv, seqlen=seqlen):
            cuda_seed(0)
            q, k, v, dout = batched_inputs(batch, seqlen, seqlen, num_head, num_head_kv, head_dim)
            check_batched(q, k, v, dout, *window_for(window_name, seqlen))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_packed_gqa_at_every_group_size(monkeypatch, restrict_candidates, head_dim):
    pack_divides = specialization.pack_divides
    monkeypatch.setattr(specialization, "pack_pays", lambda info, facts, arch: pack_divides(info, facts))
    for group, window_name, (seqlen_q, seqlen_k) in itertools.product(
        PACKED_GQA_QHEAD_PER_KVHEAD, ["dense", "causal", "local"], SHORT_QUERY_SEQLENS
    ):
        with case(group=group, window=window_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k):
            window, causal = window_for(window_name, seqlen_k)
            results = []
            for variants in (("_batched_gqa",), ()):
                restrict_candidates(variants)
                cuda_seed(seqlen_q + seqlen_k + head_dim + group)
                q, k, v, dout = batched_inputs(3, seqlen_q, seqlen_k, 2 * group, 2, head_dim)
                results.append(launch_forward_backward(q, k, v, dout, (3, 2 * group, seqlen_q), window))
            (out, lse, *gradients), dense = results
            queue_attention_checks(
                [(out, lse)], [gradients], attention_ref, q, k, v, dout, causal=causal, window_size=window
            )
            for name, packed_tensor, dense_tensor in zip(("out", "lse"), (out, lse), dense):
                equal = (packed_tensor == dense_tensor).all()
                queue_requirement(f"{name} of the gqa instance equals the dense instance", equal)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_forward_and_backward(head_dim):
    for layout, window_name, (seqlen_q, seqlen_k) in itertools.product(
        HEAD_LAYOUTS, ["dense", "causal", "local"], VARLEN_SEQLENS
    ):
        with case(layout=layout, window=window_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k):
            cuda_seed(seqlen_q + seqlen_k + head_dim)
            generator = torch.Generator().manual_seed(seqlen_q + seqlen_k)
            batch = 49 if seqlen_q <= 1024 else 7
            lengths_q = near_maximum_lengths(batch, seqlen_q, generator)
            lengths_k = near_maximum_lengths(batch, seqlen_k, generator, zero_lengths=True)
            cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
            q, k, v, dout = varlen_inputs(int(lengths_q.sum()), int(lengths_k.sum()), *HEAD_LAYOUTS[layout], head_dim)
            window, causal = window_for(window_name, seqlen_k)
            check_varlen(q, k, v, dout, cu_q, cu_k, causal, window, hint_variants(lengths_q, lengths_k))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_zero_length_queries(head_dim):
    for layout, window_name, (seqlen_q, seqlen_k) in itertools.product(
        HEAD_LAYOUTS, ["dense", "causal", "local"], ZERO_LENGTH_QUERY_SEQLENS
    ):
        with case(layout=layout, window=window_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k):
            cuda_seed(seqlen_q + seqlen_k + head_dim)
            generator = torch.Generator().manual_seed(seqlen_q + seqlen_k)
            batch = 49 if seqlen_q <= 1024 else 7
            lengths_q = near_maximum_lengths(batch, seqlen_q, generator, zero_lengths=True).flip(0)
            lengths_k = near_maximum_lengths(batch, seqlen_k, generator, zero_lengths=True)
            cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
            total_q = int(lengths_q.sum())
            q, k, v, dout = varlen_inputs(
                total_q + ZERO_LENGTH_QUERY_PADDING_ROWS, int(lengths_k.sum()), *HEAD_LAYOUTS[layout], head_dim
            )
            window, causal = window_for(window_name, seqlen_k)
            variants = hint_variants(lengths_q, lengths_k)
            check_varlen(q, k, v, dout, cu_q, cu_k, causal, window, variants, total_q)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_small_varlen_sequences(head_dim):
    for (num_head, num_head_kv), (batch, minimum, maximum), window_name in itertools.product(
        SMALL_VARLEN_HEAD_COUNTS, SMALL_VARLEN_SHAPES, ["dense", "causal"]
    ):
        with case(
            num_head=num_head, num_head_kv=num_head_kv, batch=batch, minimum=minimum, maximum=maximum, window=window_name
        ):
            cuda_seed(0)
            lengths = uniform_lengths(batch, minimum, maximum, torch.Generator().manual_seed(0))
            cu = cu_seqlens_of(lengths)
            total = int(lengths.sum())
            q, k, v, dout = varlen_inputs(total, total, num_head, num_head_kv, head_dim)
            window, causal = window_for(window_name, maximum)
            check_varlen(q, k, v, dout, cu, cu, causal, window, hint_variants(lengths, lengths))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_narrow_windows_at_long_lengths(head_dim):
    for window, seqlen in itertools.product(NARROW_WINDOWS, LONG_SEQLENS):
        with case(window=window, seqlen=seqlen):
            cuda_seed(seqlen)
            q, k, v, dout = batched_inputs(2, seqlen, seqlen, *HEAD_LAYOUTS["gqa"], head_dim)
            check_batched(q, k, v, dout, window)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_narrow_windows_at_long_lengths(head_dim):
    for window, seqlen in itertools.product(NARROW_WINDOWS, LONG_SEQLENS):
        with case(window=window, seqlen=seqlen):
            cuda_seed(seqlen)
            generator = torch.Generator().manual_seed(seqlen)
            lengths_q = near_maximum_lengths(2, seqlen, generator)
            lengths_k = near_maximum_lengths(2, seqlen, generator)
            cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
            q, k, v, dout = varlen_inputs(int(lengths_q.sum()), int(lengths_k.sum()), *HEAD_LAYOUTS["gqa"], head_dim)
            check_varlen(q, k, v, dout, cu_q, cu_k, False, window, [{}])


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_leaves_rows_past_the_last_boundary_unwritten(head_dim):
    for layout, padding_rows in itertools.product(["mha", "gqa"], PADDING_ROWS):
        with case(layout=layout, padding_rows=padding_rows):
            cuda_seed(head_dim + padding_rows)
            generator = torch.Generator().manual_seed(padding_rows)
            lengths_q, lengths_k = near_maximum_lengths(5, 200, generator), near_maximum_lengths(5, 200, generator)
            cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
            total_q, total_k = int(lengths_q.sum()), int(lengths_k.sum())
            q, k, v, dout = varlen_inputs(total_q + padding_rows, total_k, *HEAD_LAYOUTS[layout], head_dim)
            window, causal = window_for("causal", total_k)
            check_varlen(q, k, v, dout, cu_q, cu_k, causal, window, [{}], total_q)
