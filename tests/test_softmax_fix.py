import itertools
import math
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

from checks import case, queue, queue_requirement
from reference import (
    FULLY_MASKED_LSE,
    StableAttention,
    inter_sequence_visible,
    stable_attention_f64,
    varlen_visible,
)
from shapes import HEAD_DIMS, cu_seqlens_of, cuda_seed
from test_inter_sequence import CAUSAL, DENSE, build_layout, layout_tensors, sample_layout
from zigfa import inter_sequence, runtime, specialization
from zigfa.backward import HEAD_SPLIT_MINIMUM_GROUP, SM80_HEAD_SPLIT_MINIMUM_SEQLEN, backward_variant
from zigfa.forward import SPLIT_KV_KEYS_PER_QUERY, SPLIT_KV_MIN_BLOCKS, forward_variant, split_kv
from zigfa.specialization import SCHEDULER_METADATA_MIN_BATCH
from zigfa.torch import flash_attn_func, flash_attn_inter_sequence_func, flash_attn_varlen_func

DEVICE_ID = runtime.current_device().device_id
INTER_SEQUENCE_FORWARD_HEAD_DIMS = inter_sequence.forward_head_dims(DEVICE_ID)
INTER_SEQUENCE_HEAD_DIMS = inter_sequence.head_dims(DEVICE_ID)
PERSISTENT_HEAD_DIMS = inter_sequence.persistent_head_dims(DEVICE_ID)
EXP2_ERROR = 2.0**-13 if runtime.arch_name(DEVICE_ID) == "sm100" else 0.0

PUSH = 8.0
QUERY_SPREAD = 0.25
ZERO_MAXIMUM_MARGINS = (17.0, 80.0)
PEAK_MARGINS = (2.0, 60.0)
PEAKED_FRACTION = 0.75
# ex2.approx.ftz and the .ftz products flush results below FLUSH_LIMIT = 2**-126 (e**-87.3) to zero. The
# checked margins keep the terms of a row above that limit. The bounds of out and dv allow FLUSHED_TERM per
# summand: a flushed term times a row of v or dout, whose norms stay below 256. The bounds of dq and dk take
# their allowance from `flush_allowances`.
FLUSH_MARGIN = 84.0
RECOVERABLE_MARGIN = 76.0
FLUSH_LIMIT = 2.0**-126
FLUSHED_TERM = 256 * FLUSH_LIMIT
# Rounding a score z to FP32 changes exp(z) by up to about |z| * 2**-24 relative. The lse of a zero-maximum
# row is about the sum of the exponentials of its other keys, so its bound allows LSE_UNIT = 8 * 2**-24 per
# unit of the mean |z| of those keys, weighted by their exponentials, plus 8 * LSE_UNIT.
LSE_UNIT = 2.0**-21
# The gradient bounds allow four bfloat16 unit roundoffs (2**-8 each) per unit of the error weights of the
# reference.
GRADIENT_UNIT = 2.0**-6
SHIFT_UNIT = 2.0**-7
NORM_FLOOR_FACTOR = 2.5
KEY_SHIFT = 96.0
# Two keys with the score -(83 + 7/64) and dP of 1 and -(1 - 2**-7) get BF16 dS whose sum is 2**-127, an FP32
# subnormal number. A key with the score -(76 + 21/128) carries most of the remaining weight of its row.
PAIR_SCORE = (83.0, 7 / 64)
PAIR_VALUE = -(1 - 2.0**-7)
HEAVY_SCORE = (76.0, 21 / 128)
FAR_SCORE = (1000.0, 0.0)
ORTHOGONAL_SHIFT = 2.0**40
ROW_SUM_UNIT = 2.0**-20
OTHER_SCALE = 0.2
REFERENCE_ELEMENTS = 2**24
HEAD_AXES = StableAttention(
    out=2, lse=1, dq=2, dk=2, dv=2, top=1, margin=1, spread=1, visible=1, pivot=1, out_scale=2, dq_scale=2,
    dk_scale=2, dv_scale=2, corrected_keys=2,
)
GQA_HEADS = (4, 2)
WINDOWS = {"dense": (-1, -1), "causal": (-1, 0), "local": (64, 32)}


@dataclass
class Problem:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: torch.Tensor
    mask: torch.Tensor
    attend: Callable
    softmax_scale: Optional[float] = None
    exp2_error: float = EXP2_ERROR


def random_tensors(rows_q, rows_k, num_head, num_head_kv, head_dim, query_scale=1.0):
    q = query_scale * torch.randn(rows_q, num_head, head_dim, device="cuda", dtype=torch.float64)
    k, v = (torch.randn(rows_k, num_head_kv, head_dim, device="cuda", dtype=torch.float64) for _ in range(2))
    dout = torch.randn(rows_q, num_head, head_dim, device="cuda", dtype=torch.float64)
    return q, k, v, dout


def grid_keys(rows_k, num_head_kv, head_dim):
    return (2 * torch.randn(rows_k, num_head_kv, head_dim, device="cuda", dtype=torch.float64)).round().clamp(-8, 8) / 2


def with_zero_maximum(q, k, mask, zero_keys, scale):
    visible = mask.sum(-1, keepdim=True).clamp(min=1).double().log()
    low, high = ZERO_MAXIMUM_MARGINS[0], ZERO_MAXIMUM_MARGINS[1] - visible
    targets = low + (high - low) * torch.rand(q.shape[:2], device=q.device, dtype=torch.float64)
    q, k = QUERY_SPREAD * q, k.clone()
    q[..., -1] = -(targets + visible) / (PUSH * scale)
    k[..., -1] = PUSH
    k[zero_keys] = 0.0
    return q, k


def first_visible_keys(mask, num_head):
    return mask.int().argmax(-1, keepdim=True).expand(-1, num_head)


def random_visible_keys(mask, num_head):
    weights = (mask | ~mask.any(-1, keepdim=True)).double()
    return torch.multinomial(weights, num_head, replacement=True)


def peak_targets(mask, pivots):
    low, high = PEAK_MARGINS
    targets = low + (high - low) * torch.rand(pivots.shape, device=pivots.device, dtype=torch.float64)
    peaked = (torch.rand(pivots.shape, device=pivots.device) < PEAKED_FRACTION) & mask.gather(1, pivots)
    return targets.masked_fill(~peaked, float("nan"))


def with_peaks(q, k, mask, pivots, targets, scale):
    groups = q.shape[1] // k.shape[1]
    keys = k.repeat_interleave(groups, dim=1)
    direction = keys.gather(0, pivots[..., None].expand(-1, -1, keys.shape[-1]))
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    base = scale * torch.einsum("qhd,khd->qhk", q, keys)
    slope = scale * torch.einsum("qhd,khd->qhk", direction, keys)
    others = mask[:, None, :] & (torch.arange(keys.shape[0], device=q.device) != pivots[..., None])
    low, high = torch.zeros_like(targets), torch.full_like(targets, 4096.0)
    for _ in range(64):
        middle = (low + high) / 2
        z = base + middle[..., None] * slope
        margin = z.gather(-1, pivots[..., None])[..., 0] - z.masked_fill(~others, float("-inf")).logsumexp(-1)
        below = margin < targets
        low, high = torch.where(below, middle, low), torch.where(below, high, middle)
    return q + torch.where(targets.isnan(), 0.0, high)[..., None] * direction


def bf16_tensors(*tensors):
    return tuple(tensor.bfloat16() for tensor in tensors)


def uniform_cu_seqlens(batch, seqlen):
    return cu_seqlens_of(torch.full((batch,), seqlen, dtype=torch.long))


def batched_attention(batch, window, softmax_scale=None):
    def attend(q, k, v):
        out, lse = flash_attn_func(
            *(x.unflatten(0, (batch, -1)) for x in (q, k, v)), softmax_scale=softmax_scale, causal=window[1] == 0,
            window_size=window, return_attn_probs=True,
        )
        return out.flatten(0, 1), lse.transpose(0, 1).flatten(1)

    return attend


def batched_mask(batch, seqlen_q, seqlen_k, window):
    return varlen_visible(uniform_cu_seqlens(batch, seqlen_q), uniform_cu_seqlens(batch, seqlen_k), window).cuda()


def varlen_attention(cu_q, cu_k, window, softmax_scale=None):
    lengths_q, lengths_k = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
    arguments = (cu_q.cuda(), cu_k.cuda(), max(int(lengths_q.max()), 1), max(int(lengths_k.max()), 1))

    def attend(q, k, v):
        return flash_attn_varlen_func(
            q, k, v, *arguments, softmax_scale=softmax_scale, causal=window[1] == 0, window_size=window,
            return_attn_probs=True,
        )

    return attend


def inter_sequence_problem(layout, num_head, num_head_kv, head_dim):
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    arguments = (cu_q, cu_k, max(layout.lengths_q), max(layout.lengths_k), bits)

    def attend(q, k, v):
        return flash_attn_inter_sequence_func(
            q, k, v, *arguments, window_sizes=window_sizes, return_attn_probs=True
        )

    mask = inter_sequence_visible(cu_q, cu_k, layout.visible, layout.windows).cuda()
    tensors = random_tensors(sum(layout.lengths_q), sum(layout.lengths_k), num_head, num_head_kv, head_dim)
    return tensors, mask, attend, cu_k.cpu().long()


def reference_of(problem):
    q, k, v, dout = (x[None] for x in (problem.q, problem.k, problem.v, problem.dout))
    groups = q.shape[2] // k.shape[2]
    chunk = max(1, REFERENCE_ELEMENTS // (groups * q.shape[1] * max(k.shape[1], 1)))
    parts = []
    for start in range(0, k.shape[2], chunk):
        heads_kv, heads = slice(start, start + chunk), slice(start * groups, (start + chunk) * groups)
        parts.append(
            stable_attention_f64(
                q[:, :, heads], k[:, :, heads_kv], v[:, :, heads_kv], dout[:, :, heads], problem.mask[None],
                problem.softmax_scale,
            )
        )
    return StableAttention(*(torch.cat(fields, axis)[0] for fields, axis in zip(zip(*parts), HEAD_AXES)))


def run(problem, gradients=True):
    leaves = [x.detach().clone().requires_grad_(gradients) for x in (problem.q, problem.k, problem.v)]
    out, lse = problem.attend(*leaves)
    if not gradients:
        return out, lse, torch.zeros_like(out), None, None
    return out.detach(), lse, *torch.autograd.grad(out, leaves, problem.dout)


def queue_worst(name, ratio, selected, required=False):
    if required:
        queue_requirement(f"{name}: the inputs exercise this check", selected.any())
    worst = torch.where(selected, ratio, 0.0).amax()
    queue("bound", f"{name}: worst error / allowed error", torch.stack([worst, torch.ones_like(worst)]))


def bf16_spacing(x):
    _, exponent = torch.frexp(x)
    return torch.where(x == 0, 0.0, torch.ldexp(torch.ones_like(x), exponent - 8))


def zero_maximum_rows(ref):
    margin = ref.margin.nan_to_num(nan=-1.0)
    return (ref.top == 0) & (margin >= ZERO_MAXIMUM_MARGINS[0]) & (margin <= FLUSH_MARGIN - ref.visible.log())


def queue_lse_checks(lse, ref, exp2_error, zero_maximum=False):
    error = (lse.double() - ref.lse).abs()
    finite = torch.isfinite(ref.lse)
    allowed = ref.lse * (LSE_UNIT * (8 + ref.spread) + exp2_error)
    queue_worst("lse of rows with a zero maximum", error / allowed, zero_maximum_rows(ref), zero_maximum)
    general = 2 * LSE_UNIT * (1 + ref.top.abs() + ref.lse.abs()) + exp2_error
    queue_worst("lse", error / general, finite)
    queue_requirement("lse is -inf exactly on the rows without keys", ((lse == FULLY_MASKED_LSE) == ~finite).all())


def checked_rows(ref):
    margin = ref.margin.nan_to_num(nan=math.inf)
    return (torch.isfinite(ref.lse) & (margin <= RECOVERABLE_MARGIN - ref.visible.log())).T


def queue_output_checks(out, dq, ref):
    empty = ~torch.isfinite(ref.lse).T
    ratio = (out.double() - ref.out).norm(dim=-1) / (GRADIENT_UNIT * ref.out_scale + FLUSHED_TERM * ref.visible.T)
    queue_worst("out", ratio, ~empty)
    queue_requirement("out and dq are zero on the rows without keys", (out[empty] == 0).all() & (dq[empty] == 0).all())


def queue_unseen_key_checks(dk, dv, seers):
    unseen = seers == 0
    zeros = (dk[unseen] == 0).all() & (dv[unseen] == 0).all()
    queue_requirement("dk and dv are zero for the keys that no row sees", zeros)


def flush_allowances(problem):
    """Returns, for each row of dq and each row of dk, the largest change that flushing can make: a flushed P_ij,
    or a flushed product with P_ij, takes at most FLUSH_LIMIT * max(1, |dP_ij - D_i|) from dS_ij, and
    |dP_ij - D_i| <= |dout_i| (|v_j| + |out_i|) <= 2 |dout_i| max_j |v_j|."""
    q, k, v, dout = (x.double() for x in (problem.q, problem.k, problem.v, problem.dout))
    groups = q.shape[1] // k.shape[1]
    scale = q.shape[-1] ** -0.5 if problem.softmax_scale is None else problem.softmax_scale
    largest_v = v.norm(dim=-1).amax(0).repeat_interleave(groups)
    ds_limit = FLUSH_LIMIT * scale * (2 * dout.norm(dim=-1) * largest_v).clamp(min=1)
    mask = problem.mask.to(q.device, torch.float64)
    dq_allowance = ds_limit * (mask @ k.norm(dim=-1).repeat_interleave(groups, dim=1))
    dk_allowance = (mask.T @ (ds_limit * q.norm(dim=-1))).unflatten(1, (-1, groups)).sum(2)
    return dq_allowance, dk_allowance


def queue_gradient_checks(dq, dk, dv, ref, seers, flushed, pivots_required=False):
    rows, pivot = checked_rows(ref), ref.pivot.T
    dq_flushed, dk_flushed = flushed
    dq_ratio = (dq.double() - ref.dq).norm(dim=-1) / (GRADIENT_UNIT * ref.dq_scale + dq_flushed)
    queue_worst("dq of rows with a pivot", dq_ratio, rows & pivot, pivots_required)
    queue_worst("dq of rows without a pivot", dq_ratio, rows & ~pivot)
    half_units = torch.where(ref.corrected_keys, 0.5 * bf16_spacing(ref.dk).norm(dim=-1), 0.0)
    dk_ratio = (dk.double() - ref.dk).norm(dim=-1) / (GRADIENT_UNIT * ref.dk_scale + half_units + dk_flushed)
    seen = ref.dk_scale > 0
    queue_worst("dk of pivot keys", dk_ratio, seen & ref.corrected_keys, pivots_required)
    queue_worst("dk of other keys", dk_ratio, seen & ~ref.corrected_keys)
    flushed = FLUSHED_TERM * seers[:, None] * (dq.shape[1] // dk.shape[1])
    dv_ratio = (dv.double() - ref.dv).norm(dim=-1) / (GRADIENT_UNIT * ref.dv_scale + flushed)
    queue_worst("dv", dv_ratio, ref.dv_scale > 0)


def queue_norm_checks(dq, dk, dv, ref):
    for name, gradient, exact in (("dq", dq, ref.dq), ("dk", dk, ref.dk), ("dv", dv, ref.dv)):
        allowed = NORM_FLOOR_FACTOR * (exact.bfloat16().double() - exact).norm()
        if name == "dk":
            allowed += 0.5 * bf16_spacing(exact)[ref.corrected_keys].norm()
        error = (gradient.double() - exact).norm()
        queue("bound", f"{name} norm-wise error against the bfloat16 rounding of the exact {name}",
              torch.stack([error, allowed]))


def check(problem, gradients=True, accuracy=True, zero_maximum=False, pivots_required=False):
    ref = reference_of(problem)
    out, lse, dq, dk, dv = run(problem, gradients)
    queue_lse_checks(lse, ref, problem.exp2_error, zero_maximum)
    queue_output_checks(out, dq, ref)
    if gradients:
        queue_unseen_key_checks(dk, dv, problem.mask.sum(0))
    if gradients and accuracy:
        queue_gradient_checks(dq, dk, dv, ref, problem.mask.sum(0), flush_allowances(problem), pivots_required)
    return ref, (out, lse, dq, dk, dv)


def zero_maximum_problem(tensors, mask, attend, zero_keys, softmax_scale=None, exp2_error=EXP2_ERROR):
    q, k, v, dout = tensors
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    q, k = with_zero_maximum(q, k, mask, zero_keys, scale)
    return Problem(*bf16_tensors(q, k, v, dout), mask, attend, softmax_scale, exp2_error)


def peaked_problem(tensors, mask, attend, pivot_choice, softmax_scale=None, exp2_error=EXP2_ERROR):
    q, k, v, dout = tensors
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    pivots = pivot_choice(mask, q.shape[1])
    q = with_peaks(q, k, mask, pivots, peak_targets(mask, pivots), scale)
    return Problem(*bf16_tensors(q, k, v, dout), mask, attend, softmax_scale, exp2_error)


def zero_keys_of_batch(batch, seqlen_q, seqlen_k, num_head_kv, window):
    late = max(64, (seqlen_k - seqlen_q) // 64 * 64) if window[1] == 0 else seqlen_k // 2
    if window[0] >= 0:
        period = window[0] + max(window[1], 0) + 1
        positions = [list(range(0, seqlen_k, period))] * batch
    else:
        positions = [[0] if sequence % 2 == 0 else [min(late, seqlen_k - 1)] for sequence in range(batch)]
    zero_keys = torch.zeros(batch * seqlen_k, num_head_kv, dtype=torch.bool)
    for sequence, keys in enumerate(positions):
        for key in keys:
            zero_keys[sequence * seqlen_k + key] = True
    zero_keys[(batch - 1) * seqlen_k + positions[-1][0] + 1, 0] = True
    return zero_keys.cuda()


def chosen_forward(q_shape, k_shape, window, cu_seqlens_shape=None, max_seqlen_q=None, max_seqlen_k=None):
    return forward_variant(DEVICE_ID, q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k, window)


def instance_info(direction, head_dim, variant):
    return runtime.manifest()[1][runtime.instance_name(DEVICE_ID, direction, head_dim, variant)]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("window_name", WINDOWS)
@pytest.mark.parametrize("seqlen_q, seqlen_k", [(200, 600), (300, 520)])
def test_zero_maximum_rows_keep_their_lse(monkeypatch, restrict_candidates, head_dim, window_name, seqlen_q, seqlen_k):
    pack_divides = specialization.pack_divides
    monkeypatch.setattr(specialization, "pack_pays", lambda info, facts, arch: pack_divides(info, facts))
    window, batch, (num_head, num_head_kv) = WINDOWS[window_name], 2, GQA_HEADS
    q_shape, k_shape = (batch, seqlen_q, num_head, head_dim), (batch, seqlen_k, num_head_kv, head_dim)
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    zero_keys = zero_keys_of_batch(batch, seqlen_q, seqlen_k, num_head_kv, window)
    served = set()
    for variant in runtime.variant_names(DEVICE_ID, "fwd", head_dim):
        restrict_candidates((variant,))
        if chosen_forward(q_shape, k_shape, window) != variant or variant in served:
            continue
        served.add(variant)
        with case(forward=variant or "unrestricted"):
            cuda_seed(head_dim + seqlen_q)
            tensors = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim)
            problem = zero_maximum_problem(tensors, mask, batched_attention(batch, window), zero_keys)
            check(problem, zero_maximum=True, pivots_required=True)


SPLIT_KV_SHAPES = {
    "gqa_decode": ((1, 1, 4096, 8, 2), (-1, 0)),
    "mha_decode": ((2, 1, 2048, 4, 4), (-1, -1)),
    "windowed_extend": ((1, 16, 2048, 8, 2), (512, 0)),
    "uneven_lengths": ((1, 100, 3001, 4, 1), (-1, 0)),
}


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("shape_name", SPLIT_KV_SHAPES)
def test_zero_maximum_rows_keep_their_lse_across_key_splits(head_dim, shape_name):
    (batch, seqlen_q, seqlen_k, num_head, num_head_kv), window = SPLIT_KV_SHAPES[shape_name]
    q_shape, k_shape = (batch, seqlen_q, num_head, head_dim), (batch, seqlen_k, num_head_kv, head_dim)
    if not any(info.split_kv for _, info in specialization.declared_variants(DEVICE_ID, "fwd", head_dim)):
        pytest.skip("no forward instance splits the keys on this GPU")
    assert "split_kv" in chosen_forward(q_shape, k_shape, window)
    cuda_seed(head_dim + seqlen_k)
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    zero_keys = torch.zeros(batch * seqlen_k, num_head_kv, dtype=torch.bool, device="cuda")
    reach = window[0] // 2 if window[0] >= 0 else seqlen_k // 3
    zero_keys[seqlen_k - seqlen_q - reach :: seqlen_k] = True
    tensors = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim)
    problem = zero_maximum_problem(tensors, mask, batched_attention(batch, window), zero_keys)
    check(problem, zero_maximum=True, pivots_required=True)


VARLEN_CASES = {"short": ((1, 160), None), "long_scaled": ((1, 700), OTHER_SCALE)}


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("window_name", ["dense", "causal"])
@pytest.mark.parametrize("case_name", VARLEN_CASES)
def test_zero_maximum_and_peaked_varlen_rows(head_dim, window_name, case_name):
    window, (lengths, softmax_scale) = WINDOWS[window_name], VARLEN_CASES[case_name]
    generator = torch.Generator().manual_seed(head_dim + lengths[1])
    lengths_q = torch.randint(*lengths, (6,), generator=generator)
    lengths_k = lengths_q + torch.randint(0, 64, (6,), generator=generator)
    cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
    mask = varlen_visible(cu_q, cu_k, window).cuda()
    attend = varlen_attention(cu_q, cu_k, window, softmax_scale)
    num_head, num_head_kv = GQA_HEADS
    rows_q, rows_k = int(cu_q[-1]), int(cu_k[-1])
    with case(forward=chosen_forward((rows_q, num_head, head_dim), (rows_k, num_head_kv, head_dim), window,
                                     (len(cu_q),), int(lengths_q.max()), int(lengths_k.max())) or "unrestricted"):
        cuda_seed(head_dim)
        zero_keys = torch.zeros(rows_k, num_head_kv, dtype=torch.bool, device="cuda")
        zero_keys[cu_k[:-1].long().cuda()] = True
        tensors = random_tensors(rows_q, rows_k, num_head, num_head_kv, head_dim)
        with case(rows="zero maximum"):
            problem = zero_maximum_problem(tensors, mask, attend, zero_keys, softmax_scale)
            check(problem, zero_maximum=True, pivots_required=True)
        for name, pivot_choice in (("sink", first_visible_keys), ("peaked", random_visible_keys)):
            with case(rows=name):
                check(peaked_problem(tensors, mask, attend, pivot_choice, softmax_scale), pivots_required=True)


@dataclass(frozen=True)
class VariantCase:
    varlen: bool
    batch: int
    seqlen_q: int
    seqlen_k: int
    num_head: int
    num_head_kv: int
    window: tuple

    def dispatch_arguments(self, head_dim):
        """Returns the q shape, k shape, cu_seqlens shape and maximum lengths that the dispatch of this case sees."""
        if self.varlen:
            return (
                (self.batch * self.seqlen_q, self.num_head, head_dim),
                (self.batch * self.seqlen_k, self.num_head_kv, head_dim),
                (self.batch + 1,), self.seqlen_q, self.seqlen_k,
            )
        return (
            (self.batch, self.seqlen_q, self.num_head, head_dim),
            (self.batch, self.seqlen_k, self.num_head_kv, head_dim),
            None, None, None,
        )


def variant_cases(direction, head_dim):
    """Returns cases whose lengths, windows and batches come from the limits of the variants declared for this
    direction and head dim and from the thresholds of the dispatch: smallest first, after the cases whose rows and
    keys fill at least one query tile, which give the checks rows with margins to measure."""
    infos = [info for _, info in specialization.declared_variants(DEVICE_ID, direction, head_dim)]
    tile = min(info.block_m for info in infos)
    lengths = {1, tile, *SM80_HEAD_SPLIT_MINIMUM_SEQLEN.get(head_dim, ())}
    for info in infos:
        lengths |= {info.min_seqlen_q, info.max_seqlen_q, info.max_seqlen_k}
    lengths.discard(0)
    split_keys = 2 * SPLIT_KV_MIN_BLOCKS * max(info.block_n for info in infos)
    windows = {specialization.UNBOUNDED, (-1, 0), (tile, tile), (tile, 0)}
    for info in infos:
        if info.max_window_size:
            windows |= {(info.max_window_size // 4, info.max_window_size // 4), (info.max_window_size // 2, 0)}
    heads = ((4, 4), (8, 2), (HEAD_SPLIT_MINIMUM_GROUP, 1))
    cases = set()
    for seqlen_q in lengths:
        keys = {seqlen_q, max(SPLIT_KV_KEYS_PER_QUERY * seqlen_q, split_keys)}
        keys |= {info.max_seqlen_k for info in infos if info.max_seqlen_k >= seqlen_q}
        for seqlen_k, varlen, batch, (num_head, num_head_kv), window in itertools.product(
            keys, (False, True), (2, SCHEDULER_METADATA_MIN_BATCH), heads, windows
        ):
            cases.add(VariantCase(varlen, batch, seqlen_q, seqlen_k, num_head, num_head_kv, window))
    return sorted(cases, key=lambda c: (
        min(c.seqlen_q, c.seqlen_k) < tile, c.batch * c.seqlen_q * c.seqlen_k * c.num_head, c.varlen, c.window
    ))


def selected_variant(direction, head_dim, variant_case):
    q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k = variant_case.dispatch_arguments(head_dim)
    if direction == "fwd":
        return chosen_forward(q_shape, k_shape, variant_case.window, cu_seqlens_shape, max_seqlen_q, max_seqlen_k)
    return backward_variant(DEVICE_ID, q_shape, k_shape, *variant_case.window, cu_seqlens_shape, max_seqlen_q,
                            max_seqlen_k)


def variant_problem_inputs(variant_case):
    """Returns the cu_seqlens of the keys, the mask and the attention function of a case."""
    window = variant_case.window
    cu_q = uniform_cu_seqlens(variant_case.batch, variant_case.seqlen_q)
    cu_k = uniform_cu_seqlens(variant_case.batch, variant_case.seqlen_k)
    mask = varlen_visible(cu_q, cu_k, window).cuda()
    if variant_case.varlen:
        return cu_k, mask, varlen_attention(cu_q, cu_k, window)
    return cu_k, mask, batched_attention(variant_case.batch, window)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("direction", ["fwd", "bwd"])
def test_each_variant_keeps_zero_maximum_and_peaked_rows_accurate(restrict_candidates, head_dim, direction):
    variants = [
        variant for variant in runtime.variant_names(DEVICE_ID, direction, head_dim)
        if variant and runtime.INTER_SEQUENCE not in variant
    ]
    cases = variant_cases(direction, head_dim)
    for variant in variants:
        restrict_candidates((variant,))
        variant_case = next((c for c in cases if selected_variant(direction, head_dim, c) == variant), None)
        queue_requirement(f"{direction} {variant}: a case built from the variant limits selects it",
                          torch.tensor(variant_case is not None, device="cuda"))
        if variant_case is None:
            continue
        rows_q = variant_case.batch * variant_case.seqlen_q
        rows_k = variant_case.batch * variant_case.seqlen_k
        cu_k, mask, attend = variant_problem_inputs(variant_case)
        with case(**{"forward" if direction == "fwd" else "backward": variant}):
            cuda_seed(head_dim + rows_q)
            tensors = random_tensors(rows_q, rows_k, variant_case.num_head, variant_case.num_head_kv, head_dim)
            if direction == "fwd":
                zero_keys = torch.zeros(rows_k, variant_case.num_head_kv, dtype=torch.bool, device="cuda")
                zero_keys[cu_k[:-1].long().cuda()] = True
                with case(rows="zero maximum"):
                    check(zero_maximum_problem(tensors, mask, attend, zero_keys), zero_maximum=True)
            with case(rows="sink"):
                check(peaked_problem(tensors, mask, attend, first_visible_keys), pivots_required=True)


def inter_sequence_layouts(head_dim):
    layouts = {"i2t": build_layout("i2t", head_dim), "i2t_mha": (build_layout("i2t", head_dim), GQA_HEADS[0])}
    if head_dim in PERSISTENT_HEAD_DIMS:
        instance = inter_sequence.persistent_instance(head_dim, 1)
        ctas = inter_sequence.resident_ctas(DEVICE_ID, instance)
        blocks = 1024 // max(inter_sequence.BLOCK_TABLE_ROWS)
        layouts["persistent_splits"] = (
            sample_layout([(37, 1024)], [CAUSAL, DENSE]), -(-(ctas + ctas // 4) // blocks)
        )
        seen = tuple((0, part) if part else (0,) for part in range(6))
        slices = sample_layout([(384, 1, 1, 3, 1, 2)], [DENSE] + [CAUSAL] * 5, seen)
        layouts["key_slices"] = (slices, -(-ctas // (384 // max(inter_sequence.BLOCK_TABLE_ROWS))))
    return {name: value if isinstance(value, tuple) else (value, None) for name, value in layouts.items()}


def first_keys_of_samples(layout, cu_k):
    first_parts = [query for query in range(layout.count) if layout.visible[query].nonzero()[0, 0] == query]
    return cu_k[first_parts]


@pytest.mark.parametrize("head_dim", INTER_SEQUENCE_FORWARD_HEAD_DIMS)
@pytest.mark.parametrize("layout_name", ["i2t", "i2t_mha", "persistent_splits", "key_slices"])
def test_zero_maximum_and_peaked_inter_sequence_rows(head_dim, layout_name):
    layouts = inter_sequence_layouts(head_dim)
    if layout_name not in layouts:
        pytest.skip("this layout exercises the persistent kernel only")
    layout, heads = layouts[layout_name]
    num_head, num_head_kv = (heads, heads) if heads else GQA_HEADS
    cuda_seed(head_dim)
    tensors, mask, attend, cu_k = inter_sequence_problem(layout, num_head, num_head_kv, head_dim)
    gradients = head_dim in INTER_SEQUENCE_HEAD_DIMS
    zero_keys = torch.zeros(tensors[1].shape[:2], dtype=torch.bool, device="cuda")
    zero_keys[first_keys_of_samples(layout, cu_k).cuda()] = True
    with case(rows="zero maximum"):
        problem = zero_maximum_problem(tensors, mask, attend, zero_keys)
        check(problem, gradients, zero_maximum=True, pivots_required=gradients)
    if gradients:
        for name, pivot_choice in (("sink", first_visible_keys), ("peaked", random_visible_keys)):
            with case(rows=name):
                check(peaked_problem(tensors, mask, attend, pivot_choice), pivots_required=True)


PEAKED_CASES = {"short": ((3, 128, 128), None), "scaled": ((2, 256, 320), OTHER_SCALE)}


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("window_name", ["dense", "causal"])
@pytest.mark.parametrize("case_name", PEAKED_CASES)
def test_peaked_rows_keep_accurate_gradients(head_dim, window_name, case_name):
    ((batch, seqlen_q, seqlen_k), softmax_scale), num_head, num_head_kv = PEAKED_CASES[case_name], *GQA_HEADS
    window = WINDOWS[window_name]
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    attend = batched_attention(batch, window, softmax_scale)
    for name, pivot_choice in (("sink", first_visible_keys), ("peaked", random_visible_keys)):
        with case(rows=name):
            cuda_seed(head_dim + seqlen_q + len(name))
            tensors = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim)
            check(peaked_problem(tensors, mask, attend, pivot_choice, softmax_scale), pivots_required=True)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_peaked_rows_with_the_head_split_backward(head_dim):
    dense_seqlen, causal_seqlen = SM80_HEAD_SPLIT_MINIMUM_SEQLEN.get(head_dim, (2048, 2048))
    window = WINDOWS["causal"] if causal_seqlen <= dense_seqlen else WINDOWS["dense"]
    seqlen, (num_head, num_head_kv) = min(dense_seqlen, causal_seqlen), (4, 1)
    q_shape, k_shape = (1, seqlen, num_head, head_dim), (1, seqlen, num_head_kv, head_dim)
    variant = backward_variant(DEVICE_ID, q_shape, k_shape, *window, None, None, None)
    if not instance_info("bwd", head_dim, variant).head_split:
        pytest.skip("no head-split backward serves this head dimension")
    cuda_seed(head_dim)
    mask = batched_mask(1, seqlen, seqlen, window)
    tensors = random_tensors(seqlen, seqlen, num_head, num_head_kv, head_dim)
    check(peaked_problem(tensors, mask, batched_attention(1, window), first_visible_keys), pivots_required=True)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_peaked_rows_with_the_unrestricted_backward(head_dim):
    # The log-sum-exp of this shape has 3 * 1999 elements, no multiple of 4, so the pivot workspace
    # must pad its rows to keep its second row 16-byte aligned.
    seqlen, num_head, window = 1999, 3, WINDOWS["causal"]
    shape = (1, seqlen, num_head, head_dim)
    if backward_variant(DEVICE_ID, shape, shape, *window, None, None, None):
        pytest.skip("a restricted backward serves this shape")
    cuda_seed(head_dim)
    mask = batched_mask(1, seqlen, seqlen, window)
    tensors = random_tensors(seqlen, seqlen, num_head, num_head, head_dim)
    check(peaked_problem(tensors, mask, batched_attention(1, window), first_visible_keys), pivots_required=True)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_query_gradient_ignores_a_common_key_shift(head_dim):
    batch, seqlen_q, seqlen_k, (num_head, num_head_kv), window = 2, 192, 256, GQA_HEADS, WINDOWS["causal"]
    cuda_seed(head_dim)
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    q, _, v, dout = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim)
    k = grid_keys(batch * seqlen_k, num_head_kv, head_dim)
    problem = peaked_problem((q, k, v, dout), mask, batched_attention(batch, window), first_visible_keys)
    ref, (_, _, dq, _, _) = check(problem, pivots_required=True)
    shifted = Problem(problem.q, problem.k + KEY_SHIFT, problem.v, problem.dout, mask, problem.attend)
    exact = (shifted.k.double() == problem.k.double() + KEY_SHIFT).all()
    queue_requirement("the shifted keys are exact in bfloat16", exact)
    _, _, shifted_dq, _, _ = run(shifted)
    flushed = flush_allowances(problem)[0] + flush_allowances(shifted)[0]
    allowed = SHIFT_UNIT * (ref.dq.norm(dim=-1) + ref.dq_scale / 8) + flushed
    ratio = (shifted_dq.double() - dq.double()).norm(dim=-1) / allowed
    queue_worst("dq change under a common key shift, rows with a pivot", ratio, checked_rows(ref) & ref.pivot.T, True)


def subnormal_pair_problem(head_dim, shift):
    batch, seqlen_q, seqlen_k = 2, 16, 130
    layouts = (
        {1: (PAIR_SCORE, 1.0), 2: (PAIR_SCORE, PAIR_VALUE)},
        {1: (HEAVY_SCORE, 1.0), 128: (PAIR_SCORE, 1.0), 129: (PAIR_SCORE, PAIR_VALUE)},
    )
    q = torch.zeros(batch * seqlen_q, 1, head_dim, device="cuda", dtype=torch.float64)
    q[..., :2] = 1.0
    k = torch.zeros(batch * seqlen_k, 1, head_dim, device="cuda", dtype=torch.float64)
    v = torch.zeros_like(k)
    k[..., 0] = -FAR_SCORE[0]
    for sequence, keys in enumerate(layouts):
        first = sequence * seqlen_k
        k[first, :, 0] = 0.0
        for key, ((whole, fraction), value) in keys.items():
            k[first + key, :, 0], k[first + key, :, 1], v[first + key, :, 0] = -whole, -fraction, value
    k[..., 2] = shift
    dout = torch.zeros_like(q)
    dout[..., 0] = 1.0
    window = WINDOWS["dense"]
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    return Problem(*bf16_tensors(q, k, v, dout), mask, batched_attention(batch, window, 1.0), 1.0)


def exact_ds(problem):
    q, k, v, dout = (x.double()[:, 0] for x in (problem.q, problem.k, problem.v, problem.dout))
    p = (problem.softmax_scale * q @ k.T).masked_fill(~problem.mask, float("-inf")).softmax(-1)
    dp = dout @ v.T
    return p * (dp - (p * dp).sum(-1, keepdim=True))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_query_gradient_ignores_a_key_shift_on_rows_with_subnormal_ds_sums(head_dim):
    problem = subnormal_pair_problem(head_dim, 0.0)
    shifted = subnormal_pair_problem(head_dim, ORTHOGONAL_SHIFT)
    queue_requirement("the shifted keys are exact in bfloat16", (shifted.k[..., 2] == ORTHOGONAL_SHIFT).all())
    ds = exact_ds(problem)
    pair = ds[0, 1:3].bfloat16().double().sum().abs()
    queue_requirement("the BF16 dS of the key pair sum to an FP32 subnormal number", (0 < pair) & (pair < 2.0**-126))
    _, _, dq, _, _ = run(problem)
    _, _, shifted_dq, _, _ = run(shifted)
    allowed = ROW_SUM_UNIT * ORTHOGONAL_SHIFT * problem.softmax_scale * ds.abs().sum(-1, keepdim=True)
    ratio = (shifted_dq.double() - dq.double()).norm(dim=-1) / allowed
    every_row = torch.ones_like(ratio, dtype=torch.bool)
    queue_worst("dq change under a key shift orthogonal to the queries", ratio, every_row)


# Every row of the first sequence of the subnormal pair problem gives its pivot key the BF16 dS -2**-127. Query
# columns that every key ignores give that key the dK terms -2**-133 and -2**-137 per row, which sum to the FP32
# subnormal numbers -2**-129 and -2**-133 over the 16 rows of the sequence.
PIVOT_TERM_COLUMNS = {3: 2.0**-6, 4: 2.0**-10}


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_pivot_key_gradient_keeps_subnormal_sums_of_terms(head_dim):
    pair = subnormal_pair_problem(head_dim, 0.0)
    q = pair.q.clone()
    columns = list(PIVOT_TERM_COLUMNS)
    q[..., columns] = q.new_tensor(list(PIVOT_TERM_COLUMNS.values()))
    problem = Problem(q, pair.k, pair.v, pair.dout, pair.mask, pair.attend, pair.softmax_scale)
    rows = slice(0, q.shape[0] // 2)
    row_sums = exact_ds(problem)[rows, 1:].bfloat16().double().sum(-1)
    expected = -problem.softmax_scale * (row_sums @ q[rows, 0, columns].double())
    queue_requirement("the dK terms of the pivot key sum to FP32 subnormal numbers", (expected.abs() < 2.0**-126).all())
    _, _, _, dk, _ = run(problem)
    exact = (dk[0, 0, columns].double() == expected.bfloat16().double()).all()
    queue_requirement("dk of the pivot key is the BF16 rounding of the sum of its terms", exact & (expected != 0).all())


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("query_scale", [1.0, 3.0])
def test_random_rows_keep_their_accuracy(head_dim, query_scale):
    batch, seqlen_q, seqlen_k, (num_head, num_head_kv), window = 2, 256, 384, GQA_HEADS, WINDOWS["causal"]
    cuda_seed(head_dim + int(query_scale))
    mask = batched_mask(batch, seqlen_q, seqlen_k, window)
    tensors = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim, query_scale)
    problem = Problem(*bf16_tensors(*tensors), mask, batched_attention(batch, window))
    ref, (_, _, dq, dk, dv) = check(problem)
    queue_norm_checks(dq, dk, dv, ref)


def test_rows_without_keys_keep_an_lse_of_minus_infinity_and_zero_gradients():
    head_dim, (num_head, num_head_kv) = HEAD_DIMS[0], GQA_HEADS
    cuda_seed(5)
    with case(layout="batched causal, more queries than keys"):
        batch, seqlen_q, seqlen_k, window = 2, 300, 160, WINDOWS["causal"]
        mask = batched_mask(batch, seqlen_q, seqlen_k, window)
        tensors = random_tensors(batch * seqlen_q, batch * seqlen_k, num_head, num_head_kv, head_dim)
        check(peaked_problem(tensors, mask, batched_attention(batch, window), first_visible_keys), accuracy=False)
    with case(layout="varlen with empty key sequences"):
        lengths_q, lengths_k = torch.tensor([40, 7, 0, 90, 33]), torch.tensor([50, 0, 12, 0, 70])
        cu_q, cu_k = cu_seqlens_of(lengths_q), cu_seqlens_of(lengths_k)
        mask = varlen_visible(cu_q, cu_k, WINDOWS["dense"]).cuda()
        tensors = random_tensors(int(cu_q[-1]), int(cu_k[-1]), num_head, num_head_kv, head_dim)
        attend = varlen_attention(cu_q, cu_k, WINDOWS["dense"])
        check(peaked_problem(tensors, mask, attend, first_visible_keys), accuracy=False)
    if INTER_SEQUENCE_HEAD_DIMS:
        with case(layout="inter-sequence rows that see no sequence"):
            layout = build_layout("i2t", 4)
            layout.visible[2] = False
            layout.visible[5, 5] = False
            tensors, mask, attend, _ = inter_sequence_problem(layout, *GQA_HEADS, INTER_SEQUENCE_HEAD_DIMS[0])
            check(peaked_problem(tensors, mask, attend, first_visible_keys), accuracy=False)


# The tied scores lie on a grid of 1/8 and the other scores on a grid of 1/2 at least 22 nats lower, so k holds every
# score and the shift exactly in bfloat16, and the FP32 product of q and k is exact. At a shift of 2**16 or 2**17, the
# FP32 product of a tied score and log2(e) rounds by up to 2**-8 or 2**-7, so the exponential of a tied key, which the
# P V product uses, differs from 1 by up to about 0.3% or 0.5%. Counting such a key as an exact 1 in the row sum moves
# O by up to about one bfloat16 unit, while a different order of the FP32 additions changes few elements of O.
TIE_SHIFTS = (2.0**16, 2.0**17)
TIE_SCORES = torch.arange(-64, -7, dtype=torch.float64) / 8
OTHER_SCORES = (60, 120)
TIED_KEYS = 4
# Lane c of a quad holds the columns 2c and 2c + 1 of each eight columns of an mma accumulator (PTX ISA, matrix
# fragments for mma.m16n8k16 with floating point type), so the gathered ties lie in lanes 1, 1, 3 and 2.
GATHERED_POSITIONS = (2, 3, 6, 13)
FAR_ROUNDING = 0.375
TIE_TOLERANCE = 0.05
SPLIT_ROWS, SPLIT_KEYS, SPLIT_HEADS = 16, 4096, 2
SLICED_HEAD_DIM, SLICED_SEQUENCES, SLICED_ROWS, SLICED_KEYS, SLICED_HEADS = 128, 16, 8, 1024, 2
# In a row whose largest score ties on the keys at TIED_LSE_POSITIONS and whose other keys score FAR_SCORE[0] below
# the shift, the forward rounds the base 2 log-sum-exp v, which it forms with the scale f32(log2(e)), to FP32 and then
# rounds the product of v and f32(ln(2)) to FP32. Each rounding costs up to half an FP32 unit. LOG2_SUM_SLACK bounds
# the other errors in v, which come from ex2 (2 units), from lg2 (2**-22 relative for inputs above 2; PTX ISA) and from
# the FP32 sums of the exponentials.
LOG2_SUM_SLACK = 2.0**-17
TIED_LSE_ROWS, TIED_LSE_KEYS, TIED_LSE_HEADS = 16, 320, 4
TIED_LSE_POSITIONS = (66, 67, 70, 200)
# A diffuse row has its maximum score on one key, scores up to 2 nats lower on eight keys of the same aligned group of
# 16 keys, which no key block splits, and scores 6 and 6.5 nats (more than 8 log2 units) lower on two other keys. A
# forward raises the offset of its exponentials whenever the maximum rises by more than rescale_threshold = 8 log2
# units, so in any block order it offsets the row by the FP32 product of the maximum and log2(e), which rounds by at
# least FAR_ROUNDING units. With the sum l' of the exponentials of the other keys relative to the maximum, an LSE that
# adds this rounding to log2(1 + l') is off by about the rounding times l' / (1 + l').
DIFFUSE_ROWS, DIFFUSE_KEYS, DIFFUSE_HEADS = 32, 320, 4
DIFFUSE_GROUPS = (64, 192)
DIFFUSE_TOP = 2
DIFFUSE_NEAR = (3, 5, 6, 9, 10, 12, 13, 15)
DIFFUSE_GAP_STEPS = 16
DIFFUSE_FAR = {10: 6.0, 300: 6.5}


def far_rounding_tie_scores(shift):
    log2_e = torch.tensor(math.log2(math.e), dtype=torch.float32).double()
    exact = (shift + TIE_SCORES) * log2_e
    _, exponent = torch.frexp(exact.float().double())
    units = (exact - exact.float().double()).abs() / torch.ldexp(torch.ones_like(exact), exponent - 24)
    return TIE_SCORES[units >= FAR_ROUNDING]


def tied_scores(generator, rows, keys, positions, shift):
    scores = -torch.randint(*OTHER_SCORES, (rows, keys), generator=generator).double() / 2
    candidates = far_rounding_tie_scores(shift)
    ties = candidates[torch.randint(len(candidates), (rows, 1), generator=generator)]
    return scores.scatter(1, positions.expand(rows, -1), ties.expand(-1, positions.shape[-1]))


def one_hot_queries(rows, heads, head_dim):
    q = torch.zeros(rows, heads, head_dim, dtype=torch.float64)
    q[torch.arange(rows), :, torch.arange(rows)] = 1.0
    q[..., -1] = 1.0
    return q


def score_keys(scores, shift, head_dim):
    heads, rows, keys = scores.shape
    k = torch.zeros(keys, heads, head_dim, dtype=torch.float64)
    k[..., :rows] = scores.permute(2, 0, 1)
    k[..., -1] = shift
    return k


def bfloat16_values(generator, keys, heads, head_dim):
    return torch.randn(keys, heads, head_dim, generator=generator, dtype=torch.float64).bfloat16().double()


def queue_tie_difference(name, out, reference):
    unit = 2.0**-8 * reference.double().abs().amax(-1, keepdim=True)
    difference = ((out.double() - reference.double()).abs() / unit).mean()
    bound = torch.stack([difference, difference.new_tensor(TIE_TOLERANCE)])
    queue("bound", f"{name}: mean |O - unsplit O| in bfloat16 units", bound)


def unsplit_variants(head_dim):
    declared = specialization.declared_variants(DEVICE_ID, "fwd", head_dim)
    return tuple(variant for variant, info in declared if not info.split_kv)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_keys_tied_across_key_splits_at_a_large_shift_keep_the_unsplit_output(restrict_candidates, head_dim):
    if not any(info.split_kv for _, info in specialization.declared_variants(DEVICE_ID, "fwd", head_dim)):
        pytest.skip("no forward instance splits the keys on this GPU")
    shape_q, shape_k = (1, SPLIT_ROWS, SPLIT_HEADS, head_dim), (1, SPLIT_KEYS, SPLIT_HEADS, head_dim)
    variant = chosen_forward(shape_q, shape_k, WINDOWS["dense"])
    info = instance_info("fwd", head_dim, variant)
    splits = split_kv(DEVICE_ID, info, shape_q, shape_k).splits if info.split_kv else 1
    in_different_splits = torch.tensor(splits >= TIED_KEYS, device="cuda")
    queue_requirement("the keys of each tied score lie in different splits", in_different_splits)
    spread = torch.arange(TIED_KEYS) * (SPLIT_KEYS // TIED_KEYS) + 5
    gathered = torch.tensor(GATHERED_POSITIONS)
    order = torch.arange(SPLIT_KEYS)
    order[spread], order[gathered] = gathered, spread
    generator = torch.Generator().manual_seed(head_dim)
    attend = batched_attention(1, WINDOWS["dense"], 1.0)
    q = one_hot_queries(SPLIT_ROWS, SPLIT_HEADS, head_dim)
    v = bfloat16_values(generator, SPLIT_KEYS, SPLIT_HEADS, head_dim)
    keys = {}
    for shift in TIE_SHIFTS:
        scores = [tied_scores(generator, SPLIT_ROWS, SPLIT_KEYS, spread, shift) for _ in range(SPLIT_HEADS)]
        keys[shift] = score_keys(torch.stack(scores), shift, head_dim)
    tensors = {shift: [x.cuda().bfloat16() for x in (q, k, v)] for shift, k in keys.items()}
    split = {shift: attend(*x)[0] for shift, x in tensors.items()}
    restrict_candidates(unsplit_variants(head_dim))
    reference_variant = chosen_forward(shape_q, shape_k, WINDOWS["dense"])
    unsplit_reference = torch.tensor("split_kv" not in reference_variant, device="cuda")
    queue_requirement("the reference does not split the keys", unsplit_reference)
    for shift, (q, k, v) in tensors.items():
        with case(shift=shift):
            unsplit = attend(q, k, v)[0]
            queue_tie_difference("split keys", split[shift], unsplit)
            queue_tie_difference("tied keys in one key block", attend(q, k[order], v[order])[0], unsplit)


def test_keys_tied_across_key_slices_at_a_large_shift_keep_the_unsplit_output(restrict_candidates):
    if SLICED_HEAD_DIM not in PERSISTENT_HEAD_DIMS:
        pytest.skip("no persistent inter-sequence forward slices the key blocks of this head dimension")
    layout = sample_layout([((SLICED_ROWS, SLICED_KEYS),)] * SLICED_SEQUENCES, [DENSE])
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    part = torch.arange(TIED_KEYS)
    sliced = part * (SLICED_KEYS // TIED_KEYS) + part * 16 + 3
    generator = torch.Generator().manual_seed(SLICED_HEAD_DIM)
    q = one_hot_queries(SLICED_ROWS, SLICED_HEADS, SLICED_HEAD_DIM).repeat(SLICED_SEQUENCES, 1, 1)
    v = bfloat16_values(generator, SLICED_SEQUENCES * SLICED_KEYS, SLICED_HEADS, SLICED_HEAD_DIM)
    tensors = {}
    for shift in TIE_SHIFTS:
        k = []
        for _ in range(SLICED_SEQUENCES):
            scores = [tied_scores(generator, SLICED_ROWS, SLICED_KEYS, sliced, shift) for _ in range(SLICED_HEADS)]
            k.append(score_keys(torch.stack(scores), shift, SLICED_HEAD_DIM))
        k = torch.cat(k)
        tensors[shift] = [x.cuda().bfloat16() for x in (q, k, v)]
    inter = {
        shift: flash_attn_inter_sequence_func(
            q, k, v, cu_q, cu_k, SLICED_ROWS, SLICED_KEYS, bits, softmax_scale=1.0, window_sizes=window_sizes,
            return_attn_probs=True,
        )[0]
        for shift, (q, k, v) in tensors.items()
    }
    restrict_candidates(unsplit_variants(SLICED_HEAD_DIM))
    attend = batched_attention(SLICED_SEQUENCES, WINDOWS["dense"], 1.0)
    for shift, (q, k, v) in tensors.items():
        with case(shift=shift):
            queue_tie_difference("sliced keys", inter[shift], attend(q, k, v)[0])


def fp32_spacing(x):
    _, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), exponent - 24)


def float32(x):
    return float(torch.tensor(x, dtype=torch.float32))


def restrict_to_unsplit_forward(restrict_candidates, head_dim, heads, rows, keys):
    restrict_candidates(unsplit_variants(head_dim))
    variant = chosen_forward((1, rows, heads, head_dim), (1, keys, heads, head_dim), WINDOWS["dense"])
    queue_requirement("the forward does not split the keys", torch.tensor("split_kv" not in variant, device="cuda"))


def lse_of_scores(scores, shift, head_dim):
    heads, rows, keys = scores.shape
    q = one_hot_queries(rows, heads, head_dim)
    k = score_keys(scores, shift, head_dim)
    v = bfloat16_values(torch.Generator().manual_seed(head_dim), keys, heads, head_dim)
    attend = batched_attention(1, WINDOWS["dense"], 1.0)
    return attend(*(x.cuda().bfloat16() for x in (q, k, v)))[1].double()


def base2_log_sum_exp(scores, shift):
    x = (shift + scores.cuda()) * float32(math.log2(math.e))
    top = x.amax(-1, keepdim=True)
    return (top + (x - top).exp2().sum(-1, keepdim=True).log2())[..., 0]


def queue_log2_sum_check(name, lse, log2_sum, exp2_error=0.0):
    ln2 = float32(math.log(2))
    allowed = ln2 * (fp32_spacing(log2_sum) / 2 + LOG2_SUM_SLACK) + fp32_spacing(lse) / 2 + exp2_error
    ratio = (lse - log2_sum * ln2).abs() / allowed
    queue_worst(name, ratio, torch.ones_like(ratio, dtype=torch.bool))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_rows_tied_at_a_large_shift_keep_their_lse_within_fp32_rounding(restrict_candidates, head_dim):
    heads, rows, keys = TIED_LSE_HEADS, TIED_LSE_ROWS, TIED_LSE_KEYS
    restrict_to_unsplit_forward(restrict_candidates, head_dim, heads, rows, keys)
    ties = TIE_SCORES[torch.arange(heads * rows) % len(TIE_SCORES)].reshape(heads, rows, 1)
    positions = torch.tensor(TIED_LSE_POSITIONS).expand(heads, rows, -1)
    scores = torch.full((heads, rows, keys), -FAR_SCORE[0], dtype=torch.float64)
    scores = scores.scatter(2, positions, ties.expand_as(positions))
    for shift in TIE_SHIFTS:
        with case(shift=shift):
            lse = lse_of_scores(scores, shift, head_dim)
            queue_log2_sum_check("lse of rows tied at the maximum", lse, base2_log_sum_exp(scores, shift))


def diffuse_scores(generator, shift):
    heads, rows, keys = DIFFUSE_HEADS, DIFFUSE_ROWS, DIFFUSE_KEYS
    candidates = far_rounding_tie_scores(shift)
    tops = candidates[torch.randint(len(candidates), (heads, rows, 1), generator=generator)]
    gaps = torch.randint(1, DIFFUSE_GAP_STEPS + 1, (heads, rows, len(DIFFUSE_NEAR)), generator=generator) / 8
    scores = torch.full((heads, rows, keys), -FAR_SCORE[0], dtype=torch.float64)
    for head in range(heads):
        group = DIFFUSE_GROUPS[head % len(DIFFUSE_GROUPS)]
        scores[head, :, group + DIFFUSE_TOP] = tops[head, :, 0]
        scores[head, :, [group + offset for offset in DIFFUSE_NEAR]] = tops[head] - gaps[head]
    for position, gap in DIFFUSE_FAR.items():
        scores[..., position] = tops[..., 0] - gap
    return scores


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_diffuse_rows_at_a_large_shift_keep_their_lse_within_fp32_rounding(restrict_candidates, head_dim):
    restrict_to_unsplit_forward(restrict_candidates, head_dim, DIFFUSE_HEADS, DIFFUSE_ROWS, DIFFUSE_KEYS)
    generator = torch.Generator().manual_seed(head_dim)
    for shift in TIE_SHIFTS:
        with case(shift=shift):
            scores = diffuse_scores(generator, shift)
            lse = lse_of_scores(scores, shift, head_dim)
            queue_log2_sum_check("lse of diffuse rows", lse, base2_log_sum_exp(scores, shift), EXP2_ERROR)


# The two keys of a cancelling pair have equal scores and the values +-CANCEL_VALUE along one axis. The sum of
# their bfloat16 dS loses the dS of the top key, which the backward must form as P (dP - D) when P <= 3/4.
CANCEL_VALUE = 2.0**12
TOP_PROBABILITIES = {"below_pivot": (0.56, 0.74), "above_pivot": (0.76, 0.8)}


def cancelling_pair_problem(batch, seqlen_q, seqlen_k, head_dim, ranges):
    num_head, num_head_kv = GQA_HEADS
    scale = head_dim**-0.5
    top, pair = seqlen_k - 1, (0, seqlen_k - 2)
    q = torch.randn(batch, seqlen_q, num_head, head_dim, device="cuda", dtype=torch.float64)
    k = torch.zeros(batch, seqlen_k, num_head_kv, head_dim, device="cuda", dtype=torch.float64)
    v = torch.zeros_like(k)
    dout = torch.randn_like(q)
    for sequence, (low, high) in enumerate(ranges):
        probability = low + (high - low) * torch.rand(seqlen_q, num_head, device="cuda", dtype=torch.float64)
        q[sequence, ..., 0] = (2 * probability / (1 - probability)).log() / scale
    q[..., 3], dout[..., 1:3] = 1.0, 1.0
    k[:, top, :, 0], k[:, :, :, 3] = 1.0, -1000.0 / scale
    k[:, [top, *pair], :, 3] = 0.0
    v[:, top, :, 1], v[:, pair[0], :, 2], v[:, pair[1], :, 2] = 1.0, CANCEL_VALUE, -CANCEL_VALUE
    tensors = bf16_tensors(*(x.flatten(0, 1) for x in (q, k, v, dout)))
    mask = batched_mask(batch, seqlen_q, seqlen_k, WINDOWS["dense"])
    return Problem(*tensors, mask, batched_attention(batch, WINDOWS["dense"]))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seqlen_k", [3, 131])
def test_pivot_threshold_is_three_quarters(head_dim, seqlen_k):
    cuda_seed(head_dim + seqlen_k)
    seqlen_q, ranges = 192, tuple(TOP_PROBABILITIES.values())
    ref, _ = check(cancelling_pair_problem(len(ranges), seqlen_q, seqlen_k, head_dim, ranges), pivots_required=True)
    pivot = ref.pivot.unflatten(1, (len(ranges), seqlen_q))
    queue_requirement("no row below 3/4 has a pivot", ~pivot[:, 0].any())
    queue_requirement("every row above 3/4 has a pivot", pivot[:, 1].all())


# The rows of the first sequence of the subnormal pair problem all have the BF16 dS sum 2**-127, so query column
# MIXED_COLUMN gives the pivot key the dK term -2**-133 per row of the first half of the sequence and -2**-27 per row of
# the second half. The kernel stores the sum of the first query block that reaches the word of the pivot key, adds the
# sums of the other blocks of the first half, which lie below `exact_add_limit`, with compare-and-swap, and adds the
# sums of the other blocks of the second half with red.add.
MIXED_COLUMN, MIXED_SMALL, MIXED_LARGE = 3, 2.0**-6, 2.0**100
SMALL_ONLY_COLUMN, SMALL_ONLY = 4, 2.0**-10
MIXED_SEQLEN_Q = 512


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("repeat", range(4))
def test_pivot_key_gradient_mixes_subnormal_and_large_sums_in_one_word(head_dim, repeat):
    pair = subnormal_pair_problem(head_dim, 0.0)
    seqlen_k = pair.k.shape[0] // 2
    q = pair.q[:1].expand(2 * MIXED_SEQLEN_Q, -1, -1).clone()
    half = MIXED_SEQLEN_Q // 2
    for sequence in range(2):
        first = sequence * MIXED_SEQLEN_Q
        q[first : first + half, :, MIXED_COLUMN] = MIXED_SMALL
        q[first + half : first + MIXED_SEQLEN_Q, :, MIXED_COLUMN] = MIXED_LARGE
        q[first : first + MIXED_SEQLEN_Q, :, SMALL_ONLY_COLUMN] = SMALL_ONLY
    dout = pair.dout[:1].expand(2 * MIXED_SEQLEN_Q, -1, -1).clone()
    window = WINDOWS["dense"]
    mask = batched_mask(2, MIXED_SEQLEN_Q, seqlen_k, window)
    problem = Problem(q, pair.k, pair.v, dout, mask, batched_attention(2, window, 1.0), 1.0)
    rows = slice(0, MIXED_SEQLEN_Q)
    columns = [MIXED_COLUMN, SMALL_ONLY_COLUMN]
    row_sums = exact_ds(problem)[rows, 1:].bfloat16().double().sum(-1)
    queue_requirement("every row of the first sequence has the dS sum 2**-127", (row_sums == 2.0**-127).all())
    expected = -problem.softmax_scale * (row_sums @ q[rows, 0, columns].double())
    queue_requirement("the large terms dominate the mixed column", expected[0] == -half * 2.0**-27)
    queue_requirement("the small-only column sums to an FP32 subnormal number", expected[1].abs() < 2.0**-126)
    _, _, _, dk, _ = run(problem)
    exact = (dk[0, 0, columns].double() == expected.bfloat16().double()).all()
    queue_requirement("dk of the pivot key is the BF16 rounding of the sum of its terms", exact)
