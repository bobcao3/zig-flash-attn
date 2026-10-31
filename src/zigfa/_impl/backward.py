import functools
import heapq
import math
from typing import NamedTuple

from . import runtime, specialization
from .tensormap import as_four_dimensional, tensor_map, tiled_tensor_map
from .tensors import (
    LOG2_E,
    View,
    build_scheduler_metadata,
    default_softmax_scale,
    require_aligned,
    require_window,
    scheduler_metadata_entries,
    sequence_layout,
    varlen_blocks,
    view_of,
)

PREPROCESS_SIGNATURE = "p" * 9 + "u" + "p" * 11 + "upu"
SCALAR_SIGNATURE = "uuu" + "ff" + "ii"
BACKWARD_SIGNATURE = "p" * 13 + SCALAR_SIGNATURE + "p" * 23 + "u"
TENSOR_MAP_BACKWARD_SIGNATURE = "m" * 4 + "p" * 9 + SCALAR_SIGNATURE + "p" * 11 + "u"
DQ_ACCUM_MAP_BACKWARD_SIGNATURE = "m" * 5 + "p" * 8 + SCALAR_SIGNATURE + "p" * 8 + "u"
DQ_SIGNATURE = "m" * 4 + "p" * 8 + SCALAR_SIGNATURE + "p" * 8 + "u"
POSTPROCESS_SIGNATURE = "p" * 9 + "uuf" + "p" * 17 + "u"
FIX_DK_SIGNATURE = "ppp" + "uuu" + "ppp"
PREPROCESS_ROWS = 32
POSTPROCESS_ROWS = 16
AUXILIARY_THREADS = 256


ACCUMULATOR_PADDING_ROWS = 128
# The SM80 query gradient accumulator starts each sequence of a varlen batch at a multiple of 8 rows,
# which takes up to 8 more rows per sequence.
SM80_SEQUENCE_ROW_ALIGNMENT = 8
HEAD_SPLITS = 4
HEAD_SPLIT_MINIMUM_GROUP = 16
# The shortest sequences from which an SM80 head split pays, by head dimension, without and with a
# causal mask.
SM80_HEAD_SPLIT_MINIMUM_SEQLEN = {
    64: (2048, 2048),
    72: (1024, 1024),
    96: (4096, 2048),
    128: (2048, 2048),
    256: (1024, 2048),
}
HEAD_SPLIT_EFFICIENCY = 0.92
WAVE_SPLIT_TRIGGER_EFFICIENCY = 0.85
WAVE_SPLIT_TARGET_EFFICIENCY = 0.9
TWO_CTA_HEAD_SPLIT_MIN_ROWS = 4096
CLEAR_SIGNATURE = "pu"
CLEAR_BLOCKS = 512
FIX_DK_BLOCKS = 2048


class BackwardWorkspace(NamedTuple):
    facts: specialization.ShapeFacts
    lse_padding: int
    accumulator_padding: int
    dkv_accum: bool
    # An FP32 buffer collects the dK products of the pivot keys for the kernels that write BF16 dK,
    # followed by one flag per buffer row. A workspace with dkv_accum holds the buffer in its dK half
    # and the flags in its dV half.
    dk_fix: bool
    scheduler_metadata_rows: int


def accumulator_shape(q_shape, padding_rows):
    total = q_shape[0] * q_shape[1] if len(q_shape) == 4 else q_shape[0]
    return q_shape[-2], total + padding_rows, q_shape[-1]


def dk_fix_shape(k_shape, padding_rows):
    heads, rows, head_dim = accumulator_shape(k_shape, padding_rows)
    return (heads * rows * head_dim + 4 * dk_state_chunks(heads * rows, head_dim),)


def dk_state_chunks(rows, head_dim):
    return -(-rows * (head_dim // 8) // 4)


def pivots_shape(lse_elements):
    # The SM100 backward with head dimension 256 loads the pivot keys in 16-byte vectors.
    return 2, -(-lse_elements // 4) * 4


@functools.cache
def backward_workspace_layout(
    device_id, q_shape, k_shape, window_size_left, window_size_right, cu_seqlens_shape, max_seqlen_q, max_seqlen_k
):
    facts = specialization.facts(q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k)
    infos = specialization.reachable(device_id, "bwd", q_shape[-1], facts)
    aligned = cu_seqlens_shape is not None and any(info.arch == 80 for info in infos)
    alignment_rows = SM80_SEQUENCE_ROW_ALIGNMENT * facts.batch_size if aligned else 0
    dkv_accum = k_shape is not None and any(
        info.head_split and head_split_pays(device_id, info, q_shape, k_shape, window_size_left, window_size_right)
        for info in infos
    )
    return BackwardWorkspace(
        facts,
        max(max(info.block_m, info.dq_block_m) for info in infos),
        max(ACCUMULATOR_PADDING_ROWS, *(info.block_m for info in infos)) + alignment_rows,
        dkv_accum,
        not dkv_accum and any(not info.dq_block_m for info in infos),
        min(((info.cluster_ctas or 1) * info.block_n for info in infos if info.scheduler_metadata), default=0),
    )


@functools.cache
def backward_variant(
    device_id, q_shape, k_shape, window_size_left, window_size_right, cu_seqlens_shape, max_seqlen_q, max_seqlen_k
):
    layout = backward_workspace_layout(
        device_id, q_shape, k_shape, window_size_left, window_size_right, cu_seqlens_shape, max_seqlen_q, max_seqlen_k
    )
    regular = specialization.unrestricted(device_id, "bwd", q_shape[-1])
    two_cta_split = bool(regular and regular.head_split and regular.cluster_ctas) and head_split_pays(
        device_id, regular, q_shape, k_shape, window_size_left, window_size_right
    )
    variant = "" if two_cta_split else specialization.choose(
        device_id,
        "bwd",
        q_shape[-1],
        layout.facts,
        lambda info, facts: specialization.accepts(info, facts)
        and (
            not info.head_split
            or head_split_pays(device_id, info, q_shape, k_shape, window_size_left, window_size_right)
        ),
    )
    specialization.log_choice(
        "backward", device_id, q_shape[-1], variant, q=q_shape, k=k_shape,
        window=(window_size_left, window_size_right), cu_seqlens=cu_seqlens_shape,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    )
    return variant


def compute_scratch_shapes(
    q_shape,
    lse_shape,
    k_shape,
    window_size_left=-1,
    window_size_right=-1,
    *,
    cu_seqlens_shape=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    require_window(window_size_left, window_size_right)
    layout = backward_workspace_layout(
        runtime.current_device().device_id,
        q_shape,
        k_shape,
        window_size_left,
        window_size_right,
        cu_seqlens_shape,
        max_seqlen_q,
        max_seqlen_k,
    )
    lse_elements = math.prod(lse_shape) + layout.lse_padding
    shapes = (
        (lse_elements,),
        (lse_elements,),
        pivots_shape(lse_elements),
        accumulator_shape(q_shape, layout.accumulator_padding),
    )
    if layout.dkv_accum:
        shapes += ((2, *accumulator_shape(k_shape, layout.accumulator_padding)),)
    elif layout.dk_fix:
        if k_shape is None:
            raise ValueError("compute_scratch_shapes needs k_shape")
        shapes += (dk_fix_shape(k_shape, layout.accumulator_padding),)
    if layout.scheduler_metadata_rows:
        if k_shape is None:
            raise ValueError("compute_scratch_shapes needs k_shape for varlen inputs")
        shapes += ((scheduler_metadata_entries(k_shape[0], layout.facts.batch_size, layout.scheduler_metadata_rows), 2),)
    return shapes


@functools.cache
def schedule_efficiency(batch_size, num_n_blocks, num_head_kv, qhead_per_kvhead, num_m_blocks, block_ratio, num_sm):
    works = [
        qhead_per_kvhead * (num_m_blocks - block * block_ratio)
        for _ in range(batch_size)
        for _ in range(num_head_kv)
        for block in range(num_n_blocks)
    ]
    finish = [0] * num_sm
    for work in works:
        heapq.heappush(finish, heapq.heappop(finish) + work)
    return sum(works) / num_sm / max(finish)


def two_cta_head_splits(device_id, info, q_shape, k_shape, window_size_left, window_size_right):
    if len(q_shape) != 4 or window_size_left >= 0 or window_size_right >= 0 or q_shape[1] < TWO_CTA_HEAD_SPLIT_MIN_ROWS:
        return 1
    batch_size, _, num_head, _ = q_shape
    seqlen_k, num_head_kv = k_shape[1], k_shape[2]
    clusters = batch_size * num_head_kv * -(-seqlen_k // (info.cluster_ctas * info.block_n))
    slots = runtime.num_sm(device_id) // info.cluster_ctas
    num_qhead_splits = 1
    while (
        specialization.fill(clusters * num_qhead_splits, slots) < WAVE_SPLIT_TARGET_EFFICIENCY
        and num_head // num_head_kv % (2 * num_qhead_splits) == 0
    ):
        num_qhead_splits *= 2
    return num_qhead_splits


def sm80_head_split_pays(q_shape, k_shape, window_size_left, window_size_right):
    # A split adds dK and dV reductions that grow with the sequence length, and it shortens the
    # longest CTA, whose work grows with the square of the sequence length.
    if len(q_shape) != 4 or window_size_left >= 0 or window_size_right > 0:
        return False
    _, seqlen_q, num_head, head_dim = q_shape
    seqlen_k, num_head_kv = k_shape[1], k_shape[2]
    qhead_per_kvhead = num_head // num_head_kv
    if head_dim not in SM80_HEAD_SPLIT_MINIMUM_SEQLEN:
        return False
    dense_minimum, causal_minimum = SM80_HEAD_SPLIT_MINIMUM_SEQLEN[head_dim]
    minimum_seqlen = causal_minimum if window_size_right == 0 else dense_minimum
    return (
        qhead_per_kvhead >= HEAD_SPLITS and qhead_per_kvhead % HEAD_SPLITS == 0 and seqlen_q == seqlen_k
        and seqlen_k >= minimum_seqlen
    )


def head_split_pays(device_id, info, q_shape, k_shape, window_size_left, window_size_right):
    if info.cluster_ctas:
        return two_cta_head_splits(device_id, info, q_shape, k_shape, window_size_left, window_size_right) > 1
    if info.arch == 80 and sm80_head_split_pays(q_shape, k_shape, window_size_left, window_size_right):
        return True
    if len(q_shape) != 4 or window_size_left >= 0 or window_size_right != 0:
        return False
    batch_size, seqlen_q, num_head, _ = q_shape
    seqlen_k, num_head_kv = k_shape[1], k_shape[2]
    qhead_per_kvhead = num_head // num_head_kv
    if qhead_per_kvhead < HEAD_SPLIT_MINIMUM_GROUP or qhead_per_kvhead % HEAD_SPLITS or seqlen_q != seqlen_k:
        return False
    efficiency = schedule_efficiency(
        batch_size,
        (seqlen_k + info.block_n - 1) // info.block_n,
        num_head_kv,
        qhead_per_kvhead,
        (seqlen_q + info.block_m - 1) // info.block_m,
        info.block_n // info.block_m,
        runtime.num_sm(device_id),
    )
    return efficiency < HEAD_SPLIT_EFFICIENCY


def accumulator_strides(q_shape, padding_rows):
    heads, rows, head_dim = accumulator_shape(q_shape, padding_rows)
    batch_stride = q_shape[1] * head_dim if len(q_shape) == 4 else 0
    return batch_stride, head_dim, rows * head_dim


def dq_accum_tensor_map(info, pointer, q_shape, padding_rows):
    """Returns the tensor map through which the backward of `info` reduces into the query gradient
    accumulator of `accumulator_shape(q_shape, padding_rows)`."""
    heads, rows, head_dim = accumulator_shape(q_shape, padding_rows)
    if info.dq_accum_interleave:
        return tiled_tensor_map(
            pointer,
            (heads, info.block_m // 8, rows, 2, head_dim),
            (rows * head_dim, 8 * head_dim, head_dim, 4 * head_dim, 1),
            (1, info.block_m // 8, 4, 2, info.dq_accum_box_columns),
            info.dq_accum_box_columns * 4,
            element_bytes=4,
        )
    return tensor_map(
        pointer,
        (1, rows, heads, head_dim),
        (heads * rows * head_dim, head_dim, rows * head_dim, 1),
        info.dq_accum_box_rows or info.block_m,
        info.dq_accum_box_columns,
        info.dq_accum_box_columns * 4,
        element_bytes=4,
    )


def qhead_splits(device_id, info, qhead_per_kvhead, window_size_left, window_size_right, ctas):
    """Returns how many CTAs of a cluster share the query heads of one key block, for a grid of `ctas`
    CTAs without splits."""
    num_qhead_splits = 1
    while (
        info.min_qhead_per_split and window_size_left < 0 and 2 * num_qhead_splits <= info.max_qhead_splits
        and qhead_per_kvhead % (2 * num_qhead_splits) == 0 and qhead_per_kvhead // (2 * num_qhead_splits) >= info.min_qhead_per_split
        and (not info.qhead_split_bounded_only or window_size_right >= 0)
    ):
        num_qhead_splits *= 2
    if num_qhead_splits == 1 and info.qhead_split_by_waves:
        num_sm = runtime.num_sm(device_id)

        def efficiency(count):
            return specialization.fill(ctas * count, num_sm)

        allowed = [
            count for count in (1 << power for power in range(max(1, info.max_qhead_splits).bit_length()))
            if qhead_per_kvhead % count == 0
            and (count == 1 or qhead_per_kvhead // count >= info.min_qhead_per_split)
        ]
        if efficiency(1) < WAVE_SPLIT_TRIGGER_EFFICIENCY:
            num_qhead_splits = next(
                (count for count in allowed if efficiency(count) >= WAVE_SPLIT_TARGET_EFFICIENCY),
                max(allowed, key=efficiency),
            )
    return num_qhead_splits


def backward(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    dq,
    dk,
    dv,
    workspace,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    window_size_left=-1,
    window_size_right=-1,
    stream=None,
):
    lse_log2, dpsum, pivots, dq_accum, *extra = workspace
    views = {name: view_of(t) for name, t in dict(
        q=q, k=k, v=v, out=out, dout=dout, lse=softmax_lse, dq=dq, dk=dk, dv=dv,
        lse_log2=lse_log2, dpsum=dpsum, pivots=pivots, dq_accum=dq_accum,
    ).items()}
    head_dim = views["q"].shape[-1]
    num_head, num_head_kv = views["q"].shape[-2], views["k"].shape[-2]
    qhead_per_kvhead = num_head // num_head_kv
    require_window(window_size_left, window_size_right)
    for name in ("q", "k", "v", "out", "dout", "dq", "dk", "dv"):
        require_aligned(name, views[name], head_dim)
    varlen = cu_seqlens_q is not None
    device_id = runtime.current_device().device_id
    key = (
        device_id,
        views["q"].shape,
        views["k"].shape,
        window_size_left,
        window_size_right,
        view_of(cu_seqlens_q).shape if varlen else None,
        max_seqlen_q,
        max_seqlen_k,
    )
    layout = backward_workspace_layout(*key)
    if len(extra) != (layout.dkv_accum or layout.dk_fix) + bool(layout.scheduler_metadata_rows):
        raise ValueError("pass the workspace that compute_scratch_shapes returns for these shapes and this window")
    (
        batch_size, seqlen_q, seqlen_k, cu_seqlens_q_pointer, cu_seqlens_k_pointer, row_axis_len_q, row_axis_len_k,
        lse_batch_stride, lse_head_stride,
    ) = sequence_layout(views["q"], views["k"], views["lse"], cu_seqlens_q, cu_seqlens_k)
    if not row_axis_len_q and not row_axis_len_k:
        return
    variant = backward_variant(*key)
    main, info = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd", variant)
    dkv_accum = extra[: layout.dkv_accum] if info.head_split else ()
    fixes_dk = not dkv_accum and not info.dq_block_m
    dk_accum = view_of(extra[0]).pointer if dkv_accum or fixes_dk else 0
    dk_accum_strides = accumulator_strides(tuple(views["k"].shape), layout.accumulator_padding)
    pivot_sum, pivot_key = pivot_pointers(views["pivots"])

    def blocks(rows, block):
        return varlen_blocks(rows, batch_size, block) if varlen else (rows + block - 1) // block

    def row_blocks(rows, block):
        return (rows + block - 1) // block if info.scheduler_metadata else blocks(rows, block)

    cluster_ctas = info.cluster_ctas or 1
    num_n_blocks = (
        scheduler_metadata_entries(row_axis_len_k, batch_size, cluster_ctas * info.block_n)
        if info.scheduler_metadata
        else blocks(row_axis_len_k, cluster_ctas * info.block_n)
    )

    grid_z = 1 if varlen else batch_size
    softmax_scale = default_softmax_scale(softmax_scale, head_dim)
    head_splits = (
        (two_cta_head_splits(device_id, info, tuple(views["q"].shape), tuple(views["k"].shape), window_size_left, window_size_right) if info.cluster_ctas else HEAD_SPLITS)
        if dkv_accum
        else 1
    )
    outputs = {name: views[name] for name in ("dk", "dv")}
    dk_states, state_chunks = 0, 0
    if dkv_accum or fixes_dk:
        _, accumulator_rows, _ = accumulator_shape(views["k"].shape, layout.accumulator_padding)
        half = num_head_kv * accumulator_rows * head_dim
    if dkv_accum:
        strides = (seqlen_k * head_dim, head_dim, accumulator_rows * head_dim, 1)
        for index, name in enumerate(("dk", "dv")):
            views[name] = View(dk_accum + index * half * 4, views["k"].shape, strides)
        clear, _ = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd_clear", variant, dynamic_shared=False)
        runtime.run(clear, (CLEAR_BLOCKS, 1, 1), AUXILIARY_THREADS, 0, CLEAR_SIGNATURE, (dk_accum, half // 2), stream)
    elif fixes_dk:
        dk_states, state_chunks = dk_accum + half * 4, dk_state_chunks(num_head_kv * accumulator_rows, head_dim)

    def row_strides(name):
        view = views[name]
        return (0 if varlen else view.strides[0]), view.strides[-3], view.strides[-2]

    accum_strides = accumulator_strides(tuple(views["q"].shape), layout.accumulator_padding)
    preprocess, _ = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd_preprocess", variant)
    if info.box_columns:
        maps = tuple(
            tensor_map(views[name].pointer, *as_four_dimensional(views[name]), rows, info.box_columns, info.swizzle_bytes)
            for name, rows in (("q", info.block_m), ("k", info.block_n), ("v", info.block_n), ("dout", info.block_m))
        )
        scalars = (
            cu_seqlens_q_pointer, cu_seqlens_k_pointer, seqlen_q, seqlen_k, qhead_per_kvhead,
            softmax_scale * LOG2_E, softmax_scale, window_size_left, window_size_right,
        )
        if info.bulk_reduce:
            signature = DQ_ACCUM_MAP_BACKWARD_SIGNATURE
            maps += (dq_accum_tensor_map(info, views["dq_accum"].pointer, tuple(views["q"].shape), layout.accumulator_padding),)
            main_values = (
                *maps, views["lse_log2"].pointer, views["dpsum"].pointer, pivot_sum, pivot_key,
                views["dk"].pointer, views["dv"].pointer,
                *scalars, *row_strides("dk"), *row_strides("dv"), lse_batch_stride, lse_head_stride, batch_size,
            )
        else:
            signature = TENSOR_MAP_BACKWARD_SIGNATURE
            main_values = (
                *maps, views["lse_log2"].pointer, views["dpsum"].pointer, pivot_sum, pivot_key,
                views["dq_accum"].pointer, views["dk"].pointer, views["dv"].pointer,
                *scalars, *row_strides("dk"), *row_strides("dv"), *accum_strides,
                lse_batch_stride, lse_head_stride, batch_size,
            )
    else:
        signature = BACKWARD_SIGNATURE
        main_values = (
            views["q"].pointer, views["k"].pointer, views["v"].pointer, views["dout"].pointer,
            views["lse_log2"].pointer, views["dpsum"].pointer, pivot_sum, pivot_key, views["dq_accum"].pointer,
            views["dk"].pointer, views["dv"].pointer, cu_seqlens_q_pointer, cu_seqlens_k_pointer,
            seqlen_q, seqlen_k, qhead_per_kvhead, softmax_scale * LOG2_E, softmax_scale,
            window_size_left, window_size_right,
            *row_strides("q"), *row_strides("k"), *row_strides("v"), *row_strides("dout"),
            *row_strides("dk"), *row_strides("dv"), *accum_strides, lse_batch_stride, lse_head_stride,
            batch_size,
        )
    if info.scheduler_metadata:
        signature += "p"
        main_values += (
            build_scheduler_metadata(
                extra[-1], "bwd", head_dim, variant, "flash_attn_bwd", cu_seqlens_k_pointer, cu_seqlens_q_pointer,
                batch_size, num_n_blocks, "compute_scratch_shapes", stream,
            ),
        )
    if info.dkv_box_rows:
        signature += "mm"
        main_values += tuple(
            tensor_map(view.pointer, *as_four_dimensional(view), info.dkv_box_rows, info.box_columns, info.swizzle_bytes)
            for view in (outputs["dk"], outputs["dv"])
        )
    runtime.run(
        preprocess,
        (row_blocks(row_axis_len_q, PREPROCESS_ROWS), num_head, grid_z),
        AUXILIARY_THREADS,
        0,
        PREPROCESS_SIGNATURE,
        (
            views["out"].pointer, views["dout"].pointer, views["lse"].pointer,
            views["lse_log2"].pointer, views["dpsum"].pointer, pivot_sum, pivot_key, views["dq_accum"].pointer,
            cu_seqlens_q_pointer, seqlen_q,
            *row_strides("out"), *row_strides("dout"), lse_batch_stride, lse_head_stride, *accum_strides,
            batch_size, dk_states, state_chunks,
        ),
        stream,
    )
    num_qhead_splits = qhead_splits(
        device_id, info, qhead_per_kvhead, window_size_left, window_size_right,
        cluster_ctas * num_n_blocks * num_head_kv * grid_z,
    )
    main_launch = (
        main,
        (cluster_ctas * num_n_blocks * head_splits, num_head_kv * num_qhead_splits, grid_z),
        info.threads,
        info.shared_bytes,
        signature,
        main_values,
        stream,
    )
    main_options = dict(pdl=info.pdl, cluster_dims=(1, num_qhead_splits, 1) if num_qhead_splits > 1 else None)
    if info.dq_block_m:
        # The dK kernel reads the pivot keys and row sums that the dQ kernel writes.
        dq_kernel, _ = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd_dq", variant)
        runtime.run(
            dq_kernel,
            (cluster_ctas * blocks(row_axis_len_q, info.dq_block_m), num_head, grid_z),
            info.threads,
            info.shared_bytes,
            DQ_SIGNATURE,
            (
                *maps, views["lse_log2"].pointer, views["dpsum"].pointer, pivot_sum, pivot_key, views["dq"].pointer,
                views["k"].pointer, cu_seqlens_q_pointer, cu_seqlens_k_pointer, seqlen_q, seqlen_k, qhead_per_kvhead,
                softmax_scale * LOG2_E, softmax_scale, window_size_left, window_size_right,
                *row_strides("dq"), *row_strides("k"), lse_batch_stride, lse_head_stride, batch_size,
            ),
            stream,
            pdl=info.pdl,
        )
        runtime.run(*main_launch, **main_options)
        return
    runtime.run(*main_launch, **main_options)
    postprocess, _ = runtime.load_for(
        device_id, "bwd", head_dim, "flash_attn_bwd_postprocess", variant, dynamic_shared=False
    )
    runtime.run(
        postprocess,
        (row_blocks(row_axis_len_q, POSTPROCESS_ROWS), num_head, grid_z),
        AUXILIARY_THREADS,
        0,
        POSTPROCESS_SIGNATURE,
        (
            views["dq_accum"].pointer, views["dq"].pointer, pivot_sum, pivot_key, views["q"].pointer,
            views["k"].pointer, dk_accum, dk_states, cu_seqlens_q_pointer, seqlen_q, qhead_per_kvhead, softmax_scale,
            *accum_strides, *row_strides("dq"), *row_strides("q"), *row_strides("k"), *dk_accum_strides,
            lse_batch_stride, lse_head_stride, batch_size,
        ),
        stream,
        pdl=info.pdl,
    )
    for name, output in outputs.items() if dkv_accum else ():
        runtime.run(
            postprocess,
            (blocks(row_axis_len_k, POSTPROCESS_ROWS), num_head_kv, grid_z),
            AUXILIARY_THREADS,
            0,
            POSTPROCESS_SIGNATURE,
            (
                views[name].pointer, output.pointer, 0, 0, 0, 0, 0, 0, cu_seqlens_k_pointer, seqlen_k, 1, 1.0,
                *dk_accum_strides, (0 if varlen else output.strides[0]), output.strides[-3], output.strides[-2],
                *(0,) * 11, batch_size,
            ),
            stream,
            pdl=info.pdl,
        )
    if fixes_dk:
        fix_dk, _ = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd_fix_dk", variant, dynamic_shared=False)
        launch_fix_dk(fix_dk, dk_accum, dk_states, views["dk"], layout.accumulator_padding, varlen, stream, info.pdl)


def launch_fix_dk(kernel, dk_accum, dk_states, dk, padding_rows, varlen, stream, pdl):
    num_head_kv, accumulator_rows, head_dim = accumulator_shape(dk.shape, padding_rows)
    rows = num_head_kv * accumulator_rows
    blocks = min(-(-rows * (head_dim // 8) // AUXILIARY_THREADS), FIX_DK_BLOCKS)
    batch_rows, batch_stride = (accumulator_rows, 0) if varlen else (dk.shape[1], dk.strides[0])
    runtime.run(
        kernel, (blocks, 1, 1), AUXILIARY_THREADS, 0, FIX_DK_SIGNATURE,
        (dk_accum, dk_states, dk.pointer, rows, accumulator_rows, batch_rows, batch_stride, *dk.strides[-3:-1]),
        stream, pdl=pdl,
    )


def pivot_pointers(pivots):
    # The workspace keeps the scaled FP32 row sums of dS in its first row and the int32 pivot keys
    # in its second.
    return pivots.pointer, pivots.pointer + pivots.strides[0] * 4
