import functools
import math
import os
from typing import NamedTuple, Optional

from . import runtime, specialization
from .backward import (
    ACCUMULATOR_PADDING_ROWS,
    AUXILIARY_THREADS,
    BACKWARD_SIGNATURE,
    POSTPROCESS_ROWS,
    POSTPROCESS_SIGNATURE,
    PREPROCESS_ROWS,
    PREPROCESS_SIGNATURE,
    SM80_SEQUENCE_ROW_ALIGNMENT,
    accumulator_shape,
    accumulator_strides,
)
from .tensors import (
    LOG2_E,
    build_scheduler_metadata,
    default_softmax_scale,
    require_aligned,
    require_supported_heads,
    require_window,
    scheduler_metadata_entries,
    view_of,
)

VARIANT = "_varlen_inter_sequence"
DIRECTIONS = ("fwd", "bwd")
TRANSPOSE_WARPS = 8
COUNT_THREADS = 1024
FORWARD_SIGNATURE = "p" * 7 + "uuufii" + "p" * 14 + "u" + "p" * 6 + "u"
RUN_TABLE_SIGNATURE = "pppp" + "ii" + "ppp" + "u"
RUN_ENTRY_WORDS = 4
RUN_QUEUE_WORDS = 8
QUEUE_BYTES = 8
PLAN_SIGNATURE = "ppppp" + "ii" + "pppp" + "uuuuu"
WIDE_MIN_ROWS = 768
NO_WIDE_BLOCKS = 2**32 - 1
PLANNED = os.environ.get("ZIGFA_INTER_SEQUENCE_PLAN") == "device"
MEGAKERNEL = os.environ.get("ZIGFA_INTER_SEQUENCE_PLAN") == "megakernel"
MEGAKERNEL_NAME = "inter_sequence_forward_megakernel"
MEGAKERNEL_SIGNATURE = "p" * 19 + "iif" + "uuu" + "p" + "uu"
SMALL_BLOCK_MAX_SEQLEN_Q = 768
SMALL_BLOCK_MAX_WAVES = 8
INTER_SEQUENCE_BACKWARD_SIGNATURE = BACKWARD_SIGNATURE + "ppp"


class Sequences(NamedTuple):

    cu_seqlens_q: object
    cu_seqlens_k: object
    visibility_bits: object
    visibility_bits_t: object
    cu_visible_k: object
    cu_visible_q: object
    sequence_order: object
    window_sizes: Optional[object] = None


def words_per_row(num_sequences):
    return (num_sequences + 31) // 32


def metadata_shapes(num_sequences):
    return (num_sequences, words_per_row(num_sequences)), (num_sequences + 1,), (num_sequences + 1,), (num_sequences,)


@functools.cache
def head_dims(device_id):
    entries = runtime.manifest()[1]
    try:
        runtime.instance_name(device_id, "fwd", 0, VARIANT)
    except RuntimeError:
        return ()
    return tuple(
        head_dim
        for head_dim in sorted({int(name.rsplit("_hd", 1)[1]) for name in entries})
        if all(runtime.instance_name(device_id, direction, head_dim, VARIANT) in entries for direction in DIRECTIONS)
    )


def require_supported(device_id, head_dim):
    supported = head_dims(device_id)
    if head_dim not in supported:
        raise NotImplementedError(f"inter-sequence attention supports head_dim {list(supported)}, got {head_dim}")


def kernel_info(device_id, direction, head_dim):
    require_supported(device_id, head_dim)
    return runtime.manifest()[1][runtime.instance_name(device_id, direction, head_dim, VARIANT)]


def prepare_metadata(
    cu_seqlens_q, cu_seqlens_k, visibility_bits, visibility_bits_t, cu_visible_k, cu_visible_q, sequence_order,
    stream=None,
):
    num_sequences = view_of(cu_seqlens_q).shape[0] - 1
    device_id = runtime.current_device().device_id
    supported = head_dims(device_id)
    if not supported:
        raise NotImplementedError("no inter-sequence kernels are built for this GPU")
    load = functools.partial(runtime.load_for, device_id, "fwd", supported[0], variant=VARIANT, dynamic_shared=False)
    if not num_sequences:
        return
    words = words_per_row(num_sequences)
    bits, bits_t = (view_of(tensor).pointer for tensor in (visibility_bits, visibility_bits_t))
    transpose, _ = load(kernel_name="inter_sequence_transpose")
    runtime.run(
        transpose, (-(-words * words // TRANSPOSE_WARPS), 1, 1), TRANSPOSE_WARPS * 32, 0, "ppu",
        (bits, bits_t, num_sequences), stream,
    )
    count, _ = load(kernel_name="inter_sequence_count_visible_tokens")
    runtime.run(
        count, (1, 1, 1), COUNT_THREADS, 0, "ppppppp" + "u",
        (
            bits, bits_t, view_of(cu_seqlens_q).pointer, view_of(cu_seqlens_k).pointer,
            view_of(cu_visible_k).pointer, view_of(cu_visible_q).pointer, view_of(sequence_order).pointer,
            num_sequences,
        ),
        stream,
    )


@functools.cache
def forward_variants(device_id, head_dim):
    require_supported(device_id, head_dim)
    entries = runtime.manifest()[1]
    names = runtime.variant_names(device_id, "fwd", head_dim, inter_sequence=True)
    instances = ((variant, runtime.instance_name(device_id, "fwd", head_dim, variant)) for variant in names)
    return tuple(
        (variant, entries[instance]) for variant, instance in instances
        if variant.startswith(VARIANT) and instance in entries
    )


@functools.cache
def megakernel_head_dims(device_id):
    entries = runtime.manifest()[1]
    supported = []
    for head_dim in head_dims(device_id):
        try:
            runtime.load_for(device_id, "fwd", head_dim, MEGAKERNEL_NAME, VARIANT)
        except Exception:
            continue
        supported.append(head_dim)
    return tuple(supported) if entries else ()


@functools.cache
def resident_ctas(device_id, head_dim, variant, kernel_name):
    kernel, info = runtime.load_for(device_id, "fwd", head_dim, kernel_name, variant)
    occupancy = kernel.occupancy.max_active_blocks_per_multiprocessor(info.threads, info.shared_bytes)
    return runtime.num_sm(device_id) * max(1, occupancy)


def forward_variant(device_id, head_dim, max_seqlen_q, rows, num_head):
    variants = forward_variants(device_id, head_dim)
    regular = next(entry for entry in variants if entry[0] == VARIANT)
    if max_seqlen_q is None:
        return regular
    blocks = -(-rows // regular[1].cluster_block_m) * num_head
    waves = blocks / resident_ctas(device_id, head_dim, VARIANT, "flash_attn_fwd")
    if max_seqlen_q > SMALL_BLOCK_MAX_SEQLEN_Q and waves >= SMALL_BLOCK_MAX_WAVES:
        return regular
    for variant, info in variants:
        if specialization.within(max_seqlen_q, info.min_seqlen_q, info.max_seqlen_q):
            return variant, info
    return regular


def forward_workspace_shapes(q_shape, num_sequences):
    variants = forward_variants(runtime.current_device().device_id, q_shape[-1])
    block_m = min(info.cluster_block_m for _, info in variants)
    blocks = scheduler_metadata_entries(q_shape[0], num_sequences, block_m)
    return (blocks, 2), (num_sequences, RUN_ENTRY_WORDS), (blocks, 2), (RUN_QUEUE_WORDS,)


def backward_workspace_shapes(q_shape, lse_shape, k_shape, num_sequences):
    device_id = runtime.current_device().device_id
    info = kernel_info(device_id, "bwd", q_shape[-1])
    lse_elements = math.prod(lse_shape) + info.block_m
    padding = max(ACCUMULATOR_PADDING_ROWS, info.block_m) + SM80_SEQUENCE_ROW_ALIGNMENT * num_sequences
    return (
        (lse_elements,),
        (lse_elements,),
        accumulator_shape(q_shape, padding),
        (scheduler_metadata_entries(k_shape[0], num_sequences, info.block_n), 2),
    )


def window_pointer(sequences):
    return 0 if sequences.window_sizes is None else view_of(sequences.window_sizes).pointer


def row_strides(view):
    return 0, view.strides[-3], view.strides[-2]


def forward(
    q, k, v, out, softmax_lse, workspace, sequences, *, max_seqlen_q=None, softmax_scale=None, window_size_left=-1,
    window_size_right=-1, stream=None,
):
    q_view, k_view, v_view, out_view, lse_view = (view_of(t) for t in (q, k, v, out, softmax_lse))
    head_dim = q_view.shape[-1]
    require_supported_heads(q_view.shape, k_view.shape)
    require_window(window_size_left, window_size_right)
    for name, view in (("q", q_view), ("k", k_view), ("v", v_view), ("out", out_view)):
        require_aligned(name, view, head_dim)
    device_id = runtime.current_device().device_id
    require_supported(device_id, head_dim)
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    _, run_table, run_list, run_queue = (view_of(t).pointer for t in workspace)
    if not q_view.shape[0]:
        return
    if MEGAKERNEL and head_dim in megakernel_head_dims(device_id):
        return forward_megakernel(
            q_view, k_view, v_view, out_view, lse_view, workspace, sequences, softmax_scale, window_size_left,
            window_size_right, stream,
        )
    if PLANNED:
        return forward_planned(
            q_view, k_view, v_view, out_view, lse_view, workspace, sequences, softmax_scale, window_size_left,
            window_size_right, stream,
        )
    hint = specialization.usable_max_seqlen(max_seqlen_q, q_view.shape[0])
    variant, _ = forward_variant(device_id, head_dim, hint, q_view.shape[0], q_view.shape[-2])
    load = functools.partial(runtime.load_for, device_id, "fwd", head_dim, variant=variant)
    kernel, info = load(kernel_name="flash_attn_fwd")
    listed, _ = load(kernel_name="flash_attn_fwd_runs")
    runs, _ = load(kernel_name="inter_sequence_forward_runs", dynamic_shared=False)
    num_m_blocks = scheduler_metadata_entries(q_view.shape[0], num_sequences, info.cluster_block_m)
    cu_seqlens_q, cu_seqlens_k = (view_of(t).pointer for t in (sequences.cu_seqlens_q, sequences.cu_seqlens_k))
    visibility_bits = view_of(sequences.visibility_bits).pointer
    runtime.run(
        runs, (1, 1, 1), COUNT_THREADS, 0, RUN_TABLE_SIGNATURE,
        (
            cu_seqlens_q, cu_seqlens_k, visibility_bits, window_pointer(sequences), window_size_left,
            window_size_right, run_table, run_list, run_queue, num_sequences,
        ),
        stream,
    )
    values = (
        q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer, cu_seqlens_q, cu_seqlens_k,
        0, 0, q_view.shape[-2] // k_view.shape[-2], default_softmax_scale(softmax_scale, head_dim) * LOG2_E,
        window_size_left, window_size_right,
        *row_strides(q_view), *row_strides(k_view), *row_strides(v_view), *row_strides(out_view),
        0, lse_view.strides[0], num_sequences,
        build_scheduler_metadata(
            workspace[0], "fwd", head_dim, variant, "flash_attn_fwd", cu_seqlens_q,
            view_of(sequences.cu_visible_k).pointer, num_sequences, num_m_blocks, "forward_workspace_shapes", stream,
        ),
        visibility_bits, window_pointer(sequences), run_table, run_list, run_queue, q_view.shape[-2],
    )
    runtime.run(
        kernel, (num_m_blocks, q_view.shape[-2], 1), info.threads, info.shared_bytes, FORWARD_SIGNATURE, values, stream
    )
    listed_grid = (resident_ctas(device_id, head_dim, variant, "flash_attn_fwd_runs"), 1, 1)
    runtime.run(listed, listed_grid, info.threads, info.shared_bytes, FORWARD_SIGNATURE, values, stream)


def forward_planned(
    q_view, k_view, v_view, out_view, lse_view, workspace, sequences, softmax_scale, window_size_left,
    window_size_right, stream,
):
    device_id = runtime.current_device().device_id
    head_dim, num_head, rows = q_view.shape[-1], q_view.shape[-2], q_view.shape[0]
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    variants = sorted(forward_variants(device_id, head_dim), key=lambda entry: entry[1].cluster_block_m)
    (narrow, narrow_info), (wide, wide_info) = variants[0], variants[-1]
    narrow_list, run_table, wide_list, queues = (view_of(t).pointer for t in workspace)
    wide_workers = resident_ctas(device_id, head_dim, wide, "flash_attn_fwd_planned")
    has_wide = wide_info.block_m > narrow_info.block_m and rows >= WIDE_MIN_ROWS
    plan, _ = runtime.load_for(device_id, "fwd", head_dim, "inter_sequence_plan_blocks", wide, dynamic_shared=False)
    cu_seqlens_q, cu_seqlens_k = (view_of(t).pointer for t in (sequences.cu_seqlens_q, sequences.cu_seqlens_k))
    runtime.run(
        plan, (1, 1, 1), COUNT_THREADS, 0, PLAN_SIGNATURE,
        (
            cu_seqlens_q, cu_seqlens_k, view_of(sequences.cu_visible_k).pointer,
            view_of(sequences.visibility_bits).pointer, window_pointer(sequences), window_size_left, window_size_right,
            run_table, narrow_list, wide_list, queues, num_sequences,
            narrow_info.block_m, wide_info.block_m, WIDE_MIN_ROWS if has_wide else NO_WIDE_BLOCKS,
            -(-SMALL_BLOCK_MAX_WAVES * wide_workers // num_head),
        ),
        stream,
    )
    workers = ((wide, wide_info, wide_list, queues + QUEUE_BYTES),) if has_wide else ()
    for variant, info, run_list, run_queue in (*workers, (narrow, narrow_info, narrow_list, queues)):
        values = (
            q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer, cu_seqlens_q,
            cu_seqlens_k, 0, 0, num_head // k_view.shape[-2],
            default_softmax_scale(softmax_scale, head_dim) * LOG2_E, window_size_left, window_size_right,
            *row_strides(q_view), *row_strides(k_view), *row_strides(v_view), *row_strides(out_view),
            0, lse_view.strides[0], num_sequences, 0, view_of(sequences.visibility_bits).pointer,
            window_pointer(sequences), run_table, run_list, run_queue, num_head,
        )
        worker, _ = runtime.load_for(device_id, "fwd", head_dim, "flash_attn_fwd_planned", variant)
        grid = (resident_ctas(device_id, head_dim, variant, "flash_attn_fwd_planned"), 1, 1)
        runtime.run(worker, grid, info.threads, info.shared_bytes, FORWARD_SIGNATURE, values, stream)


def forward_megakernel(
    q_view, k_view, v_view, out_view, lse_view, workspace, sequences, softmax_scale, window_size_left,
    window_size_right, stream,
):
    device_id = runtime.current_device().device_id
    head_dim, num_head = q_view.shape[-1], q_view.shape[-2]
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    kernel, info = runtime.load_for(device_id, "fwd", head_dim, MEGAKERNEL_NAME, VARIANT)
    grid = resident_ctas(device_id, head_dim, VARIANT, MEGAKERNEL_NAME)
    values = (
        q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer,
        *row_strides(q_view)[1:], *row_strides(k_view)[1:], *row_strides(v_view)[1:], *row_strides(out_view)[1:],
        lse_view.strides[0], view_of(sequences.cu_seqlens_q).pointer, view_of(sequences.cu_seqlens_k).pointer,
        view_of(sequences.cu_visible_k).pointer, view_of(sequences.visibility_bits).pointer,
        window_pointer(sequences), window_size_left, window_size_right,
        default_softmax_scale(softmax_scale, head_dim) * LOG2_E, num_sequences, num_head,
        num_head // k_view.shape[-2], view_of(sequences.sequence_order).pointer,
        WIDE_MIN_ROWS, -(-SMALL_BLOCK_MAX_WAVES * grid // num_head),
    )
    runtime.run(kernel, (grid, 1, 1), info.threads, info.shared_bytes, MEGAKERNEL_SIGNATURE, values, stream)


def backward(
    q, k, v, out, dout, softmax_lse, dq, dk, dv, workspace, sequences, *, softmax_scale=None, window_size_left=-1,
    window_size_right=-1, stream=None,
):
    lse_log2, dpsum, dq_accum, table = workspace
    views = {name: view_of(t) for name, t in dict(
        q=q, k=k, v=v, out=out, dout=dout, lse=softmax_lse, dq=dq, dk=dk, dv=dv,
        lse_log2=lse_log2, dpsum=dpsum, dq_accum=dq_accum,
    ).items()}
    head_dim = views["q"].shape[-1]
    num_head, num_head_kv = views["q"].shape[-2], views["k"].shape[-2]
    require_supported_heads(views["q"].shape, views["k"].shape)
    require_window(window_size_left, window_size_right)
    for name in ("q", "k", "v", "out", "dout", "dq", "dk", "dv"):
        require_aligned(name, views[name], head_dim)
    device_id = runtime.current_device().device_id
    require_supported(device_id, head_dim)
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    rows_q, rows_k = views["q"].shape[0], views["k"].shape[0]
    if not rows_q and not rows_k:
        return
    main, info = runtime.load_for(device_id, "bwd", head_dim, "flash_attn_bwd", VARIANT)
    load = functools.partial(runtime.load_for, device_id, "bwd", head_dim, variant=VARIANT, dynamic_shared=False)
    preprocess, _ = load(kernel_name="flash_attn_bwd_preprocess")
    postprocess, _ = load(kernel_name="flash_attn_bwd_postprocess")
    softmax_scale = default_softmax_scale(softmax_scale, head_dim)
    cu_seqlens_q, cu_seqlens_k = (view_of(t).pointer for t in (sequences.cu_seqlens_q, sequences.cu_seqlens_k))
    padding = max(ACCUMULATOR_PADDING_ROWS, info.block_m) + SM80_SEQUENCE_ROW_ALIGNMENT * num_sequences
    accum_strides = accumulator_strides(tuple(views["q"].shape), padding)
    lse_strides = (0, views["lse"].strides[0])
    if rows_q:
        runtime.run(
            preprocess, (-(-rows_q // PREPROCESS_ROWS), num_head, 1), AUXILIARY_THREADS, 0, PREPROCESS_SIGNATURE,
            (
                views["out"].pointer, views["dout"].pointer, views["lse"].pointer, views["lse_log2"].pointer,
                views["dpsum"].pointer, views["dq_accum"].pointer, cu_seqlens_q, 0,
                *row_strides(views["out"]), *row_strides(views["dout"]), *lse_strides, *accum_strides, num_sequences,
            ),
            stream,
        )
    num_n_blocks = scheduler_metadata_entries(rows_k, num_sequences, info.block_n)
    values = (
        views["q"].pointer, views["k"].pointer, views["v"].pointer, views["dout"].pointer,
        views["lse_log2"].pointer, views["dpsum"].pointer, views["dq_accum"].pointer,
        views["dk"].pointer, views["dv"].pointer, cu_seqlens_q, cu_seqlens_k,
        0, 0, num_head // num_head_kv, softmax_scale * LOG2_E, softmax_scale, window_size_left, window_size_right,
        *row_strides(views["q"]), *row_strides(views["k"]), *row_strides(views["v"]), *row_strides(views["dout"]),
        *row_strides(views["dk"]), *row_strides(views["dv"]), *accum_strides, *lse_strides, num_sequences,
        build_scheduler_metadata(
            table, "bwd", head_dim, VARIANT, "flash_attn_bwd", cu_seqlens_k, view_of(sequences.cu_visible_q).pointer,
            num_sequences, num_n_blocks, "backward_workspace_shapes", stream,
        ),
        view_of(sequences.visibility_bits_t).pointer, window_pointer(sequences),
    )
    if num_n_blocks:
        runtime.run(
            main, (num_n_blocks, num_head_kv, 1), info.threads, info.shared_bytes, INTER_SEQUENCE_BACKWARD_SIGNATURE,
            values, stream,
        )
    if rows_q:
        runtime.run(
            postprocess, (-(-rows_q // POSTPROCESS_ROWS), num_head, 1), AUXILIARY_THREADS, 0, POSTPROCESS_SIGNATURE,
            (
                views["dq_accum"].pointer, views["dq"].pointer, cu_seqlens_q, 0, softmax_scale, *accum_strides,
                *row_strides(views["dq"]), num_sequences,
            ),
            stream,
        )
