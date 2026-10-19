import functools
import math
from typing import NamedTuple, Optional

from . import runtime
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
PACKED_VARIANT = "_varlen_gqa_inter_sequence"
TRANSPOSE_WARPS = 8
COUNT_THREADS = 1024
TABLE_THREADS = 256
FORWARD_SIGNATURE = "p" * 7 + "uuufii" + "p" * 14 + "u" + "pppp" + "u"
BLOCK_TABLE_ROWS = (64, 128)
BLOCK_TABLE_HEADER_BYTES = 16
BLOCK_TABLE_BLOCK_BYTES = 32
KEY_VALUE_ELEMENT_BYTES = 2
PERSISTENT_SOURCE = "flash_fwd_persistent_sm80"
PERSISTENT_PACKED_VARIANT = "_gqa"
PERSISTENT_SIGNATURE = "p" * 18 + "iif" + "uuu" + "ppp" + "uu" + "pu" + "u"
O_ACCUM_ALIGNMENT = 256
SEQUENCE_RECORD_BYTES = 64
ACCUMULATOR_BYTES = 16
MAX_WARP_HEAD_DIM = 256
SPLITS_PER_CTA = 4
INTER_SEQUENCE_BACKWARD_SIGNATURE = BACKWARD_SIGNATURE + "ppp"


class Sequences(NamedTuple):

    cu_seqlens_q: object
    cu_seqlens_k: object
    visibility_bits: object
    visibility_bits_t: object
    cu_visible_k: object
    cu_visible_q: object
    window_sizes: Optional[object] = None


def words_per_row(num_sequences):
    return (num_sequences + 31) // 32


def block_table_capacity(total_q, num_sequences, packing=1):
    return scheduler_metadata_entries(total_q * packing, num_sequences, min(BLOCK_TABLE_ROWS))


def packing_of(q_shape, num_head_kv):
    head_dim, num_head = q_shape[-1], q_shape[-2]
    if num_head_kv is None or num_head_kv == num_head:
        return 1
    device_id = runtime.current_device().device_id
    if head_dim in persistent_head_dims(device_id):
        packed = persistent_instance(head_dim, num_head // num_head_kv)
    else:
        packed = runtime.instance_name(device_id, "fwd", head_dim, PACKED_VARIANT)
    return num_head // num_head_kv if packed in runtime.manifest()[1] else 1


def persistent_instance(head_dim, packing):
    return f"{PERSISTENT_SOURCE}{PERSISTENT_PACKED_VARIANT if packing > 1 else ''}_hd{head_dim}"


def metadata_shapes(num_sequences):
    return (num_sequences, words_per_row(num_sequences)), (num_sequences + 1,), (num_sequences + 1,)


def block_table_bytes(capacity):
    return BLOCK_TABLE_HEADER_BYTES + BLOCK_TABLE_BLOCK_BYTES * capacity


class OAccum(NamedTuple):

    offset: int
    splits: int
    split_bytes: int

    def split_counters(self, workspace):
        return workspace + self.offset + self.splits * self.split_bytes


def sequence_records_offset(q_shape, num_sequences, num_head_kv=None):
    capacity = block_table_capacity(q_shape[0], num_sequences, packing_of(q_shape, num_head_kv))
    order_end = len(BLOCK_TABLE_ROWS) * block_table_bytes(capacity) + 4 * num_sequences
    return -(-order_end // SEQUENCE_RECORD_BYTES) * SEQUENCE_RECORD_BYTES


def o_accum_of(q_shape, num_sequences, num_head_kv=None):
    records_end = (
        sequence_records_offset(q_shape, num_sequences, num_head_kv) + SEQUENCE_RECORD_BYTES * num_sequences
    )
    offset = -(-records_end // O_ACCUM_ALIGNMENT) * O_ACCUM_ALIGNMENT
    device_id = runtime.current_device().device_id
    head_dim = q_shape[-1]
    if head_dim not in persistent_head_dims(device_id):
        return OAccum(records_end, 0, 0)
    instance = persistent_instance(head_dim, packing_of(q_shape, num_head_kv))
    ctas = resident_ctas(device_id, instance)
    if ctas > runtime.num_sm(device_id):
        return OAccum(records_end, 0, 0)
    threads = runtime.manifest()[1][instance].threads
    accumulators = min(head_dim, MAX_WARP_HEAD_DIM) // 8 + 1
    return OAccum(offset, SPLITS_PER_CTA * ctas, accumulators * threads * ACCUMULATOR_BYTES)


def forward_workspace_shapes(q_shape, num_sequences, num_head_kv=None):
    o_accum = o_accum_of(q_shape, num_sequences, num_head_kv)
    end = o_accum.split_counters(0) + 4 * o_accum.splits
    return ((-(-end // 4),),)


def section_heads(device_id, num_head, num_head_kv, head_dim, rows_k):
    kv_head_bytes = 2 * rows_k * head_dim * KEY_VALUE_ELEMENT_BYTES
    fitting = max(1, runtime.l2_bytes(device_id) // 2 // max(1, kv_head_bytes))
    kv_heads = max(
        divisor for divisor in range(1, num_head_kv + 1) if num_head_kv % divisor == 0 and divisor <= fitting
    )
    return kv_heads * (num_head // num_head_kv)


def block_table(workspace, capacity, block_m):
    return workspace + BLOCK_TABLE_ROWS.index(block_m) * block_table_bytes(capacity)


def forward_workspace(workspace, q_shape, num_sequences, num_head_kv=None):
    (shape,) = forward_workspace_shapes(q_shape, num_sequences, num_head_kv)
    view = view_of(workspace[0])
    if len(workspace) != 1 or math.prod(view.shape) < shape[0]:
        raise ValueError("the workspace does not match forward_workspace_shapes")
    packing = packing_of(q_shape, num_head_kv)
    return view.pointer, block_table_capacity(q_shape[0], num_sequences, packing)


@functools.cache
def forward_head_dims(device_id):
    entries = runtime.manifest()[1]
    try:
        runtime.instance_name(device_id, "fwd", 0, VARIANT)
    except RuntimeError:
        return ()
    single_tile = {
        head_dim for head_dim in runtime.head_dims()
        if runtime.instance_name(device_id, "fwd", head_dim, VARIANT) in entries
    }
    return tuple(sorted(single_tile.union(persistent_head_dims(device_id))))


def built_head_dims(device_id):
    return tuple(sorted(set(runtime.head_dims()).union(forward_head_dims(device_id))))


@functools.cache
def head_dims(device_id):
    entries = runtime.manifest()[1]
    return tuple(
        head_dim
        for head_dim in forward_head_dims(device_id)
        if runtime.instance_name(device_id, "bwd", head_dim, VARIANT) in entries
    )


def require_supported(device_id, head_dim, supported_head_dims=head_dims):
    supported = supported_head_dims(device_id)
    if head_dim not in supported:
        raise NotImplementedError(f"inter-sequence attention supports head_dim {list(supported)}, got {head_dim}")


def kernel_info(device_id, direction, head_dim):
    require_supported(device_id, head_dim)
    return runtime.manifest()[1][runtime.instance_name(device_id, direction, head_dim, VARIANT)]


def prepare_metadata(
    cu_seqlens_q, cu_seqlens_k, visibility_bits, visibility_bits_t, cu_visible_k, cu_visible_q, stream=None
):
    num_sequences = view_of(cu_seqlens_q).shape[0] - 1
    device_id = runtime.current_device().device_id
    supported = head_dims(device_id)
    if not supported:
        raise NotImplementedError("no inter-sequence kernels are built for this GPU")
    persistent = persistent_head_dims(device_id)
    if persistent:
        instance = persistent_instance(persistent[0], 1)
    else:
        instance = runtime.instance_name(device_id, "fwd", supported[0], VARIANT)
    load = functools.partial(runtime.load_instance, device_id, instance, dynamic_shared=False)
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
        count, (1, 1, 1), COUNT_THREADS, 0, "pppppp" + "u",
        (
            bits, bits_t, view_of(cu_seqlens_q).pointer, view_of(cu_seqlens_k).pointer,
            view_of(cu_visible_k).pointer, view_of(cu_visible_q).pointer, num_sequences,
        ),
        stream,
    )


@functools.cache
def persistent_head_dims(device_id):
    try:
        runtime.arch_name(device_id, inter_sequence=True)
    except RuntimeError:
        return ()
    names = (name for name in runtime.manifest()[1] if name.startswith(PERSISTENT_SOURCE))
    return tuple(sorted({int(name.rsplit("_hd", 1)[1]) for name in names}))


@functools.cache
def resident_ctas(device_id, instance):
    kernel, info = runtime.load_instance(device_id, instance, "flash_attn_fwd")
    occupancy = kernel.occupancy.max_active_blocks_per_multiprocessor(info.threads, info.shared_bytes)
    return runtime.num_sm(device_id) * max(1, occupancy)


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
    q, k, v, out, softmax_lse, workspace, sequences, *, softmax_scale=None, window_size_left=-1, window_size_right=-1,
    stream=None,
):
    q_view, k_view, v_view, out_view, lse_view = (view_of(t) for t in (q, k, v, out, softmax_lse))
    head_dim, num_head, rows = q_view.shape[-1], q_view.shape[-2], q_view.shape[0]
    device_id = runtime.current_device().device_id
    require_supported_heads(q_view.shape, k_view.shape, built_head_dims(device_id))
    require_window(window_size_left, window_size_right)
    for name, view in (("q", q_view), ("k", k_view), ("v", v_view), ("out", out_view)):
        require_aligned(name, view, head_dim)
    require_supported(device_id, head_dim, forward_head_dims)
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    num_head_kv = k_view.shape[-2]
    workspace, capacity = forward_workspace(workspace, q_view.shape, num_sequences, num_head_kv)
    if not rows:
        return
    if head_dim in persistent_head_dims(device_id):
        return forward_persistent(
            q_view, k_view, v_view, out_view, lse_view, workspace, capacity, sequences, softmax_scale,
            window_size_left, window_size_right, stream,
        )
    packing = packing_of(q_view.shape, num_head_kv)
    variant = PACKED_VARIANT if packing > 1 else VARIANT
    load = functools.partial(runtime.load_for, device_id, "fwd", head_dim, variant=variant)
    kernel, info = load(kernel_name="flash_attn_fwd")
    build, _ = load(kernel_name="inter_sequence_block_table", dynamic_shared=False)
    records = workspace + sequence_records_offset(q_view.shape, num_sequences, num_head_kv)
    heads = section_heads(device_id, num_head, num_head_kv, head_dim, k_view.shape[0]) // packing
    sections = num_head // packing // heads
    build_block_tables(
        build, workspace, capacity, sequences, window_size_left, window_size_right, stream, records=records,
        packing=packing,
    )
    values = (
        q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer,
        view_of(sequences.cu_seqlens_q).pointer, view_of(sequences.cu_seqlens_k).pointer, 0, 0,
        num_head // k_view.shape[-2], default_softmax_scale(softmax_scale, head_dim) * LOG2_E, window_size_left,
        window_size_right, *row_strides(q_view), *row_strides(k_view), *row_strides(v_view), *row_strides(out_view),
        0, lse_view.strides[0], num_sequences, view_of(sequences.visibility_bits).pointer, window_pointer(sequences),
        block_table(workspace, capacity, info.cluster_block_m), records, num_head,
    )
    grid = (capacity * heads, sections, 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, FORWARD_SIGNATURE, values, stream)


def build_block_tables(
    kernel, workspace, capacity, sequences, window_size_left, window_size_right, stream, records=0, split_counters=0,
    split_counter_count=0, packing=1,
):
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    pointers = (sequences.cu_seqlens_q, sequences.cu_seqlens_k, sequences.cu_visible_k, sequences.visibility_bits)
    values = (
        *(view_of(tensor).pointer for tensor in pointers), workspace, capacity, num_sequences, split_counters,
        split_counter_count, records, window_pointer(sequences), window_size_left, window_size_right, packing,
    )
    record_ctas = -(-num_sequences // TABLE_THREADS) if records else 0
    grid = (1 + record_ctas, 1, 1)
    signature = "ppppp" + "uu" + "pu" + "pp" + "ii" + "u"
    runtime.run(kernel, grid, TABLE_THREADS, 0, signature, values, stream)


def forward_persistent(
    q_view, k_view, v_view, out_view, lse_view, workspace, capacity, sequences, softmax_scale, window_size_left,
    window_size_right, stream,
):
    device_id = runtime.current_device().device_id
    head_dim, num_head = q_view.shape[-1], q_view.shape[-2]
    num_sequences = view_of(sequences.cu_seqlens_q).shape[0] - 1
    num_head_kv = k_view.shape[-2]
    packing = packing_of(q_view.shape, num_head_kv)
    instance = persistent_instance(head_dim, packing)
    kernel, info = runtime.load_instance(device_id, instance, "flash_attn_fwd")
    grid = resident_ctas(device_id, instance)
    o_accum = o_accum_of(q_view.shape, num_sequences, num_head_kv)
    records = workspace + sequence_records_offset(q_view.shape, num_sequences, num_head_kv)
    values = (
        q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer,
        *row_strides(q_view)[1:], *row_strides(k_view)[1:], *row_strides(v_view)[1:], *row_strides(out_view)[1:],
        lse_view.strides[0], view_of(sequences.cu_seqlens_q).pointer, view_of(sequences.cu_seqlens_k).pointer,
        view_of(sequences.visibility_bits).pointer, window_pointer(sequences), window_size_left, window_size_right,
        default_softmax_scale(softmax_scale, head_dim) * LOG2_E, num_sequences, num_head, num_head // num_head_kv,
        *(block_table(workspace, capacity, rows) for rows in BLOCK_TABLE_ROWS), records,
        section_heads(device_id, num_head, num_head_kv, head_dim, k_view.shape[0]), capacity,
        workspace + o_accum.offset, o_accum.splits, runtime.num_sm(device_id),
    )
    build, _ = runtime.load_instance(
        device_id, persistent_instance(head_dim, 1), "inter_sequence_block_table", dynamic_shared=False
    )
    build_block_tables(
        build, workspace, capacity, sequences, window_size_left, window_size_right, stream, records=records,
        split_counters=o_accum.split_counters(workspace), split_counter_count=o_accum.splits, packing=packing,
    )
    runtime.run(kernel, (grid, 1, 1), info.threads, info.shared_bytes, PERSISTENT_SIGNATURE, values, stream)


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
