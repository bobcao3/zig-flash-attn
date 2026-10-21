import functools
import math
from typing import NamedTuple

from cuda.bindings import driver

from . import runtime, specialization
from .tensormap import as_four_dimensional, tensor_map
from .tensors import (
    LOG2_E,
    build_scheduler_metadata,
    default_softmax_scale,
    require_aligned,
    require_supported_heads,
    require_window,
    scheduler_metadata_entries,
    sequence_layout,
    varlen_blocks,
    view_of,
)


SPLIT_KV_MIN_BLOCKS = 4
SPLIT_KV_KEYS_PER_QUERY = 4
SPLIT_KV_MAX_SPLITS = 64
ACCUMULATOR_BYTES = 16


class ForwardWorkspace(NamedTuple):
    facts: specialization.ShapeFacts
    semaphore: bool
    scheduler_metadata_rows: int
    split_kv_words: int


class SplitKv(NamedTuple):
    tiles: int
    splits: int
    o_accum_warps: int
    counter_words: int
    words: int


@functools.cache
def resident_ctas(device_id, info):
    kernel, _ = runtime.load_instance(device_id, info.name, "flash_attn_fwd")
    occupancy = kernel.occupancy.max_active_blocks_per_multiprocessor(info.threads, info.shared_bytes)
    return runtime.num_sm(device_id) * max(1, occupancy)


@functools.cache
def split_kv(device_id, info, q_shape, k_shape, window=specialization.UNBOUNDED):
    batch, seqlen_q, num_head, head_dim = q_shape
    seqlen_k, num_head_kv = k_shape[1], k_shape[2]
    packing = num_head // num_head_kv if info.pack_gqa else 1
    rows = seqlen_q * packing
    tiles = batch * num_head // packing * -(-rows // info.block_m)
    keys = seqlen_k if window[0] < 0 else min(seqlen_k, window[0] + seqlen_q + max(window[1], 0))
    splits = 1
    if keys >= SPLIT_KV_KEYS_PER_QUERY * seqlen_q:
        splits = max(1, min(resident_ctas(device_id, info) // tiles, keys // info.block_n // SPLIT_KV_MIN_BLOCKS,
                            SPLIT_KV_MAX_SPLITS))
    warps = info.threads // 32
    warp_rows = info.block_m // warps
    o_accum_warps = min(warps, -(-min(rows, info.block_m) // warp_rows))
    partial_bytes = warp_rows // 16 * (head_dim // 8 + 1) * 32 * ACCUMULATOR_BYTES
    counter_words = -(-tiles // 4) * 4
    words = counter_words + tiles * splits * o_accum_warps * partial_bytes // 4
    return SplitKv(tiles, splits, o_accum_warps, counter_words, words)


@functools.cache
def forward_workspace_layout(device_id, q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k):
    facts = specialization.facts(
        q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k, runtime.num_sm(device_id)
    )
    infos = specialization.reachable(device_id, "fwd", q_shape[-1], facts)
    plans = (split_kv(device_id, info, q_shape, k_shape) for info in infos if info.split_kv and k_shape)
    return ForwardWorkspace(
        facts,
        any(info.scheduling_mode == "dynamic" for info in infos),
        min((info.cluster_block_m for info in infos if info.scheduler_metadata), default=0),
        max((plan.words for plan in plans if plan.splits > 1), default=0),
    )


@functools.cache
def forward_variant(
    device_id, q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k, window=specialization.UNBOUNDED
):
    layout = forward_workspace_layout(device_id, q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k)

    def rule(info, facts):
        splits = info.split_kv and split_kv(device_id, info, q_shape, k_shape, window).splits > 1
        return specialization.accepts(info, facts, window) and (splits or not info.split_kv)

    variant = specialization.choose(device_id, "fwd", q_shape[-1], layout.facts, rule=rule)
    specialization.log_choice(
        "forward", device_id, q_shape[-1], variant, q=q_shape, k=k_shape, window=window,
        cu_seqlens=cu_seqlens_shape, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    )
    return variant


def forward_workspace_shapes(q_shape, *, k_shape=None, cu_seqlens_shape=None, max_seqlen_q=None, max_seqlen_k=None):
    layout = forward_workspace_layout(
        runtime.current_device().device_id, q_shape, k_shape, cu_seqlens_shape, max_seqlen_q, max_seqlen_k
    )
    shapes = ((1,),) if layout.semaphore else ()
    if layout.scheduler_metadata_rows:
        rows = q_shape[0] if layout.facts.varlen else q_shape[1]
        shapes += ((scheduler_metadata_entries(rows, layout.facts.batch_size, layout.scheduler_metadata_rows), 2),)
    if layout.split_kv_words:
        shapes += ((layout.split_kv_words,),)
    return shapes


def forward(
    q,
    k,
    v,
    out,
    softmax_lse,
    workspace=(),
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
    q_view, k_view, v_view, out_view, lse_view = (view_of(t) for t in (q, k, v, out, softmax_lse))
    varlen = cu_seqlens_q is not None
    head_dim = q_view.shape[-1]
    num_head = q_view.shape[-2]
    num_head_kv = k_view.shape[-2]
    require_supported_heads(q_view.shape, k_view.shape)
    require_window(window_size_left, window_size_right)
    for name, view in (("q", q_view), ("k", k_view), ("v", v_view), ("out", out_view)):
        require_aligned(name, view, head_dim)
    layout = sequence_layout(q_view, k_view, lse_view, cu_seqlens_q, cu_seqlens_k)
    batch_strides = (0,) * 4 if varlen else tuple(view.strides[0] for view in (q_view, k_view, v_view, out_view))
    softmax_scale = default_softmax_scale(softmax_scale, head_dim)
    device_id = runtime.current_device().device_id
    key = (
        device_id,
        q_view.shape,
        k_view.shape,
        view_of(cu_seqlens_q).shape if varlen else None,
        max_seqlen_q,
        max_seqlen_k,
    )
    workspace_layout = forward_workspace_layout(*key)
    metadata_index = int(workspace_layout.semaphore)
    split_kv_index = metadata_index + bool(workspace_layout.scheduler_metadata_rows)
    if len(workspace) != split_kv_index + bool(workspace_layout.split_kv_words):
        raise ValueError("pass the workspace that forward_workspace_shapes returns for these shapes")
    if not layout.row_axis_len_q:
        return
    variant = forward_variant(*key, (window_size_left, window_size_right))
    kernel, info = runtime.load_for(device_id, "fwd", head_dim, "flash_attn_fwd", variant)
    qhead_per_kvhead = num_head // num_head_kv
    packed_heads = qhead_per_kvhead if info.pack_gqa else 1
    if info.pack_gqa and varlen:
        raise ValueError(f"{info.name} packs the query heads of batched inputs only")
    num_head_packed = num_head // packed_heads
    row_axis = -3
    num_m_blocks = (
        scheduler_metadata_entries(layout.row_axis_len_q, layout.batch_size, info.cluster_block_m)
        if info.scheduler_metadata
        else varlen_blocks(layout.row_axis_len_q, layout.batch_size, info.cluster_block_m)
        if varlen
        else (layout.row_axis_len_q * packed_heads + info.cluster_block_m - 1) // info.cluster_block_m
    )
    common = (
        layout.cu_seqlens_q_pointer,
        layout.cu_seqlens_k_pointer,
        layout.seqlen_q,
        layout.seqlen_k,
        qhead_per_kvhead,
        softmax_scale * LOG2_E,
        window_size_left,
        window_size_right,
    )
    output_strides = (batch_strides[3], out_view.strides[row_axis], out_view.strides[-2])
    lse_strides = (layout.lse_batch_stride, layout.lse_head_stride)
    if info.box_columns:
        query_rows = info.q_box_rows or info.block_m
        boxes = ((q_view, query_rows, packed_heads), (k_view, info.block_n, 1), (v_view, info.block_n, 1))
        if info.q_box_rows:
            boxes += ((out_view, info.q_box_rows, packed_heads),)
        maps = tuple(
            tensor_map(view.pointer, *as_four_dimensional(view), rows, info.box_columns, info.swizzle_bytes, heads=heads)
            for view, rows, heads in boxes
        )
        pointers = (out_view.pointer, lse_view.pointer)
        if info.scheduling_mode != "single_tile":
            dynamic = info.scheduling_mode == "dynamic"
            signature = "m" * len(maps) + "pppp" + "uuufii" + "uuu" + "p" * (6 if dynamic else 5)
            tiles = num_m_blocks * num_head_packed * (1 if varlen else layout.batch_size)
            windowed = window_size_left != -1 or window_size_right != -1
            cluster_ctas = max(info.cluster_ctas, 1)
            resident = (
                tiles
                if info.grid_per_tile_when_windowed and windowed
                else runtime.num_sm(device_id) // cluster_ctas
            )
            grid = (min(tiles, resident) * cluster_ctas, 1, 1)
            if dynamic:
                if not workspace or math.prod(view_of(workspace[0]).shape) < 1:
                    raise ValueError(
                        "this kernel needs the int32 tile_count_semaphore at the head of the workspace that "
                        "forward_workspace_shapes returns"
                    )
                tile_count_semaphore_pointer = view_of(workspace[0]).pointer
                status, = driver.cuMemsetD32Async(tile_count_semaphore_pointer, 0, 1, runtime.stream_of(stream).handle)
                if status != driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"cuMemsetD32Async failed: {status}")
            values = (
                *maps, *pointers, *common, num_m_blocks, num_head_packed, layout.batch_size,
                *output_strides, *lse_strides,
                *((tile_count_semaphore_pointer,) if dynamic else ()),
            )
        else:
            signature = "m" * len(maps) + "pppp" + "uuufii" + "p" * 5 + "u"
            grid = (num_m_blocks, num_head_packed, 1 if varlen else layout.batch_size)
            values = (*maps, *pointers, *common, *output_strides, *lse_strides, layout.batch_size)
    else:
        signature = "p" * 7 + "uuufii" + "p" * 14 + "u"
        grid = (num_m_blocks, num_head_packed, 1 if varlen else layout.batch_size)
        values = (
            q_view.pointer, k_view.pointer, v_view.pointer, out_view.pointer, lse_view.pointer,
            *common,
            batch_strides[0], q_view.strides[row_axis], q_view.strides[-2],
            batch_strides[1], k_view.strides[row_axis], k_view.strides[-2],
            batch_strides[2], v_view.strides[row_axis], v_view.strides[-2],
            *output_strides, *lse_strides, layout.batch_size,
        )
        if info.split_kv:
            plan = split_kv(device_id, info, q_view.shape, k_view.shape, (window_size_left, window_size_right))
            split_workspace = view_of(workspace[split_kv_index])
            if math.prod(split_workspace.shape) < plan.words:
                raise ValueError("the split-KV workspace is smaller than forward_workspace_shapes returns")
            counters = split_workspace.pointer
            status, = driver.cuMemsetD32Async(counters, 0, plan.tiles, runtime.stream_of(stream).handle)
            if status != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemsetD32Async failed: {status}")
            signature += "ppuu"
            grid = (num_m_blocks * plan.splits, *grid[1:])
            values += (counters + 4 * plan.counter_words, counters, plan.splits, plan.o_accum_warps)
    if info.scheduler_metadata:
        signature += "p"
        values += (
            build_scheduler_metadata(
                workspace[metadata_index], "fwd", head_dim, variant, "flash_attn_fwd", layout.cu_seqlens_q_pointer,
                layout.cu_seqlens_k_pointer, layout.batch_size, num_m_blocks, "forward_workspace_shapes", stream,
            ),
        )
    runtime.run(
        kernel, grid, info.threads, info.shared_bytes, signature, values, stream,
        pdl=info.pdl,
    )
