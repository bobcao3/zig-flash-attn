import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple

from . import runtime
from .tensormap import BFLOAT16_BYTES

ALIGNMENT_BYTES = 16
ALIGNMENT_ELEMENTS = ALIGNMENT_BYTES // BFLOAT16_BYTES
LOG2_E = math.log2(math.e)
SCHEDULER_METADATA_THREADS = 1024
SCHEDULER_METADATA_SIGNATURE = "ppp" + "uu"
SCHEDULER_METADATA_ALIGNMENT_BYTES = 8


@dataclass(frozen=True)
class View:
    pointer: int
    shape: tuple
    strides: tuple


def view_of(tensor):
    if hasattr(tensor, "data_ptr"):
        return View(tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
    from cuda.core.utils import StridedMemoryView

    memory = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)
    strides = memory.strides or _contiguous_strides(memory.shape)
    return View(memory.ptr, tuple(memory.shape), tuple(strides))


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


class SequenceLayout(NamedTuple):
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    cu_seqlens_q_pointer: int
    cu_seqlens_k_pointer: int
    row_axis_len_q: int
    row_axis_len_k: int
    lse_batch_stride: int
    lse_head_stride: int


def sequence_layout(q_view, k_view, lse_view, cu_seqlens_q, cu_seqlens_k):
    if cu_seqlens_q is None:
        return SequenceLayout(
            batch_size=q_view.shape[0],
            seqlen_q=q_view.shape[1],
            seqlen_k=k_view.shape[1],
            cu_seqlens_q_pointer=0,
            cu_seqlens_k_pointer=0,
            row_axis_len_q=q_view.shape[1],
            row_axis_len_k=k_view.shape[1],
            lse_batch_stride=lse_view.strides[0],
            lse_head_stride=lse_view.strides[1],
        )
    return SequenceLayout(
        batch_size=cu_seqlens_q.shape[0] - 1,
        seqlen_q=0,
        seqlen_k=0,
        cu_seqlens_q_pointer=view_of(cu_seqlens_q).pointer,
        cu_seqlens_k_pointer=view_of(cu_seqlens_k).pointer,
        row_axis_len_q=q_view.shape[0],
        row_axis_len_k=k_view.shape[0],
        lse_batch_stride=0,
        lse_head_stride=lse_view.strides[0],
    )


def varlen_blocks(rows, batch_size, block):
    return (rows + batch_size * (block - 1)) // block


def scheduler_metadata_entries(rows, batch_size, block_m):
    return min(varlen_blocks(rows, batch_size, block_m), rows) if batch_size else rows


def build_scheduler_metadata(
    table, direction, head_dim, mode, kernel_name, cu_seqlens, cu_seqlens_other, batch_size, num_block, shapes_function, stream
):
    view = view_of(table)
    if math.prod(view.shape) < 2 * num_block or view.pointer % SCHEDULER_METADATA_ALIGNMENT_BYTES:
        raise ValueError(
            f"{kernel_name} needs scheduler metadata of {num_block} entries at a {SCHEDULER_METADATA_ALIGNMENT_BYTES}-byte "
            f"aligned address; pass the workspace that {shapes_function} returns"
        )
    if num_block:
        kernel, _ = runtime.load(direction, head_dim, f"{kernel_name}_prepare_varlen_num_blocks", mode, dynamic_shared=False)
        runtime.run(
            kernel,
            ((num_block + SCHEDULER_METADATA_THREADS - 1) // SCHEDULER_METADATA_THREADS, 1, 1),
            SCHEDULER_METADATA_THREADS,
            0,
            SCHEDULER_METADATA_SIGNATURE,
            (view.pointer, cu_seqlens, cu_seqlens_other, batch_size, num_block),
            stream,
        )
    return view.pointer


def default_softmax_scale(softmax_scale, head_dim):
    return head_dim**-0.5 if softmax_scale is None else softmax_scale


def is_kernel_ready(view):
    return (
        view.pointer % ALIGNMENT_BYTES == 0
        and view.strides[-1] == 1
        and all(stride % ALIGNMENT_ELEMENTS == 0 for stride in view.strides[:-1])
    )


def require_aligned(name, view, head_dim):
    if head_dim % ALIGNMENT_ELEMENTS or not is_kernel_ready(view):
        raise ValueError(
            f"{name} must have a {ALIGNMENT_BYTES}-byte aligned pointer, a contiguous last dimension "
            f"and strides that are multiples of {ALIGNMENT_ELEMENTS} elements"
        )


def require_supported_heads(q_shape, k_shape, head_dims=None):
    supported = runtime.head_dims() if head_dims is None else head_dims
    if q_shape[-1] != k_shape[-1] or q_shape[-1] not in supported:
        raise ValueError(f"head_dim must be one of {list(supported)} and equal for q and k")
    if q_shape[-2] % k_shape[-2]:
        raise ValueError("the query head count must be a multiple of the key head count")


def require_window(window_size_left, window_size_right):
    if not all(isinstance(bound, numbers.Integral) for bound in (window_size_left, window_size_right)):
        raise TypeError(
            "window bounds must be integers, "
            f"got left={type(window_size_left).__name__}, right={type(window_size_right).__name__}"
        )
    if min(window_size_left, window_size_right) < -1:
        raise ValueError(
            "window bounds must each be -1 or non-negative, "
            f"got left={window_size_left}, right={window_size_right}"
        )
