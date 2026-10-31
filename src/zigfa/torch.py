from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class InterSequenceMetadata:
    """Prepared sequence visibility. Callers must leave the contents unchanged."""

    _tensors: tuple[Tensor, Tensor, Tensor]


def flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    qv: Tensor | None = None,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_q: Tensor | None = None,
    seqused_k: Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    qv: Tensor | None = None,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_qkvpacked_func(
    qkv: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    num_heads_q: int | None = None,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash_attn_inter_sequence_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    visibility_bits: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    window_sizes: Tensor | None = None,
    metadata: InterSequenceMetadata | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnInterSequenceFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        visibility_bits,
        softmax_scale,
        causal,
        window_size,
        window_sizes,
        metadata,
        deterministic,
        return_attn_probs,
    )


def flash_attn_inter_sequence_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    visibility_bits: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    window_sizes: Tensor | None = None,
    metadata: InterSequenceMetadata | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    from ._impl import ops

    return ops.FlashAttnInterSequenceKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        visibility_bits,
        softmax_scale,
        causal,
        window_size,
        window_sizes,
        metadata,
        deterministic,
        return_attn_probs,
    )


def prepare_inter_sequence_metadata(
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    visibility_bits: Tensor,
) -> InterSequenceMetadata:
    """Prepare reusable metadata for inter-sequence attention.

    Pass the result unchanged as metadata to inter-sequence attention calls.
    Prepare new metadata when sequence boundaries, visibility contents, or the
    device change. Preparation uses the current Torch CUDA stream.
    """
    from ._impl import ops

    return ops.prepare_inter_sequence_metadata(cu_seqlens_q, cu_seqlens_k, visibility_bits)


__all__ = [
    "InterSequenceMetadata",
    "flash_attn_func",
    "flash_attn_inter_sequence_func",
    "flash_attn_inter_sequence_kvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "prepare_inter_sequence_metadata",
]
