from typing import NamedTuple, Optional, Tuple

import torch
from cuda.core import Device

from . import inter_sequence
from .backward import backward_workspace_shapes
from .backward import backward as launch_backward
from .forward import forward_workspace_shapes
from .forward import forward as launch_forward
from .tensors import default_softmax_scale, is_kernel_ready, require_supported_heads, view_of


def current_stream_handle():
    return torch.cuda.current_stream().cuda_stream


def bind_current_device(tensor):
    Device(tensor.device.index).set_current()


def softmax_lse_shape(q, varlen):
    if varlen:
        return (q.shape[1], q.shape[0])
    return (q.shape[0], q.shape[2], q.shape[1])


def allocate_forward_outputs(q, varlen):
    out = q.new_empty(q.shape)
    softmax_lse = q.new_empty(softmax_lse_shape(q, varlen), dtype=torch.float32)
    return out, softmax_lse


def allocate_gradients(q, k, v):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def clone_unless_kernel_ready(tensor):
    if is_kernel_ready(view_of(tensor)):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def empty_unless_kernel_ready(tensor):
    if is_kernel_ready(view_of(tensor)):
        return tensor
    return tensor.new_empty(tensor.shape)


def maybe_contiguous(tensor):
    return tensor.contiguous() if tensor is not None and tensor.stride(-1) != 1 else tensor


def require_bfloat16(**tensors):
    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")


def require_cu_seqlens(cu_seqlens_q, cu_seqlens_k):
    for name, cu_seqlens in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
            raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")


def validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, head_dims=None):
    varlen = cu_seqlens_q is not None
    if varlen != (cu_seqlens_k is not None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be given together")
    require_bfloat16(q=q, k=k, v=v)
    expected_dimensions = 3 if varlen else 4
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != expected_dimensions:
            raise ValueError(f"{name} must have {expected_dimensions} dimensions, got {tensor.dim()}")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    require_supported_heads(q.shape, k.shape, head_dims)
    if varlen:
        require_cu_seqlens(cu_seqlens_q, cu_seqlens_k)
    elif q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same batch size")


def validate_backward_operands(dout, q, k, v, out, softmax_lse, dq, dk, dv, varlen):
    require_bfloat16(dout=dout, out=out, dq=dq, dk=dk, dv=dv)
    if dout.shape != q.shape or out.shape != q.shape or dq.shape != q.shape:
        raise ValueError("dout, out and dq must have the shape of q")
    if dk.shape != k.shape or dv.shape != v.shape:
        raise ValueError("dk and dv must have the shape of k")
    if softmax_lse.dtype != torch.float32 or tuple(softmax_lse.shape) != tuple(softmax_lse_shape(q, varlen)):
        raise ValueError(
            "softmax_lse must be a float32 tensor shaped (batch, num_heads_q, seqlen_q) or (num_heads_q, total_q)"
        )


def cu_seqlens_shape(cu_seqlens):
    return None if cu_seqlens is None else tuple(cu_seqlens.shape)


def backward_into(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
    dq,
    dk,
    dv,
):
    validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    validate_backward_operands(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q is not None)
    bind_current_device(q)
    dout, q, k, v, out = (clone_unless_kernel_ready(tensor) for tensor in (dout, q, k, v, out))
    softmax_lse = softmax_lse.contiguous()
    gradients = tuple(empty_unless_kernel_ready(tensor) for tensor in (dq, dk, dv))
    hints = dict(max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)
    workspace = tuple(
        torch.empty(shape, dtype=torch.float32, device=q.device)
        for shape in backward_workspace_shapes(
            tuple(q.shape),
            tuple(softmax_lse.shape),
            tuple(k.shape),
            window_size_left,
            window_size_right,
            cu_seqlens_shape=cu_seqlens_shape(cu_seqlens_q),
            **hints,
        )
    )
    launch_backward(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        *gradients,
        workspace,
        cu_seqlens_q=maybe_contiguous(cu_seqlens_q),
        cu_seqlens_k=maybe_contiguous(cu_seqlens_k),
        **hints,
        softmax_scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        stream=current_stream_handle(),
    )
    for target, source in zip((dq, dk, dv), gradients):
        if source is not target:
            target.copy_(source)


@torch.library.custom_op("zigfa::flash_attn_forward", mutates_args=(), device_types="cuda")
def flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    bind_current_device(q)
    q, k, v = (clone_unless_kernel_ready(tensor) for tensor in (q, k, v))
    out, softmax_lse = allocate_forward_outputs(q, cu_seqlens_q is not None)
    hints = dict(max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)
    workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device=q.device)
        for shape in forward_workspace_shapes(
            tuple(q.shape), k_shape=tuple(k.shape), cu_seqlens_shape=cu_seqlens_shape(cu_seqlens_q), **hints
        )
    )
    launch_forward(
        q,
        k,
        v,
        out,
        softmax_lse,
        workspace,
        cu_seqlens_q=maybe_contiguous(cu_seqlens_q),
        cu_seqlens_k=maybe_contiguous(cu_seqlens_k),
        **hints,
        softmax_scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        stream=current_stream_handle(),
    )
    return out, softmax_lse


@flash_attn_forward.register_fake
def flash_attn_forward_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
):
    return allocate_forward_outputs(q, cu_seqlens_q is not None)


@torch.library.custom_op("zigfa::flash_attn_backward", mutates_args=(), device_types="cuda")
def flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradients = allocate_gradients(q, k, v)
    backward_into(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        window_size_left,
        window_size_right,
        *gradients,
    )
    return gradients


@flash_attn_backward.register_fake
def flash_attn_backward_fake(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
):
    return allocate_gradients(q, k, v)


@torch.library.custom_op(
    "zigfa::flash_attn_backward_packed", mutates_args=("dq", "dpacked"), device_types="cuda"
)
def flash_attn_backward_packed(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
    dq: Optional[torch.Tensor],
    dpacked: torch.Tensor,
) -> None:
    gradients = packed_gradient_views(q, dq, dpacked)
    backward_into(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        window_size_left,
        window_size_right,
        *gradients,
    )


@flash_attn_backward_packed.register_fake
def flash_attn_backward_packed_fake(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
    dq,
    dpacked,
):
    return None


def reject_unsupported_options(
    dropout_p=0.0,
    alibi_slopes=None,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    seqused_q=None,
    seqused_k=None,
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    deterministic=False,
):
    unsupported_tensors = dict(
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
    )
    for name, value in unsupported_tensors.items():
        if value is not None:
            raise NotImplementedError(f"{name} is not supported")
    if dropout_p != 0:
        raise NotImplementedError(f"dropout_p={dropout_p} is not supported; only 0.0 is")
    if attention_chunk != 0:
        raise NotImplementedError(f"attention_chunk={attention_chunk} is not supported; only 0 is")
    if softcap != 0:
        raise NotImplementedError(f"softcap={softcap} is not supported; only 0.0 is")
    if num_splits not in (0, 1):
        raise NotImplementedError(f"num_splits={num_splits} is not supported; only 0 and 1 are")
    if deterministic:
        raise NotImplementedError(
            "deterministic=True is not supported: the backward pass accumulates dQ with floating-point atomics"
        )


def pad_gradients(ctx, *gradients):
    return (*gradients, *(None,) * (len(ctx.needs_input_grad) - len(gradients)))


def run_forward(
    ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, return_softmax
):
    softmax_scale = default_softmax_scale(softmax_scale, q.shape[-1])
    window_size_left, window_size_right = window_size
    window = (window_size_left, 0 if causal else window_size_right)
    if cu_seqlens_q is None:
        max_seqlen_q, max_seqlen_k = q.shape[1], k.shape[1]
    out, softmax_lse = flash_attn_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, *window
    )
    ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.window = window
    if not return_softmax:
        return out
    ctx.mark_non_differentiable(softmax_lse)
    return out, softmax_lse


def backward_arguments(ctx, dout):
    q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    return (
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.softmax_scale,
        *ctx.window,
    )


def split_qkv(qkv, num_heads_q):
    if qkv.dim() == 5:
        if qkv.shape[-3] != 3:
            raise ValueError(f"a five-dimensional qkv must have size 3 in dimension -3, got {qkv.shape[-3]}")
        return qkv.unbind(dim=-3)
    if qkv.dim() != 4 or num_heads_q is None:
        raise ValueError("qkv must have five dimensions, or four dimensions together with num_heads_q")
    num_heads_k = (qkv.shape[2] - num_heads_q) // 2
    if num_heads_q + 2 * num_heads_k != qkv.shape[2]:
        raise ValueError("qkv head dimension must equal num_heads_q plus twice the key head count")
    return qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)


def stacked_shape(tensor, count):
    return (*tensor.shape[:-2], count, *tensor.shape[-2:])


def packed_gradient_views(q, dq, dpacked):
    if dq is None:
        return split_qkv(dpacked, q.shape[-2])
    dk, dv = dpacked.unbind(dim=-3)
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        reject_unsupported_options(
            qv=qv,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            deterministic=deterministic,
        )
        return run_forward(ctx, q, k, v, None, None, None, None, softmax_scale, causal, window_size, return_softmax)

    @staticmethod
    def backward(ctx, dout, *unused):
        dq, dk, dv = flash_attn_backward(*backward_arguments(ctx, dout))
        return pad_gradients(ctx, dq, dk, dv)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        reject_unsupported_options(
            qv=qv,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            deterministic=deterministic,
        )
        return run_forward(
            ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, return_softmax
        )

    @staticmethod
    def backward(ctx, dout, *unused):
        dq, dk, dv = flash_attn_backward(*backward_arguments(ctx, dout))
        return pad_gradients(ctx, dq, dk, dv)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_softmax=False,
    ):
        reject_unsupported_options(
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_chunk=attention_chunk,
            softcap=softcap,
            deterministic=deterministic,
        )
        q, k, v = split_qkv(qkv, num_heads_q)
        ctx.qkv_dimensions = qkv.dim()
        return run_forward(ctx, q, k, v, None, None, None, None, softmax_scale, causal, window_size, return_softmax)

    @staticmethod
    def backward(ctx, dout, *unused):
        arguments = backward_arguments(ctx, dout)
        q, k = arguments[1], arguments[2]
        if ctx.qkv_dimensions == 5:
            dqkv = q.new_empty(stacked_shape(q, 3))
        else:
            dqkv = q.new_empty((*q.shape[:-2], q.shape[-2] + 2 * k.shape[-2], q.shape[-1]))
        flash_attn_backward_packed(*arguments, None, dqkv)
        return pad_gradients(ctx, dqkv)


def kvpacked_backward(ctx, dout):
    arguments = backward_arguments(ctx, dout)
    q, k = arguments[1], arguments[2]
    dq = q.new_empty(q.shape)
    dkv = k.new_empty(stacked_shape(k, 2))
    flash_attn_backward_packed(*arguments, dq, dkv)
    return pad_gradients(ctx, dq, dkv)


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_softmax=False,
    ):
        reject_unsupported_options(
            dropout_p=dropout_p, alibi_slopes=alibi_slopes, softcap=softcap, deterministic=deterministic
        )
        k, v = kv.unbind(dim=-3)
        return run_forward(ctx, q, k, v, None, None, None, None, softmax_scale, causal, window_size, return_softmax)

    @staticmethod
    def backward(ctx, dout, *unused):
        return kvpacked_backward(ctx, dout)


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_softmax=False,
    ):
        reject_unsupported_options(
            dropout_p=dropout_p, alibi_slopes=alibi_slopes, softcap=softcap, deterministic=deterministic
        )
        k, v = kv.unbind(dim=-3)
        return run_forward(
            ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, return_softmax
        )

    @staticmethod
    def backward(ctx, dout, *unused):
        return kvpacked_backward(ctx, dout)


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    return FlashAttnFunc.apply(
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
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    return FlashAttnVarlenFunc.apply(
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
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
):
    return FlashAttnQKVPackedFunc.apply(
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
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnKVPackedFunc.apply(
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
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnVarlenKVPackedFunc.apply(
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


class InterSequenceSchedulerMetadata(NamedTuple):

    visibility_bits_t: torch.Tensor
    cu_visible_k: torch.Tensor
    cu_visible_q: torch.Tensor


def require_int32(name, tensor, shape):
    if tensor.dtype != torch.int32 or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous int32 tensor of shape {tuple(shape)}, "
            f"got {tensor.dtype} {tuple(tensor.shape)}"
        )


def validate_inter_sequence_layout(cu_seqlens_q, cu_seqlens_k, visibility_bits, window_sizes=None):
    require_cu_seqlens(cu_seqlens_q, cu_seqlens_k)
    num_sequences = cu_seqlens_q.shape[0] - 1
    if num_sequences < 1:
        raise ValueError("cu_seqlens_q must describe at least one sequence")
    require_int32("visibility_bits", visibility_bits, (num_sequences, inter_sequence.words_per_row(num_sequences)))
    if window_sizes is not None:
        require_int32("window_sizes", window_sizes, (2, num_sequences))
    return num_sequences


def allocate_inter_sequence_metadata(cu_seqlens_q):
    shapes = inter_sequence.metadata_shapes(cu_seqlens_q.shape[0] - 1)
    return tuple(cu_seqlens_q.new_empty(shape) for shape in shapes)


@torch.library.custom_op("zigfa::inter_sequence_metadata", mutates_args=(), device_types="cuda")
def inter_sequence_metadata(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, visibility_bits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_inter_sequence_layout(cu_seqlens_q, cu_seqlens_k, visibility_bits)
    bind_current_device(visibility_bits)
    metadata = allocate_inter_sequence_metadata(cu_seqlens_q)
    inter_sequence.prepare_metadata(
        cu_seqlens_q, cu_seqlens_k, visibility_bits, *metadata, stream=current_stream_handle()
    )
    return metadata


@inter_sequence_metadata.register_fake
def inter_sequence_metadata_fake(cu_seqlens_q, cu_seqlens_k, visibility_bits):
    return allocate_inter_sequence_metadata(cu_seqlens_q)


def get_inter_sequence_scheduler_metadata(cu_seqlens_q, cu_seqlens_k, visibility_bits):
    return InterSequenceSchedulerMetadata(*inter_sequence_metadata(cu_seqlens_q, cu_seqlens_k, visibility_bits))


def inter_sequence_layout(
    cu_seqlens_q, cu_seqlens_k, visibility_bits, visibility_bits_t, cu_visible_k, cu_visible_q, window_sizes
):
    return inter_sequence.Sequences(
        maybe_contiguous(cu_seqlens_q),
        maybe_contiguous(cu_seqlens_k),
        visibility_bits,
        visibility_bits_t,
        cu_visible_k,
        cu_visible_q,
        window_sizes,
    )


@torch.library.custom_op("zigfa::inter_sequence_forward", mutates_args=(), device_types="cuda")
def inter_sequence_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    visibility_bits: torch.Tensor,
    visibility_bits_t: torch.Tensor,
    cu_visible_k: torch.Tensor,
    cu_visible_q: torch.Tensor,
    window_sizes: Optional[torch.Tensor],
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dims = inter_sequence.built_head_dims(q.device.index)
    validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, head_dims)
    num_sequences = validate_inter_sequence_layout(cu_seqlens_q, cu_seqlens_k, visibility_bits, window_sizes)
    bind_current_device(q)
    q, k, v = (clone_unless_kernel_ready(tensor) for tensor in (q, k, v))
    out, softmax_lse = allocate_forward_outputs(q, True)
    workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device=q.device)
        for shape in inter_sequence.forward_workspace_shapes(tuple(q.shape), num_sequences, k.shape[-2])
    )
    inter_sequence.forward(
        q,
        k,
        v,
        out,
        softmax_lse,
        workspace,
        inter_sequence_layout(
            cu_seqlens_q, cu_seqlens_k, visibility_bits, visibility_bits_t, cu_visible_k, cu_visible_q, window_sizes
        ),
        softmax_scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        stream=current_stream_handle(),
    )
    return out, softmax_lse


@inter_sequence_forward.register_fake
def inter_sequence_forward_fake(q, k, v, *unused):
    return allocate_forward_outputs(q, True)


def inter_sequence_backward_into(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    visibility_bits,
    visibility_bits_t,
    cu_visible_k,
    cu_visible_q,
    window_sizes,
    softmax_scale,
    window_size_left,
    window_size_right,
    dq,
    dk,
    dv,
):
    validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    validate_backward_operands(dout, q, k, v, out, softmax_lse, dq, dk, dv, True)
    num_sequences = validate_inter_sequence_layout(cu_seqlens_q, cu_seqlens_k, visibility_bits, window_sizes)
    bind_current_device(q)
    dout, q, k, v, out = (clone_unless_kernel_ready(tensor) for tensor in (dout, q, k, v, out))
    softmax_lse = softmax_lse.contiguous()
    gradients = tuple(empty_unless_kernel_ready(tensor) for tensor in (dq, dk, dv))
    *accumulators, table = inter_sequence.backward_workspace_shapes(
        tuple(q.shape), tuple(softmax_lse.shape), tuple(k.shape), num_sequences
    )
    workspace = (
        *(torch.empty(shape, dtype=torch.float32, device=q.device) for shape in accumulators),
        torch.empty(table, dtype=torch.int32, device=q.device),
    )
    inter_sequence.backward(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        *gradients,
        workspace,
        inter_sequence_layout(
            cu_seqlens_q, cu_seqlens_k, visibility_bits, visibility_bits_t, cu_visible_k, cu_visible_q, window_sizes
        ),
        softmax_scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        stream=current_stream_handle(),
    )
    for target, source in zip((dq, dk, dv), gradients):
        if source is not target:
            target.copy_(source)


@torch.library.custom_op("zigfa::inter_sequence_backward", mutates_args=(), device_types="cuda")
def inter_sequence_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    visibility_bits: torch.Tensor,
    visibility_bits_t: torch.Tensor,
    cu_visible_k: torch.Tensor,
    cu_visible_q: torch.Tensor,
    window_sizes: Optional[torch.Tensor],
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradients = allocate_gradients(q, k, v)
    inter_sequence_backward_into(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        visibility_bits,
        visibility_bits_t,
        cu_visible_k,
        cu_visible_q,
        window_sizes,
        softmax_scale,
        window_size_left,
        window_size_right,
        *gradients,
    )
    return gradients


@inter_sequence_backward.register_fake
def inter_sequence_backward_fake(dout, q, k, v, *unused):
    return allocate_gradients(q, k, v)


@torch.library.custom_op(
    "zigfa::inter_sequence_backward_packed", mutates_args=("dq", "dpacked"), device_types="cuda"
)
def inter_sequence_backward_packed(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    visibility_bits: torch.Tensor,
    visibility_bits_t: torch.Tensor,
    cu_visible_k: torch.Tensor,
    cu_visible_q: torch.Tensor,
    window_sizes: Optional[torch.Tensor],
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
    dq: torch.Tensor,
    dpacked: torch.Tensor,
) -> None:
    inter_sequence_backward_into(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        visibility_bits,
        visibility_bits_t,
        cu_visible_k,
        cu_visible_q,
        window_sizes,
        softmax_scale,
        window_size_left,
        window_size_right,
        *packed_gradient_views(q, dq, dpacked),
    )


@inter_sequence_backward_packed.register_fake
def inter_sequence_backward_packed_fake(*unused):
    return None


def inter_sequence_window(causal, window_size, window_sizes):
    left, right = window_size
    if window_sizes is not None and (causal or (left, right) != (-1, -1)):
        raise ValueError("window_sizes cannot be combined with causal=True or a window_size")
    return left, 0 if causal else right


def run_inter_sequence_forward(
    ctx,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    visibility_bits,
    softmax_scale,
    causal,
    window_size,
    window_sizes,
    scheduler_metadata,
    deterministic,
    return_softmax,
):
    reject_unsupported_options(deterministic=deterministic)
    softmax_scale = default_softmax_scale(softmax_scale, q.shape[-1])
    window = inter_sequence_window(causal, window_size, window_sizes)
    if scheduler_metadata is None:
        scheduler_metadata = get_inter_sequence_scheduler_metadata(cu_seqlens_q, cu_seqlens_k, visibility_bits)
    layout = (cu_seqlens_q, cu_seqlens_k, visibility_bits, *scheduler_metadata, window_sizes)
    out, softmax_lse = inter_sequence_forward(q, k, v, *layout, softmax_scale, *window)
    ctx.save_for_backward(q, k, v, out, softmax_lse, *layout)
    ctx.softmax_scale = softmax_scale
    ctx.window = window
    if not return_softmax:
        return out
    ctx.mark_non_differentiable(softmax_lse)
    return out, softmax_lse


def inter_sequence_backward_arguments(ctx, dout):
    return (dout, *ctx.saved_tensors, ctx.softmax_scale, *ctx.window)


class FlashAttnInterSequenceFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        window_size=(-1, -1),
        window_sizes=None,
        scheduler_metadata=None,
        deterministic=False,
        return_softmax=False,
    ):
        return run_inter_sequence_forward(
            ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, visibility_bits, softmax_scale, causal, window_size, window_sizes,
            scheduler_metadata, deterministic, return_softmax,
        )

    @staticmethod
    def backward(ctx, dout, *unused):
        dq, dk, dv = inter_sequence_backward(*inter_sequence_backward_arguments(ctx, dout))
        return pad_gradients(ctx, dq, dk, dv)


class FlashAttnInterSequenceKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        visibility_bits,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        window_sizes=None,
        scheduler_metadata=None,
        deterministic=False,
        return_softmax=False,
    ):
        k, v = kv.unbind(dim=-3)
        return run_inter_sequence_forward(
            ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, visibility_bits, softmax_scale, causal, window_size, window_sizes,
            scheduler_metadata, deterministic, return_softmax,
        )

    @staticmethod
    def backward(ctx, dout, *unused):
        arguments = inter_sequence_backward_arguments(ctx, dout)
        q, k = arguments[1], arguments[2]
        dq = q.new_empty(q.shape)
        dkv = k.new_empty(stacked_shape(k, 2))
        inter_sequence_backward_packed(*arguments, dq, dkv)
        return pad_gradients(ctx, dq, dkv)


def flash_attn_inter_sequence_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    visibility_bits,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    window_sizes=None,
    scheduler_metadata=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnInterSequenceFunc.apply(
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
        scheduler_metadata,
        deterministic,
        return_attn_probs,
    )


def flash_attn_inter_sequence_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    visibility_bits,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    window_sizes=None,
    scheduler_metadata=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnInterSequenceKVPackedFunc.apply(
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
        scheduler_metadata,
        deterministic,
        return_attn_probs,
    )


__all__ = [
    "FlashAttnFunc",
    "FlashAttnInterSequenceFunc",
    "FlashAttnInterSequenceKVPackedFunc",
    "FlashAttnKVPackedFunc",
    "FlashAttnQKVPackedFunc",
    "FlashAttnVarlenFunc",
    "FlashAttnVarlenKVPackedFunc",
    "flash_attn_backward",
    "flash_attn_backward_packed",
    "flash_attn_forward",
    "flash_attn_func",
    "flash_attn_inter_sequence_func",
    "flash_attn_inter_sequence_kvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "get_inter_sequence_scheduler_metadata",
    "InterSequenceSchedulerMetadata",
]
