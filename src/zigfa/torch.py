from typing import Optional, Tuple

import torch
from cuda.core import Device

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


def validate_attention_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k):
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
    require_supported_heads(q.shape, k.shape)
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


__all__ = [
    "FlashAttnFunc",
    "FlashAttnKVPackedFunc",
    "FlashAttnQKVPackedFunc",
    "FlashAttnVarlenFunc",
    "FlashAttnVarlenKVPackedFunc",
    "flash_attn_backward",
    "flash_attn_backward_packed",
    "flash_attn_forward",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
]
