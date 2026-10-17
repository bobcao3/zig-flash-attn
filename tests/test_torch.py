import inspect
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext
from torch.library import opcheck

from checks import queue_forward_checks, queue_gradient_checks
from conftest import clear_selection
from reference import attention_ref, varlen_attention_ref
from shapes import HEAD_DIM, HEAD_DIMS, HEAD_LAYOUTS, cu_seqlens_of, cuda_seed, uniform_lengths, window_for
from zigfa.torch import (
    flash_attn_backward,
    flash_attn_backward_packed,
    flash_attn_forward,
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
)

OPCHECK_TESTS = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
DYNAMIC_SHAPES = [(2, 100, 130), (3, 64, 64), (4, 200, 96)]
SOFTMAX_SCALE = HEAD_DIM**-0.5


def keep(tensor):
    return tensor


def stride_two(tensor):
    holder = tensor.new_zeros(*tensor.shape, 2)
    holder[..., 0] = tensor
    return holder[..., 0]


def padded_rows(tensor):
    holder = tensor.new_zeros(*tensor.shape[:-1], tensor.shape[-1] + 4)
    holder[..., : tensor.shape[-1]] = tensor
    return holder[..., : tensor.shape[-1]]


def misaligned_pointer(tensor):
    holder = tensor.new_zeros(tensor.numel() + 3)
    view = holder[3 : 3 + tensor.numel()].view(tensor.shape)
    view.copy_(tensor)
    return view


def rows_major_in_batch(tensor):
    return tensor.movedim(-3, 0).contiguous().movedim(0, -3)


LAYOUTS = {
    "stride_two": stride_two,
    "padded_rows": padded_rows,
    "misaligned_pointer": misaligned_pointer,
    "rows_major_in_batch": rows_major_in_batch,
}


@dataclass(frozen=True)
class Case:
    arguments: tuple
    options: dict
    apply: Callable
    unpack: Callable
    reference: Callable
    dynamic_dims: tuple

    @property
    def leaves(self):
        return [a for a in self.arguments if isinstance(a, torch.Tensor) and a.requires_grad]


def random_leaf(shape, layout):
    return layout(torch.randn(*shape, device="cuda", dtype=torch.bfloat16)).requires_grad_()


def unbind_kv(tensors):
    return tensors[0], *tensors[1].unbind(dim=-3)


def reference_of(function, *arguments, **options):
    def reference(q, k, v, upcast=True):
        return function(q, k, v, *arguments, upcast=upcast, **options)

    return reference


def batched_case(
    batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, layout=keep, softmax_scale=None
):
    q = random_leaf((batch, seqlen_q, num_head, head_dim), layout)
    k = random_leaf((batch, seqlen_k, num_head_kv, head_dim), layout)
    v = random_leaf((batch, seqlen_k, num_head_kv, head_dim), layout)
    return Case(
        (q, k, v),
        dict(causal=causal, window_size=window, softmax_scale=softmax_scale),
        flash_attn_func,
        lambda tensors: tensors,
        reference_of(attention_ref, causal=causal, window_size=window, softmax_scale=softmax_scale),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    )


def varlen_tensors(batch, seqlen_q, seqlen_k, seed):
    cu_seqlens_q = cu_seqlens_of(uniform_lengths(batch, 1, seqlen_q, torch.Generator().manual_seed(seed))).cuda(non_blocking=True)
    cu_seqlens_k = cu_seqlens_of(uniform_lengths(batch, 1, seqlen_k, torch.Generator().manual_seed(seed + 1))).cuda(non_blocking=True)
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
    max_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())
    return cu_seqlens_q, cu_seqlens_k, max_q, max_k


def varlen_case(
    batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, layout=keep, softmax_scale=None
):
    cu_q, cu_k, max_q, max_k = varlen_tensors(batch, seqlen_q, seqlen_k, batch + seqlen_q)
    q = random_leaf((int(cu_q[-1]), num_head, head_dim), layout)
    k = random_leaf((int(cu_k[-1]), num_head_kv, head_dim), layout)
    v = random_leaf((int(cu_k[-1]), num_head_kv, head_dim), layout)
    return Case(
        (q, k, v, cu_q, cu_k, max_q, max_k),
        dict(causal=causal, window_size=window, softmax_scale=softmax_scale),
        flash_attn_varlen_func,
        lambda tensors: tensors,
        reference_of(varlen_attention_ref, cu_q, cu_k, causal, window, softmax_scale=softmax_scale),
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    )


def qkvpacked_case(
    batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, layout=keep, softmax_scale=None
):
    if num_head == num_head_kv:
        qkv = random_leaf((batch, seqlen_q, 3, num_head, head_dim), layout)
        num_heads_q = None
        unpack = lambda tensors: tensors[0].unbind(dim=-3)
    else:
        qkv = random_leaf((batch, seqlen_q, num_head + 2 * num_head_kv, head_dim), layout)
        num_heads_q = num_head
        unpack = lambda tensors: tensors[0].split([num_head, num_head_kv, num_head_kv], dim=-2)
    return Case(
        (qkv,),
        dict(causal=causal, window_size=window, softmax_scale=softmax_scale, num_heads_q=num_heads_q),
        flash_attn_qkvpacked_func,
        unpack,
        reference_of(attention_ref, causal=causal, window_size=window, softmax_scale=softmax_scale),
        ((0, 0), (0, 1)),
    )


def kvpacked_case(
    batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, layout=keep, softmax_scale=None
):
    q = random_leaf((batch, seqlen_q, num_head, head_dim), layout)
    kv = random_leaf((batch, seqlen_k, 2, num_head_kv, head_dim), layout)
    return Case(
        (q, kv),
        dict(causal=causal, window_size=window, softmax_scale=softmax_scale),
        flash_attn_kvpacked_func,
        unbind_kv,
        reference_of(attention_ref, causal=causal, window_size=window, softmax_scale=softmax_scale),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )


def varlen_kvpacked_case(
    batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, layout=keep, softmax_scale=None
):
    cu_q, cu_k, max_q, max_k = varlen_tensors(batch, seqlen_q, seqlen_k, batch + seqlen_q)
    q = random_leaf((int(cu_q[-1]), num_head, head_dim), layout)
    kv = random_leaf((int(cu_k[-1]), 2, num_head_kv, head_dim), layout)
    return Case(
        (q, kv, cu_q, cu_k, max_q, max_k),
        dict(causal=causal, window_size=window, softmax_scale=softmax_scale),
        flash_attn_varlen_kvpacked_func,
        unbind_kv,
        reference_of(varlen_attention_ref, cu_q, cu_k, causal, window, softmax_scale=softmax_scale),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    )


VARIANTS = {
    "batched": batched_case,
    "varlen": varlen_case,
    "qkvpacked": qkvpacked_case,
    "kvpacked": kvpacked_case,
    "varlen_kvpacked": varlen_kvpacked_case,
}


def build_case(
    variant, head_dim, window_name, layout_name="gqa", shape=(2, 113, 203), tensor_layout=keep, softmax_scale=None
):
    batch, seqlen_q, seqlen_k = shape
    window, causal = window_for(window_name, seqlen_q if variant == "qkvpacked" else seqlen_k)
    num_head, num_head_kv = HEAD_LAYOUTS[layout_name]
    return VARIANTS[variant](
        batch, seqlen_q, seqlen_k, num_head, num_head_kv, head_dim, causal, window, tensor_layout, softmax_scale
    )


def check_case(case, apply=None, gradient_layout=keep, return_attn_probs=False, **options):
    returned = (apply or case.apply)(
        *case.arguments, return_attn_probs=return_attn_probs, **case.options, **options
    )
    out, lse = returned if return_attn_probs else (returned, None)
    if return_attn_probs:
        assert not lse.requires_grad
    dout = gradient_layout(torch.randn_like(out))
    out.backward(dout)
    inputs = case.unpack([leaf.detach() for leaf in case.leaves])
    gradients = case.unpack([leaf.grad for leaf in case.leaves])
    queue_forward_checks([(out, lse)], case.reference, *inputs)
    queue_gradient_checks([gradients], case.reference, *inputs, dout)


MATRIX_LAYOUTS = {
    "batched": ["gqa"],
    "varlen": list(HEAD_LAYOUTS),
    "qkvpacked": ["mha", "gqa"],
    "kvpacked": ["gqa"],
    "varlen_kvpacked": ["gqa"],
}


@pytest.mark.parametrize(
    "variant,layout_name,window_name",
    [
        (variant, layout_name, window_name)
        for variant, layout_names in MATRIX_LAYOUTS.items()
        for layout_name in layout_names
        for window_name in ("dense", "causal", "local")
        if variant != "varlen" or window_name != "dense"
    ],
)
def test_forward_and_backward_match_reference(variant, layout_name, window_name):
    cuda_seed(HEAD_DIM + len(variant) + len(layout_name))
    softmax_scale = 0.1 if window_name == "local" else None
    case = build_case(variant, HEAD_DIM, window_name, layout_name, softmax_scale=softmax_scale)
    check_case(case, return_attn_probs=True)


ACCEPTED_OPTIONS = {
    "batched": dict(pack_gqa=True, sm_margin=4, num_splits=0),
    "varlen": dict(pack_gqa=True, sm_margin=4, num_splits=0),
    "qkvpacked": dict(sm_margin=4),
}


@pytest.mark.parametrize("variant", ACCEPTED_OPTIONS)
def test_accepted_options_match_reference(variant):
    cuda_seed(len(variant))
    case = build_case(variant, HEAD_DIM, "causal", shape=(2, 64, 64))
    check_case(case, **ACCEPTED_OPTIONS[variant])


@pytest.mark.parametrize("variant", ["batched", "varlen"])
@pytest.mark.parametrize("window", [(16, 5)])
def test_causal_ignores_the_right_window_bound(variant, window):
    cuda_seed(window[0])
    num_head, num_head_kv = HEAD_LAYOUTS["gqa"]
    check_case(VARIANTS[variant](2, 256, 256, num_head, num_head_kv, HEAD_DIM, True, window), return_attn_probs=True)


@pytest.mark.parametrize(
    "variant,shape",
    [
        ("batched", (1, 300, 129)),
        ("batched", (1, 129, 300)),
        ("qkvpacked", (1, 1, 1)),
        ("qkvpacked", (1, 300, 300)),
        ("qkvpacked", (1, 129, 129)),
    ],
)
def test_unusual_sequence_lengths(variant, shape):
    cuda_seed(sum(shape))
    check_case(build_case(variant, HEAD_DIM, "causal", "gqa", shape))


@pytest.mark.parametrize(
    "variant,layout_name",
    [
        (variant, layout_name)
        for variant in ("batched", "varlen", "qkvpacked")
        for layout_name in LAYOUTS
        if (variant, layout_name) != ("varlen", "rows_major_in_batch")
    ],
)
def test_irregular_input_layouts(variant, layout_name):
    cuda_seed(len(variant) + len(layout_name))
    layout = LAYOUTS[layout_name]
    check_case(build_case(variant, HEAD_DIM, "local", "gqa", tensor_layout=layout), gradient_layout=layout)


@pytest.mark.parametrize("variant", ["qkvpacked", "kvpacked", "varlen_kvpacked"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_every_head_dim_matches_reference(variant, head_dim):
    cuda_seed(head_dim + len(variant))
    check_case(build_case(variant, head_dim, "causal", shape=(2, 64, 64)), return_attn_probs=True)


def test_strided_packed_gradient_buffer_receives_the_gradients():
    cuda_seed(13)
    q, k, v = (torch.randn(2, 64, 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    window = (-1, -1)
    out, lse = flash_attn_forward(q, k, v, None, None, 64, 64, SOFTMAX_SCALE, *window)
    dout = torch.randn_like(q)
    padded = torch.zeros(2, 64, 3, 2, HEAD_DIM + 4, device="cuda", dtype=torch.bfloat16)[..., :HEAD_DIM]
    flash_attn_backward_packed(dout, q, k, v, out, lse, None, None, 64, 64, SOFTMAX_SCALE, *window, None, padded)
    queue_gradient_checks([padded.unbind(dim=-3)], attention_ref, q, k, v, dout)


UNSUPPORTED_VALUES = dict(dropout_p=0.1, attention_chunk=128, softcap=30.0, num_splits=4, deterministic=True)
BATCHED_UNSUPPORTED_OPTIONS = (
    "qv", "q_descale", "k_descale", "v_descale", "attention_chunk", "softcap", "num_splits", "deterministic"
)
UNSUPPORTED_OPTIONS = {
    "batched": BATCHED_UNSUPPORTED_OPTIONS,
    "varlen": (*BATCHED_UNSUPPORTED_OPTIONS, "seqused_q", "seqused_k"),
    "qkvpacked": ("q_descale", "k_descale", "v_descale", "attention_chunk", "softcap", "deterministic"),
    "kvpacked": ("dropout_p", "alibi_slopes", "softcap", "deterministic"),
    "varlen_kvpacked": ("dropout_p", "alibi_slopes", "softcap", "deterministic"),
}


@pytest.mark.parametrize("variant", UNSUPPORTED_OPTIONS)
def test_unsupported_options_raise(variant):
    case = build_case(variant, HEAD_DIM, "dense", shape=(2, 64, 64))
    for option in UNSUPPORTED_OPTIONS[variant]:
        value = UNSUPPORTED_VALUES.get(option, case.arguments[0].detach())
        with pytest.raises(NotImplementedError, match=option):
            case.apply(*case.arguments, **case.options, **{option: value})


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_option_is_accepted_or_unsupported(variant):
    case = build_case(variant, HEAD_DIM, "dense", shape=(2, 64, 64))
    options = list(inspect.signature(case.apply).parameters)[len(case.arguments) :]
    tested = {*case.options, *ACCEPTED_OPTIONS.get(variant, ()), *UNSUPPORTED_OPTIONS[variant], "return_attn_probs"}
    assert set(options) <= tested, set(options) - tested


def test_invalid_inputs_raise_value_error():
    q, k, v = (torch.randn(1, 64, 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match="bfloat16"):
        flash_attn_func(q.half(), k.half(), v.half())
    unsupported = (torch.randn(1, 64, 2, max(HEAD_DIMS) + 8, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match="head_dim"):
        flash_attn_func(*unsupported)
    with pytest.raises(ValueError, match="multiple"):
        flash_attn_func(torch.randn(1, 64, 3, HEAD_DIM, device="cuda", dtype=torch.bfloat16), k, v)
    with pytest.raises(ValueError, match="num_heads_q"):
        flash_attn_qkvpacked_func(torch.randn(1, 64, 6, HEAD_DIM, device="cuda", dtype=torch.bfloat16))


@pytest.fixture
def fresh_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "variant,layout_name",
    [
        ("batched", "gqa"),
        ("varlen", "gqa"),
        ("qkvpacked", "mha"),
        ("qkvpacked", "gqa"),
        ("kvpacked", "gqa"),
        ("varlen_kvpacked", "gqa"),
    ],
)
def test_compiled_attention_keeps_sizes_symbolic(fresh_compiler, variant, layout_name):
    compiled = None
    for call, shape in enumerate(DYNAMIC_SHAPES):
        cuda_seed(call)
        case = build_case(variant, HEAD_DIM, "causal", layout_name, shape)
        compiled = compiled or torch.compile(case.apply, fullgraph=True, dynamic=True)
        for index, dim in case.dynamic_dims:
            torch._dynamo.mark_dynamic(case.arguments[index], dim)
        with torch.compiler.set_stance("fail_on_recompile" if call else "default"):
            check_case(case, apply=compiled)


def test_compiled_return_attn_probs(fresh_compiler):
    compiled = torch.compile(flash_attn_func, fullgraph=True)
    case = build_case("batched", HEAD_DIM, "causal", shape=(2, 64, 64))
    check_case(case, apply=compiled, return_attn_probs=True)


def symbolic_inputs(fake_mode, shape, dynamic_dimensions):
    dynamic_sizes = [DimDynamic.DYNAMIC if index in dynamic_dimensions else DimDynamic.STATIC for index in range(len(shape))]
    context = StatelessSymbolicContext(dynamic_sizes=dynamic_sizes)
    return fake_mode.from_tensor(torch.empty(shape, device="cuda", dtype=torch.bfloat16), symbolic_context=context)


def test_fake_outputs_stay_symbolic_without_guards():
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    q = symbolic_inputs(fake_mode, (4, 128, 6, 64), (0, 1))
    k = symbolic_inputs(fake_mode, (4, 160, 2, 64), (0, 1))
    v = symbolic_inputs(fake_mode, (4, 160, 2, 64), (0, 1))
    guards_before = len(shape_env.guards)
    with fake_mode:
        out, lse = torch.ops.zigfa.flash_attn_forward(q, k, v, None, None, q.shape[1], k.shape[1], SOFTMAX_SCALE, -1, 0)
        dq, dk, dv = torch.ops.zigfa.flash_attn_backward(
            out, q, k, v, out, lse, None, None, q.shape[1], k.shape[1], SOFTMAX_SCALE, -1, 0
        )
        dkv = k.new_empty((*k.shape[:-2], 2, *k.shape[-2:]))
        torch.ops.zigfa.flash_attn_backward_packed(
            out, q, k, v, out, lse, None, None, q.shape[1], k.shape[1], SOFTMAX_SCALE, -1, 0, dq, dkv
        )
    assert all(isinstance(size, torch.SymInt) for size in (out.shape[0], out.shape[1], lse.shape[0], lse.shape[2]))
    assert all(isinstance(size, torch.SymInt) for size in (dk.shape[0], dk.shape[1], dq.shape[1]))
    assert len(shape_env.guards) == guards_before


def test_fake_varlen_outputs_stay_symbolic_without_guards():
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    q = symbolic_inputs(fake_mode, (300, 6, 64), (0,))
    k = symbolic_inputs(fake_mode, (400, 2, 64), (0,))
    v = symbolic_inputs(fake_mode, (400, 2, 64), (0,))
    cu_q = fake_mode.from_tensor(torch.zeros(5, dtype=torch.int32, device="cuda"))
    guards_before = len(shape_env.guards)
    with fake_mode:
        out, lse = torch.ops.zigfa.flash_attn_forward(q, k, v, cu_q, cu_q, 128, 128, SOFTMAX_SCALE, -1, 0)
    assert isinstance(out.shape[0], torch.SymInt) and isinstance(lse.shape[1], torch.SymInt)
    assert len(shape_env.guards) == guards_before


def op_arguments(variant, shape=(2, 64, 96)):
    case = build_case(variant, HEAD_DIM, "causal", "gqa", shape)
    q, k, v = (tensor.detach() for tensor in case.arguments[:3])
    if variant == "varlen":
        cu_q, cu_k, max_q, max_k = case.arguments[3:7]
    else:
        cu_q, cu_k, max_q, max_k = None, None, q.shape[1], k.shape[1]
    return q, k, v, cu_q, cu_k, max_q, max_k


@pytest.mark.parametrize("variant", ["batched", "varlen"])
def test_forward_op_passes_opcheck(variant):
    cuda_seed(21)
    q, k, v, cu_q, cu_k, max_q, max_k = op_arguments(variant)
    opcheck(flash_attn_forward, (q, k, v, cu_q, cu_k, max_q, max_k, SOFTMAX_SCALE, -1, 0), test_utils=OPCHECK_TESTS)


@pytest.mark.parametrize("variant", ["batched", "varlen"])
def test_backward_ops_pass_opcheck(variant):
    cuda_seed(23)
    q, k, v, cu_q, cu_k, max_q, max_k = op_arguments(variant)
    out, lse = flash_attn_forward(q, k, v, cu_q, cu_k, max_q, max_k, SOFTMAX_SCALE, -1, 0)
    dout = torch.randn_like(out)
    arguments = (dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, SOFTMAX_SCALE, -1, 0)
    opcheck(flash_attn_backward, arguments, test_utils=OPCHECK_TESTS)
    dq = torch.empty_like(q)
    dkv = k.new_empty((*k.shape[:-2], 2, *k.shape[-2:]))
    opcheck(flash_attn_backward_packed, (*arguments, dq, dkv), test_utils=OPCHECK_TESTS)


LAUNCH_SELECTION_CASES = {
    "batched_causal_mqa_grouped": lambda: batched_case(2, 2048, 2048, 32, 1, 64, True, (-1, 0)),
    "batched_causal_gqa": lambda: batched_case(2, 256, 256, 6, 3, 128, True, (-1, 0)),
    "batched_local_mha": lambda: batched_case(3, 256, 256, 6, 6, 64, False, (64, 32)),
    "batched_unmasked_mqa": lambda: batched_case(2, 512, 512, 6, 1, 96, False, (-1, -1)),
    "batched_unmasked_mqa_long": lambda: batched_case(1, 4096, 4096, 8, 1, 64, False, (-1, -1)),
    "batched_unmasked_mqa_short": lambda: batched_case(2, 64, 64, 16, 1, 64, False, (-1, -1)),
    "batched_causal_gqa_short": lambda: batched_case(2, 32, 200, 8, 2, 128, True, (-1, 0)),
    "batched_causal_gqa_hd72": lambda: batched_case(2, 256, 256, 6, 3, 72, True, (-1, 0)),
    "batched_causal_mqa_hd256": lambda: batched_case(2, 256, 256, 4, 1, 256, True, (-1, 0)),
    "varlen_causal_gqa": lambda: varlen_case(5, 256, 256, 6, 3, 64, True, (-1, 0)),
    "varlen_local_mqa": lambda: varlen_case(5, 256, 256, 6, 1, 128, False, (64, 32)),
    "varlen_causal_mqa_grouped": lambda: varlen_case(3, 512, 512, 32, 1, 64, True, (-1, 0)),
    "qkvpacked_causal": lambda: qkvpacked_case(2, 128, 128, 6, 6, 64, True, (-1, 0)),
    "qkvpacked_causal_gqa": lambda: qkvpacked_case(2, 128, 128, 6, 3, 64, True, (-1, 0)),
    "kvpacked_causal_gqa": lambda: kvpacked_case(2, 128, 128, 6, 3, 64, True, (-1, 0)),
    "varlen_kvpacked_causal": lambda: varlen_kvpacked_case(3, 128, 128, 6, 3, 64, True, (-1, 0)),
}


@pytest.mark.parametrize("name", list(LAUNCH_SELECTION_CASES))
def test_launch_selection_never_synchronizes_through_torch(name):
    cuda_seed(31)
    case = LAUNCH_SELECTION_CASES[name]()
    dout = torch.randn_like(case.apply(*case.arguments, **case.options))

    def launch():
        out = case.apply(*case.arguments, **case.options)
        return out, torch.autograd.grad(out, case.leaves, dout)

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        launch()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    clear_selection()
    torch.cuda.set_sync_debug_mode("error")
    try:
        launch()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, gradients = launch()
    with torch.no_grad():
        for tensor in (*case.leaves, dout):
            tensor.copy_(torch.randn_like(tensor))
    graph.replay()
    torch.cuda.synchronize()
    inputs = case.unpack([leaf.detach() for leaf in case.leaves])
    queue_forward_checks([(out, None)], case.reference, *inputs)
    queue_gradient_checks([case.unpack(gradients)], case.reference, *inputs, dout)


def test_backward_packed_op_with_concatenated_heads_matches_reference():
    cuda_seed(27)
    q, k, v, _, _, max_q, max_k = op_arguments("batched", (2, 64, 64))
    out, lse = flash_attn_forward(q, k, v, None, None, max_q, max_k, SOFTMAX_SCALE, -1, 0)
    dout = torch.randn_like(out)
    arguments = (dout, q, k, v, out, lse, None, None, max_q, max_k, SOFTMAX_SCALE, -1, 0)
    dqkv = torch.empty_like(torch.cat((q, k, v), dim=-2))
    opcheck(flash_attn_backward_packed, (*arguments, None, dqkv), test_utils=OPCHECK_TESTS)
    flash_attn_backward_packed(*arguments, None, dqkv)
    gradients = dqkv.split([q.shape[-2], k.shape[-2], v.shape[-2]], dim=-2)
    queue_gradient_checks([gradients], attention_ref, q, k, v, dout, window_size=(-1, 0))
