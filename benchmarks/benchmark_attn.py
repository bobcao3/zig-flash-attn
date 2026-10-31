import argparse
import contextlib
import functools
import importlib.metadata
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn

import timing
from zigfa._impl import backward as backward_impl, forward as forward_impl, runtime
from zigfa._impl.ops import softmax_lse_shape

SHAPES = ("mha", "gqa", "varlen", "sliding")
MASKS = ("dense", "causal")
SEQLENS = (512, 1024, 2048, 4096, 8192)
GQA_QHEAD_PER_KVHEAD = 4
VARLEN_LENGTH_SPREAD = 2
SDPA_BACKENDS = {"cuDNN": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}
FA4_MAJORS = (9, 10, 12)
SM80_ON_MAJORS = (12,)
SELECTION_MODULES = ("zigfa._impl.forward", "zigfa._impl.backward", "zigfa._impl.specialization")
NATIVE_ARCH = os.environ.get("ZIGFA_ARCH")
selected_arch = NATIVE_ARCH


def cu_seqlens_on_device(lengths):
    return torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32).cuda()


@dataclass(frozen=True)
class Case:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    lengths_q: list
    lengths_k: list
    window: tuple
    varlen: bool

    @property
    def num_head(self):
        return self.q.shape[-2]

    @property
    def num_head_kv(self):
        return self.k.shape[-2]

    @property
    def head_dim(self):
        return self.q.shape[-1]

    @property
    def total_q(self):
        return sum(self.lengths_q)


def visible_pairs(seqlen_q, seqlen_k, window):
    left, right = window
    rows = torch.arange(seqlen_q, dtype=torch.int64) + (seqlen_k - seqlen_q)
    first = (rows - left).clamp(min=0) if left >= 0 else torch.zeros_like(rows)
    last = (rows + right).clamp(max=seqlen_k - 1) if right >= 0 else torch.full_like(rows, seqlen_k - 1)
    return int((last - first + 1).clamp(min=0).sum())


def head_counts(shape, head_dim, dim):
    if shape == "gqa":
        num_head = max(1, round(dim / head_dim / GQA_QHEAD_PER_KVHEAD)) * GQA_QHEAD_PER_KVHEAD
        return num_head, num_head // GQA_QHEAD_PER_KVHEAD
    return dim // head_dim, dim // head_dim


def make_case(batch, seqlen, num_head, num_head_kv, head_dim, window, varlen):
    if varlen:
        generator = torch.Generator().manual_seed(seqlen)
        lower = seqlen // VARLEN_LENGTH_SPREAD
        lengths = torch.randint(lower, seqlen + lower + 1, (batch,), generator=generator).tolist()
    else:
        lengths = [seqlen] * batch
    total_q = sum(lengths)
    shape = (total_q, num_head, head_dim) if varlen else (batch, seqlen, num_head, head_dim)
    shape_kv = (total_q, num_head_kv, head_dim) if varlen else (batch, seqlen, num_head_kv, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    return Case(q, k, v, lengths, lengths, window, varlen)


def shape_case(shape, mask, head_dim, seqlen, arguments):
    """Builds the case of one point of the report: `seqlen` is the sequence length, or the window size of the
    sliding shape, which attends within one sequence of `--tokens` tokens."""
    num_head, num_head_kv = head_counts(shape, head_dim, arguments.dim)
    if shape == "sliding":
        window = {"dense": (seqlen, seqlen), "causal": (seqlen, 0)}[mask]
        return make_case(1, arguments.total_q, num_head, num_head_kv, head_dim, window, False)
    window = {"dense": (-1, -1), "causal": (-1, 0)}[mask]
    batch = max(1, arguments.total_q // seqlen)
    return make_case(batch, seqlen, num_head, num_head_kv, head_dim, window, shape == "varlen")


def attention_flops(pairs, num_head, head_dim, backward):
    forward = 4 * num_head * head_dim * pairs
    return 2.5 * forward if backward else forward


def attention_bytes(total_q, total_k, num_head, num_head_kv, head_dim, backward):
    query_bytes = total_q * num_head * head_dim * 2
    key_value_bytes = total_k * num_head_kv * head_dim * 2
    lse_bytes = total_q * num_head * 4
    if backward:
        return 4 * query_bytes + 4 * key_value_bytes + lse_bytes
    return 2 * query_bytes + 2 * key_value_bytes + lse_bytes


def flops(case, backward):
    pairs = sum(visible_pairs(sq, sk, case.window) for sq, sk in zip(case.lengths_q, case.lengths_k))
    return attention_flops(pairs, case.num_head, case.head_dim, backward)


def compulsory_bytes(case, backward):
    return attention_bytes(case.total_q, sum(case.lengths_k), case.num_head, case.num_head_kv, case.head_dim, backward)


def lazily(make):
    """Returns a function that calls the function that `make` returns, and calls `make` at its first call."""
    made = []

    def call():
        if not made:
            made.append(make())
        return made[0]()

    return call


def gradient_of(forward, inputs, grad):
    """Returns a function that computes the gradients of `inputs` through one output of `forward`, which runs
    once at the first call."""

    def make():
        result = forward()
        out = result[0] if isinstance(result, tuple) else result
        return lambda: torch.autograd.grad(out, inputs, grad, retain_graph=True)

    return lazily(make)


class Preferred:
    """Calls the first of `candidates`, named functions, whose first call does not raise `RuntimeError`, and
    names it in `backend`."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.backend = None

    def __call__(self):
        if self.backend is not None:
            return self.candidates[self.backend]()
        for backend, function in self.candidates.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    result = function()
            except RuntimeError:
                continue
            self.backend = backend
            return result
        raise RuntimeError(f"none of {', '.join(self.candidates)} supports this case")


def sdpa_candidates(case, dout, causal):
    """Returns the forward and backward functions of torch SDPA with each backend, cuDNN first."""
    padded = not case.varlen and case.window in ((-1, -1), (-1, 0))
    if padded:
        inputs = [x.detach().transpose(1, 2).requires_grad_() for x in (case.q, case.k, case.v)]
        grad = dout.transpose(1, 2)
        options = {"enable_gqa": True} if case.num_head != case.num_head_kv else {}

        def with_backend(backend):
            def call():
                with sdpa_kernel(SDPA_BACKENDS[backend]):
                    return functional.scaled_dot_product_attention(*inputs, is_causal=causal, **options)

            return call

        forward = {backend: with_backend(backend) for backend in SDPA_BACKENDS}
        return forward, {backend: gradient_of(function, inputs, grad) for backend, function in forward.items()}
    lengths = case.lengths_q if case.varlen else [case.q.shape[1]] * case.q.shape[0]
    cu_seqlens = cu_seqlens_on_device(lengths)
    longest = max(lengths)
    packed = [x.reshape(-1, *x.shape[-2:]) for x in (case.q, case.k, case.v)]
    inputs = [x.detach().requires_grad_() for x in packed]
    grad = dout.reshape(-1, *dout.shape[-2:])
    grouped = case.num_head != case.num_head_kv
    forward, backward = {}, {}
    if case.window in ((-1, -1), (-1, 0)) and not grouped:

        def cudnn_forward():
            return torch.ops.aten._cudnn_attention_forward(
                *packed, None, cu_seqlens, cu_seqlens, longest, longest, True, is_causal=causal
            )

        def cudnn_backward():
            out, lse, _, _, _, _, seed, offset, _ = cudnn_forward()
            return lambda: torch.ops.aten._cudnn_attention_backward(
                grad, *packed, out, lse, seed, offset, None, cu_seqlens, cu_seqlens, longest, longest, 0.0, causal
            )

        forward["cuDNN"], backward["cuDNN"] = cudnn_forward, lazily(cudnn_backward)

    def flash_forward():
        return varlen_attn(
            *inputs, cu_seqlens, cu_seqlens, longest, longest, window_size=case.window, enable_gqa=grouped
        )

    forward["flash"], backward["flash"] = flash_forward, gradient_of(flash_forward, inputs, grad)
    return forward, backward


@functools.cache
def fa4_interface():
    if torch.cuda.get_device_capability()[0] not in FA4_MAJORS:
        return None
    try:
        return importlib.import_module("flash_attn.cute.interface")
    except ImportError:
        return None


def fa4_functions(case, dout, causal):
    """Returns the forward and backward functions of FlashAttention-4, or None where it is not installed."""
    interface = fa4_interface()
    if interface is None:
        return None
    inputs = [x.detach().requires_grad_() for x in (case.q, case.k, case.v)]
    window = (None, None) if case.window[0] < 0 else case.window
    options = {"causal": causal, "window_size": window}
    if case.varlen:
        cu_seqlens = cu_seqlens_on_device(case.lengths_q)
        longest = max(case.lengths_q)

        def forward():
            return interface.flash_attn_varlen_func(
                *inputs,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=longest,
                max_seqlen_k=longest,
                **options,
            )
    else:

        def forward():
            return interface.flash_attn_func(*inputs, **options)

    return forward, gradient_of(forward, inputs, dout)


def use_arch(arch):
    """Makes zigfa select kernels of `arch`, or of the GPU when `arch` is None, by setting `ZIGFA_ARCH` and
    clearing the cached selections. Loaded kernels stay loaded, so graphs that captured them stay valid."""
    global selected_arch
    if selected_arch == arch:
        return
    if arch is None:
        os.environ.pop("ZIGFA_ARCH", None)
    else:
        os.environ["ZIGFA_ARCH"] = arch
    runtime.arch_name.cache_clear()
    runtime.variant_names.cache_clear()
    runtime.load_for.cache_clear()
    for name in SELECTION_MODULES:
        for value in vars(sys.modules[name]).values():
            if hasattr(value, "cache_clear"):
                value.cache_clear()
    selected_arch = arch


def zigfa_functions(case, dout, arch):
    """Returns the forward and backward functions of zigfa with the kernels of `arch`."""
    use_arch(arch)
    cu_seqlens_q = cu_seqlens_on_device(case.lengths_q) if case.varlen else None
    cu_seqlens_k = cu_seqlens_on_device(case.lengths_k) if case.varlen else None
    max_seqlen_q = max(case.lengths_q) if case.varlen else None
    max_seqlen_k = max(case.lengths_k) if case.varlen else None
    cu_seqlens_shape = tuple(cu_seqlens_q.shape) if case.varlen else None
    options = dict(
        window_size_left=case.window[0],
        window_size_right=case.window[1],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    out = torch.empty_like(case.q)
    lse = torch.empty(softmax_lse_shape(case.q, case.varlen), device="cuda", dtype=torch.float32)
    forward_workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in forward_impl.compute_scratch_shapes(
            tuple(case.q.shape),
            k_shape=tuple(case.k.shape),
            cu_seqlens_shape=cu_seqlens_shape,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    )
    dq, dk, dv = torch.empty_like(case.q), torch.empty_like(case.k), torch.empty_like(case.v)
    backward_workspace = tuple(
        torch.empty(shape, device="cuda", dtype=torch.float32)
        for shape in backward_impl.compute_scratch_shapes(
            tuple(case.q.shape),
            tuple(lse.shape),
            tuple(case.k.shape),
            *case.window,
            cu_seqlens_shape=cu_seqlens_shape,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    )

    def forward():
        use_arch(arch)
        stream = torch.cuda.current_stream().cuda_stream
        forward_impl.forward(case.q, case.k, case.v, out, lse, forward_workspace, stream=stream, **options)

    def backward():
        use_arch(arch)
        stream = torch.cuda.current_stream().cuda_stream
        backward_impl.backward(case.q, case.k, case.v, out, dout, lse, dq, dk, dv, backward_workspace, stream=stream, **options)

    def backward_after_forward():
        forward()
        return backward

    return forward, lazily(backward_after_forward)


def implementations(case):
    """Returns the forward and backward functions of each implementation, zigfa first."""
    dout = torch.randn_like(case.q)
    causal = case.window == (-1, 0)
    forward, backward = {}, {}
    forward["zigfa"], backward["zigfa"] = zigfa_functions(case, dout, NATIVE_ARCH)
    if torch.cuda.get_device_capability()[0] in SM80_ON_MAJORS:
        forward["zigfa sm80"], backward["zigfa sm80"] = zigfa_functions(case, dout, "sm80")
    use_arch(NATIVE_ARCH)
    sdpa_forward, sdpa_backward = sdpa_candidates(case, dout, causal)
    forward["torch SDPA"], backward["torch SDPA"] = Preferred(sdpa_forward), Preferred(sdpa_backward)
    fa4 = fa4_functions(case, dout, causal)
    if fa4 is not None:
        forward["FA4"], backward["FA4"] = fa4
    return forward, backward


def run_case(name, point, case, arguments, output):
    forward, backward = implementations(case)
    groups = {"fwd": forward, "bwd": backward}
    case_seconds = timing.CASE_SECONDS / len(arguments.directions)
    line = name
    for direction in arguments.directions:
        functions = groups[direction]
        is_backward = direction == "bwd"
        flop_count, byte_count = flops(case, is_backward), compulsory_bytes(case, is_backward)
        estimate = timing.estimated_seconds(flop_count, byte_count, is_backward)
        measurement = timing.measure(f"{name} {direction}", functions, estimate, case_seconds)
        backends = {label: function.backend for label, function in functions.items() if hasattr(function, "backend")}
        display = {label: f"{label} ({backend})" for label, backend in backends.items() if backend}
        line += f" | {direction} " + timing.summary(measurement, list(functions), flop_count, byte_count, display)
        if output is not None:
            record = timing.record(measurement, flop_count, byte_count, backends)
            output.write(json.dumps({"benchmark": "attention", **point, "direction": direction, **record}) + "\n")
            output.flush()
    return line


def versions():
    found = {"cudnn": str(torch.backends.cudnn.version())}
    for name in ("torch", "flash-attn-4"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found[name] = importlib.metadata.version(name)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Times zigfa against torch SDPA (cuDNN first, FlashAttention otherwise) and FlashAttention-4"
    )
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--head-dims", type=int, nargs="+", default=list(runtime.head_dims()))
    parser.add_argument("--seqlens", type=int, nargs="+", default=list(SEQLENS))
    parser.add_argument("--directions", nargs="+", choices=("fwd", "bwd"), default=["fwd", "bwd"])
    parser.add_argument("--tokens", dest="total_q", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--json", help="append one JSON record per case and direction to this file")
    arguments = parser.parse_args()
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    warnings.filterwarnings("ignore", message="Argument .* cannot be converted to a JitArgument")
    started = time.perf_counter()
    durations = []
    print(timing.header(), flush=True)
    print(
        "Inputs are bfloat16 values drawn from N(0, 1). torch SDPA runs cuDNN where cuDNN takes the case and "
        "FlashAttention otherwise; the backend follows its name. A percent after a rate is of the roofline, and "
        "(zigfa N%) is the speed of zigfa as a percent of the speed of that implementation.",
        flush=True,
    )
    output = open(arguments.json, "a") if arguments.json else None
    common = {"device": torch.cuda.get_device_name(), "versions": versions()}
    for shape in arguments.shapes:
        for mask in arguments.masks:
            for head_dim in arguments.head_dims:
                for seqlen in arguments.seqlens:
                    case = shape_case(shape, mask, head_dim, seqlen, arguments)
                    batch = 1 if shape == "sliding" else len(case.lengths_q)
                    name = f"{shape:7} {mask:6} hd={head_dim:3} s={seqlen:5} b={batch:3}"
                    point = {**common, "shape": shape, "mask": mask, "head_dim": head_dim, "seqlen": seqlen}
                    case_started = time.perf_counter()
                    line = run_case(name, point, case, arguments, output)
                    durations.append(time.perf_counter() - case_started)
                    print(f"{line} [{durations[-1]:.2f} s]", flush=True)
    print(timing.totals(durations, time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
