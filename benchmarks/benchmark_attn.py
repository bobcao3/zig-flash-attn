import argparse
import contextlib
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn

import zigfa
from zigfa import runtime
from zigfa.torch import softmax_lse_shape

LAYOUTS = ("mha", "gqa", "mqa")
GQA_QHEAD_PER_KVHEAD = 4
WINDOWS = ("dense", "causal", "local")
LOCAL_WINDOW_DIVISOR = 8
VARLEN_LENGTH_SPREAD = 2


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


def head_counts(layout, head_dim, dim):
    if layout == "gqa":
        num_head = max(1, round(dim / head_dim / GQA_QHEAD_PER_KVHEAD)) * GQA_QHEAD_PER_KVHEAD
        return num_head, num_head // GQA_QHEAD_PER_KVHEAD
    num_head = dim // head_dim
    return num_head, {"mha": num_head, "mqa": 1}[layout]


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


def flops(case, backward):
    pairs = sum(visible_pairs(sq, sk, case.window) for sq, sk in zip(case.lengths_q, case.lengths_k))
    forward = 4 * case.num_head * case.head_dim * pairs
    return 2.5 * forward if backward else forward


def compulsory_bytes(case, backward):
    query_bytes = case.total_q * case.num_head * case.head_dim * 2
    key_value_bytes = sum(case.lengths_k) * case.num_head_kv * case.head_dim * 2
    lse_bytes = case.total_q * case.num_head * 4
    if backward:
        return 4 * query_bytes + 4 * key_value_bytes + lse_bytes
    return 2 * query_bytes + 2 * key_value_bytes + lse_bytes


def time_seconds(function, warmup=5, repeats=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats * 1e-3


def rate(flop_count, byte_count, seconds):
    return f"{flop_count / seconds * 1e-12:7.1f} TF/s {byte_count / seconds * 1e-12:6.2f} TB/s"


def padded_baselines(case, dout, causal):
    transposed = [x.detach().transpose(1, 2).requires_grad_() for x in (case.q, case.k, case.v)]
    grad = dout.transpose(1, 2)
    options = {"enable_gqa": True} if case.num_head != case.num_head_kv else {}

    def forward():
        return functional.scaled_dot_product_attention(*transposed, is_causal=causal, **options)

    def backward_of():
        result = forward()
        return lambda: torch.autograd.grad(result, transposed, grad, retain_graph=True)

    return {
        "pytorch sdpa": (contextlib.nullcontext, forward, backward_of),
        "cudnn": (lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION), forward, backward_of),
    }


def packed_baselines(case, dout, causal):
    lengths = case.lengths_q if case.varlen else [case.q.shape[1]] * case.q.shape[0]
    cu_seqlens = cu_seqlens_on_device(lengths)
    longest = max(lengths)
    packed = [x.reshape(-1, *x.shape[-2:]) for x in (case.q, case.k, case.v)]
    leaves = [x.detach().requires_grad_() for x in packed]
    grad = dout.reshape(-1, *dout.shape[-2:])
    grouped = case.num_head != case.num_head_kv

    def forward():
        return varlen_attn(*leaves, cu_seqlens, cu_seqlens, longest, longest, window_size=case.window, enable_gqa=grouped)

    def backward_of():
        result = forward()
        return lambda: torch.autograd.grad(result, leaves, grad, retain_graph=True)

    def cudnn_forward():
        return torch.ops.aten._cudnn_attention_forward(
            *packed, None, cu_seqlens, cu_seqlens, longest, longest, True, is_causal=causal
        )

    def cudnn_backward_of():
        out, lse, _, _, _, _, seed, offset, _ = cudnn_forward()
        return lambda: torch.ops.aten._cudnn_attention_backward(
            grad, *packed, out, lse, seed, offset, None, cu_seqlens, cu_seqlens, longest, longest, 0.0, causal
        )

    baselines = {"pytorch sdpa": (contextlib.nullcontext, forward, backward_of)}
    if case.window in ((-1, -1), (-1, 0)) and not grouped:
        baselines["cudnn"] = (contextlib.nullcontext, cudnn_forward, cudnn_backward_of)
    return baselines


def torch_baselines(case, dout, backward):
    causal = case.window == (-1, 0)
    padded = not case.varlen and case.window in ((-1, -1), (-1, 0))
    baselines = (padded_baselines if padded else packed_baselines)(case, dout, causal)
    results = {}
    for name, (context, forward, backward_of) in baselines.items():
        try:
            with context():
                forward_seconds = time_seconds(forward)
                backward_seconds = None
                if backward:
                    with contextlib.suppress(RuntimeError):
                        backward_seconds = time_seconds(backward_of())
            results[name] = (forward_seconds, backward_seconds)
        except RuntimeError:
            results[name] = None
    return results


def run_case(case, arguments):
    cu_seqlens_q = cu_seqlens_on_device(case.lengths_q) if case.varlen else None
    cu_seqlens_k = cu_seqlens_on_device(case.lengths_k) if case.varlen else None
    max_seqlen_q = max(case.lengths_q) if case.varlen else None
    max_seqlen_k = max(case.lengths_k) if case.varlen else None
    cu_seqlens_shape = tuple(cu_seqlens_q.shape) if case.varlen else None
    options = dict(
        window_size_left=case.window[0],
        window_size_right=case.window[1],
        stream=torch.cuda.current_stream().cuda_stream,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    out = torch.empty_like(case.q)
    lse = torch.empty(softmax_lse_shape(case.q, case.varlen), device="cuda", dtype=torch.float32)
    forward_workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in zigfa.forward_workspace_shapes(
            tuple(case.q.shape),
            k_shape=tuple(case.k.shape),
            cu_seqlens_shape=cu_seqlens_shape,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
    )
    forward_seconds = time_seconds(
        lambda: zigfa.forward(case.q, case.k, case.v, out, lse, forward_workspace, **options)
    )
    line = f" zigfa fwd {rate(flops(case, False), compulsory_bytes(case, False), forward_seconds)}"
    dout = torch.randn_like(case.q)
    if arguments.backward:
        dq, dk, dv = torch.empty_like(case.q), torch.empty_like(case.k), torch.empty_like(case.v)
        workspace = tuple(
            torch.empty(shape, device="cuda", dtype=torch.float32)
            for shape in zigfa.backward_workspace_shapes(
                tuple(case.q.shape),
                tuple(lse.shape),
                tuple(case.k.shape),
                *case.window,
                cu_seqlens_shape=cu_seqlens_shape,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
        )
        backward_seconds = time_seconds(
            lambda: zigfa.backward(case.q, case.k, case.v, out, dout, lse, dq, dk, dv, workspace, **options)
        )
        line += f" bwd {rate(flops(case, True), compulsory_bytes(case, True), backward_seconds)}"
    if arguments.baselines:
        for name, result in torch_baselines(case, dout, arguments.backward).items():
            if result is None:
                line += f"  {name}: n/a"
                continue
            line += f"  {name} fwd {rate(flops(case, False), compulsory_bytes(case, False), result[0])}"
            if result[1] is not None:
                line += f" bwd {rate(flops(case, True), compulsory_bytes(case, True), result[1])}"
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dims", type=int, nargs="+", default=list(runtime.head_dims()))
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=["mha"])
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=["dense", "causal"])
    parser.add_argument("--tokens", dest="total_q", type=int, default=16384)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--local-window", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--backward", action="store_true")
    arguments = parser.parse_args()
    if arguments.local_window and (arguments.varlen or "local" in arguments.windows):
        parser.error("--local-window runs one sequence of --tokens tokens with each --seqlens value as the window size")
    for window_name in arguments.windows:
        for layout in arguments.layouts:
            for head_dim in arguments.head_dims:
                num_head, num_head_kv = head_counts(layout, head_dim, arguments.dim)
                for seqlen in arguments.seqlens:
                    batch = 1 if arguments.local_window else max(1, arguments.total_q // seqlen)
                    length = arguments.total_q if arguments.local_window else seqlen
                    span = max(1, seqlen // LOCAL_WINDOW_DIVISOR)
                    windows = (
                        {"dense": (seqlen, seqlen), "causal": (seqlen, 0)}
                        if arguments.local_window
                        else {"dense": (-1, -1), "causal": (-1, 0), "local": (span, span)}
                    )
                    window = windows[window_name]
                    case = make_case(batch, length, num_head, num_head_kv, head_dim, window, arguments.varlen)
                    shape = "local_window" if arguments.local_window else "varlen" if arguments.varlen else "batched"
                    prefix = f"{window_name:6} {layout} {shape:7} hd={head_dim:3} s={seqlen:5} b={batch:3}"
                    print(prefix + run_case(case, arguments), flush=True)
                    time.sleep(0.5)


if __name__ == "__main__":
    main()
