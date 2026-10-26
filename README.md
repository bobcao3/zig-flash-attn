# zigfa

FlashAttention-4 forward and backward kernels written in Zig and compiled to
PTX, for SM80, SM90, SM100 and SM120 GPUs, with a drop-in torch interface.

## Usage

```python
import torch
from zigfa.torch import flash_attn_func, flash_attn_inter_sequence_func

q = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = flash_attn_func(q, k, v, causal=True)  # (batch, seqlen, heads, dim)

# Packed sequences of 256 and 64 tokens; the second sees the first and itself.
cu = torch.tensor([0, 256, 320], dtype=torch.int32, device="cuda")
bits = torch.tensor([[0b01], [0b11]], dtype=torch.int32, device="cuda")
q, k, v = (x[0, :320] for x in (q, k, v))
out = flash_attn_inter_sequence_func(q, k, v, cu, cu, 256, 256, bits)
```

The arguments follow flash-attn's `flash_attn_func` and
`flash_attn_varlen_func`; autograd, GQA and sliding windows are supported.
In `flash_attn_inter_sequence_func`, bit `j` of row `i` lets query sequence
`i` see key sequence `j`; a query sequence may be empty or shorter than its
keys, as when extending a KV cache. It takes head dimensions 64, 128 and 256,
and 512 without autograd.

## Profiling

Kernels built with `zig build -Dprofile`, or packaged with `ZIGFA_PROFILE=1`,
record the begin and end of named zones in each warp. Builds without the
option compile the same code as before:

```python
from zigfa import profile

with profile.capture() as cap:
    out = flash_attn_func(q, k, v)
cap.save("fwd.jsonl.gz")
print(cap.summary(by="sm"))
```

The SM80 and SM120 forward kernels and the persistent inter-sequence forward
mark their zones. `python -m zigfa.view fwd.jsonl.gz --open` serves the trace
viewer on localhost and opens the trace; without `--open`, it prints the link. A
link takes `group` (`sm`, `cluster`, `cta`, `wg` or `warp`), `start` and `end`
in microseconds, and `expand`, the groups to unfold (or `all`, with `fold`
naming the exceptions), for example
`http://127.0.0.1:8732/?trace=traces/0/fwd.jsonl.gz&group=sm&start=100&end=140&expand=sm2`.
`reports/profile_viewer.html` also opens a trace that you drop onto it, and
`reports/profile_design.html` describes the record and file formats.

## Benchmarks

`benchmarks/benchmark_attn.py` times zigfa against torch SDPA, which runs cuDNN where cuDNN accepts
the case and FlashAttention otherwise, and against FlashAttention-4 on SM90, SM100 and SM120 GPUs.
`benchmarks/benchmark_inter_sequence.py` times inter-sequence attention against compiled
FlexAttention and FlashAttention-4 with a `mask_mod` of the same visibility. Their default
arguments measure the points of `reports/perf_report.html`; with `--json PATH`, each script writes
one record per point, and `python reports/update_perf_report.py DEVICE PATH...` writes the records
of one device into the report.

## Build from source

`pip install .` needs the Zig compiler with the self-hosted PTX backend from
https://forge.bc3.moe/bob/zig-nvptx; set `ZIG` (and `PTXAS`) to the binaries.
