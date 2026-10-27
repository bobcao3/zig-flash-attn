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

The lower-level `zigfa.backward` takes a workspace that the caller allocates.
`zigfa.backward_workspace_shapes(q_shape, lse_shape, k_shape, window_size_left=-1,
window_size_right=-1, *, cu_seqlens_shape=None, max_seqlen_q=None,
max_seqlen_k=None)` returns the shapes of its buffers and requires `k_shape`,
because the backward keeps the dK terms of keys whose probability exceeds 3/4 in
a buffer sized by the keys. Allocate each shape as a `torch.float32` tensor on
the device of `q` and pass the tensors as a tuple in the returned order. The
number and the shapes of the buffers depend on the GPU, the shapes and the
window, so do not unpack the tuple by position.

## Numerics

The kernels apply the fix of "A Fix for Softmax and Cross-Entropy" (Barron,
Gnanapragasam and Tschannen) to attention. The forward leaves the exponential of
a key that scores the stored maximum of its row out of the running sum of the
row and adds it back as an exact 1, so the log-sum-exp of a row that puts almost
all of its weight on one key keeps the small terms of the other keys. The SM100
forward can leave out several keys that score the stored maximum and adds each
of them back as an exact 1. During its pass over the keys of a row, the forward
raises the stored maximum only when a score exceeds it by more than about
8 / (softmax_scale * log2(e)), so the stored maximum can lie up to that amount
below the largest score of the row. In a row whose other scores all lie more
than that amount below its largest score, the key with the largest score is left
out. In the backward, the score of a key whose probability exceeds 3/4 gets the
negated sum of the other score gradients of its row as its gradient, which keeps
dQ and dK accurate on such rows, for example rows dominated by an attention
sink. The SM80, SM90 and SM120 backward kernels compare the probability
rounded to BF16 with 3/4. Kernels that write dK in BF16 add the dK terms of
such a key in a second pass, which can cost half a BF16 unit of the dK before
those terms. The SM100 backward for head dimension 256 instead rounds the
gradient of such a key to BF16 before it multiplies the gradient into dK, which
costs less than one BF16 unit of each such term.

A row that splits its weight over several keys with nearly equal scores has no
such key and keeps the standard backward formula. The exponentials flush values
below 2^-126 to zero, so the fix holds up to margins of about 87 - ln(n) nats
when n keys carry the rest of the weight; the SM100 backward also flushes the
products that form the gradients and loses accuracy a few nats earlier. In the
forward, the exponential of a score can exceed that of the stored maximum by a
factor of up to about 2^8. When the exponential of the stored maximum grows by a
factor above 2^126, the factor that rescales the earlier sum flushes to zero and
drops the terms of that sum, which lie below 2^-118 times the largest
exponential of the row. The forward log-sum-exp therefore holds only up to
margins of about 118 ln(2) - ln(n), or 82 - ln(n), nats. The SM100 forward
computes some exponentials with a polynomial whose relative error reaches about
9e-5. Any key that the SM100 forward does not leave out can carry that error in
its exponential, so the SM100 log-sum-exp can be off by up to about 9e-5 * l'
nats, where l' is the sum of the exponentials of those keys relative to the
exponential of the stored maximum.

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

The SM80 and SM120 forward kernels and the persistent inter-sequence forwards
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
FlexAttention with a block mask of the same visibility, on its Triton kernels and, on SM90 and SM100
GPUs, on its FlashAttention-4 kernels (`kernel_options={"BACKEND": "FLASH"}`); `--gqa 4` gives the
keys and values a quarter of the query heads. Their default arguments measure the points of
`reports/perf_report.html`; with `--json PATH`, each script writes one record per point, and
`python reports/update_perf_report.py DEVICE PATH...` writes the records of one device into the
report.

## Tests

`pytest tests` runs the test suite on the current GPU. At the end it lists the built kernel
instances that the library can dispatch to on that GPU and whose main kernel no test launched;
`--kernel-coverage=require` fails the run when it lists any.

## Build from source

`pip install .` needs the Zig compiler with the self-hosted PTX backend from
https://forge.bc3.moe/bob/zig-nvptx; set `ZIG` (and `PTXAS`) to the binaries.
