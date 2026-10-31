# zigfa

FlashAttention-4 forward and backward kernels for SM80, SM90, SM100 and SM120 GPUs,
with PyTorch interfaces. The kernels are written in Zig and compiled to PTX with
[this Zig compiler fork](https://forge.bc3.moe/bob/zig-nvptx). The Python host code
loads and launches kernels through NVIDIA's
[`cuda-core` bindings](https://nvidia.github.io/cuda-python/cuda-core/latest/).

## Install

Download a wheel from [releases](https://github.com/bobcao3/zig-flash-attn/releases)
and install it with the `cu12` or `cu13` extra to select the CUDA Python dependencies:

```sh
pip install './zigfa-2026.10.30-py3-none-any.whl[cu13]'
```

Use CUDA-enabled PyTorch for the same CUDA major. Python 3.10 or later is required.
The NVIDIA driver must support the compiled kernels, regardless of the extra.

## Usage

Inputs must be CUDA BF16 tensors. Autograd, grouped-query attention and sliding
windows are supported; dropout, ALiBi and deterministic backward are not.

```python
import torch
from zigfa import flash_attn_func, flash_attn_varlen_func

q = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = flash_attn_func(q, k, v, causal=True)

cu_seqlens = torch.tensor([0, 1024, 2048], dtype=torch.int32, device="cuda")
out = flash_attn_varlen_func(
    q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
    cu_seqlens, cu_seqlens, 1024, 1024, causal=True,
)
```

The package also exports `flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func`
and `flash_attn_varlen_kvpacked_func`. See [the public signatures](src/zigfa/torch.py)
for the supported arguments.

## Inter-sequence attention

`flash_attn_inter_sequence_func` lets each packed query sequence attend to a chosen
set of key sequences. This example has a shared prefix of 256 tokens and two
independent continuations of 64 tokens each. Each continuation sees the prefix and
itself, but not the other continuation:

```python
from zigfa import flash_attn_inter_sequence_func, prepare_inter_sequence_metadata

cu_seqlens = torch.tensor([0, 256, 320, 384], dtype=torch.int32, device="cuda")
visibility_bits = torch.tensor([[0b001], [0b011], [0b101]], dtype=torch.int32, device="cuda")
metadata = prepare_inter_sequence_metadata(cu_seqlens, cu_seqlens, visibility_bits)

q = torch.randn(384, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(384, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = flash_attn_inter_sequence_func(
    q, k, v,
    cu_seqlens, cu_seqlens,
    max_seqlen_q=256,
    max_seqlen_k=256,
    visibility_bits=visibility_bits,
    causal=True,
    metadata=metadata,
)
```

Use `q` of shape `(total_q, heads_q, dim)` and `k` and `v` of shape
`(total_k, heads_kv, dim)`. For `n` sequences, construct the layout tensors as follows:

- `cu_seqlens_q` and `cu_seqlens_k` each contain `n + 1` cumulative token offsets,
  starting at zero and ending at the corresponding token total. Both describe
  the same number of sequences. Query sequences may be empty or shorter than their
  matching key sequences, for example when using cached keys and values.
- `visibility_bits` has shape `(n, (n + 31) // 32)`. Bit `j % 32` of word `j // 32`
  in row `i` lets query sequence `i` attend to key sequence `j`. All other pairs
  are masked out. Leave unused bits zero. If a word sets bit 31, construct it as
  `int64` and cast to `int32` to obtain the signed representation.
- All layout tensors must be contiguous CUDA `int32` tensors on the input device.

Set `max_seqlen_q` and `max_seqlen_k` to the largest individual query and key
sequence lengths.

Causal and sliding-window masks apply only within a matching query/key sequence.
Other visible key sequences remain fully visible. For a shorter query sequence,
the causal mask aligns its queries to the end of the matching key sequence.

Prepare metadata once and reuse the handle with new Q/K/V tensors for the same
layout. Prepare a new handle when sequence boundaries, visibility contents or the
device change. Leave the handle's contents unchanged, and keep passing the
boundaries, visibility and maximum sequence lengths on every call. Omitting
`metadata` makes each call prepare its own handle.

Inter-sequence attention supports head dimensions 64, 128 and 256, plus 512 for
forward only. `flash_attn_inter_sequence_kvpacked_func` accepts packed keys and
values.

## Build and test

Set `ZIG` to the compiler fork's executable. The Pixi build environment supplies
CUDA 13.4 tools:

```sh
pixi run -e build --locked python -m build --wheel
pixi run -e cu13 --locked pytest tests --kernel-coverage=require
```

Use `-e cu12` to test with CUDA 12 dependencies. The default environment combines
the build tools and CUDA 13 test dependencies. Development versions append a
source hash to the base in `VERSION`; tagged builds use the release version.

See [the benchmark scripts](benchmarks/) and [performance report](reports/perf_report.html)
for measurements.
