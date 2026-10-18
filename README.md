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
keys, as when extending a KV cache.

## Build from source

`pip install .` needs the Zig compiler with the self-hosted PTX backend from
https://forge.bc3.moe/bob/zig-nvptx; set `ZIG` (and `PTXAS`) to the binaries.
