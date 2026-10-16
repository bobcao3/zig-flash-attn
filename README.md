# zigfa

FlashAttention-4 forward and backward kernels written in Zig and compiled to
PTX, for SM80, SM90, SM100 and SM120 GPUs, with a drop-in torch interface.

## Usage

```python
import torch
from zigfa.torch import flash_attn_func, flash_attn_varlen_func

q = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = flash_attn_func(q, k, v, causal=True)  # (batch, seqlen, heads, dim)
```

The arguments follow flash-attn's `flash_attn_func` and
`flash_attn_varlen_func`; autograd, GQA and sliding windows are supported.

## Build from source

`pip install .` needs the Zig compiler with the self-hosted PTX backend from
https://forge.bc3.moe/bob/zig-nvptx; set `ZIG` (and `PTXAS`) to the binaries.
