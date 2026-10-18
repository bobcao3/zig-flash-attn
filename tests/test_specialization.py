import pytest
import torch

from shapes import HEAD_DIM
from zigfa import runtime
from zigfa.tensors import build_scheduler_metadata, varlen_blocks, view_of

SCHEDULER_METADATA_KERNELS = {"forward": ("fwd", "flash_attn_fwd"), "backward": ("bwd", "flash_attn_bwd")}
SCHEDULER_METADATA_BATCHES = [1, 4, 31, 32, 33, 1023, 1024, 1025, 2049, 3000]
SCHEDULER_METADATA_LENGTHS = ["heavy_tail", "zero", "tile_multiples"]


def every_block(lengths, block_m, num_block):
    entries = [(sequence, block) for sequence, length in enumerate(lengths) for block in range(-(-length // block_m))]
    return sorted(entries + [(len(lengths), 0)] * (num_block - len(entries)))


def scheduler_metadata_lengths(kind, batch, block_m, generator):
    if kind == "zero":
        lengths = torch.randint(0, 3, (batch,), generator=generator)
        lengths *= torch.randint(1, 300, (batch,), generator=generator)
        lengths[: batch // 2] = 0
        return lengths
    if kind == "tile_multiples":
        return torch.randint(0, 4, (batch,), generator=generator) * block_m
    return (torch.rand(batch, generator=generator) ** 6 * 4000).to(torch.int64) + 1


@pytest.mark.parametrize("kernel", list(SCHEDULER_METADATA_KERNELS))
@pytest.mark.parametrize("batch", SCHEDULER_METADATA_BATCHES)
@pytest.mark.parametrize("kind", SCHEDULER_METADATA_LENGTHS)
def test_scheduler_metadata_lists_every_block_once(kernel, batch, kind):
    direction, kernel_name = SCHEDULER_METADATA_KERNELS[kernel]
    variant = "_varlen"
    _, info = runtime.load(direction, HEAD_DIM, kernel_name, variant)
    assert info.scheduler_metadata
    block_m = info.cluster_block_m if direction == "fwd" else (info.cluster_ctas or 1) * info.block_n
    generator = torch.Generator().manual_seed(batch * 7 + len(kind))
    lengths = scheduler_metadata_lengths(kind, batch, block_m, generator)
    other_lengths = torch.randint(0, 5000, (batch,), generator=generator)
    offset = 17
    cu_seqlens, cu_seqlens_other = (
        torch.cat([torch.zeros(1, dtype=torch.int64), values.cumsum(0)]).add(offset).to(torch.int32).cuda()
        for values in (lengths, other_lengths)
    )
    num_block = varlen_blocks(int(lengths.sum()), batch, block_m)
    table = torch.full((max(num_block, 1), 2), -1, dtype=torch.int32, device="cuda")
    build_scheduler_metadata(
        table, direction, HEAD_DIM, variant, kernel_name, view_of(cu_seqlens).pointer, view_of(cu_seqlens_other).pointer,
        batch, num_block, "the test", torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    listed = sorted(tuple(entry) for entry in table[:num_block].tolist())
    assert listed == every_block(lengths.tolist(), block_m, num_block)
