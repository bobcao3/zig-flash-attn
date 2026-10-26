import torch

from zigfa import runtime

HEAD_DIMS = runtime.head_dims()
HEAD_DIM = 64
NUM_HEAD = 6
HEAD_LAYOUTS = {"mha": (NUM_HEAD, 6), "mqa": (NUM_HEAD, 1), "gqa": (NUM_HEAD, 3)}
WINDOW_NAMES = ["dense", "causal", "local", "local_left", "local_right"]

BATCHED_SEQLENS = [
    (1, 1), (3, 3), (64, 32), (64, 128), (128, 128), (128, 192), (256, 256), (239, 1), (799, 3),
    (113, 203), (113, 128), (128, 217), (113, 211), (108, 256), (256, 512), (384, 256), (640, 128),
    (1024, 1024), (1023, 1024), (1024, 1023), (2048, 2048), (4096, 4096), (4224, 4224),
]

VARLEN_SEQLENS = [
    (1, 1), (1, 3), (2, 1), (511, 1), (3, 513), (64, 128), (128, 128), (256, 256), (113, 203),
    (128, 217), (113, 211), (108, 256), (256, 512), (640, 128), (512, 256),
    (1024, 1024), (1024, 1023), (2048, 2048),
]

SMALL_VARLEN_HEADS = [1, 4, 6]
SMALL_VARLEN_HEAD_COUNTS = sorted(
    {
        head_counts
        for heads in SMALL_VARLEN_HEADS
        for head_counts in ((heads, heads), (heads, 1), (3 * heads, heads))
    }
)
SMALL_VARLEN_SHAPES = [
    (1, 1, 8), (1, 1, 64), (1, 1, 512), (1, 32, 64), (1, 32, 512), (7, 1, 512), (7, 32, 64),
    (20, 1, 64), (20, 1, 512), (20, 32, 512), (20, 128, 512),
]
PADDING_ROWS = [1, 37, 128]
LONG_SEQLENS = [1024, 2048, 4096]
NARROW_WINDOWS = [(0, 0), (63, 0), (0, 63), (127, 128)]
ZERO_LENGTH_QUERY_SEQLENS = [(1024, 1024), (4096, 4096)]
ZERO_LENGTH_QUERY_PADDING_ROWS = 128
GQA_QHEAD_PER_KVHEAD_SHAPES = [(3, 18, 6, 4096), (8, 16, 2, 2048), (1, 8, 2, 4096)]
PACKED_GQA_QHEAD_PER_KVHEAD = [2, 3, 4, 8, 16, 32]
SHORT_QUERY_SEQLENS = [(1, 1), (5, 77), (33, 1000), (77, 33), (96, 128), (128, 256), (192, 192)]


def cuda_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def window_for(name, seqlen_k):
    generator = torch.Generator().manual_seed(seqlen_k)
    left, right = torch.randint(0, seqlen_k, (2,), generator=generator).tolist()
    return {
        "dense": ((-1, -1), False),
        "causal": ((-1, 0), True),
        "local": ((left, right), False),
        "local_left": ((left, -1), False),
        "local_right": ((-1, right), False),
    }[name]


BATCHED_WINDOW_SEQLENS = [
    (window_name, seqlen_q, seqlen_k)
    for window_name in WINDOW_NAMES
    for seqlen_q, seqlen_k in BATCHED_SEQLENS
    if window_name == "causal" or window_for(window_name, seqlen_k)[0] != window_for("causal", seqlen_k)[0]
]


def cu_seqlens_of(lengths):
    cumulative = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    return cumulative.to(torch.int32).pin_memory()


def near_maximum_lengths(batch, maximum, generator, zero_lengths=False):
    lengths = torch.randint(max(0 if zero_lengths else 1, maximum - 20), maximum + 1, (batch,), generator=generator)
    if zero_lengths:
        lengths[::5] = 0
        lengths[-1] = 0
    return lengths


def uniform_lengths(batch, minimum, maximum, generator):
    return torch.randint(minimum, maximum + 1, (batch,), generator=generator)
