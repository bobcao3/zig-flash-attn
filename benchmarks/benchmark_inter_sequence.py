import argparse
import functools
import importlib
import json
import time
import warnings

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import timing
from benchmark_attn import attention_bytes, attention_flops, gradient_of, versions, visible_pairs
from zigfa import flash_attn_inter_sequence_func, prepare_inter_sequence_metadata
from zigfa._impl import inter_sequence, runtime

DENSE = (-1, -1)
CAUSAL = (-1, 0)
TEXT, CLEAN_IMAGE, NOISED_IMAGE, SECOND_TEXT, SECOND_IMAGE = range(5)
CAPTION = (
    ("text", CAUSAL, (0,)),
    ("image", DENSE, (0, 1)),
    ("image", DENSE, (0, 2)),
    ("image", DENSE, (0, 3)),
    ("text", CAUSAL, (0, 4)),
    ("text", CAUSAL, (0, 5)),
    ("text", CAUSAL, (0, 1, 2, 3, 4, 5, 6)),
)
IMAGE_TO_TEXT = (("image", DENSE, (0,)), ("text", CAUSAL, (0, 1)))
TEXT_TO_IMAGE = (("text", CAUSAL, (0,)), ("image", DENSE, (0, 1)), ("image", DENSE, (0, 1, 2)))
INTERLEAVED = (
    ("text", CAUSAL, (TEXT,)),
    ("image", DENSE, (TEXT, CLEAN_IMAGE)),
    ("image", DENSE, (TEXT, NOISED_IMAGE)),
    ("text", CAUSAL, (TEXT, CLEAN_IMAGE, SECOND_TEXT)),
    ("image", DENSE, (TEXT, CLEAN_IMAGE, SECOND_TEXT, SECOND_IMAGE)),
)
# Each layout names the sequences of a sample, the tokens of an image, and the range of the tokens of a text.
LAYOUTS = {
    "captions": (CAPTION, 280, (32, 512)),
    "image → text": (IMAGE_TO_TEXT, 560, (32, 512)),
    "text → image": (TEXT_TO_IMAGE, 1024, (64, 512)),
    "interleaved": (INTERLEAVED, 1024, (64, 512)),
    "4096-token images": (INTERLEAVED, 4096, (64, 512)),
}
HEAD_DIMS = (64, 128, 256, 512)
MAX_PARTS = 8
# FlashAttention-4 supports block sparsity on SM90 and SM100 GPUs; on SM120 GPUs it rejects block-sparse tensors.
# Its SM100 forward needs query blocks of a multiple of 256 rows, two query tiles of 128 rows.
FLASH_BLOCK_SIZES = {9: (128, 128), 10: (256, 128)}
# With PyTorch 2.14.1, the Triton kernels of torch.compile(flex_attention) above head dim 256 fail to compile on most
# GPUs, and on a B300 they fault with a misaligned address, which ends the CUDA context of the process.
# FlashAttention-4 rejects head dims above 256.
FLEX_MAX_HEAD_DIM = 256
RECOMPILE_LIMIT = 128
# Autotuning makes the Triton FlexAttention faster in some cases and slower in others, so each case times both
# modes and keeps the faster.
AUTOTUNE_MODE = "max-autotune-no-cudagraphs"
AUTOTUNED_FLEX = "FlexAttention autotuned"


def visibility_bits_of(visible):
    count = visible.shape[0]
    words = (count + 31) // 32
    padded = torch.zeros(count, words * 32, dtype=torch.int64)
    padded[:, :count] = visible.long()
    packed = (padded.view(count, words, 32) << torch.arange(32)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def layout(samples, parts):
    lengths, sequence_windows, seen = [], [], []
    for sample in samples:
        first = len(lengths)
        for length, (_, window, parts_seen) in zip(sample, parts):
            lengths.append(length)
            sequence_windows.append(window)
            seen.append([first + part for part in parts_seen])
    visible = torch.zeros(len(lengths), len(lengths), dtype=torch.bool)
    for query, keys in enumerate(seen):
        visible[query, keys] = True
    return lengths, visible, sequence_windows


def useful_pairs(lengths_q, lengths_k, visible, windows):
    """Counts the query-key pairs that inter-sequence attention computes: all pairs of a visible pair of
    sequences, and the pairs inside the window of a sequence that sees itself."""
    pairs = 0
    for query, key in visible.nonzero().tolist():
        if query == key:
            pairs += visible_pairs(lengths_q[query], lengths_k[key], windows[query])
        else:
            pairs += lengths_q[query] * lengths_k[key]
    return pairs


def cu_seqlens_of(lengths):
    return torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32).cuda()


def samples_of(parts, image, text_range, tokens, generator):
    samples = []
    while sum(map(sum, samples)) < tokens:
        samples.append(
            tuple(
                image if kind == "image" else int(torch.randint(*text_range, (1,), generator=generator))
                for kind, _, _ in parts
            )
        )
    return samples


def mask_tensors(samples, parts):
    """Returns, for each token, its sample and its part of the sample, and a table whose entry
    `MAX_PARTS * query_part + key_part` is 1 where the query part sees the key part in full, 2 where a part sees
    itself causally, and 0 elsewhere. These define the visibility for FlexAttention."""
    sample_of_token, part_of_token = [], []
    for index, sample in enumerate(samples):
        for part, length in enumerate(sample):
            sample_of_token += [index] * length
            part_of_token += [part] * length
    table = torch.zeros(MAX_PARTS * MAX_PARTS, dtype=torch.int32)
    for query, (_, window, parts_seen) in enumerate(parts):
        for key in parts_seen:
            table[MAX_PARTS * query + key] = 2 if key == query and window == CAUSAL else 1
    return [torch.tensor(values, dtype=torch.int32, device="cuda") for values in (sample_of_token, part_of_token)] + [
        table.cuda()
    ]


def block_mask_of(tensors, total, block_size=128):
    """Returns the FlexAttention block mask of the visibility of `mask_tensors`, with blocks of `block_size` query
    and key rows."""
    sample, part, table = tensors

    def mask_mod(batch, head, query, key):
        rule = table[part[query] * MAX_PARTS + part[key]]
        return (sample[query] == sample[key]) & ((rule == 1) | ((rule == 2) & (key <= query)))

    return create_block_mask(mask_mod, None, None, total, total, BLOCK_SIZE=block_size)


def flex_functions(q, k, v, dout, block_mask, backend, mode=None):
    """Returns the forward and backward functions of FlexAttention with `block_mask`, compiled in the torch.compile
    `mode`, on the kernel backend `backend`: "TRITON" for the Triton kernels, or "FLASH" for the FlashAttention-4
    kernels."""
    inputs = [x.detach().transpose(0, 1).unsqueeze(0).contiguous().requires_grad_() for x in (q, k, v)]
    grad = dout.transpose(0, 1).unsqueeze(0).contiguous()
    compiled = compiled_flex_attention(mode)
    grouped = k.shape[1] != q.shape[1]
    options = {"BACKEND": backend}

    def forward():
        return compiled(*inputs, block_mask=block_mask, enable_gqa=grouped, kernel_options=options)

    return forward, gradient_of(forward, inputs, grad)


@functools.cache
def compiled_flex_attention(mode):
    """Returns `torch.compile(flex_attention, mode=mode)`. Each case has its own token count, so the compiler may
    compile once per case and backend; a backward that donates its saved buffers could not run a second time."""
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, RECOMPILE_LIMIT)
    torch._functorch.config.donated_buffer = False
    return torch.compile(flex_attention, mode=mode)


def keep_faster_flex(results):
    """Keeps, as FlexAttention, the faster of the default and the autotuned Triton FlexAttention in `results`, and
    names the torch.compile mode of the one it keeps."""
    autotuned = results.pop(AUTOTUNED_FLEX, None)
    default = results.get("FlexAttention")
    if autotuned and (default is None or autotuned["seconds"] < default["seconds"]):
        results["FlexAttention"] = {**autotuned, "mode": AUTOTUNE_MODE}
    elif default:
        default["mode"] = "default"


@functools.cache
def flash_block_size():
    """Returns the block size of the block mask for FlexAttention's FLASH backend on this GPU, or None where
    FlashAttention-4 is not installed or has no block sparsity."""
    block_size = FLASH_BLOCK_SIZES.get(torch.cuda.get_device_capability()[0])
    if block_size is None:
        return None
    try:
        importlib.import_module("flash_attn.cute.interface")
    except ImportError:
        return None
    return block_size


def run(name, point, samples, parts, head_dim, arguments, output):
    lengths, visible, windows = layout(samples, parts)
    total = sum(lengths)
    heads = arguments.dim // head_dim
    heads_kv = heads // arguments.gqa
    cu_seqlens = cu_seqlens_of(lengths)
    bits = visibility_bits_of(visible).cuda()
    window_sizes = torch.tensor(windows, dtype=torch.int32).t().contiguous().cuda()
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = (
        torch.randn(total, heads_kv, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    )
    dout = torch.randn_like(q)
    longest = max(lengths)
    metadata = prepare_inter_sequence_metadata(cu_seqlens, cu_seqlens, bits)

    def zigfa_forward():
        return flash_attn_inter_sequence_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            longest,
            longest,
            bits,
            window_sizes=window_sizes,
            metadata=metadata,
        )

    tensors = mask_tensors(samples, parts)
    forward = {"zigfa": zigfa_forward}
    backward = {"zigfa": gradient_of(zigfa_forward, (q, k, v), dout)}
    if head_dim <= FLEX_MAX_HEAD_DIM:
        block_mask = block_mask_of(tensors, total)
        for label, mode in (("FlexAttention", None), (AUTOTUNED_FLEX, AUTOTUNE_MODE)):
            forward[label], backward[label] = flex_functions(q, k, v, dout, block_mask, "TRITON", mode)
    if head_dim <= FLEX_MAX_HEAD_DIM and flash_block_size():
        forward["FA4 FlexAttention"], backward["FA4 FlexAttention"] = flex_functions(
            q, k, v, dout, block_mask_of(tensors, total, flash_block_size()), "FLASH"
        )
    pairs = useful_pairs(lengths, lengths, visible, windows)
    case_seconds = timing.CASE_SECONDS / len(arguments.directions)
    line = f"{name:30} tokens={total:6} sequences={len(lengths):4}"
    for direction in arguments.directions:
        functions = {"fwd": forward, "bwd": backward}[direction]
        is_backward = direction == "bwd"
        flop_count = attention_flops(pairs, heads, head_dim, is_backward)
        byte_count = attention_bytes(total, total, heads, heads_kv, head_dim, is_backward)
        estimate = timing.estimated_seconds(flop_count, byte_count, is_backward)
        measurement = timing.measure(f"{name} {direction}", functions, estimate, case_seconds)
        line += f" | {direction} " + timing.summary(measurement, list(functions), flop_count, byte_count)
        if output is not None:
            record = timing.record(measurement, flop_count, byte_count)
            keep_faster_flex(record["results"])
            output.write(json.dumps({"benchmark": "inter_sequence", **point, "direction": direction, **record}) + "\n")
            output.flush()
    return line


def main():
    parser = argparse.ArgumentParser(
        description="Times inter-sequence attention against compiled FlexAttention with a block mask of the same "
        "visibility, on its Triton kernels and on its FlashAttention-4 kernels"
    )
    parser.add_argument("--layouts", nargs="+", choices=list(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--head-dims", type=int, nargs="+", default=list(HEAD_DIMS))
    parser.add_argument("--directions", nargs="+", choices=("fwd", "bwd"), default=["fwd", "bwd"])
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--gqa", type=int, default=1, help="query heads per key and value head")
    parser.add_argument("--json", help="append one JSON record per case and direction to this file")
    arguments = parser.parse_args()
    if not inter_sequence.forward_head_dims(runtime.current_device().device_id):
        print(f"zigfa builds no inter-sequence kernels for {torch.cuda.get_device_name()}.", flush=True)
        return
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    warnings.filterwarnings("ignore", message="Argument .* cannot be converted to a JitArgument")
    started = time.perf_counter()
    durations = []
    print(timing.header(), flush=True)
    print(
        "Inputs are bfloat16 values drawn from N(0, 1). Each line times inter-sequence attention against compiled "
        "FlexAttention with a block mask of the same visibility, on its Triton kernels (FlexAttention) and on its "
        "FlashAttention-4 kernels (FA4 FlexAttention) where they run. TF/s count "
        "only the visible query-key pairs; a percent after a rate is of the roofline, and (zigfa N%) is the speed "
        "of zigfa as a percent of the speed of that implementation.",
        flush=True,
    )
    output = open(arguments.json, "a") if arguments.json else None
    common = {"device": torch.cuda.get_device_name(), "versions": versions()}
    generator = torch.Generator().manual_seed(1)
    for name, (parts, image, text_range) in LAYOUTS.items():
        samples = samples_of(parts, image, text_range, arguments.tokens, generator)
        if name not in arguments.layouts:
            continue
        for head_dim in arguments.head_dims:
            case_started = time.perf_counter()
            point = {
                **common,
                "layout": name,
                "head_dim": head_dim,
                "gqa": arguments.gqa,
                "tokens": sum(map(sum, samples)),
                "sequences": sum(map(len, samples)),
            }
            line = run(f"{name} hd={head_dim:3}", point, samples, parts, head_dim, arguments, output)
            durations.append(time.perf_counter() - case_started)
            print(f"{line} [{durations[-1]:.2f} s]", flush=True)
    print(timing.totals(durations, time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
