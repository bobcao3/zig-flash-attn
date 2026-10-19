import argparse
import itertools
import statistics

import torch

from zigfa.torch import flash_attn_inter_sequence_func, flash_attn_varlen_func, get_inter_sequence_scheduler_metadata

DENSE = (-1, -1)
CAUSAL = (-1, 0)
TEXT, CLEAN_IMAGE, NOISED_IMAGE, SECOND_TEXT, SECOND_IMAGE = range(5)
TASKS = {
    "i2t": (("image", DENSE, (0,)), ("text", CAUSAL, (0, 1))),
    "it2i": (("text", CAUSAL, (0,)), ("image", DENSE, (0, 1)), ("image", DENSE, (0, 1, 2))),
    "titi": (
        ("text", CAUSAL, (TEXT,)),
        ("image", DENSE, (TEXT, CLEAN_IMAGE)),
        ("image", DENSE, (TEXT, NOISED_IMAGE)),
        ("text", CAUSAL, (TEXT, CLEAN_IMAGE, SECOND_TEXT)),
        ("image", DENSE, (TEXT, CLEAN_IMAGE, SECOND_TEXT, SECOND_IMAGE)),
    ),
    "caption": (
        ("text", CAUSAL, (0,)),
        ("image", DENSE, (0, 1)),
        ("image", DENSE, (0, 2)),
        ("image", DENSE, (0, 3)),
        ("text", CAUSAL, (0, 4)),
        ("text", CAUSAL, (0, 5)),
        ("text", CAUSAL, (0, 1, 2, 3, 4, 5, 6)),
    ),
}
EXTEND_TASKS = {
    "i2t_extend": ("image", "text"),
    "titi_extend": ("text", "image", "text"),
    "it2i_generate": ("text", "image"),
    "titi_generate": ("text", "image", "text", "image"),
    "caption_extend": ("text", "image", "image", "image", "text", "text", "text"),
}
NEW_TOKENS_AND_REQUESTS = ((1, 64), (128, 16), (1024, 4))
IMAGE_REQUESTS = (1, 4)
MODELS = {
    "gemma": (
        ((16, 8, 256), (8, 4, 256), (32, 16, 128)),
        ("i2t", "titi", "i2t_extend", "titi_extend"),
        (256, 560, 1120),
        ((32, 512), (256, 2048)),
    ),
    "qwen-image": (((24, 24, 128),), ("it2i", "titi", "it2i_generate", "titi_generate"), (1024, 4096), ((64, 512),)),
    "gemma4-12b": (((16, 8, 256),), ("caption", "caption_extend"), (280,), ((32, 512),)),
}


def paired_milliseconds(functions, rounds, repeats=3, warmup=5):
    for _ in range(warmup):
        for function in functions:
            function()
    torch.cuda.synchronize()
    times = [[] for _ in functions]
    for round_index in range(rounds):
        order = list(range(len(functions)))[:: -1 if round_index % 2 else 1]
        events = {}
        for index in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                functions[index]()
            end.record()
            events[index] = (start, end)
        torch.cuda.synchronize()
        for index, (start, end) in events.items():
            times[index].append(start.elapsed_time(end) / repeats)
    medians = [statistics.median(samples) for samples in times]
    speedups = [statistics.median(a / b for a, b in zip(times[0], samples)) for samples in times]
    return medians, speedups


def graph_replay(function, copies=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(copies):
            function()
    return graph.replay


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
    return lengths, visibility_bits_of(visible), torch.tensor(sequence_windows, dtype=torch.int32).t().contiguous()


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


def run(name, samples, parts, heads, heads_kv, head_dim, rounds):
    lengths, bits, window_sizes = layout(samples, parts)
    sample_lengths = [sum(sample) for sample in samples]
    total = sum(lengths)
    cu_seqlens, cu_samples = cu_seqlens_of(lengths), cu_seqlens_of(sample_lengths)
    bits, window_sizes = bits.cuda(), window_sizes.cuda()
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = (torch.randn_like(q[:, :heads_kv]).requires_grad_() for _ in range(2))
    longest, longest_sample = max(lengths), max(sample_lengths)
    metadata = get_inter_sequence_scheduler_metadata(cu_seqlens, cu_seqlens, bits)

    def dense():
        return flash_attn_varlen_func(q, k, v, cu_samples, cu_samples, longest_sample, longest_sample)

    def inter(scheduler_metadata=metadata):
        return flash_attn_inter_sequence_func(
            q, k, v, cu_seqlens, cu_seqlens, longest, longest, bits, window_sizes=window_sizes,
            scheduler_metadata=scheduler_metadata,
        )

    (dense_forward, inter_forward, _), (_, forward_speedup, rebuilt_speedup) = paired_milliseconds(
        [dense, inter, lambda: inter(None)], rounds
    )
    outputs = [dense(), inter()]
    dout = torch.randn_like(outputs[0])
    backward = [lambda out=out: torch.autograd.grad(out, (q, k, v), dout, retain_graph=True) for out in outputs]
    (dense_backward, inter_backward), (_, backward_speedup) = paired_milliseconds(backward, rounds)
    print(
        f"{name:44} tokens={total:6} sequences={len(lengths):4} | forward dense {dense_forward:7.3f} ms "
        f"inter {inter_forward:7.3f} ms x{forward_speedup:.2f} (metadata rebuilt x{rebuilt_speedup:.2f}) | "
        f"backward dense {dense_backward:7.3f} ms inter {inter_backward:7.3f} ms x{backward_speedup:.2f}",
        flush=True,
    )


def extend_requests(parts, image, text_range, new_tokens, count, generator):
    requests = []
    for _ in range(count):
        lengths = [
            image if kind == "image" else int(torch.randint(*text_range, (1,), generator=generator)) for kind in parts
        ]
        last = (image, image) if parts[-1] == "image" else (new_tokens, lengths[-1] + new_tokens)
        requests.append([*((0, length) for length in lengths[:-1]), last])
    return requests


def run_extend(name, requests, parts, heads, heads_kv, head_dim, rounds, copies=10):
    lengths_q = [length_q for lengths in requests for length_q, _ in lengths]
    lengths_k = [length_k for lengths in requests for _, length_k in lengths]
    request_q = [sum(length_q for length_q, _ in lengths) for lengths in requests]
    request_k = [sum(length_k for _, length_k in lengths) for lengths in requests]
    visible = torch.zeros(len(lengths_q), len(lengths_q), dtype=torch.bool)
    for first in range(0, len(lengths_q), len(parts)):
        for part in range(len(parts)):
            visible[first + part, first : first + part + 1] = True
    windows = [DENSE if kind == "image" else CAUSAL for _ in requests for kind in parts]
    window_sizes = torch.tensor(windows, dtype=torch.int32).t().contiguous().cuda()
    bits = visibility_bits_of(visible).cuda()
    cu_q, cu_k, cu_request_q, cu_request_k = map(cu_seqlens_of, (lengths_q, lengths_k, request_q, request_k))
    q = torch.randn(sum(lengths_q), heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(sum(lengths_k), heads_kv, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    metadata = get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits)

    def dense():
        return flash_attn_varlen_func(
            q, k, v, cu_request_q, cu_request_k, max(request_q), max(request_k), causal=parts[-1] == "text"
        )

    def inter():
        return flash_attn_inter_sequence_func(
            q, k, v, cu_q, cu_k, max(lengths_q), max(lengths_k), bits, window_sizes=window_sizes,
            scheduler_metadata=metadata,
        )

    replays = [graph_replay(function, copies) for function in (dense, inter)]
    (dense_forward, inter_forward), (_, speedup) = paired_milliseconds(replays, rounds)
    print(
        f"{name:62} queries={sum(lengths_q):6} keys={sum(lengths_k):7} | forward dense "
        f"{dense_forward / copies * 1e3:8.1f} us inter {inter_forward / copies * 1e3:8.1f} us x{speedup:.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compares inter-sequence attention with dense varlen attention over the same samples"
    )
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=[*TASKS, *EXTEND_TASKS], default=[*TASKS, *EXTEND_TASKS])
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=20)
    arguments = parser.parse_args()
    generator = torch.Generator().manual_seed(1)
    for model in arguments.models:
        shapes, tasks, images, text_ranges = MODELS[model]
        for task, (heads, heads_kv, head_dim), image, text_range in itertools.product(
            (task for task in tasks if task in arguments.tasks), shapes, images, text_ranges
        ):
            name = f"{model} {task} h{heads}/{heads_kv} d{head_dim} image={image} text<{text_range[1]}"
            if task in TASKS:
                samples = samples_of(TASKS[task], image, text_range, arguments.tokens, generator)
                run(name, samples, TASKS[task], heads, heads_kv, head_dim, arguments.rounds)
                continue
            parts = EXTEND_TASKS[task]
            steps = NEW_TOKENS_AND_REQUESTS if parts[-1] == "text" else ((None, count) for count in IMAGE_REQUESTS)
            for new_tokens, count in steps:
                requests = extend_requests(parts, image, text_range, new_tokens, count, generator)
                step = f"requests={count}" + (f" new={new_tokens}" if new_tokens else "")
                run_extend(f"{name} {step}", requests, parts, heads, heads_kv, head_dim, arguments.rounds)


if __name__ == "__main__":
    main()
