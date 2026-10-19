import itertools
from dataclasses import dataclass

import pytest
import torch
from torch.library import opcheck

from checks import case, queue_forward_checks, queue_gradient_checks, queue_requirement
from reference import FULLY_MASKED_LSE, inter_sequence_attention_ref, inter_sequence_visible
from shapes import HEAD_LAYOUTS, cu_seqlens_of, cuda_seed
from zigfa import inter_sequence, runtime
from zigfa.torch import (
    flash_attn_inter_sequence_func,
    flash_attn_inter_sequence_kvpacked_func,
    get_inter_sequence_scheduler_metadata,
    inter_sequence_backward,
    inter_sequence_backward_packed,
    inter_sequence_forward,
    inter_sequence_metadata,
)

HEAD_DIMS = inter_sequence.head_dims(runtime.current_device().device_id)
FORWARD_HEAD_DIMS = inter_sequence.forward_head_dims(runtime.current_device().device_id)
SINGLE_TILE_HEAD_DIMS = tuple(
    head_dim for head_dim in HEAD_DIMS
    if head_dim not in inter_sequence.persistent_head_dims(runtime.current_device().device_id)
)
LAYOUTS = [
    "i2t", "it2i", "titi", "titi_noised_first", "i2t_extend", "titi_extend", "random", "gapped", "separate_keys"
]
DENSE, CAUSAL = (-1, -1), (-1, 0)
M64_MAX_ROWS = 384
SHORT_MAX_ROWS = 8
TEXT, CLEAN_IMAGE, NOISED_IMAGE, SECOND_TEXT, SECOND_IMAGE = range(5)
TITI_WINDOWS = [CAUSAL, DENSE, DENSE, CAUSAL, DENSE]
TITI_PARTS_SEEN = {
    "titi": (
        (TEXT,),
        (TEXT, CLEAN_IMAGE),
        (TEXT, NOISED_IMAGE),
        (TEXT, CLEAN_IMAGE, SECOND_TEXT),
        (TEXT, CLEAN_IMAGE, SECOND_TEXT, SECOND_IMAGE),
    ),
    "titi_noised_first": (
        (TEXT,),
        (TEXT, CLEAN_IMAGE),
        (TEXT, NOISED_IMAGE),
        (TEXT, NOISED_IMAGE, SECOND_TEXT),
        (TEXT, NOISED_IMAGE, SECOND_TEXT, SECOND_IMAGE),
    ),
}
WINDOWS = [(-1, -1), (-1, 0), (-1, 7), (5, -1), (9, 3), (0, 0)]
OPCHECK_TESTS = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

pytestmark = pytest.mark.skipif(not HEAD_DIMS, reason="no inter-sequence kernels run on this GPU")


@dataclass(frozen=True)
class Layout:
    lengths_q: list
    lengths_k: list
    visible: torch.Tensor
    windows: list

    @property
    def count(self):
        return len(self.lengths_q)


def visibility_bits_of(visible):
    count = visible.shape[0]
    words = (count + 31) // 32
    padded = torch.zeros(count, words * 32, dtype=torch.int64)
    padded[:, :count] = visible.long()
    packed = (padded.view(count, words, 32) << torch.arange(32)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def sample_layout(samples, windows_of_parts, parts_seen=None):
    lengths_q, lengths_k, windows, seen = [], [], [], []
    for sample in samples:
        first = len(lengths_q)
        for part, length in enumerate(sample):
            length_q, length_k = length if isinstance(length, tuple) else (length, length)
            lengths_q.append(length_q)
            lengths_k.append(length_k)
            windows.append(windows_of_parts[part])
            seen.append([first + p for p in (range(part + 1) if parts_seen is None else parts_seen[part])])
    visible = torch.zeros(len(lengths_q), len(lengths_q), dtype=torch.bool)
    for query, keys in enumerate(seen):
        visible[query, keys] = True
    return Layout(lengths_q, lengths_k, visible, windows)


def extend_layout(name, generator):
    if name == "i2t_extend":
        images, cached_texts = torch.randint(16, 300, (2, 5), generator=generator).tolist()
        decoding, waiting = 1, 0
        new_tokens = [decoding, waiting, *torch.randint(2, 200, (3,), generator=generator).tolist()]
        samples = [
            ((0, image), (new, text + new)) for image, text, new in zip(images, cached_texts, new_tokens)
        ]
        return sample_layout(samples, [DENSE, CAUSAL])
    samples = []
    parts = torch.randint(8, 260, (4, 4), generator=generator).tolist()
    for request, (text, image, second_text, new) in enumerate(parts):
        cached = ((0, text), (0, image))
        generates_image, decodes = request % 2 == 1, request == 2
        if generates_image:
            samples.append((*cached, (0, second_text), (image, image)))
        else:
            samples.append((*cached, (1 if decodes else new, second_text + new), (0, 0)))
    return sample_layout(samples, [CAUSAL, DENSE, CAUSAL, DENSE])


def random_layout(count, generator, gaps=False, separate_keys=False):
    lengths_q = torch.randint(1, 160, (count,), generator=generator).tolist()
    lengths_k = list(lengths_q)
    if separate_keys:
        lengths_k = [length + int(torch.randint(0, 40, (), generator=generator)) for length in lengths_q]
        lengths_q[:: max(1, count // 3)] = [0] * len(lengths_q[:: max(1, count // 3)])
    visible = torch.rand(count, count, generator=generator) < 0.4
    if gaps:
        visible = torch.zeros(count, count, dtype=torch.bool)
        visible[:, ::2] = True
        visible.fill_diagonal_(True)
    windows = [WINDOWS[int(index)] for index in torch.randint(0, len(WINDOWS), (count,), generator=generator)]
    return Layout(lengths_q, lengths_k, visible, windows)


def build_layout(name, seed):
    generator = torch.Generator().manual_seed(seed)
    if name == "i2t":
        samples = [(int(image), int(text)) for image, text in torch.randint(16, 300, (5, 2), generator=generator)]
        return sample_layout(samples, [DENSE, CAUSAL])
    if name == "it2i":
        samples = [tuple(int(x) for x in torch.randint(8, 260, (3,), generator=generator)) for _ in range(3)]
        return sample_layout(samples, [CAUSAL, DENSE, DENSE])
    if name.endswith("_extend"):
        return extend_layout(name, generator)
    if name in TITI_PARTS_SEEN:
        samples = []
        for text, image, second_text, second_image in torch.randint(8, 260, (3, 4), generator=generator).tolist():
            samples.append((text, image, image, second_text, second_image))
        return sample_layout(samples, TITI_WINDOWS, TITI_PARTS_SEEN[name])
    return random_layout(
        11 if name != "gapped" else 9, generator, gaps=name == "gapped", separate_keys=name == "separate_keys"
    )


def layout_tensors(layout):
    cu_q = cu_seqlens_of(torch.tensor(layout.lengths_q)).cuda(non_blocking=True)
    cu_k = cu_seqlens_of(torch.tensor(layout.lengths_k)).cuda(non_blocking=True)
    bits = visibility_bits_of(layout.visible).cuda()
    window_sizes = torch.tensor(layout.windows, dtype=torch.int32).t().contiguous().cuda()
    return cu_q, cu_k, bits, window_sizes


def leaves(layout, num_head, num_head_kv, head_dim):
    rows_q, rows_k = sum(layout.lengths_q), sum(layout.lengths_k)
    shapes = ((rows_q, num_head), (rows_k, num_head_kv), (rows_k, num_head_kv))
    return [
        torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for rows, heads in shapes
    ]


def check_layout(layout, num_head, num_head_kv, head_dim, **options):
    q, k, v = leaves(layout, num_head, num_head_kv, head_dim)
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    windows = layout.windows
    if "causal" in options or "window_size" in options:
        left, right = options.get("window_size", (-1, -1))
        windows = [(left, 0 if options.get("causal") else right)] * layout.count
    else:
        options["window_sizes"] = window_sizes
    out, lse = flash_attn_inter_sequence_func(
        q, k, v, cu_q, cu_k, max(layout.lengths_q), max(layout.lengths_k), bits, return_attn_probs=True, **options
    )
    mask = inter_sequence_visible(cu_q, cu_k, layout.visible, windows)
    queue_forward_checks([(out, lse)], inter_sequence_attention_ref, q, k, v, mask)
    if head_dim not in HEAD_DIMS:
        return out, lse, None, mask
    dout = torch.randn_like(out)
    gradients = torch.autograd.grad(out, (q, k, v), dout)
    queue_gradient_checks([gradients], inter_sequence_attention_ref, q, k, v, dout, mask)
    return out, lse, gradients, mask


@pytest.mark.parametrize("head_dim", FORWARD_HEAD_DIMS)
@pytest.mark.parametrize("layout_name", LAYOUTS)
def test_layouts_match_reference(head_dim, layout_name):
    for heads_name, seed in itertools.product(HEAD_LAYOUTS, range(2)):
        with case(heads=heads_name, seed=seed):
            cuda_seed(seed)
            check_layout(build_layout(layout_name, seed), *HEAD_LAYOUTS[heads_name], head_dim)


@pytest.mark.parametrize("head_dim", FORWARD_HEAD_DIMS)
@pytest.mark.parametrize("options", [dict(causal=True), dict(window_size=(6, 2)), dict(window_size=(-1, -1))])
def test_scalar_windows_match_reference(head_dim, options):
    cuda_seed(3)
    with case(**options):
        check_layout(build_layout("random", 3), 6, 2, head_dim, **dict(options))


def test_rows_that_see_no_keys_return_zeros():
    head_dim = HEAD_DIMS[0]
    layout = build_layout("i2t", 4)
    layout.visible[2] = False
    layout.visible[5, 5] = False
    layout.windows[3] = (0, -1)
    cuda_seed(4)
    out, lse, gradients, mask = check_layout(layout, 4, 4, head_dim)
    blind = ~mask.any(dim=-1).cuda()
    queue_requirement("rows without keys return zeros", (out[blind] == 0).all())
    queue_requirement("rows without keys have an lse of -inf", (lse[:, blind] == FULLY_MASKED_LSE).all())
    queue_requirement("rows without keys get no query gradient", (gradients[0][blind] == 0).all())


@pytest.mark.parametrize("lengths_q, lengths_k", [([0, 0], [300, 256]), ([10, 30], [0, 0])])
def test_batches_without_queries_or_keys_return_zeros(lengths_q, lengths_k):
    layout = Layout(lengths_q, lengths_k, torch.tensor([[True, False], [True, True]]), [CAUSAL, DENSE])
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    for head_dim in HEAD_DIMS:
        with case(head_dim=head_dim):
            cuda_seed(12)
            q, k, v = leaves(layout, 4, 2, head_dim)
            out, lse = flash_attn_inter_sequence_func(
                q, k, v, cu_q, cu_k, max(lengths_q), max(lengths_k), bits, window_sizes=window_sizes,
                return_attn_probs=True,
            )
            gradients = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
            queue_requirement("the output is zero", (out == 0).all())
            queue_requirement("the lse is -inf", (lse == FULLY_MASKED_LSE).all())
            for name, gradient in zip(("dq", "dk", "dv"), gradients):
                queue_requirement(f"{name} is zero", (gradient == 0).all())


def test_many_sequences_span_several_words():
    generator = torch.Generator().manual_seed(5)
    layout = random_layout(70, generator)
    cuda_seed(5)
    lengths = [length // 4 + 1 for length in layout.lengths_q]
    check_layout(Layout(lengths, lengths, layout.visible, layout.windows), 4, 2, HEAD_DIMS[0])


def test_kvpacked_matches_unpacked():
    head_dim = HEAD_DIMS[-1]
    layout = build_layout("gapped", 6)
    cuda_seed(6)
    q, k, v = leaves(layout, 6, 2, head_dim)
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    arguments = (cu_q, cu_k, max(layout.lengths_q), max(layout.lengths_k), bits)
    out = flash_attn_inter_sequence_func(q, k, v, *arguments, window_sizes=window_sizes)
    kv = torch.stack([k.detach(), v.detach()], dim=1).requires_grad_()
    packed = flash_attn_inter_sequence_kvpacked_func(q, kv, *arguments, window_sizes=window_sizes)
    dout = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    packed_dq, dkv = torch.autograd.grad(packed, (q, kv), dout)
    queue_requirement("kvpacked output equals unpacked output", (packed == out).all())
    queue_requirement("kvpacked dq is close to unpacked dq", (packed_dq - dq).abs().max() < 1e-2)
    queue_requirement("kvpacked dkv stacks dk and dv", (dkv == torch.stack([dk, dv], dim=1)).all())


def test_scheduler_metadata_is_reusable():
    layout = build_layout("it2i", 7)
    cuda_seed(7)
    q, k, v = (tensor.detach() for tensor in leaves(layout, 4, 4, HEAD_DIMS[0]))
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    metadata = get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits)
    arguments = (q, k, v, cu_q, cu_k, max(layout.lengths_q), max(layout.lengths_k), bits)
    fresh = flash_attn_inter_sequence_func(*arguments, window_sizes=window_sizes)
    reused = flash_attn_inter_sequence_func(*arguments, window_sizes=window_sizes, scheduler_metadata=metadata)
    queue_requirement("reused metadata gives the same output", (fresh == reused).all())


@pytest.mark.parametrize("count", [1, 31, 32, 33, 100])
def test_metadata_matches_its_definition(count):
    generator = torch.Generator().manual_seed(count)
    layout = random_layout(count, generator, separate_keys=True)
    cu_q, cu_k, bits, _ = layout_tensors(layout)
    visible_bits_t, cu_visible_k, cu_visible_q = get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits)
    lengths_q, lengths_k = torch.tensor(layout.lengths_q), torch.tensor(layout.lengths_k)
    visible = layout.visible.long()
    expected_k = torch.cat([torch.zeros(1, dtype=torch.long), (visible @ lengths_k).cumsum(0)])
    expected_q = torch.cat([torch.zeros(1, dtype=torch.long), (visible.t() @ lengths_q).cumsum(0)])
    assert torch.equal(visible_bits_t.cpu(), visibility_bits_of(layout.visible.t()))
    assert torch.equal(cu_visible_k.cpu().long(), expected_k)
    assert torch.equal(cu_visible_q.cpu().long(), expected_q)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("count", [6, 31, 32, 33, 100, 300, 1100])
def test_block_tables_match_their_definition(count, persistent):
    device_id = runtime.current_device().device_id
    head_dims = inter_sequence.persistent_head_dims(device_id) if persistent else SINGLE_TILE_HEAD_DIMS
    if not head_dims:
        pytest.skip("no kernel of this path runs on this GPU")
    generator = torch.Generator().manual_seed(count)
    layout = random_layout(count, generator, separate_keys=True)
    layout.lengths_q[1::7] = [M64_MAX_ROWS - 70 + 37 * index for index in range(len(layout.lengths_q[1::7]))]
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    sequences = inter_sequence.Sequences(
        cu_q, cu_k, bits, *get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits), window_sizes
    )
    q, k, v = (tensor.detach() for tensor in leaves(layout, 2, 2, head_dims[0]))
    out, lse = torch.empty_like(q), torch.empty(q.shape[1], q.shape[0], dtype=torch.float32, device=q.device)
    workspace = tuple(
        torch.empty(shape, dtype=torch.int32, device=q.device)
        for shape in inter_sequence.forward_workspace_shapes(tuple(q.shape), count)
    )
    inter_sequence.forward(q, k, v, out, lse, workspace, sequences)
    capacity = inter_sequence.block_table_capacity(q.shape[0], count)
    words = workspace[0].cpu().long()
    lengths_q = torch.tensor(layout.lengths_q)
    visible_keys = layout.visible.long() @ torch.tensor(layout.lengths_k)
    last_block_rows = lengths_q % 128
    m128 = (lengths_q > 64) & ((last_block_rows == 0) | (last_block_rows > 64) | (lengths_q >= M64_MAX_ROWS))
    short = lengths_q <= SHORT_MAX_ROWS
    group = torch.zeros_like(lengths_q) if persistent else torch.where(short, 2, torch.where(m128, 0, 1))
    if persistent:
        table_rows = inter_sequence.BLOCK_TABLE_ROWS
    else:
        table_rows = (inter_sequence.kernel_info(device_id, "fwd", head_dims[0]).cluster_block_m,)
    for rows in table_rows:
        table = inter_sequence.block_table(0, capacity, rows) // 4
        blocks = (lengths_q + rows - 1) // rows
        assert words[table] == int(blocks.sum())
        first_entry = table + inter_sequence.BLOCK_TABLE_HEADER_BYTES // 4
        entries = words[first_entry : first_entry + 2 * int(blocks.sum())].view(-1, 2)
        sequences = entries[:, 0].unique_consecutive().tolist()
        assert sorted(sequences) == [sequence for sequence in range(count) if blocks[sequence]]
        first_block = 0
        for sequence in sequences:
            expected = torch.tensor([sequence, first_block]).expand(int(blocks[sequence]), 2)
            assert torch.equal(entries[first_block : first_block + int(blocks[sequence])], expected)
            first_block += int(blocks[sequence])
        order = [(int(group[sequence]), -length_class(int(visible_keys[sequence])), sequence) for sequence in sequences]
        assert order == sorted(order)


def length_class(length):
    if length == 0:
        return 0
    exponent = length.bit_length() - 1
    return exponent << 2 | (length << 2 >> exponent) & 3


def test_invalid_arguments_raise():
    layout = build_layout("i2t", 8)
    q, k, v = (tensor.detach() for tensor in leaves(layout, 4, 2, HEAD_DIMS[0]))
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    arguments = (q, k, v, cu_q, cu_k, 1, 1)
    with pytest.raises(ValueError, match="window_sizes"):
        flash_attn_inter_sequence_func(*arguments, bits, causal=True, window_sizes=window_sizes)
    with pytest.raises(NotImplementedError, match="deterministic"):
        flash_attn_inter_sequence_func(*arguments, bits, deterministic=True)
    with pytest.raises(ValueError, match="visibility_bits"):
        flash_attn_inter_sequence_func(*arguments, bits[:-1])
    with pytest.raises(ValueError, match="window_sizes"):
        flash_attn_inter_sequence_func(*arguments, bits, window_sizes=window_sizes[:, :-1])
    unsupported = [head_dim for head_dim in runtime.head_dims() if head_dim not in HEAD_DIMS]
    if unsupported:
        q, k, v = (t.new_zeros(*t.shape[:2], unsupported[0]) for t in (q, k, v))
        with pytest.raises(NotImplementedError, match="head_dim"):
            flash_attn_inter_sequence_func(q, k, v, cu_q, cu_k, 1, 1, bits)


def op_arguments():
    layout = build_layout("gapped", 9)
    cuda_seed(9)
    q, k, v = (tensor.detach() for tensor in leaves(layout, 4, 2, HEAD_DIMS[0]))
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)
    metadata = get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits)
    return q, k, v, (cu_q, cu_k, bits, *metadata, window_sizes), HEAD_DIMS[0] ** -0.5


def test_ops_pass_opcheck():
    q, k, v, layout, scale = op_arguments()
    opcheck(inter_sequence_metadata, layout[:3], test_utils=OPCHECK_TESTS)
    opcheck(inter_sequence_forward, (q, k, v, *layout, scale, -1, -1), test_utils=OPCHECK_TESTS)
    out, lse = inter_sequence_forward(q, k, v, *layout, scale, -1, -1)
    dout = torch.randn_like(out)
    arguments = (dout, q, k, v, out, lse, *layout, scale, -1, -1)
    opcheck(inter_sequence_backward, arguments, test_utils=OPCHECK_TESTS)
    dq, dkv = q.new_empty(q.shape), k.new_empty((k.shape[0], 2, *k.shape[1:]))
    opcheck(inter_sequence_backward_packed, (*arguments, dq, dkv), test_utils=OPCHECK_TESTS)


def test_compiled_attention_matches_eager():
    torch._dynamo.reset()
    layout = build_layout("i2t", 10)
    cuda_seed(10)
    q, k, v = leaves(layout, 4, 2, HEAD_DIMS[0])
    cu_q, cu_k, bits, window_sizes = layout_tensors(layout)

    def attention(q, k, v):
        metadata = get_inter_sequence_scheduler_metadata(cu_q, cu_k, bits)
        return flash_attn_inter_sequence_func(
            q, k, v, cu_q, cu_k, 1, 1, bits, window_sizes=window_sizes, scheduler_metadata=metadata
        )

    compiled = torch.compile(attention, fullgraph=True)
    out = compiled(q, k, v)
    dout = torch.randn_like(out)
    gradients = torch.autograd.grad(out, (q, k, v), dout)
    eager = attention(q, k, v)
    eager_gradients = torch.autograd.grad(eager, (q, k, v), dout)
    queue_requirement("compiled output equals eager output", (out == eager).all())
    for name, compiled_gradient, eager_gradient in zip(("dq", "dk", "dv"), gradients, eager_gradients):
        close = (compiled_gradient - eager_gradient).abs().max() < 1e-2
        queue_requirement(f"compiled {name} is close to eager {name}", close)
    torch._dynamo.reset()


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_long_image_and_text_sequences(head_dim):
    layouts = {
        "i2t": sample_layout([(1024, 300), (560, 777)], [DENSE, CAUSAL]),
        "it2i": sample_layout([(200, 1024, 1024)], [CAUSAL, DENSE, DENSE]),
        "titi": sample_layout([(77, 1024, 1024, 300, 1024)], TITI_WINDOWS, TITI_PARTS_SEEN["titi"]),
    }
    for name, layout in layouts.items():
        with case(layout=name):
            cuda_seed(len(name))
            check_layout(layout, 8, 2, head_dim)


@pytest.mark.parametrize("head_dim", inter_sequence.persistent_head_dims(runtime.current_device().device_id))
@pytest.mark.parametrize("packing", [1, 4])
def test_persistent_slices_the_keys_of_single_warp_blocks(head_dim, packing):
    device_id = runtime.current_device().device_id
    instance = inter_sequence.persistent_instance(head_dim, packing)
    ctas = inter_sequence.resident_ctas(device_id, instance)
    image_rows = 384
    num_head_kv = -(-ctas // (image_rows * packing // max(inter_sequence.BLOCK_TABLE_ROWS)))
    cuda_seed(19)
    seen = tuple((0, part) if part else (0,) for part in range(6))
    layout = sample_layout([(image_rows, 1, 1, 3, 1, 2)], [DENSE] + [CAUSAL] * 5, seen)
    check_layout(layout, packing * num_head_kv, num_head_kv, head_dim)


@pytest.mark.parametrize("head_dim", inter_sequence.persistent_head_dims(runtime.current_device().device_id))
def test_persistent_splits_the_last_round_of_long_blocks(head_dim):
    device_id = runtime.current_device().device_id
    instance = inter_sequence.persistent_instance(head_dim, 1)
    ctas = inter_sequence.resident_ctas(device_id, instance)
    image_rows = 1024
    blocks = image_rows // max(inter_sequence.BLOCK_TABLE_ROWS)
    heads = -(-(ctas + ctas // 4) // blocks)
    cuda_seed(13)
    check_layout(sample_layout([(37, image_rows)], [CAUSAL, DENSE]), heads, heads, head_dim)
