import functools
import logging
import numbers
import os
from dataclasses import dataclass
from typing import Optional

from . import runtime

PACK_EFFICIENCY = 0.75
PACK_MAX_TILES = {80: 1, 90: 2, 100: 1, 120: 1}
PACK_BY_WAVES = {80: False, 90: True, 100: False, 120: False}
PACK_WAVES_MIN_ROWS = 1024
PACK_BOUNDED_ARCHS = (90, 100)
UNBOUNDED = (-1, -1)
MASKS = ("dense", "causal", "local_right", "local_left", "local_two_sided")
SCHEDULER_METADATA_MIN_BATCH = 4
DEBUG_LEVELS = ("INFO",)

logger = logging.getLogger("zigfa")


@dataclass(frozen=True)
class ShapeFacts:
    varlen: bool
    qhead_per_kvhead: Optional[int]
    batch_size: Optional[int]
    mean_seqlen_q: int
    max_seqlen_q_hint: Optional[int]
    max_seqlen_k_hint: Optional[int]
    max_seqlen_q: Optional[int]
    max_seqlen_k: Optional[int]
    num_head_kv: Optional[int] = None
    num_sm: Optional[int] = None


def usable_max_seqlen(value, row_axis_len):
    if value is not None and not isinstance(value, numbers.Integral):
        raise TypeError(f"max_seqlen_q and max_seqlen_k must be integers, got {type(value).__name__}")
    return value if value is not None and 0 < value <= row_axis_len else None


def facts(q_shape, k_shape=None, cu_seqlens_shape=None, max_seqlen_q=None, max_seqlen_k=None, num_sm=None):
    varlen = len(q_shape) == 3
    qhead_per_kvhead = None if k_shape is None else q_shape[-2] // k_shape[-2]
    if varlen:
        batch_size = None if cu_seqlens_shape is None else cu_seqlens_shape[0] - 1
        mean_seqlen_q = q_shape[0] // batch_size if batch_size else q_shape[0]
        row_axis_len_k = None if k_shape is None else k_shape[0]
    else:
        batch_size, mean_seqlen_q = q_shape[0], q_shape[1]
        row_axis_len_k = None if k_shape is None else k_shape[1]
    hint_q = usable_max_seqlen(max_seqlen_q, q_shape[0] if varlen else q_shape[1])
    hint_k = usable_max_seqlen(max_seqlen_k, row_axis_len_k or 0)
    return ShapeFacts(
        varlen=varlen,
        qhead_per_kvhead=qhead_per_kvhead,
        batch_size=batch_size,
        mean_seqlen_q=mean_seqlen_q,
        max_seqlen_q_hint=hint_q,
        max_seqlen_k_hint=hint_k,
        max_seqlen_q=hint_q if varlen else q_shape[1],
        max_seqlen_k=hint_k if varlen else row_axis_len_k,
        num_head_kv=None if k_shape is None else k_shape[-2],
        num_sm=num_sm,
    )


def serves(info, facts):
    if info.inter_sequence or info.inputs != "any" and info.inputs != ("varlen" if facts.varlen else "batched"):
        return False
    return not (info.grouped and facts.qhead_per_kvhead == 1)


def candidates(variants, facts):
    return tuple((variant, info) for variant, info in variants if variant and serves(info, facts))


def fill(rows, tile):
    return rows / (-(-rows // tile) * tile) if rows else 1.0


def pack_divides(info, facts):
    return bool(facts.qhead_per_kvhead) and not (info.box_columns and (info.q_box_rows or info.block_m) % facts.qhead_per_kvhead)


def fewer_waves(info, facts, rows):
    if rows < PACK_WAVES_MIN_ROWS or facts.varlen or info.scheduling_mode == "single_tile":
        return False
    if not (facts.batch_size and facts.num_head_kv and facts.num_sm):
        return False
    unpacked = facts.batch_size * facts.num_head_kv * facts.qhead_per_kvhead * -(-rows // info.cluster_block_m)
    packed = facts.batch_size * facts.num_head_kv * -(-rows * facts.qhead_per_kvhead // info.cluster_block_m)
    return -(-packed // facts.num_sm) < -(-unpacked // facts.num_sm)


def pack_pays(info, facts, arch):
    if not pack_divides(info, facts):
        return False
    rows = min(facts.mean_seqlen_q, facts.max_seqlen_q_hint or facts.mean_seqlen_q)
    if PACK_BY_WAVES[arch] and fewer_waves(info, facts, rows):
        return True
    if rows > PACK_MAX_TILES[arch] * info.cluster_block_m:
        return False
    return fill(rows, info.cluster_block_m) < PACK_EFFICIENCY * fill(rows * facts.qhead_per_kvhead, info.cluster_block_m)


def within(seqlen, low, high):
    return not high or (seqlen is not None and low <= seqlen <= high)


def bounded(window):
    return window[0] >= 0 or window[1] >= 0


def local_two_sided(window):
    return window[0] >= 0 and window[1] >= 0


def mask(window):
    left, right = window
    if left >= 0:
        return MASKS.index("local_two_sided" if right >= 0 else "local_left")
    if right < 0:
        return MASKS.index("dense")
    return MASKS.index("causal" if right == 0 else "local_right")


def accepts(info, facts, window=UNBOUNDED):
    if info.two_sided_window and not local_two_sided(window):
        return False
    if not (info.masks >> mask(window)) & 1:
        return False
    if not (local_two_sided(window) and window[0] + window[1] < info.max_window_size):
        packs_bounded = info.pack_gqa and info.max_seqlen_q and bounded(window)
        query_limit = 2 * info.max_seqlen_q if packs_bounded else info.max_seqlen_q
        if not within(facts.max_seqlen_q, info.min_seqlen_q, query_limit):
            return False
        if not within(facts.max_seqlen_k, 0, info.max_seqlen_k):
            return False
    packs_window = bounded(window) and info.arch in PACK_BOUNDED_ARCHS and pack_divides(info, facts)
    packs_within_limit = bool(info.max_seqlen_q) and pack_divides(info, facts)
    if info.pack_gqa and not packs_window and not packs_within_limit and not pack_pays(info, facts, info.arch):
        return False
    if info.scheduler_metadata and (facts.batch_size or 0) < SCHEDULER_METADATA_MIN_BATCH:
        return False
    return True


def declared_variants(device_id, direction, head_dim):
    _, entries = runtime.manifest()
    partial, _ = runtime.declared()
    for variant in runtime.variant_names(device_id, direction, head_dim):
        name = runtime.instance_name(device_id, direction, head_dim, variant)
        if name in entries:
            yield variant, entries[name]
        elif not partial:
            raise RuntimeError(f"{name} is declared but not built")


def unrestricted(device_id, direction, head_dim):
    return next((info for variant, info in declared_variants(device_id, direction, head_dim) if not variant), None)


def choose(device_id, direction, head_dim, facts, rule=accepts):
    for variant, info in candidates(tuple(declared_variants(device_id, direction, head_dim)), facts):
        if rule(info, facts):
            return variant
    return ""


def reachable(device_id, direction, head_dim, facts):
    variants = tuple(declared_variants(device_id, direction, head_dim))
    regular = tuple(info for variant, info in variants if not variant)
    return regular + tuple(info for _, info in candidates(variants, facts))


@functools.cache
def debug_enabled():
    setting = os.environ.get("ZIGFA_DEBUG", "")
    if setting and setting not in DEBUG_LEVELS:
        raise ValueError(f"ZIGFA_DEBUG must be unset or one of {list(DEBUG_LEVELS)}, got {setting!r}")
    if setting:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return bool(setting)


def log_choice(direction, device_id, head_dim, variant, **request):
    if not debug_enabled():
        return
    info = runtime.manifest()[1][runtime.instance_name(device_id, direction, head_dim, variant)]
    described = " ".join(f"{name}={value}" for name, value in request.items())
    logger.info(
        "%s device=%d %s -> %s threads=%d block_m=%d block_n=%d",
        direction, device_id, described, info.name, info.threads, info.block_m, info.block_n,
    )
