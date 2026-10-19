import contextlib
import functools
import gzip
import heapq
import io
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cuda.bindings import driver

CHUNK_RECORDS = 32
TICK_BITS = 24
TICK_MASK = (1 << TICK_BITS) - 1
HEADER_OFFSET = 64
LAYOUT_VERSION = 1
FORMAT = "zigfa.zones"
CONTROL = np.dtype([("next_chunk", "<u4"), ("capacity", "<u4"), ("dropped", "<u4"), ("format", "<u4")])
HEADER = np.dtype([("clock", "<u8"), ("globaltimer", "<u8"), ("cta", "<u4"), ("origin", "<u4")])
LEVELS = ("sm", "cluster", "cta", "wg", "warp")

active = None


def records_offset(capacity):
    return -(-(HEADER_OFFSET + HEADER.itemsize * (capacity + 1)) // 128) * 128


def buffer_bytes(capacity):
    return records_offset(capacity) + CHUNK_RECORDS * 4 * (capacity + 1)


def checked(name, result):
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


def copy_to_host(address, count, dtype):
    host = np.empty(count, dtype)
    if host.nbytes:
        checked("cuMemcpyDtoH", driver.cuMemcpyDtoH(host.ctypes.data, address, host.nbytes))
    return host


class DeviceBuffer:
    def __init__(self, capacity, stream_handle=None):
        self.capacity = capacity
        self.address = int(checked("cuMemAlloc", driver.cuMemAlloc(buffer_bytes(capacity))))
        if stream_handle is None:
            checked("cuMemsetD8", driver.cuMemsetD8(self.address, 0, records_offset(capacity)))
            checked("cuMemsetD32", driver.cuMemsetD32(self.address + 4, capacity, 1))
            checked("cuMemsetD32", driver.cuMemsetD32(self.address + 12, LAYOUT_VERSION, 1))
            checked("cuCtxSynchronize", driver.cuCtxSynchronize())
        else:
            checked("cuMemsetD8Async", driver.cuMemsetD8Async(self.address, 0, records_offset(capacity), stream_handle))
            checked("cuMemsetD32Async", driver.cuMemsetD32Async(self.address + 4, capacity, 1, stream_handle))
            checked("cuMemsetD32Async", driver.cuMemsetD32Async(self.address + 12, LAYOUT_VERSION, 1, stream_handle))

    def read(self):
        control = copy_to_host(self.address, 1, CONTROL)[0]
        self.claimed = int(control["next_chunk"])
        used = min(self.claimed, self.capacity)
        headers = copy_to_host(self.address + HEADER_OFFSET, used, HEADER)
        records = copy_to_host(self.address + records_offset(self.capacity), used * CHUNK_RECORDS, "<u4")
        return headers, records.reshape(used, CHUNK_RECORDS), int(control["dropped"])

    def free(self):
        if self.address:
            checked("cuMemFree", driver.cuMemFree(self.address))
            self.address = 0


@functools.cache
def discard_buffer(context):
    return DeviceBuffer(0)


@functools.cache
def clock_ghz(context):
    device = checked("cuCtxGetDevice", driver.cuCtxGetDevice())
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE
    return checked("cuDeviceGetAttribute", driver.cuDeviceGetAttribute(attribute, device)) / 1e6


@functools.cache
def zone_names(kernel):
    library = checked("cuKernelGetLibrary", driver.cuKernelGetLibrary(kernel.handle))
    status, address, size = driver.cuLibraryGetGlobal(library, b"profile_zones")
    if status != driver.CUresult.CUDA_SUCCESS:
        return ()
    return tuple(copy_to_host(address, size, np.uint8).tobytes().decode().split("\n")[:-1])


def launch_argument(kernel, label, grid, threads, cluster_dims, stream_handle):
    """Returns the profile buffer address that a kernel built with -Dprofile takes as its last argument."""
    context = int(checked("cuCtxGetCurrent", driver.cuCtxGetCurrent()))
    if active is None:
        return discard_buffer(context).address
    launch = Launch(label, tuple(grid), tuple(cluster_dims or (1, 1, 1)), threads, zone_names(kernel))
    launch.ghz = clock_ghz(context)
    launch.buffer = DeviceBuffer(active.chunks, stream_handle)
    active.launches.append(launch)
    return launch.buffer.address


@contextlib.contextmanager
def capture(chunks=1 << 18):
    """Records the zones of each launch of a kernel built with -Dprofile. Each launch gets a buffer of
    `chunks` chunks of 32 records."""
    global active
    if active is not None:
        raise RuntimeError("a profile capture is already active")
    session = Capture(chunks)
    active = session
    try:
        yield session
        checked("cuCtxSynchronize", driver.cuCtxSynchronize())
        for launch in session.launches:
            launch.decode(*launch.buffer.read())
            if launch.dropped:
                warnings.warn(
                    f"{launch.kernel} dropped {launch.dropped} records; "
                    f"capture(chunks={launch.buffer.claimed}) holds them all",
                    stacklevel=3,
                )
    finally:
        active = None
        for launch in session.launches:
            if launch.buffer is not None:
                launch.buffer.free()
                launch.buffer = None


def unwrap(headers, records):
    """Returns the SM clock of each record, walking back from the 64-bit clock that closed its chunk."""
    origin = headers["origin"].astype(np.int64)
    count = (origin >> 13) & 0x3F
    shift = (origin >> 19) & 0xF
    anchor = headers["clock"].astype(np.int64) >> shift
    ticks = (records & TICK_MASK).astype(np.int64)
    columns = np.arange(CHUNK_RECORDS)
    last = np.maximum(count - 1, 0)
    to_close = (anchor & TICK_MASK) - ticks[np.arange(len(ticks)), last]
    steps = (np.diff(ticks, axis=1) & TICK_MASK) * (columns[None, 1:] < count[:, None])
    behind = np.flip(np.cumsum(np.flip(steps, axis=1), axis=1), axis=1)
    behind = np.concatenate([behind, np.zeros((len(ticks), 1), np.int64)], axis=1)
    clock = (anchor - (to_close & TICK_MASK))[:, None] - behind
    return clock << shift[:, None], columns[None, :] < count[:, None]


def piecewise(points, knots, values):
    rate = np.diff(values) / np.diff(knots)
    out = np.interp(points, knots, values)
    out = np.where(points < knots[0], values[0] + (points - knots[0]) * rate[0], out)
    return np.where(points > knots[-1], values[-1] + (points - knots[-1]) * rate[-1], out)


def local_rate(points, knots, values):
    rate = np.diff(values) / np.diff(knots)
    return rate[np.clip(np.searchsorted(knots, points, side="right") - 1, 0, len(rate) - 1)]


def clock_map(sm, clock, globaltimer, ghz, knot_ns=5000, per_knot=16, max_knots=1024, rounds=6):
    """Returns a function that maps SM clocks to globaltimer nanoseconds.

    All SMs share one clock rate, which the GPU changes within a launch, so the map is one linear
    spline in clock ticks, with knots at least `knot_ns` and `per_knot` headers apart, plus a clock
    offset per SM. The fit falls back to `ghz` when the headers span no globaltimer step.
    """
    sms, group = np.unique(sm, return_inverse=True)
    count = np.bincount(group)
    clock = clock.astype(np.int64)
    clock_base = np.full(len(sms), np.iinfo(np.int64).max)
    np.minimum.at(clock_base, group, clock)
    time_base = int(globaltimer.min())
    x = (clock - clock_base[group]).astype(np.float64)
    y = (globaltimer.astype(np.int64) - time_base).astype(np.float64)
    dx = x - (np.bincount(group, x) / count)[group]
    dy = y - (np.bincount(group, y) / count)[group]
    rate = dx @ dy / (dx @ dx) if dx @ dx > 0 else 0.0
    rounds = rounds if rate > 0 else 0
    rate = rate if rate > 0 else 1 / ghz
    offset = np.bincount(group, x - y / rate) / count
    knots, values = np.array([0.0, 1.0]), np.array([0.0, rate])
    for _ in range(rounds):
        tau = x - offset[group]
        low, span = tau.min(), np.ptp(tau)
        width = max(knot_ns / rate, span * per_knot / len(x), span / (max_knots - 1))
        knots = low + np.arange(int(span // width) + 2) * width
        values = spline_fit(tau, y, knots)
        steps = np.diff(values) / width
        median = np.median(steps) if np.median(steps) > 0 else rate
        steps = np.clip(steps, median / 4, median * 4)
        values = values[0] + np.r_[0.0, np.cumsum(steps * width)]
        gradient = local_rate(tau, knots, values)
        residual = y - piecewise(tau, knots, values)
        offset -= np.bincount(group, residual * gradient) / np.bincount(group, gradient * gradient)
    values = values + np.mean(y - piecewise(x - offset[group], knots, values))

    def ns(sm, clock):
        index = np.searchsorted(sms, sm)
        ticks = (np.asarray(clock, np.int64) - clock_base[index]).astype(np.float64) - offset[index]
        return time_base + np.rint(piecewise(ticks, knots, values)).astype(np.int64)

    return ns


def spline_fit(points, values, knots, smoothing=0.1):
    """Returns the knot values of the least squares linear spline through (points, values), with a
    penalty on the second differences that bridges knots without points."""
    size = len(knots)
    position = (points - knots[0]) / (knots[1] - knots[0])
    left = np.minimum(position.astype(np.int64), size - 2)
    right_weight = position - left
    left_weight = 1 - right_weight
    index = np.arange(size)
    normal = np.zeros((size, size))
    normal[index, index] = np.bincount(left, left_weight**2, size) + np.bincount(left + 1, right_weight**2, size)
    normal[index[:-1], index[1:]] = normal[index[1:], index[:-1]] = np.bincount(left, left_weight * right_weight, size)[:-1]
    second = np.diff(np.eye(size), 2, axis=0)
    normal += smoothing * len(points) / size * second.T @ second + 1e-9 * np.eye(size)
    rhs = np.bincount(left, left_weight * values, size) + np.bincount(left + 1, right_weight * values, size)
    return np.linalg.solve(normal, rhs)


def runs(keys):
    """Returns the index where the run of equal keys around each element starts."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return np.repeat(starts, np.diff(np.r_[starts, len(keys)]))


def pair(track, zone, ends):
    """Pairs each end with the latest open begin of the same zone on its track, given events in
    track and time order. Returns the begin and end indices, a display depth per pair, and the
    number of unpaired events."""
    group = track * 128 + zone
    order = np.argsort(group, kind="stable")
    key, closing = group[order], ends[order]
    depth = np.cumsum(np.where(closing, -1, 1))
    depth -= np.r_[0, depth][runs(key)]
    level = np.where(closing, depth, depth - 1)
    ranked = np.lexsort((np.arange(len(key)), level, key))
    first, second = ranked[:-1], ranked[1:]
    matched = (
        ~closing[first] & closing[second] & (key[first] == key[second]) & (level[first] == level[second])
        & (level[first] >= 0)
    )
    opened, closed = order[first[matched]], order[second[matched]]
    by_begin = np.argsort(opened)
    begin, end = opened[by_begin], closed[by_begin]
    return begin, end, display_depth(track, begin, end), len(track) - 2 * len(begin)


def display_depth(track, begin, end):
    """Returns the depth of each zone: the number of zones open on its track when it begins, or the
    lowest free depth on tracks where zones overlap without nesting."""
    step = np.zeros(len(track), np.int64)
    step[begin], step[end] = 1, -1
    open_count = np.cumsum(step)
    open_count -= np.r_[0, open_count][runs(track)]
    depth = open_count[begin] - 1
    by_depth = np.lexsort((begin, depth, track[begin]))
    group = track[begin][by_depth] * (len(track) + 1) + depth[by_depth]
    reach = np.maximum.accumulate(group * (len(track) + 1) + end[by_depth]) - group * (len(track) + 1)
    clash = np.r_[False, (group[1:] == group[:-1]) & (begin[by_depth][1:] < reach[:-1])]
    for clashing in np.unique(track[begin][by_depth][clash]):
        pairs = np.flatnonzero(track[begin] == clashing)
        events = sorted([(begin[k], 1, k) for k in pairs] + [(end[k], 0, k) for k in pairs])
        free, unused = [], 0
        for _, opens, k in events:
            if not opens:
                heapq.heappush(free, depth[k])
            elif free:
                depth[k] = heapq.heappop(free)
            else:
                depth[k], unused = unused, unused + 1
    return depth


@dataclass
class Launch:
    kernel: str
    grid: tuple
    cluster: tuple
    threads: int
    zones: tuple
    ghz: float = 1.0
    buffer: object = None
    dropped: int = 0
    unmatched: int = 0
    tracks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    zone_track: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    zone_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    zone_start: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    zone_end: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    zone_depth: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    close_track: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    close_time: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def decode(self, headers, records, dropped):
        self.dropped = dropped
        origin = headers["origin"].astype(np.int64)
        kept = ((origin >> 13) & 0x3F) > 0
        if not kept.any():
            return
        headers, records, origin = headers[kept], records[kept], origin[kept]
        sm = origin & 0xFF
        warp = (origin >> 8) & 0x1F
        cta = headers["cta"].astype(np.int64)
        keys, track = np.unique(np.stack([cta, warp]), axis=1, return_inverse=True)
        track = track.reshape(-1)
        track_sm = np.zeros(keys.shape[1], np.int64)
        track_sm[track] = sm
        self.tracks = np.stack([track_sm, keys[0], keys[1]], axis=1)
        to_ns = clock_map(sm, headers["clock"], headers["globaltimer"], self.ghz)
        clock, valid = unwrap(headers, records)
        ns = to_ns(np.broadcast_to(sm[:, None], clock.shape), clock)
        event_track = np.broadcast_to(track[:, None], clock.shape)[valid]
        event_time = ns[valid]
        words = records[valid].astype(np.int64)
        order = np.lexsort((event_time, event_track))
        event_track, event_time, words = event_track[order], event_time[order], words[order]
        begin, end, depth, self.unmatched = pair(event_track, words >> 25, (words >> 24) & 1 == 1)
        self.zone_track = event_track[begin]
        self.zone_id = words[begin] >> 25
        self.zone_start = event_time[begin]
        self.zone_end = event_time[end]
        self.zone_depth = depth
        self.close_track = track
        self.close_time = to_ns(sm, headers["clock"])

    def zone_name(self, index):
        return self.zones[index] if index < len(self.zones) else f"zone {index}"

    def clusters(self):
        gx, gy, _ = self.grid
        cx, cy, cz = self.cluster
        cta = self.tracks[:, 1]
        x, y, z = cta % gx, cta // gx % gy, cta // (gx * gy)
        return x // cx + gx // cx * (y // cy + gy // cy * (z // cz))


@dataclass
class Capture:
    chunks: int
    launches: list = field(default_factory=list)

    def t0(self):
        starts = [launch.zone_start.min() for launch in self.launches if len(launch.zone_start)]
        closes = [launch.close_time.min() for launch in self.launches if len(launch.close_time)]
        return int(min(starts + closes, default=0))

    def jsonl(self):
        t0 = self.t0()
        lines = [f'{{"format":"{FORMAT}","version":{LAYOUT_VERSION},"t0":{t0}}}']
        first_track = []
        tracks = 0
        for index, launch in enumerate(self.launches):
            fields = ["L", index, launch.kernel, list(launch.grid), list(launch.cluster), launch.threads]
            lines.append(json.dumps([*fields, list(launch.zones), launch.dropped], separators=(",", ":")))
            first_track.append(tracks)
            tracks += len(launch.tracks)
        for index, launch in enumerate(self.launches):
            for offset, (sm, cta, warp) in enumerate(launch.tracks.tolist()):
                lines.append(f'["T",{first_track[index] + offset},{index},{sm},{cta},{warp}]')
        for index, launch in enumerate(self.launches):
            order = np.lexsort((launch.zone_start, launch.zone_track))
            rows = np.stack([
                launch.zone_track[order] + first_track[index],
                launch.zone_id[order],
                launch.zone_start[order] - t0,
                launch.zone_end[order] - launch.zone_start[order],
                launch.zone_depth[order],
            ], axis=1)
            lines.extend(f'["Z",{t},{z},{s},{d},{depth}]' for t, z, s, d, depth in rows.tolist())
            order = np.lexsort((launch.close_time, launch.close_track))
            rows = np.stack([launch.close_track[order] + first_track[index], launch.close_time[order] - t0], axis=1)
            lines.extend(f'["C",{t},{time}]' for t, time in rows.tolist())
        return "\n".join(lines) + "\n"

    def save(self, path):
        """Writes the capture as gzip JSON lines, which reports/profile_viewer.html opens."""
        path = Path(path)
        if not path.name.endswith(".jsonl.gz"):
            raise ValueError(f"save writes .jsonl.gz files, not {path.name}")
        path.write_bytes(gzip.compress(self.jsonl().encode(), mtime=0))

    def summary(self, by="sm"):
        """Returns a table of the count, mean and 99th percentile duration of each zone per `by` level."""
        if by is not None and by not in LEVELS:
            raise ValueError(f"by must be None or one of {LEVELS}, got {by!r}")
        rows = []
        for index, launch in enumerate(self.launches):
            sm, cta, warp = (launch.tracks[launch.zone_track, column] for column in range(3))
            level = {
                None: np.zeros_like(sm),
                "sm": sm,
                "cluster": launch.clusters()[launch.zone_track] if len(launch.tracks) else sm,
                "cta": cta,
                "wg": cta * 8 + warp // 4,
                "warp": cta * 32 + warp,
            }[by]
            duration = (launch.zone_end - launch.zone_start) / 1000
            for key in np.unique(level):
                for zone in np.unique(launch.zone_id):
                    selected = duration[(level == key) & (launch.zone_id == zone)]
                    if len(selected):
                        rows.append((
                            index, level_label(by, int(key)), launch.zone_name(int(zone)),
                            len(selected), selected.mean(), np.percentile(selected, 99),
                        ))
        out = io.StringIO()
        out.write(f"{'launch':>6}  {by or 'all':<12}  {'zone':<12}  {'count':>8}  {'mean µs':>9}  {'p99 µs':>9}\n")
        for row in rows:
            out.write(f"{row[0]:>6}  {row[1]:<12}  {row[2]:<12}  {row[3]:>8}  {row[4]:>9.3f}  {row[5]:>9.3f}\n")
        return out.getvalue()


def level_label(by, key):
    if by in (None, "sm", "cluster", "cta"):
        return "all" if by is None else f"{by} {key}"
    if by == "wg":
        return f"cta {key // 8} wg {key % 8}"
    return f"cta {key // 32} w {key % 32}"

