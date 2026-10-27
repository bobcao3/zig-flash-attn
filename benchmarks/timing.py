"""Interleaved device timing for the zigfa benchmarks.

`measure` times the implementations of one case in rounds. A round runs one block of each
implementation, a CUDA graph of calls that takes about `BLOCK_SECONDS`, and the next round starts
with the next implementation, so that clock and temperature changes affect all of them alike. The
first measurement of a process first runs the blocks for `CLOCK_WARMUP_SECONDS`, which brings the
GPU to the clock that its power limit sustains; later cases call each function once before timing.

CUDA events time a block that is a CUDA graph. A block that CUDA graphs cannot capture, such as an
autograd backward, runs its calls one after another, and the host may then launch kernels slower
than the GPU runs them. The torch profiler times such a block: the block runs alone inside a
`record_function` range, and its time is the time during which the GPU ran at least one kernel,
copy or memset that started inside the range.
"""

import bisect
import contextlib
import dataclasses
import functools
import os
import statistics
import sys
import threading
import time

import torch
from cuda.bindings import nvml
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

CLOCK_WARMUP_SECONDS = 1.0
CASE_SECONDS = 0.3
BLOCK_SECONDS = 2e-3
MAX_BLOCK_CALLS = 1000
MIN_ROUNDS = 3
MAX_CASE_SECONDS = 10.0
SETUP_SECONDS = 300.0
HOST_SECONDS = 2.0
SLOW_SETUP_SECONDS = 0.2
TIMEOUT_ESTIMATE_MULTIPLE = 200
TELEMETRY_PERIOD_SECONDS = 0.02
RANGE_PREFIX = "timing:"
clock_warm = False


@dataclasses.dataclass(frozen=True)
class Device:
    name: str
    peak_flop_rate: float
    byte_rate: float
    forward_flop_rate: float
    backward_flop_rate: float


# The peaks are the dense BF16 tensor rates and the memory bandwidths of the datasheets. The zigfa rates
# estimate the time of a call: medians of the MHA points of reports/perf_report.html for B200, H100, H200 and
# GB10, medians of the compute-bound cases of benchmark_attn.py for RTX PRO 6000, RTX 6000 Ada, L4 and A10,
# and for the others the rates of a GPU that runs the same kernels, scaled by the peaks. A name matches the
# first entry that it contains.
DEVICES = (
    Device("B300", 2250e12, 8.0e12, 900e12, 706e12),
    Device("B200", 2250e12, 8.0e12, 900e12, 706e12),
    Device("H200", 989e12, 4.8e12, 509e12, 398e12),
    Device("H100", 989e12, 3.35e12, 493e12, 377e12),
    Device("GB10", 125e12, 0.273e12, 85e12, 45e12),
    Device("RTX PRO 6000", 504e12, 1.79e12, 341e12, 212e12),
    Device("RTX 6000 Ada", 364e12, 0.96e12, 145e12, 130e12),
    Device("A100-SXM4-80GB", 312e12, 2.04e12, 190e12, 140e12),
    Device("A100 80GB", 312e12, 1.94e12, 190e12, 140e12),
    Device("A100", 312e12, 1.56e12, 190e12, 140e12),
    Device("L40S", 362e12, 0.864e12, 170e12, 120e12),
    Device("L4", 121e12, 0.3e12, 60e12, 48e12),
    Device("A10", 125e12, 0.6e12, 64e12, 51e12),
)
UNKNOWN_DEVICE = Device("unknown", float("nan"), float("nan"), 100e12, 70e12)


@dataclasses.dataclass(frozen=True)
class Timing:
    seconds: float
    ratio: float
    graph: bool
    setup_seconds: float


@dataclasses.dataclass(frozen=True)
class Measurement:
    timings: dict
    failures: dict
    clock_mhz: float | None
    power_watts: float | None


@functools.cache
def device():
    name = torch.cuda.get_device_name()
    return next((entry for entry in DEVICES if entry.name in name), UNKNOWN_DEVICE)


def roofline_seconds(flop_count, byte_count):
    entry = device()
    return max(flop_count / entry.peak_flop_rate, byte_count / entry.byte_rate)


def estimated_seconds(flop_count, byte_count, backward):
    """Returns the time that one zigfa call of `flop_count` FLOPs and `byte_count` compulsory bytes should
    take on this GPU."""
    entry = device()
    flop_rate = entry.backward_flop_rate if backward else entry.forward_flop_rate
    byte_seconds = byte_count / entry.byte_rate if entry.byte_rate == entry.byte_rate else 0.0
    return max(flop_count / flop_rate, byte_seconds)


class Telemetry:
    """Samples the SM clock and the board power of the current GPU on a background thread."""

    def __enter__(self):
        self.samples = []
        self.done = threading.Event()
        try:
            nvml.init_v2()
            self.handle = nvml.device_get_handle_by_index_v2(torch.cuda.current_device())
        except nvml.NvmlError:
            return self
        threading.Thread(target=self.sample, daemon=True).start()
        return self

    def sample(self):
        while not self.done.wait(TELEMETRY_PERIOD_SECONDS):
            clock = nvml.device_get_clock_info(self.handle, nvml.ClockType.CLOCK_SM)
            self.samples.append((clock, nvml.device_get_power_usage(self.handle) * 1e-3))

    def __exit__(self, *exception):
        self.done.set()

    def means(self):
        if not self.samples:
            return None, None
        return tuple(sum(values) / len(values) for values in zip(*self.samples))


class Watchdog:
    """Ends the process when a case runs past `seconds`, so that a hung kernel cannot hold the GPU."""

    def __init__(self, name, seconds):
        self.timer = threading.Timer(seconds, self.expire, (name, seconds))
        self.timer.daemon = True

    def expire(self, name, seconds):
        print(f"\n{name} ran past its limit of {seconds:.1f} s", file=sys.stderr, flush=True)
        os._exit(124)

    def __enter__(self):
        self.timer.start()

    def __exit__(self, *exception):
        self.timer.cancel()


def block_of(function, calls):
    """Returns a function that makes `calls` calls of `function`, from one CUDA graph if `function` allows
    capture, and whether it is a graph; or the exception that the first call of `function` raised. The first
    call runs on the capture stream, so that a function that sets up autograd at its first call records its
    backward on that stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        try:
            function()
        except Exception as error:
            torch.cuda.synchronize()
            return error
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                function()
        return graph.replay, True
    except Exception:
        torch.cuda.synchronize()

    def eager():
        for _ in range(calls):
            function()

    return eager, False


def busy_seconds(events, labels):
    """Returns for each label the GPU time of each of its `record_function` ranges in `events`, in the order of the
    ranges: the length of the union of the intervals of the kernels, copies and memsets that start inside it."""
    ranges = sorted(
        (event.time_range.start, event.time_range.end, event.name.removeprefix(RANGE_PREFIX))
        for event in events
        if event.device_type == DeviceType.CPU and event.name.startswith(RANGE_PREFIX)
    )
    starts = [start for start, _, _ in ranges]
    spans = [[] for _ in ranges]
    for event in events:
        if event.device_type != DeviceType.CUDA or getattr(event, "is_user_annotation", False):
            continue
        index = bisect.bisect_right(starts, event.time_range.start) - 1
        if index >= 0 and event.time_range.start <= ranges[index][1]:
            spans[index].append((event.time_range.start, event.time_range.end))
    busy = {label: [] for label in labels}
    for (_, _, label), intervals in zip(ranges, spans):
        total, covered = 0.0, float("-inf")
        for start, end in sorted(intervals):
            total += max(0.0, end - max(start, covered))
            covered = max(covered, end)
        busy[label].append(total * 1e-6)
    return busy


def measure(name, functions, estimate, case_seconds=CASE_SECONDS):
    """Returns the median device time of one call of each of `functions`, a dict of named functions whose
    first entry is the implementation under test, and the median ratio of the time of each to the time of
    the first, over the rounds. A function that raises `RuntimeError` on its first call has no timing.

    `estimate` is the expected time of one call, which sets the calls in a block. The first calls, which may
    compile kernels, may take `SETUP_SECONDS`. The measurement may take its warmup, twice its budget,
    `HOST_SECONDS` and `TIMEOUT_ESTIMATE_MULTIPLE` estimates for each function, but at most
    `MAX_CASE_SECONDS`."""
    global clock_warm
    warmup = 0 if clock_warm else CLOCK_WARMUP_SECONDS
    allowance = TIMEOUT_ESTIMATE_MULTIPLE * estimate * len(functions)
    limit = min(MAX_CASE_SECONDS, warmup + 2 * case_seconds + HOST_SECONDS + allowance)
    calls = max(1, min(MAX_BLOCK_CALLS, round(BLOCK_SECONDS / estimate)))
    blocks, setups, failures = {}, {}, {}
    with Watchdog(f"{name} setup", SETUP_SECONDS):
        for label, function in functions.items():
            setup_started = time.perf_counter()
            block = block_of(function, calls)
            if isinstance(block, Exception):
                failures[label] = f"{type(block).__name__}: {str(block).splitlines()[0] if str(block) else ''}"
                continue
            blocks[label] = block
            setups[label] = time.perf_counter() - setup_started
    labels = list(blocks)
    if not labels:
        return Measurement({}, failures, None, None)
    with Watchdog(name, limit):
        deadline = time.perf_counter() + warmup
        while time.perf_counter() < deadline:
            for label in labels:
                blocks[label][0]()
            torch.cuda.synchronize()
    clock_warm = True
    eager = [label for label in labels if not blocks[label][1]]
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if eager else contextlib.nullcontext()
    samples = {label: [] for label in labels}
    measured = 0.0
    with profiler, Watchdog(name, limit), Telemetry() as telemetry:
        while measured < case_seconds or len(samples[labels[0]]) < MIN_ROUNDS:
            shift = len(samples[labels[0]]) % len(labels)
            events = {}
            for label in labels[shift:] + labels[:shift]:
                if label in eager:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with record_function(RANGE_PREFIX + label):
                        blocks[label][0]()
                        torch.cuda.synchronize()
                    measured += time.perf_counter() - started
                    samples[label].append(None)
                    continue
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                blocks[label][0]()
                end.record()
                events[label] = (start, end)
            torch.cuda.synchronize()
            for label, (start, end) in events.items():
                seconds = start.elapsed_time(end) * 1e-3
                measured += seconds
                samples[label].append(seconds / calls)
    if eager:
        for label, values in busy_seconds(profiler.events(), eager).items():
            assert len(values) == len(samples[label])
            samples[label] = [seconds / calls for seconds in values]
    first = samples[labels[0]] if labels[0] == next(iter(functions)) else None
    timings = {
        label: Timing(
            statistics.median(values),
            statistics.median(own / base for own, base in zip(values, first)) if first else float("nan"),
            blocks[label][1],
            setups[label],
        )
        for label, values in samples.items()
    }
    return Measurement(timings, failures, *telemetry.means())


def header():
    entry = device()
    name = torch.cuda.get_device_name()
    if entry is UNKNOWN_DEVICE:
        return f"{name}: no roofline for this GPU in timing.DEVICES."
    return (
        f"{name}: roofline of {entry.peak_flop_rate * 1e-12:.0f} TF/s dense BF16 and "
        f"{entry.byte_rate * 1e-12:.2f} TB/s. Each case measures about {CASE_SECONDS * 1e3:.0f} ms of calls "
        f"after {CLOCK_WARMUP_SECONDS:.0f} s of warmup at the start."
    )


def summary(measurement, labels, flop_count, byte_count, display=None):
    """Formats the time of one call of each implementation, its throughput in the unit of the bound that the
    roofline predicts, its percent of the roofline, and for the other implementations the speed of the first
    one as a percent of theirs. `display` renames labels in the text."""
    display = display or {}
    entry = device()
    memory_bound = byte_count / entry.byte_rate > flop_count / entry.peak_flop_rate
    bound_seconds = roofline_seconds(flop_count, byte_count)
    parts = []
    for label in labels:
        result = measurement.timings.get(label)
        if result is None:
            parts.append(f"{display.get(label, label)} n/a")
            continue
        if memory_bound:
            rate = f"{byte_count / result.seconds * 1e-12:5.2f} TB/s"
        else:
            rate = f"{flop_count / result.seconds * 1e-12:6.1f} TF/s"
        text = f"{display.get(label, label)} {result.seconds * 1e3:8.3f} ms {rate}"
        if bound_seconds == bound_seconds:
            text += f" {100 * bound_seconds / result.seconds:3.0f}%"
        if label != labels[0] and result.ratio == result.ratio:
            text += f" ({labels[0]} {100 * result.ratio:3.0f}%)"
        parts.append(text + ("" if result.graph else " eager"))
    slow = [
        f"{label} {result.setup_seconds:.1f} s"
        for label, result in measurement.timings.items()
        if result.setup_seconds > SLOW_SETUP_SECONDS
    ]
    if slow:
        parts.append("setup " + ", ".join(slow))
    if measurement.clock_mhz is not None:
        parts.append(f"@{measurement.clock_mhz:4.0f} MHz {measurement.power_watts:3.0f} W")
    return "  ".join(parts)


def record(measurement, flop_count, byte_count, backends=None):
    """Returns the fields of a JSON record of `measurement`: the work of one call, the GPU telemetry, the time of
    one call of each implementation with the backend that ran it, and why the others did not run."""
    backends = backends or {}
    return {
        "flops": flop_count,
        "bytes": byte_count,
        "clock_mhz": measurement.clock_mhz,
        "power_watts": measurement.power_watts,
        "results": {
            label: {"seconds": result.seconds, "graph": result.graph, "backend": backends.get(label)}
            for label, result in measurement.timings.items()
        },
        "failures": measurement.failures,
    }


def totals(case_seconds, seconds):
    """Formats the number of cases, the total time, the time of the first case, which includes the warmup,
    and the median and longest time of the other cases."""
    text = f"{len(case_seconds)} cases in {seconds:.1f} s"
    if case_seconds:
        text += f"; the first took {case_seconds[0]:.2f} s with the warmup"
    if len(case_seconds) > 1:
        rest = case_seconds[1:]
        text += f", the others {statistics.median(rest):.2f} s (median) and at most {max(rest):.2f} s"
    return text
