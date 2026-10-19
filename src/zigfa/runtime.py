import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cuda.bindings import driver
from cuda.core import Device, LaunchConfig, ObjectCode, Stream, launch

ARCHS = {8: "sm80", 9: "sm90", 10: "sm100", 12: "sm120"}
INTER_SEQUENCE_ARCHS = {8: "sm80", 12: "sm80"}
INTER_SEQUENCE = "inter_sequence"
SCALAR_TYPES = {"p": np.uint64, "u": np.uint32, "i": np.int32, "f": np.float32}
FLOAT_CODES = {code for code, scalar in SCALAR_TYPES.items() if np.issubdtype(scalar, np.floating)}
TENSOR_MAP_BYTES = driver.CU_TENSOR_MAP_NUM_QWORDS * np.dtype(np.uint64).itemsize


@dataclass(frozen=True)
class KernelInfo:
    name: str
    threads: int
    shared_bytes: int
    block_m: int
    block_n: int
    box_columns: int
    swizzle_bytes: int
    scheduling_mode: str
    grid_per_tile_when_windowed: int
    q_box_rows: int
    dq_accum_box_rows: int
    bulk_reduce: int
    cluster_ctas: int
    dq_block_m: int
    dq_accum_box_columns: int
    pdl: int
    min_qhead_per_split: int
    dq_accum_interleave: int
    max_qhead_splits: int
    qhead_split_bounded_only: int
    qhead_split_by_waves: int
    scheduler_metadata: int
    pack_gqa: int
    head_split: int
    min_seqlen_q: int
    max_seqlen_q: int
    max_seqlen_k: int
    max_window_size: int
    dkv_box_rows: int
    two_sided_window: int
    masks: int
    inputs: str
    grouped: int
    inter_sequence: int
    arch: int

    @property
    def cluster_block_m(self):
        return max(self.cluster_ctas, 1) * self.block_m


def manifest_value(text):
    return int(text) if text.isdigit() else text


@functools.cache
def manifest():
    package = Path(__file__).parent
    for directory in (package / "cubin", package.parents[1] / "zig-out"):
        path = directory / "kernels.manifest"
        if path.exists():
            header, *lines = path.read_text().splitlines()
            marker, *columns = header.split()
            if marker != "#":
                raise RuntimeError(f"{path} has no header line; rebuild the package")
            entries = {}
            for line in lines:
                values = line.split()
                entries[values[0]] = KernelInfo(
                    **dict(zip(columns, (values[0], *map(manifest_value, values[1:])), strict=True))
                )
            return directory, entries
    raise FileNotFoundError("kernels.manifest not found; build the package first")


@functools.cache
def declared():
    lines = (manifest()[0] / "kernels.declared").read_text().split()
    partial = bool(lines) and lines[0] == "partial"
    return partial, tuple(lines[1:] if partial else lines)


def current_device():
    device = Device()
    device.set_current()
    return device


def load(direction, head_dim, kernel_name, variant="", dynamic_shared=True):
    return load_for(current_device().device_id, direction, head_dim, kernel_name, variant, dynamic_shared)


@functools.cache
def num_sm(device_id):
    return Device(device_id).properties.multiprocessor_count


@functools.cache
def head_dims():
    return tuple(sorted({int(info.name.rsplit("_hd", 1)[1]) for info in manifest()[1].values()}))


@functools.cache
def arch_name(device_id, inter_sequence=False):
    major = Device(device_id).compute_capability.major
    arch = os.environ.get("ZIGFA_ARCH") or (INTER_SEQUENCE_ARCHS if inter_sequence else ARCHS).get(major)
    if arch is None:
        raise RuntimeError(f"no kernel arch supports sm_{major}x")
    if arch not in ARCHS.values():
        raise RuntimeError(f"ZIGFA_ARCH must be one of {sorted(ARCHS.values())}, got {arch!r}")
    return arch


def instance_name(device_id, direction, head_dim, variant=""):
    return f"flash_{direction}_{arch_name(device_id, INTER_SEQUENCE in variant)}{variant}_hd{head_dim}"


@functools.cache
def variant_names(device_id, direction, head_dim, inter_sequence=False):
    prefix, suffix = f"flash_{direction}_{arch_name(device_id, inter_sequence)}", f"_hd{head_dim}"
    _, names = declared()
    return tuple(
        name[len(prefix) : -len(suffix)]
        for name in names
        if name.startswith(prefix) and name.endswith(suffix) and name[len(prefix) : -len(suffix)][:1] in ("", "_")
    )


@functools.cache
def load_for(device_id, direction, head_dim, kernel_name, variant="", dynamic_shared=True):
    directory, entries = manifest()
    major = Device(device_id).compute_capability.major
    instance = instance_name(device_id, direction, head_dim, variant)
    if instance not in entries:
        raise RuntimeError(f"{instance} is not built for head_dim={head_dim} on sm_{major}x")
    for cubin_directory in (directory / "cubin", directory):
        path = cubin_directory / f"{instance}.cubin"
        if not path.exists():
            continue
        kernel = ObjectCode.from_cubin(path.read_bytes()).get_kernel(kernel_name)
        if dynamic_shared:
            attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            shared_bytes = entries[instance].shared_bytes
            status, = driver.cuKernelSetAttribute(attribute, shared_bytes, kernel.handle, device_id)
            if status != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuKernelSetAttribute failed: {status}")
        return kernel, entries[instance]
    raise RuntimeError(f"{instance} has no cubin")


@functools.cache
def verify_parameters(kernel, signature):
    sizes = []
    while True:
        status, _, size = driver.cuKernelGetParamInfo(kernel.handle, len(sizes))
        if status != driver.CUresult.CUDA_SUCCESS:
            break
        sizes.append(size)
    expected = [
        TENSOR_MAP_BYTES if code == "m" else np.dtype(SCALAR_TYPES[code]).itemsize for code in signature
    ]
    if sizes != expected:
        raise RuntimeError(f"kernel parameter sizes {sizes} do not match launch signature {signature!r}")


@functools.cache
def stream_from_handle(handle):
    # Each Stream.from_handle builds a new wrapper class, which only the cycle collector frees.
    return Stream.from_handle(handle)


def stream_of(stream):
    if stream is None:
        return current_device().default_stream
    if isinstance(stream, int):
        return stream_from_handle(stream)
    return current_device().create_stream(stream)


def run(kernel, grid, threads, shared_bytes, signature, values, stream, pdl=False, cluster_dims=None, cooperative=False):
    verify_parameters(kernel, signature)
    options = {}
    if cluster_dims is not None:
        options["cluster"] = cluster_dims
        grid = tuple(blocks // ctas for blocks, ctas in zip(grid, cluster_dims, strict=True))
    config = LaunchConfig(
        grid=grid,
        block=(threads, 1, 1),
        shmem_size=shared_bytes,
        programmatic_stream_serialization=bool(pdl),
        is_cooperative=cooperative,
        **options,
    )
    arguments = []
    for code, value in zip(signature, values, strict=True):
        if code in SCALAR_TYPES and (code in FLOAT_CODES) != isinstance(value, (float, np.floating)):
            raise TypeError(f"launch signature {signature!r} gives the value {value!r} the code {code!r}")
        arguments.append(value if code == "m" else SCALAR_TYPES[code](value))
    launch(stream_of(stream), config, kernel, *arguments)
