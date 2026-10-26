"""Writes the JSON records of benchmarks/benchmark_attn.py and benchmarks/benchmark_inter_sequence.py into
perf_report.html as the points of one device, replacing the points that the device had for those shapes.

    python reports/update_perf_report.py sm90 attention.jsonl inter_sequence.jsonl
    python reports/update_perf_report.py pro6000 --name "RTX PRO 6000" --sub "SM120 and SM80 kernels" \\
        --arch sm120 records.jsonl
"""

import argparse
import datetime
import html
import json
import re
from pathlib import Path

REPORT = Path(__file__).with_name("perf_report.html")
SERIES = {
    "zigfa": "zig",
    "zigfa sm80": "zig80",
    "torch SDPA": "ref",
    "FA4": "fa4",
    "FlexAttention": "ref",
    "FA4 FlexAttention": "fa4",
}
BACKEND_NAMES = {"cuDNN": "cuDNN", "flash": "FlashAttention"}
INTER_SEQUENCE_LAYOUTS = ("captions", "image → text", "text → image", "interleaved", "4096-token images")
INTER_SEQUENCE_HEADS = (64, 128, 256, 512)
MASKS = ("dense", "causal")


def cudnn_version(number):
    value = int(number)
    return f"{value // 10000}.{value % 10000 // 100}.{value % 100}"


def reference_name(benchmark, backends):
    if benchmark == "inter_sequence":
        return "FlexAttention"
    found = {backend for values in backends.values() for backend in values if backend}
    names = [name for name in BACKEND_NAMES.values() if name in found]
    return f"torch SDPA ({' or '.join(names)})" if names else "torch SDPA"


def entry(records, keys, points, point_of):
    """Returns the report entry of the records of one shape and direction: TF/s per series, key and point, the
    torch SDPA backend of each point, and the reference name."""
    series, backends = {}, {}
    for record in records:
        key, index = point_of(record)
        for label, result in record["results"].items():
            values = series.setdefault(SERIES[label], {})
            values.setdefault(key, [None] * points)[index] = round(record["flops"] / result["seconds"] * 1e-12, 1)
            if SERIES[label] == "ref" and result.get("backend"):
                backends.setdefault(key, [None] * points)[index] = BACKEND_NAMES[result["backend"]]
    for values in series.values():
        for key in keys:
            values.setdefault(key, [None] * points)
    result = {**series, "refName": reference_name(records[0]["benchmark"], backends)}
    if backends:
        result["refBackends"] = backends
    return result


def update_data(data, device, records):
    seqs = data["seqs"]
    attention_keys = [f"{mask}/{head}" for mask in MASKS for head in data["heads"]]
    inter_keys = [f"mixed/{head}" for head in INTER_SEQUENCE_HEADS]
    groups = {}
    for record in records:
        shape = "interseq" if record["benchmark"] == "inter_sequence" else record["shape"]
        groups.setdefault((shape, record["direction"]), []).append(record)
    for (shape, direction), group in groups.items():
        if shape == "interseq":
            value = entry(
                group,
                inter_keys,
                len(INTER_SEQUENCE_LAYOUTS),
                lambda r: (f"mixed/{r['head_dim']}", INTER_SEQUENCE_LAYOUTS.index(r["layout"])),
            )
        else:
            value = entry(
                group,
                attention_keys,
                len(seqs),
                lambda r: (f"{r['mask']}/{r['head_dim']}", seqs.index(r["seqlen"])),
            )
        data["perf"].setdefault(shape, {}).setdefault(direction, {})[device] = value
    first = records[0]
    versions = first.get("versions", {})
    data.setdefault("runs", {})[device] = {
        "gpu": first["device"],
        "date": datetime.date.today().isoformat(),
        "torch": versions.get("torch"),
        "cudnn": cudnn_version(versions["cudnn"]) if versions.get("cudnn") else None,
        "fa4": versions.get("flash-attn-4"),
    }


def runs_text(data, devices):
    parts = []
    for device, run in data.get("runs", {}).items():
        name = devices.get(device, run["gpu"])
        libraries = [f"PyTorch {run['torch']}"]
        if run.get("cudnn"):
            libraries.append(f"cuDNN {run['cudnn']}")
        if run.get("fa4"):
            libraries.append(f"flash-attn-4 {run['fa4']}")
        parts.append(f"{name} ({run['gpu']}) on {run['date']} with {', '.join(libraries)}.")
    return " ".join(parts)


def device_names(text):
    return dict(re.findall(r'^\s+(\w+): \{ name: "([^"]+)"', text, re.M))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("device", help="the key of the device in DEVICES, such as sm90")
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--name", help="the name of a new device")
    parser.add_argument("--sub", help="the kernels that a new device runs, such as SM90")
    parser.add_argument("--arch", help="the architecture class of a new device, such as sm90")
    arguments = parser.parse_args()
    records = [json.loads(line) for path in arguments.records for line in path.read_text().splitlines() if line]
    text = REPORT.read_text()
    devices = device_names(text)
    if arguments.device not in devices:
        if not (arguments.name and arguments.sub and arguments.arch):
            parser.error(f"{arguments.device} is a new device: give --name, --sub and --arch")
        line = (
            f'  {arguments.device}: {{ name: "{arguments.name}", sub: "{arguments.sub}", '
            f'arch: "{arguments.arch}", ref: "torch SDPA" }},\n'
        )
        start = text.index("const DEVICES = {")
        end = text.index("\n};", start)
        text = text[: end + 1] + line + text[end + 1 :]
        devices[arguments.device] = arguments.name
    start = text.index("const DATA = ") + len("const DATA = ")
    data, length = json.JSONDecoder().raw_decode(text[start:])
    update_data(data, arguments.device, records)
    text = text[:start] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + text[start + length :]
    runs = f'<p class="note" id="perf-runs">Measured: {html.escape(runs_text(data, devices))}</p>'
    text = re.sub(r'<p class="note" id="perf-runs">.*?</p>', lambda _: runs, text, flags=re.S)
    REPORT.write_text(text)


if __name__ == "__main__":
    main()
