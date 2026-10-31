import collections

import pytest
import torch

import checks
from shapes import HEAD_DIMS
from zigfa._impl import inter_sequence, runtime, specialization
from zigfa._impl.backward import backward_variant, backward_workspace_layout
from zigfa._impl.forward import forward_variant, forward_workspace_layout

SELECTION_CACHES = (
    forward_workspace_layout,
    forward_variant,
    backward_workspace_layout,
    backward_variant,
)
MAIN_KERNELS = ("flash_attn_fwd", "flash_attn_bwd")
session_launches = collections.Counter()


def pytest_addoption(parser):
    parser.addoption(
        "--kernel-coverage",
        choices=("report", "require"),
        default="report",
        help="list the instances that this GPU dispatches to and whose main kernel no test launched, or also fail "
        "the run when any is left",
    )


def pytest_configure(config):
    torch.set_float32_matmul_precision("high")
    launch = runtime.run

    def run(kernel, *arguments, **options):
        label = runtime.kernel_labels.get(kernel, "unlabelled")
        checks.launched_kernels.add(label)
        session_launches[label] += 1
        return launch(kernel, *arguments, **options)

    runtime.run = run


def dispatched_instances():
    """Returns the built instances that the library can dispatch to on this GPU: the forward and backward instances of
    its architecture, the inter-sequence instances of its inter-sequence architecture, and the persistent forward
    instances that `inter_sequence.persistent_instance` selects."""
    device_id = runtime.current_device().device_id
    arch = runtime.arch_name(device_id)
    inter_arch = runtime.arch_name(device_id, inter_sequence=True)
    entries = runtime.manifest()[1]
    names = [name for name in runtime.declared()[1] if name in entries]
    regular = {name for name in names if name.startswith((f"flash_fwd_{arch}_", f"flash_bwd_{arch}_"))}
    inter = {
        name for name in names
        if name.startswith((f"flash_fwd_{inter_arch}_", f"flash_bwd_{inter_arch}_")) and runtime.INTER_SEQUENCE in name
    }
    persistent = {
        inter_sequence.persistent_instance(head_dim, packing)
        for head_dim in inter_sequence.persistent_head_dims(device_id)
        for packing in (1, 2)
    }
    return sorted(regular | inter | persistent - {None})


def instances_not_launched():
    launched = {label.split(".")[0] for label in session_launches if label.split(".")[-1] in MAIN_KERNELS}
    return [name for name in dispatched_instances() if name not in launched]


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("kernel_coverage") == "require" and instances_not_launched():
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    instances, missing = dispatched_instances(), instances_not_launched()
    terminalreporter.section("kernel coverage")
    terminalreporter.write_line(
        f"{len(instances) - len(missing)} of the {len(instances)} instances that this GPU dispatches to launched their "
        "main kernel"
    )
    for name in missing:
        terminalreporter.write_line(f"not launched: {name}")


def pytest_report_header(config):
    return f"head dims in the kernel manifest: {', '.join(map(str, HEAD_DIMS))}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        result = yield
    except BaseException:
        checks.queued_checks.clear()
        raise
    checks.verify_queued_checks()
    return result


def clear_selection():
    for cached in SELECTION_CACHES:
        cached.cache_clear()


@pytest.fixture(autouse=True)
def fresh_selection():
    checks.launched_kernels.clear()
    clear_selection()
    yield
    clear_selection()


@pytest.fixture
def restrict_candidates(monkeypatch):
    candidates = specialization.candidates

    def restrict(variants):
        monkeypatch.setattr(
            specialization,
            "candidates",
            lambda declared, facts: tuple(pair for pair in candidates(declared, facts) if pair[0] in variants),
        )
        clear_selection()

    return restrict
