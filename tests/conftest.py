import pytest
import torch

import checks
from shapes import HEAD_DIMS
from zigfa import specialization
from zigfa.backward import backward_mode, backward_workspace_layout
from zigfa.forward import forward_mode, forward_workspace_layout

SELECTION_CACHES = (
    forward_workspace_layout,
    forward_mode,
    backward_workspace_layout,
    backward_mode,
)


def pytest_configure(config):
    torch.set_float32_matmul_precision("high")


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
    clear_selection()
    yield
    clear_selection()


@pytest.fixture
def restrict_candidates(monkeypatch):
    def restrict(modes):
        monkeypatch.setattr(specialization, "candidates", lambda facts: modes)
        clear_selection()

    return restrict
