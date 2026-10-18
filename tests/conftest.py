import pytest
import torch

import checks
from shapes import HEAD_DIMS
from zigfa import specialization
from zigfa.backward import backward_variant, backward_workspace_layout
from zigfa.forward import forward_variant, forward_workspace_layout

SELECTION_CACHES = (
    forward_workspace_layout,
    forward_variant,
    backward_workspace_layout,
    backward_variant,
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
    candidates = specialization.candidates

    def restrict(variants):
        monkeypatch.setattr(
            specialization,
            "candidates",
            lambda declared, facts: tuple(pair for pair in candidates(declared, facts) if pair[0] in variants),
        )
        clear_selection()

    return restrict
