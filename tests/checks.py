import contextlib

import torch

from reference import FULLY_MASKED_LSE, reference_gradients

queued_checks = []
case_labels = []

OUTPUT_RTOL = 4
LSE_TOLERANCE = 1e-2


@contextlib.contextmanager
def case(**parameters):
    case_labels.append(" ".join(f"{name}={value}" for name, value in parameters.items()))
    try:
        yield
    finally:
        case_labels.pop()


def queue(kind, name, value):
    queued_checks.append((kind, " ".join([*case_labels, name]), value))


def queue_tolerance_check(tensor, tensor_ref, tensor_pt, name):
    assert tensor.shape == tensor_ref.shape, f"{name}: shape {tuple(tensor.shape)} != {tuple(tensor_ref.shape)}"
    atol = 2 * (tensor_ref + 0.3 - 0.3 - tensor_ref).abs().amax()
    error = (tensor.float() - tensor_ref.float()).abs().amax()
    allowed = OUTPUT_RTOL * (tensor_pt.float() - tensor_ref.float()).abs().amax() + atol
    queue("bound", name, torch.stack([error, allowed]))


def queue_lse_check(lse, lse_ref):
    assert lse.shape == lse_ref.shape, f"lse: shape {tuple(lse.shape)} != {tuple(lse_ref.shape)}"
    finite = torch.isfinite(lse_ref)
    excess = (lse - lse_ref).abs() - LSE_TOLERANCE - LSE_TOLERANCE * lse_ref.abs()
    queue("excess", "lse values", torch.where(finite, excess, -1.0).amax().reshape(1))
    unmatched_fully_masked = (lse == FULLY_MASKED_LSE) != (lse_ref == FULLY_MASKED_LSE)
    queue("failure", "lse of fully masked rows", unmatched_fully_masked.any().reshape(1))


def queue_requirement(name, condition):
    queue("failure", name, ~condition.reshape(1))


def verify_queued_checks():
    checks = list(queued_checks)
    queued_checks.clear()
    if not checks:
        return
    values = torch.cat([value.float().reshape(-1) for _, _, value in checks]).tolist()
    failures = []
    offset = 0
    for kind, name, value in checks:
        part = values[offset : offset + value.numel()]
        offset += value.numel()
        if kind == "bound" and not part[0] <= part[1]:
            failures.append(f"{name}: error {part[0]:.3e} exceeds {part[1]:.3e}")
        elif kind == "excess" and not part[0] <= 0:
            failures.append(f"{name}: excess {part[0]:.3e}")
        elif kind == "failure" and part[0]:
            failures.append(f"{name} failed")
    assert not failures, f"{len(failures)} of {len(checks)} checks failed:\n" + "\n".join(failures)


def queue_forward_checks(results, function, q, k, v, *arguments, **options):
    out_ref, lse_ref = function(q, k, v, *arguments, **options)
    out_pt, _ = function(q, k, v, *arguments, upcast=False, **options)
    for variant, (out, lse) in enumerate(results):
        with case(variant=variant):
            queue_tolerance_check(out, out_ref, out_pt, "out")
            if lse is not None:
                queue_lse_check(lse, lse_ref)


def queue_gradient_checks(results, function, q, k, v, dout, *arguments, **options):
    gradients_ref = reference_gradients(function, q, k, v, dout, *arguments, **options)
    gradients_pt = reference_gradients(function, q, k, v, dout, *arguments, upcast=False, **options)
    for variant, gradients in enumerate(results):
        with case(variant=variant):
            for name, gradient, ref, pt in zip(("dq", "dk", "dv"), gradients, gradients_ref, gradients_pt):
                queue_tolerance_check(gradient, ref, pt, name)
