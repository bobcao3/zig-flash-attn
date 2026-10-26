import contextlib

import torch

import reference
from reference import FULLY_MASKED_LSE

queued_checks = []
case_labels = []

OUTPUT_RTOL = 4
BF16_REFERENCE_RTOL = 3 * 2**-7
LSE_TOLERANCE = 1e-2
FLOAT32_LONG_KEYS = 1024

launched_kernels = set()
float32_checked = set()


@contextlib.contextmanager
def case(**parameters):
    case_labels.append(" ".join(f"{name}={value}" for name, value in parameters.items()))
    try:
        yield
    finally:
        case_labels.pop()


def queue(kind, name, value):
    queued_checks.append((kind, " ".join([*case_labels, name]), value))


def largest(tensor):
    return torch.linalg.vector_norm(tensor, float("inf"))


def queue_tolerance_checks(tensors, tensor_ref, tensor_pt, name):
    atol = 2 * largest((tensor_ref + 0.3).sub_(0.3).sub_(tensor_ref))
    tensor_ref = tensor_ref.float()
    bf16_allowed = BF16_REFERENCE_RTOL * largest(tensor_ref) + atol
    if tensor_pt is None:
        allowed = bf16_allowed
    else:
        pt_error = largest(torch.sub(tensor_pt, tensor_ref))
        queue("bound", f"{name} of the reference in {tensor_pt.dtype}", torch.stack([pt_error, bf16_allowed]))
        allowed = OUTPUT_RTOL * pt_error + atol
    for variant, tensor in enumerate(tensors):
        assert tensor.shape == tensor_ref.shape, f"{name}: shape {tuple(tensor.shape)} != {tuple(tensor_ref.shape)}"
        with case(variant=variant):
            queue("bound", name, torch.stack([largest(torch.sub(tensor, tensor_ref)), allowed]))


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


def reference_pass(function, q, k, v, dout, upcast, *arguments, **options):
    if dout is None:
        with torch.no_grad():
            return (*function(q, k, v, *arguments, upcast=upcast, **options), None)
    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
    out, lse = function(*inputs, *arguments, upcast=upcast, **options)
    gradients = torch.autograd.grad(out, inputs, dout.to(out.dtype).contiguous())
    return out.detach(), lse.detach(), gradients


def wants_float32(kernels, longest_keys):
    size = "long" if longest_keys >= FLOAT32_LONG_KEYS else "short"
    pairs = {(kernel, size) for kernel in kernels}
    if pairs <= float32_checked:
        return False
    float32_checked.update(pairs)
    return True


def queue_attention_checks(
    forward_results, gradient_results, function, q, k, v, dout, *arguments, float32_reference=True, **options
):
    """Checks outputs and gradients against `function` in the dtype of `q`. The first case of each launched kernel
    with fewer than FLOAT32_LONG_KEYS keys, and the first with more, also run `function` in float32. An error may then
    reach OUTPUT_RTOL times the error of the pass in the dtype of `q`, and otherwise BF16_REFERENCE_RTOL times the
    largest value of that pass."""
    kernels = set(launched_kernels)
    launched_kernels.clear()
    dout = dout if gradient_results else None
    out_ref, lse_ref, gradients_ref = reference_pass(function, q, k, v, dout, False, *arguments, **options)
    out_pt, gradients_pt = None, (None, None, None)
    if float32_reference and wants_float32(kernels, reference.longest_keys):
        out_pt, gradients_pt = out_ref, gradients_ref
        out_ref, lse_ref, gradients_ref = reference_pass(function, q, k, v, dout, True, *arguments, **options)
    with torch.no_grad():
        if forward_results:
            queue_tolerance_checks([out for out, _ in forward_results], out_ref, out_pt, "out")
            for variant, (_, lse) in enumerate(forward_results):
                if lse is not None:
                    with case(variant=variant):
                        queue_lse_check(lse, lse_ref)
        if gradient_results:
            for index, name in enumerate(("dq", "dk", "dv")):
                gradients = [gradients[index] for gradients in gradient_results]
                queue_tolerance_checks(gradients, gradients_ref[index], gradients_pt[index], name)
