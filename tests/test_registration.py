import torch
import pytest
from src.neural_shadow_art.registration import RigidRegistration


def make_square(size=64, lo=24, hi=40, shift=(0, 0)):
    m = torch.zeros(size, size)
    r0, r1 = lo + shift[0], hi + shift[0]
    c0, c1 = lo + shift[1], hi + shift[1]
    m[r0:r1, c0:c1] = 1.0
    return m


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a > 0.5, b > 0.5
    return ((a & b).sum() / ((a | b).sum() + 1e-7)).item()


def test_identity_warp_is_noop():
    reg = RigidRegistration(n_views=1)
    mask = make_square()
    assert reg.is_identity()
    assert torch.equal(reg.warp_mask(mask, 0), mask)


def test_icp_recovers_translation():
    """After update(), the registered target should overlap the rendered
    shadow much better than the original did (paper Sec. 3.3)."""
    torch.manual_seed(0)
    reg = RigidRegistration(n_views=1)
    original = make_square()
    rendered = make_square(shift=(6, -8))  # what the model currently casts

    before = iou(original, rendered)
    updated = reg.update([original], [rendered])[0]
    after = iou(updated, rendered)

    assert not reg.is_identity()
    assert after > before + 0.2, f"ICP should align target (IoU {before:.3f} → {after:.3f})"
    assert after > 0.8


def test_registration_state_roundtrip():
    torch.manual_seed(0)
    reg = RigidRegistration(n_views=2)
    reg.update(
        [make_square(), make_square()],
        [make_square(shift=(4, 4)), make_square(shift=(-3, 5))],
    )
    state = reg.state_dict()

    reg2 = RigidRegistration(n_views=2)
    reg2.load_state_dict(state)
    assert torch.allclose(reg.transforms, reg2.transforms)

    mask = make_square()
    assert torch.equal(reg.warp_mask(mask, 0), reg2.warp_mask(mask, 0))
