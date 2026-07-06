import torch
import pytest
from src.neural_shadow_art.config import Config, LossConfig
from src.neural_shadow_art.losses import (
    loss_rendering,
    loss_cohesion,
    loss_connectivity,
    loss_smoothness,
    loss_volume,
    loss_binarization,
    LossScheduler,
)


def test_rendering_loss_zero_on_perfect():
    pred = torch.tensor([0.0, 1.0, 0.5, 0.8])
    target = pred.clone()
    assert loss_rendering(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_rendering_loss_positive_on_mismatch():
    pred = torch.zeros(10)
    target = torch.ones(10)
    assert loss_rendering(pred, target).item() > 0.0


def test_binarization_zero_on_binary():
    occ = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert loss_binarization(occ).item() == pytest.approx(0.0, abs=1e-6)


def test_binarization_positive_on_intermediate():
    occ = torch.full((8,), 0.5)
    assert loss_binarization(occ).item() > 0.0


def test_cohesion_zero_on_uniform_ray():
    # If all samples along a ray are identical, there's no change → loss=0
    occ = torch.ones(4, 10) * 0.7
    assert loss_cohesion(occ).item() == pytest.approx(0.0, abs=1e-6)


def test_cohesion_positive_on_alternating():
    occ = torch.zeros(2, 4)
    occ[:, ::2] = 1.0  # alternating 1,0,1,0 → large jumps
    assert loss_cohesion(occ).item() > 0.0


def test_volume_loss_increases_with_occupancy():
    weights = torch.full((4, 5), 0.1)
    low_occ = torch.full((4, 5), 0.1)
    high_occ = torch.full((4, 5), 0.9)
    assert loss_volume(high_occ, weights).item() > loss_volume(low_occ, weights).item()


def test_volume_loss_scales_with_segment_length():
    # Same occupancy but doubled segment lengths → doubled volume (Eq. 15-16)
    occ = torch.full((4, 5), 0.9)
    w1 = torch.full((4, 5), 0.1)
    v1 = loss_volume(occ, w1).item()
    v2 = loss_volume(occ, 2.0 * w1).item()
    assert v2 == pytest.approx(2.0 * v1, rel=1e-5)


def test_loss_scheduler_weights_epoch0():
    cfg = LossConfig(beta_coh=0.1, beta_bin=0.1, beta_smo=0.05, beta_vol=0.01)
    sched = LossScheduler(cfg)
    w = sched.get_weights(epoch=0)
    # At epoch 0: coh and bin scaled by 2^0 = 1
    assert w["coh"] == pytest.approx(0.1 * 1.0)
    assert w["bin"] == pytest.approx(0.1 * 1.0)
    assert w["smo"] == 0.0
    assert w["vol"] == 0.0


def test_loss_scheduler_weights_epoch2():
    cfg = LossConfig(beta_coh=0.1, beta_bin=0.1, beta_smo=0.05, beta_vol=0.01)
    sched = LossScheduler(cfg)
    w = sched.get_weights(epoch=2)
    # At epoch 2: scale = 2^2 = 4
    assert w["coh"] == pytest.approx(0.1 * 4.0)
    assert w["bin"] == pytest.approx(0.1 * 4.0)
    assert w["smo"] == 0.0
    assert w["vol"] == 0.0


def test_loss_scheduler_weights_epoch4():
    cfg = LossConfig(beta_coh=0.1, beta_bin=0.1, beta_smo=0.05, beta_vol=0.01)
    sched = LossScheduler(cfg)
    w = sched.get_weights(epoch=4)
    # At epoch 4: scale = 2^3 = 8 (capped); smo and vol now active
    assert w["coh"] == pytest.approx(0.1 * 8.0)
    assert w["smo"] == pytest.approx(0.05)
    assert w["vol"] == pytest.approx(0.01)


def test_loss_scheduler_connectivity_gate():
    cfg = LossConfig(beta_con=0.01, con_start_epoch=5)
    sched = LossScheduler(cfg)
    assert sched.get_weights(epoch=4)["con"] == 0.0
    assert sched.get_weights(epoch=5)["con"] == pytest.approx(0.01)


def test_loss_scheduler_total():
    cfg = LossConfig()
    sched = LossScheduler(cfg)
    terms = {
        "ren": torch.tensor(1.0),
        "coh": torch.tensor(0.5),
        "smo": torch.tensor(0.0),
        "vol": torch.tensor(0.0),
        "bin": torch.tensor(0.2),
    }
    total = sched.compute_total_loss(epoch=0, loss_terms=terms)
    assert total.item() > 0.0


def make_ray_samples(n_rays=64, k=64, steepness=400.0, seed=0):
    """Synthetic ray samples crossing a sharp planar surface at z = 0.

    Rays travel along +z through the unit cube; occupancy is a steep sigmoid
    of z, mimicking a converged binary field.
    """
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n_rays, 1, 2, generator=g) - 0.5           # (N, 1, 2)
    z = torch.linspace(-0.5, 0.5, k).reshape(1, k, 1).expand(n_rays, k, 1)
    points = torch.cat([xy.expand(n_rays, k, 2), z], dim=-1)   # (N, K, 3)
    occ = torch.sigmoid(steepness * z.squeeze(-1))             # (N, K)
    return points, occ


def test_smoothness_zero_on_uniform_field():
    points, _ = make_ray_samples()
    occ = torch.full(points.shape[:2], 0.3, requires_grad=True)
    val = loss_smoothness(points, occ, img_width=64)
    assert val.item() == pytest.approx(0.0)


def test_smoothness_finds_surface_and_is_finite():
    points, occ = make_ray_samples()
    occ = occ.clone().requires_grad_(True)
    # Low θ so the finite-difference gradient of the synthetic surface
    # comfortably clears the θ·w threshold
    val = loss_smoothness(points, occ, img_width=64, theta=0.05)
    assert val.item() >= 0.0
    assert torch.isfinite(val)
    # Must be differentiable w.r.t. occupancy values
    val.backward()
    assert occ.grad is not None
    assert torch.isfinite(occ.grad).all()


def test_smoothness_flat_surface_smoother_than_bumpy():
    # A flat plane should incur less normal-variation penalty than an
    # undulating surface with the same sharpness.
    g = torch.Generator().manual_seed(1)
    n_rays, k = 128, 64
    xy = torch.rand(n_rays, 1, 2, generator=g) - 0.5
    z = torch.linspace(-0.5, 0.5, k).reshape(1, k, 1).expand(n_rays, k, 1)
    points = torch.cat([xy.expand(n_rays, k, 2), z], dim=-1)

    flat_occ = torch.sigmoid(400.0 * z.squeeze(-1))
    bump = 0.15 * torch.sin(20.0 * points[..., 0]) * torch.cos(20.0 * points[..., 1])
    bumpy_occ = torch.sigmoid(400.0 * (z.squeeze(-1) - bump))

    val_flat = loss_smoothness(points, flat_occ, img_width=64, theta=0.05)
    val_bumpy = loss_smoothness(points, bumpy_occ, img_width=64, theta=0.05)
    assert val_bumpy.item() > val_flat.item()


def make_blob_grid(g=24, centers=((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
                   radius=0.28, bridge_occ=None):
    """Near-binary spherical blobs on a g³ grid over [-1, 1]³.

    Occupancy is a steep sigmoid of distance to the nearest blob surface,
    mimicking a converged binarized field. bridge_occ (if given) fills a
    thin axis-aligned corridor between the blob centers at that occupancy.
    """
    ax = torch.linspace(-1, 1, g)
    x, y, z = torch.meshgrid(ax, ax, ax, indexing="ij")
    occ = torch.zeros(g, g, g)
    for cx, cy, cz in centers:
        d = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2).sqrt()
        occ = torch.maximum(occ, torch.sigmoid((radius - d) * 40.0))
    if bridge_occ is not None:
        corridor = (x.abs() < 0.55) & (y.abs() < 0.12) & (z.abs() < 0.12)
        occ = torch.maximum(occ, corridor.float() * bridge_occ)
    return occ


def test_connectivity_zero_when_connected():
    # A single blob: everything is (in) the largest component → tiny loss,
    # only the soft shell contributes.
    grid = make_blob_grid(centers=((0.0, 0.0, 0.0),))
    assert loss_connectivity(grid).item() < 0.1


def test_connectivity_high_when_disconnected():
    # Two equal blobs: roughly half the mass is unreachable from the seed.
    grid = make_blob_grid()
    assert loss_connectivity(grid).item() > 0.3


def test_connectivity_drops_when_bridge_added():
    # The doc's acceptance test: adding a bridge between the blobs must
    # drive the loss down (here the bridge merges the hard components).
    apart = loss_connectivity(make_blob_grid()).item()
    bridged = loss_connectivity(make_blob_grid(bridge_occ=1.0)).item()
    assert bridged < 0.5 * apart
    assert bridged < 0.1


def test_connectivity_gradients_shrink_stragglers_and_grow_bridges():
    # A weak (sub-threshold) corridor across a short gap: the straggler blob
    # is still a separate hard component, so the gradient must (a) shrink it
    # and (b) reward raising occupancy along the corridor toward the main
    # body. (The bridge-growing signal decays with the product of corridor
    # occupancies, so the gap must be short for (b) to dominate the voxel's
    # own shrink term — exactly the "grow from the nearest contact" behavior
    # wanted during training.)
    grid = make_blob_grid(
        centers=((-0.4, 0.0, 0.0), (0.4, 0.0, 0.0)), bridge_occ=0.48
    ).requires_grad_(True)
    val = loss_connectivity(grid)
    val.backward()
    assert torch.isfinite(grid.grad).all()

    g = grid.shape[0]
    ax = torch.linspace(-1, 1, g)
    seed_blob = int((ax + 0.4).abs().argmin())     # x index of first blob center
    far_blob = int((ax - 0.4).abs().argmin())      # x index of second blob center
    mid = int(ax.abs().argmin())                   # corridor midpoint
    c = g // 2                                     # y = z = center index
    assert grid.grad[far_blob, c, c].item() > 0.0   # shrink the straggler
    assert grid.grad[mid, c, c].item() < 0.0        # grow the bridge
    # The seed component itself is fully reachable — no shrink pressure there.
    assert grid.grad[seed_blob, c, c].item() <= grid.grad[far_blob, c, c].item()


def test_connectivity_empty_field_no_crash():
    # Nothing clears the threshold → falls back to an argmax-voxel seed.
    grid = torch.full((16, 16, 16), 0.01, requires_grad=True)
    val = loss_connectivity(grid)
    assert torch.isfinite(val)
    val.backward()
    assert torch.isfinite(grid.grad).all()
