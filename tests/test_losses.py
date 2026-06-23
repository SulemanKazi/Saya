import torch
import pytest
from src.neural_shadow_art.config import Config, LossConfig
from src.neural_shadow_art.losses import (
    loss_rendering,
    loss_cohesion,
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
    low_occ = torch.full((20, 1), 0.1)
    high_occ = torch.full((20, 1), 0.9)
    assert loss_volume(high_occ).item() > loss_volume(low_occ).item()


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


def make_simple_occupancy_fn(device):
    """A simple density function for testing: sphere at origin."""
    def fn(pts: torch.Tensor) -> torch.Tensor:
        r = pts.norm(dim=-1, keepdim=True)
        return torch.sigmoid(10.0 * (0.3 - r))
    return fn


def test_smoothness_loss_runs():
    device = torch.device("cpu")
    fn = make_simple_occupancy_fn(device)
    val = loss_smoothness(fn, device, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5),
                          n_samples=32, k_neighbors=4)
    assert val.item() >= 0.0
    assert not torch.isnan(val)
