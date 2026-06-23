from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig


def loss_rendering(
    pred: Tensor,
    target: Tensor,
    area_weight: float = 1.0,
) -> Tensor:
    """MSE between predicted ray occupancy and binary target mask.

    Args:
        pred: (N,) predicted occupancy in [0, 1].
        target: (N,) binary ground-truth labels {0, 1}.
        area_weight: scalar weight derived from shadow area ratio.
            Smaller shadow areas receive higher weight.
    Returns:
        Scalar loss.
    """
    return F.mse_loss(pred, target.float()) * area_weight


def loss_cohesion(per_sample_occ: Tensor) -> Tensor:
    """Penalizes occupancy jumps between adjacent samples along each ray.

    Discourages thin multi-layer geometry and promotes contiguous solid regions.

    Args:
        per_sample_occ: (N, K) occupancy at each sample along each ray.
    Returns:
        Scalar loss.
    """
    if per_sample_occ.shape[1] < 2:
        return per_sample_occ.sum() * 0.0
    diff = per_sample_occ[:, 1:] - per_sample_occ[:, :-1]  # (N, K-1)
    return (diff ** 2).mean()


def loss_smoothness(
    occupancy_fn: Callable[[Tensor], Tensor],
    device: torch.device,
    bbox_min: tuple,
    bbox_max: tuple,
    grad_threshold: float = 0.01,
    n_samples: int = 256,
    k_neighbors: int = 8,
) -> Tensor:
    """Surface normal consistency loss.

    Samples random points in the scene, identifies surface points via gradient
    magnitude thresholding, then penalizes normal deviation among k nearest
    surface neighbors.

    Args:
        occupancy_fn: callable (N, 3) → (N, 1).
        device: target device.
        bbox_min / bbox_max: scene bounding box corners as tuples.
        grad_threshold: θ_w — gradient magnitude threshold for surface detection.
        n_samples: number of random points to sample.
        k_neighbors: number of nearest neighbors to use.
    Returns:
        Scalar loss (0.0 if fewer than 2 surface points found).
    """
    bmin = torch.tensor(bbox_min, dtype=torch.float32, device=device)
    bmax = torch.tensor(bbox_max, dtype=torch.float32, device=device)

    pts = torch.rand(n_samples, 3, device=device) * (bmax - bmin) + bmin
    pts.requires_grad_(True)

    occ = occupancy_fn(pts)  # (N, 1)
    grad = torch.autograd.grad(
        occ.sum(), pts, create_graph=True
    )[0]  # (N, 3)

    grad_norm = grad.norm(dim=-1)  # (N,)
    surface_mask = grad_norm > grad_threshold

    if surface_mask.sum() < 2:
        return occ.sum() * 0.0  # keeps graph alive

    surface_pts = pts[surface_mask]  # (S, 3)
    surface_normals = F.normalize(grad[surface_mask], dim=-1)  # (S, 3)

    S = surface_pts.shape[0]
    k = min(k_neighbors, S - 1)

    # Pairwise distances for KNN among surface points
    dists = torch.cdist(surface_pts.detach(), surface_pts.detach())  # (S, S)
    dists.fill_diagonal_(float("inf"))
    _, indices = dists.topk(k, dim=1, largest=False)  # (S, k)

    neighbor_normals = surface_normals[indices]           # (S, k, 3)
    center_normals = surface_normals.unsqueeze(1).expand_as(neighbor_normals)

    cos_sim = (center_normals * neighbor_normals).sum(dim=-1)  # (S, k)
    return (1.0 - cos_sim).mean()


def loss_volume(
    occ: Tensor,
    sigmoid_beta: float = 100.0,
) -> Tensor:
    """Minimize occupied volume via a differentiable soft-sigmoid approximation.

    Uses sigmoid(β*(f - 0.5)) which approximates a step function at 0.5.
    With β=100 this is essentially a smooth Heaviside.

    Args:
        occ: (N, 1) or (N,) occupancy values from the MLP.
        sigmoid_beta: sharpness; higher → closer to hard threshold.
    Returns:
        Scalar loss.
    """
    return torch.sigmoid(sigmoid_beta * (occ.squeeze(-1) - 0.5)).mean()


def loss_binarization(occ: Tensor) -> Tensor:
    """Force occupancy toward binary values by penalizing intermediate values.

    L_bin = mean(min(f², (1-f)²))

    Args:
        occ: (N, 1) or (N,) occupancy values.
    Returns:
        Scalar loss (= 0 when all values are exactly 0 or 1).
    """
    f = occ.squeeze(-1)
    return torch.minimum(f ** 2, (1.0 - f) ** 2).mean()


class LossScheduler:
    """Manages loss weight scheduling as described in the paper.

    Schedule:
    - β_coh (cohesion) and β_bin (binarization) ramp up as 2^min(epoch, 3)
      starting from epoch 0, encouraging solid binary geometry early.
    - β_smo (smoothness) and β_vol (volume) are suppressed for the first 3
      epochs so basic geometry can form, then activated at full weight.
    """

    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    def get_weights(self, epoch: int) -> dict[str, float]:
        """Return the effective loss weight for each term at a given epoch."""
        scale_early = float(2 ** min(epoch, 3))
        return {
            "ren": self.cfg.beta_ren,
            "coh": self.cfg.beta_coh * scale_early,
            "smo": 0.0 if epoch <= 3 else self.cfg.beta_smo,
            "vol": 0.0 if epoch <= 3 else self.cfg.beta_vol,
            "bin": self.cfg.beta_bin * scale_early,
        }

    def compute_total_loss(
        self,
        epoch: int,
        loss_terms: dict[str, Tensor],
    ) -> Tensor:
        """Weighted sum of loss terms."""
        weights = self.get_weights(epoch)
        total = sum(
            weights[k] * v
            for k, v in loss_terms.items()
            if k in weights
        )
        return total
