from __future__ import annotations

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
    sample_points: Tensor,
    sample_occ: Tensor,
    img_width: int,
    theta: float = 0.4,
    k1: int = 26,
    k2: int = 6,
    max_candidates: int = 1024,
) -> Tensor:
    """Surface smoothness loss (paper Eqs. 11-14).

    Follows the paper: occupancy gradients are estimated with least-squares
    finite differences over the k₁ nearest sample points (Eqs. 12-13) —
    NOT network autograd, which the paper notes is unstable. Points whose
    gradient magnitude exceeds θ·w (Eq. 11, w = image width) are surface
    points; Eq. 14 penalizes the rate of gradient change between each surface
    point and its k₂ nearest surface neighbors.

    For tractability, candidate points are pre-selected by a cheap along-ray
    occupancy difference before the full least-squares estimate (the paper
    processes every batch sample, which is quadratic in batch size).

    Args:
        sample_points: (N, K, 3) sample positions on valid truncated rays.
        sample_occ: (N, K) occupancy at those samples (with autograd graph).
        img_width: w in Eq. 11.
        theta: θ in Eq. 11.
        k1 / k2: neighbor counts for Eqs. 13 / 14.
        max_candidates: cap on surface candidates per step (cost control).
    Returns:
        Scalar loss (0.0 if fewer than 2 surface points found).
    """
    zero = sample_occ.sum() * 0.0  # keeps graph alive on early exits
    N, K = sample_occ.shape
    if N == 0 or K < 5:
        return zero

    pts = sample_points.detach()
    threshold = theta * img_width

    # 1. Cheap along-ray gradient proxy to find candidate surface crossings.
    #    Underestimates ‖∇f‖ by the cosine between ray and normal, hence the
    #    0.5 slack factor in the pre-filter.
    seg = (pts[:, 2:] - pts[:, :-2]).norm(dim=-1).clamp(min=1e-9)  # (N, K-2)
    proxy = (sample_occ[:, 2:] - sample_occ[:, :-2]).abs().detach() / seg
    cand_mask = proxy > 0.5 * threshold  # (N, K-2), index k ↔ sample k+1

    n_cand = int(cand_mask.sum())
    if n_cand == 0:
        return zero
    if n_cand > max_candidates:
        flat = torch.where(
            cand_mask.reshape(-1),
            proxy.reshape(-1),
            torch.full_like(proxy.reshape(-1), -1.0),
        )
        keep = flat.topk(max_candidates).indices
        cand_ray = keep // (K - 2)
        cand_k = keep % (K - 2) + 1
    else:
        idx = cand_mask.nonzero(as_tuple=False)
        cand_ray, cand_k = idx[:, 0], idx[:, 1] + 1

    C = cand_ray.shape[0]
    cand_pts = pts[cand_ray, cand_k]         # (C, 3)
    cand_occ = sample_occ[cand_ray, cand_k]  # (C,)

    # 2. Neighbor pool: candidates plus their ±1/±2 along-ray samples, so the
    #    least-squares system sees the sharp variation across the surface as
    #    well as transverse structure from nearby rays.
    offsets = torch.tensor([-2, -1, 0, 1, 2], device=pts.device)
    pool_k = (cand_k.unsqueeze(1) + offsets).clamp(0, K - 1)  # (C, 5)
    pool_ray = cand_ray.unsqueeze(1).expand_as(pool_k)
    pool_pts = pts[pool_ray.reshape(-1), pool_k.reshape(-1)]          # (5C, 3)
    pool_occ = sample_occ[pool_ray.reshape(-1), pool_k.reshape(-1)]   # (5C,)

    # 3. Least-squares gradient estimate from the k₁ nearest pool points
    #    (Eqs. 12-13), solved via regularized normal equations.
    d = torch.cdist(cand_pts, pool_pts)  # (C, 5C)
    d = torch.where(d < 1e-9, torch.full_like(d, float("inf")), d)  # drop self
    k1_eff = min(k1, pool_pts.shape[0] - 1)
    if k1_eff < 3:
        return zero
    nn_idx = d.topk(k1_eff, dim=1, largest=False).indices  # (C, k1)

    K_mat = pool_pts[nn_idx] - cand_pts.unsqueeze(1)              # (C, k1, 3)
    b_vec = (pool_occ[nn_idx] - cand_occ.unsqueeze(1)).unsqueeze(-1)  # (C, k1, 1)
    KtK = K_mat.transpose(1, 2) @ K_mat                           # (C, 3, 3)
    Ktb = K_mat.transpose(1, 2) @ b_vec                           # (C, 3, 1)
    lam = KtK.diagonal(dim1=1, dim2=2).mean(dim=1) * 1e-6 + 1e-12  # (C,)
    eye = torch.eye(3, device=pts.device).unsqueeze(0)
    grads = torch.linalg.solve(
        KtK + lam.view(-1, 1, 1) * eye, Ktb
    ).squeeze(-1)  # (C, 3)

    # 4. Surface points via Eq. 11, then the Eq. 14 penalty.
    surface = grads.norm(dim=-1) > threshold
    S = int(surface.sum())
    if S < 2:
        return zero

    s_pts = cand_pts[surface]
    s_grads = grads[surface]
    sd = torch.cdist(s_pts, s_pts)
    sd.fill_diagonal_(float("inf"))
    k2_eff = min(k2, S - 1)
    knn_d, knn_i = sd.topk(k2_eff, dim=1, largest=False)  # (S, k2)

    grad_diff = (s_grads.unsqueeze(1) - s_grads[knn_i]).norm(dim=-1)  # (S, k2)
    return (grad_diff / knn_d.clamp(min=1e-7)).mean()


def loss_volume(
    sample_occ: Tensor,
    sample_weights: Tensor,
    sigmoid_beta: float = 100.0,
    tau: float = 0.5,
) -> Tensor:
    """Differentiable volume approximation (paper Eqs. 15-16).

    Each sample's soft occupancy switch sigmoid((f−τ)/T) is weighted by its
    trapezoid segment length ω, so the per-ray sum approximates the occupied
    length along the ray; the batch mean approximates total volume.

    Args:
        sample_occ: (N, K) occupancy at samples along each ray.
        sample_weights: (N, K) segment lengths ω (Eq. 16); 0 for padding.
        sigmoid_beta: 1/T — sharpness of the soft switch.
        tau: occupancy threshold τ (matches the reconstruction threshold).
    Returns:
        Scalar loss.
    """
    n_rays = sample_occ.shape[0]
    if n_rays == 0:
        return sample_occ.sum() * 0.0
    soft = torch.sigmoid(sigmoid_beta * (sample_occ - tau))
    return (sample_weights * soft).sum() / n_rays


def loss_connectivity(
    occ_grid: Tensor,
    threshold: float = 0.5,
    n_iters: int | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Soft flood-fill connectivity loss (L_con) on a coarse occupancy grid.

    Penalizes occupied mass that is not reachable from the main body, so
    stragglers must either connect or vanish — organic connections form
    during training instead of post-hoc struts (see
    research_papers/3d_print_improvements.md, Option 2).

    Reachability r is seeded on the largest connected component of the
    hard-thresholded field (computed detached — the seed choice is a
    constant w.r.t. autograd) and spread one voxel per iteration:

        r ← max(maxpool3d(r · f, 3×3×3), seed)

    Multiplying by the occupancy f *before* pooling gates the spread through
    occupied space while keeping the fill differentiable, and means a voxel's
    reachability reflects only the path leading to it, not its own softness
    (a soft surface voxel adjacent to reached solid interior gets r ≈ 1 and
    is not penalized). Pinning the seed at 1 makes r monotone in the
    iteration count. The loss is the occupied-but-unreachable fraction

        L_con = Σ f·(1 − r) / (Σ f + ε)  ∈ [0, 1].

    Gradients both shrink disconnected mass (∂L/∂f > 0 on stragglers) and
    grow bridges: raising f along a partially-occupied corridor toward the
    main body raises r over the entire straggler, so ∂L/∂f < 0 there.

    Args:
        occ_grid: (G, G, G) occupancy field sampled on a regular grid,
            with autograd graph attached.
        threshold: occupancy level defining solid voxels for seeding.
        n_iters: flood-fill iterations; None → the grid diameter (max dim).
        eps: guards the division for a near-empty field.
    Returns:
        Scalar loss.
    """
    from scipy import ndimage

    if n_iters is None:
        n_iters = max(occ_grid.shape)

    # Seed: largest connected component of the thresholded field (6-conn),
    # falling back to the single most-occupied voxel when nothing clears the
    # threshold. Seeding a whole component (not an argmax voxel) keeps the
    # seed stable across steps — argmax can jump between blobs.
    hard = (occ_grid.detach() >= threshold).cpu().numpy()
    seed = torch.zeros_like(occ_grid)
    if hard.any():
        labels, n_comp = ndimage.label(hard)
        if n_comp > 1:
            sizes = ndimage.sum_labels(hard, labels, index=range(1, n_comp + 1))
            hard = labels == (int(sizes.argmax()) + 1)
        seed[torch.from_numpy(hard).to(occ_grid.device)] = 1.0
    else:
        seed.view(-1)[int(occ_grid.detach().argmax())] = 1.0

    f = occ_grid[None, None]  # (1, 1, G, G, G) for max_pool3d
    seed = seed[None, None]
    r = seed
    for _ in range(n_iters):
        r = torch.maximum(F.max_pool3d(r * f, kernel_size=3, stride=1, padding=1), seed)
    r = r[0, 0]

    return (occ_grid * (1.0 - r)).sum() / (occ_grid.sum() + eps)


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
    - β_smo (smoothness) and β_vol (volume) are suppressed for epochs 0–2
      (paper's first 3 epochs, 1-indexed), then activated at full weight.
    - β_con (connectivity, not in the paper) is suppressed until
      con_start_epoch — the shape must exist, and L_vol should have done
      most of its shrinking, before connectivity is meaningful.
    """

    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    def get_weights(self, epoch: int) -> dict[str, float]:
        """Return the effective loss weight for each term at a given epoch."""
        scale_early = float(2 ** min(epoch, 3))
        return {
            "ren": self.cfg.beta_ren,
            "coh": self.cfg.beta_coh * scale_early,
            "smo": 0.0 if epoch < 3 else self.cfg.beta_smo,
            "vol": 0.0 if epoch < 3 else self.cfg.beta_vol,
            "bin": self.cfg.beta_bin * scale_early,
            "con": 0.0 if epoch < self.cfg.con_start_epoch else self.cfg.beta_con,
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
