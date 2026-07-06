from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import yaml


@dataclass
class ModelConfig:
    n_layers: int = 8
    hidden_dim: int = 256
    pos_enc_levels: int = 6
    # Initial logit of the occupancy field. Must start near-empty: with n
    # samples per ray, the rendering gradient scales as (1−f)^(n−1), which
    # vanishes (underflows for n = 256) if the field initializes at f ≈ 0.5.
    init_bias: float = -6.0


@dataclass
class RenderConfig:
    # Samples per ray. None → use the image width (paper: n = w).
    n_samples_per_ray: Optional[int] = None
    frustum_truncation: bool = True
    bbox_min: Tuple[float, float, float] = (-0.5, -0.5, -0.5)
    bbox_max: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class LossConfig:
    beta_ren: float = 1.0
    beta_coh: float = 0.001    # paper: 10^-3 (base, scaled by 2^min(epoch,3))
    beta_smo: float = 0.0001   # paper: 10^-4
    beta_vol: float = 0.0001   # paper: 10^-4
    beta_bin: float = 0.05     # paper: 5×10^-2 (base, scaled by 2^min(epoch,3))
    grad_threshold: float = 0.4  # paper: θ in Eq. 11 (threshold applied as θ·w)
    vol_sigmoid_beta: float = 100.0  # 1/T in Eq. 15 (temperature of the soft switch)
    smo_k1: int = 26           # paper: k₁ neighbors for finite-difference gradients (Eq. 13)
    smo_k2: int = 6            # paper: k₂ nearest surface neighbors (Eq. 14)
    n_smo_samples: int = 1024  # max surface candidates per step (cost cap for L_smo)
    # Connectivity loss L_con (not in the paper — see
    # research_papers/3d_print_improvements.md, Option 2): soft flood-fill
    # reachability on a coarse global grid; occupied mass unreachable from the
    # largest component is penalized, so stragglers connect or vanish.
    beta_con: float = 0.01     # weight on the steps where L_con is evaluated
    con_grid: int = 48         # coarse grid resolution per axis (memory ~ con_grid⁴)
    con_every: int = 20        # evaluate L_con every N steps (global term — no need per batch)
    con_start_epoch: int = 5   # inactive before this epoch (after L_vol has shrunk the shape)


@dataclass
class TrainConfig:
    epochs: int = 30
    # Multiplier on the per-epoch step count. Steps/epoch = this ×
    # (img_size² · n_views / batch_size_rays), i.e. how many times each pixel
    # is revisited per epoch (with fresh stratified samples). Raise it to add
    # optimization iterations WITHOUT touching the epoch-indexed loss schedule
    # (the 2^min(epoch,3) ramp and the smo/vol gate at epoch 3 stay proportional).
    steps_per_epoch_mult: int = 1
    lr: float = 1e-4
    lr_light: float = 1e-3
    img_size: int = 256
    batch_size_rays: int = 4096
    seed: int = 42
    checkpoint_every: int = 5
    log_every: int = 10
    use_registration: bool = False
    registration_every: int = 5  # paper Sec. 3.3: ICP registration every 5 epochs


@dataclass
class MeshConfig:
    grid_resolution: int = 200
    iso_threshold: float = 0.5
    output_format: str = "stl"
    eval_batch_size: int = 32768
    # Connectivity repair: connect disconnected components with struts routed
    # through the target visual hull (region projecting inside the shadow in
    # every view). Adding material there provably cannot change any shadow —
    # the render is a union, O = 1 − ∏(1 − f_k), so extra occupancy only
    # darkens pixels that are already dark in all targets.
    connect_components: bool = True
    strut_radius_mm: float = 2.0   # physical strut radius on the printed model
    model_size_mm: float = 100.0   # physical size the bbox maps to when printed
    # Erode target masks by this many pixels before the hull test, so material
    # is never added flush against a silhouette boundary (guards against
    # off-by-half-a-pixel effects from nearest-pixel mask sampling).
    hull_erosion_px: int = 1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    device: str = "auto"


def _merge_dict_into_dataclass(dc, d: dict):
    """Recursively merge dict d into dataclass dc, returning a new instance."""
    if not isinstance(d, dict):
        return d
    dc = copy.copy(dc)
    for key, value in d.items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown config key: '{key}'")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__"):
            setattr(dc, key, _merge_dict_into_dataclass(current, value))
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(dc, key, tuple(value))
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | None = None) -> Config:
    """Load config from a YAML file, merging over dataclass defaults."""
    cfg = Config()
    if path is None:
        return cfg
    with open(path) as f:
        overrides = yaml.safe_load(f)
    if overrides:
        cfg = _merge_dict_into_dataclass(cfg, overrides)
    return cfg


def resolve_device(cfg: Config) -> torch.device:
    """Auto-select cuda > mps > cpu when device='auto'."""
    spec = cfg.device
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)
