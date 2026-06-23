from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Tuple

import torch
import yaml


@dataclass
class ModelConfig:
    n_layers: int = 8
    hidden_dim: int = 256
    pos_enc_levels: int = 6


@dataclass
class RenderConfig:
    rays_per_pixel: int = 30
    frustum_truncation: bool = True
    bbox_min: Tuple[float, float, float] = (-0.5, -0.5, -0.5)
    bbox_max: Tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class LossConfig:
    beta_ren: float = 1.0
    beta_coh: float = 0.1
    beta_smo: float = 0.05
    beta_vol: float = 0.01
    beta_bin: float = 0.1
    grad_threshold: float = 0.01
    vol_sigmoid_beta: float = 100.0
    n_vol_samples: int = 512
    n_smo_samples: int = 256


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-4
    lr_light: float = 1e-3
    batch_size_rays: int = 4096
    seed: int = 42
    checkpoint_every: int = 5
    log_every: int = 10
    use_registration: bool = False


@dataclass
class MeshConfig:
    grid_resolution: int = 200
    iso_threshold: float = 0.5
    output_format: str = "stl"
    eval_batch_size: int = 32768


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
