from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config, ModelConfig


class PositionalEncoding(nn.Module):
    """NeRF-style positional encoding.

    Maps (N, 3) coordinates to (N, 3*(1 + 2*L)) by appending
    sin(2^k * pi * x) and cos(2^k * pi * x) for k=0..L-1 per coordinate.
    With L=6 the output dim is 3 + 6*2*3 = 39.
    """

    def __init__(self, n_levels: int = 6):
        super().__init__()
        self.n_levels = n_levels
        freqs = 2.0 ** torch.arange(n_levels)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        return 3 * (1 + 2 * self.n_levels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., 3)
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(math.pi * freq * x))
            encoded.append(torch.cos(math.pi * freq * x))
        return torch.cat(encoded, dim=-1)


class OccupancyMLP(nn.Module):
    """8-layer MLP that maps positionally-encoded 3D coordinates to occupancy in [0,1]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pos_enc = PositionalEncoding(cfg.pos_enc_levels)
        in_dim = self.pos_enc.output_dim
        layers: list[nn.Module] = []
        for i in range(cfg.n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.hidden_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3) raw coordinates → (N, 1) occupancy in [0, 1]
        return self.net(self.pos_enc(x))


class ShadowArtModel(nn.Module):
    """Top-level model: MLP + jointly optimized light directions and screen normals.

    All parameters are in a single state_dict so checkpoints are complete.
    """

    def __init__(self, cfg: Config, n_views: int):
        super().__init__()
        self.mlp = OccupancyMLP(cfg.model)

        # Initialize light directions uniformly on the sphere.
        # For n_views=2 this gives roughly opposite directions; for more views
        # they are random. screen_normals start aligned with light directions.
        init_dirs = F.normalize(torch.randn(n_views, 3), dim=-1)
        self.light_dirs = nn.Parameter(init_dirs.clone())
        self.screen_normals = nn.Parameter(init_dirs.clone())

    def get_light_dirs(self) -> Tensor:
        """Normalized light directions, (n_views, 3)."""
        return F.normalize(self.light_dirs, dim=-1)

    def get_screen_normals(self) -> Tensor:
        """Normalized screen normals, (n_views, 3)."""
        return F.normalize(self.screen_normals, dim=-1)

    def occupancy(self, points: Tensor) -> Tensor:
        """Evaluate occupancy at 3D points.

        Args:
            points: (N, 3) coordinates in scene space.
        Returns:
            (N, 1) occupancy values in [0, 1].
        """
        return self.mlp(points)
