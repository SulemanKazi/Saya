from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config, ModelConfig


class PositionalEncoding(nn.Module):
    """Positional encoding per paper Eq. 2.

    Maps (N, 3) coordinates to (N, 3*(1 + 2*L)) by appending
    sin(2^k * x) and cos(2^k * x) for k=0..L-1 per coordinate.
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
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class OccupancyMLP(nn.Module):
    """MLP that maps positionally-encoded 3D coordinates to occupancy in [0,1].

    n_layers counts all fully connected layers including the final sigmoid
    output layer (paper: 8 layers, the last with sigmoid instead of ReLU).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_layers < 2:
            raise ValueError("n_layers must be >= 2 (hidden + output)")
        self.pos_enc = PositionalEncoding(cfg.pos_enc_levels)
        in_dim = self.pos_enc.output_dim
        layers: list[nn.Module] = []
        for i in range(cfg.n_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.hidden_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
        out_layer = nn.Linear(cfg.hidden_dim, 1)
        # Start with a near-empty field (f ≈ sigmoid(init_bias) everywhere).
        # With O = 1 − ∏(1 − f_k), a field initialized at f ≈ 0.5 saturates
        # every ray (O ≈ 1) and the per-sample gradient ∏_{j≠k}(1 − f_j)
        # vanishes — for n = w = 256 samples it underflows float32 outright,
        # freezing training. Near-empty keeps ∏(1 − f_j) ≈ 1 so shadow
        # constraints can grow geometry.
        nn.init.normal_(out_layer.weight, std=1e-2)
        nn.init.constant_(out_layer.bias, cfg.init_bias)
        layers.append(out_layer)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3) raw coordinates → (N, 1) occupancy in [0, 1]
        return self.net(self.pos_enc(x))


class ShadowArtModel(nn.Module):
    """Top-level model: MLP + jointly optimized light directions and screen normals.

    All parameters are in a single state_dict so checkpoints are complete.
    """

    # Default initial light directions (direction of travel, source → scene),
    # used when the user does not supply --light-dirs. Axis-aligned defaults
    # match the perpendicular configurations used as starting points in the paper.
    _DEFAULT_DIRS = [
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]

    def __init__(self, cfg: Config, n_views: int, init_light_dirs: Tensor | None = None):
        super().__init__()
        self.mlp = OccupancyMLP(cfg.model)

        if init_light_dirs is not None:
            if init_light_dirs.shape != (n_views, 3):
                raise ValueError(
                    f"init_light_dirs must have shape ({n_views}, 3), "
                    f"got {tuple(init_light_dirs.shape)}"
                )
            init_dirs = F.normalize(init_light_dirs.float(), dim=-1)
        elif n_views <= len(self._DEFAULT_DIRS):
            init_dirs = torch.tensor(self._DEFAULT_DIRS[:n_views])
        else:
            init_dirs = F.normalize(torch.randn(n_views, 3), dim=-1)

        self.light_dirs = nn.Parameter(init_dirs.clone())
        # Screen normals point back toward the object (paper: ⟨l_i, s_i⟩ < 0);
        # perpendicular light–screen configuration initially.
        self.screen_normals = nn.Parameter(-init_dirs.clone())

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
