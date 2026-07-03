from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .config import Config, resolve_device
from .dataset import ShadowDataset
from .losses import (
    LossScheduler,
    loss_binarization,
    loss_cohesion,
    loss_rendering,
    loss_smoothness,
    loss_volume,
)
from .mesh_export import MarchingCubesMeshExporter
from .model import ShadowArtModel
from .registration import RigidRegistration
from .renderer import DifferentiableRenderer, RayGenerator


class Trainer:
    """Orchestrates the full Neural Shadow Art training loop.

    Two Adam parameter groups:
      1. MLP weights (lr = cfg.train.lr)
      2. Light directions + screen normals (lr = cfg.train.lr_light)

    When registration is enabled, targets are periodically re-registered to
    the rendered shadows via ICP (paper Sec. 3.3) rather than optimized by
    gradient descent.
    """

    def __init__(
        self,
        cfg: Config,
        model: ShadowArtModel,
        dataset: ShadowDataset,
        output_dir: str,
        registration: Optional[RigidRegistration] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.registration = registration
        self.device = resolve_device(cfg)

        self.model.to(self.device)
        if self.registration is not None:
            self.registration.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = LossScheduler(cfg.loss)
        self.ray_gen = RayGenerator(cfg)
        self.renderer = DifferentiableRenderer(cfg)

        self._bbox_min = torch.tensor(cfg.render.bbox_min, device=self.device)
        self._bbox_max = torch.tensor(cfg.render.bbox_max, device=self.device)

        # Training targets: the (possibly registered) view masks. Registration
        # replaces these periodically; the originals stay in the dataset.
        self.targets = [
            self.dataset[i]["mask"].to(self.device)
            for i in range(self.dataset.n_views)
        ]
        if self.registration is not None and not self.registration.is_identity():
            # Resuming with saved registration state: re-apply the transforms.
            self.targets = [
                self.registration.warp_mask(m, i) for i, m in enumerate(self.targets)
            ]

        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Adam:
        param_groups = [
            {"params": self.model.mlp.parameters(), "lr": self.cfg.train.lr},
            {
                "params": [self.model.light_dirs, self.model.screen_normals],
                "lr": self.cfg.train.lr_light,
            },
        ]
        return torch.optim.Adam(param_groups)

    def _sample_pixel_coords(self, n_pixels: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n_pixels random pixel coordinates in [-1, 1]².

        Returns:
            pixel_coords: (n_pixels, 2)
            pixel_indices: (n_pixels,) flat indices into H*W
        """
        H = W = self.dataset.img_size
        indices = torch.randint(0, H * W, (n_pixels,), device=self.device)
        col = (indices % W).float()
        row = (indices // W).float()
        u = col / (W - 1) * 2.0 - 1.0
        v = row / (H - 1) * 2.0 - 1.0
        coords = torch.stack([u, v], dim=1)
        return coords, indices

    def _train_step(self, epoch: int) -> dict[str, float]:
        self.model.train()

        n_views = self.dataset.n_views
        rays_per_view = max(1, self.cfg.train.batch_size_rays // n_views)
        # Paper: n = w samples per ray (image width)
        n_samples = self.cfg.render.n_samples_per_ray or self.dataset.img_size

        light_dirs = self.model.get_light_dirs()    # (n_views, 3)
        screen_normals = self.model.get_screen_normals()  # (n_views, 3)

        all_pred_occ = []
        all_target_occ = []
        all_results = []

        for view_idx in range(n_views):
            pixel_coords, pixel_indices = self._sample_pixel_coords(rays_per_view)

            bundle = self.ray_gen.generate_rays_for_view(
                light_dirs[view_idx],
                screen_normals[view_idx],
                pixel_coords,
            )

            if self.cfg.render.frustum_truncation and n_views > 1:
                bundle = self.ray_gen.apply_frustum_truncation(
                    bundle,
                    light_dirs.detach(),
                    screen_normals.detach(),
                    view_idx,
                )

            result = self.renderer.render_view(self.model, bundle, n_samples)

            target = self.targets[view_idx].flatten()[pixel_indices]

            all_pred_occ.append(result.pred_occ)
            all_target_occ.append(target)
            all_results.append(result)

        # --- Rendering loss (L_ren, paper Eq. 8) ---
        # α is the maximum image-area / shadow-bbox-area ratio over all inputs,
        # applied uniformly to the whole batch.
        alpha = self.dataset.shadow_area_ratios().max().item()
        l_ren = loss_rendering(
            torch.cat(all_pred_occ), torch.cat(all_target_occ), alpha
        )

        # --- Sample-based losses use only valid (non-truncated-away) rays ---
        sample_occ = torch.cat([r.sample_occ[r.valid] for r in all_results])
        sample_weights = torch.cat([r.sample_weights[r.valid] for r in all_results])
        sample_points = torch.cat([r.sample_points[r.valid] for r in all_results])

        if sample_occ.shape[0] == 0:
            # Every ray in the batch missed the scene (degenerate light
            # configuration) — nothing differentiable to optimize this step.
            zero = 0.0
            return {"total": zero, "ren": zero, "coh": zero,
                    "smo": zero, "vol": zero, "bin": zero}

        # --- Cohesion loss (L_coh, Eq. 10) ---
        l_coh = loss_cohesion(sample_occ)

        # --- Volume + binarization (Eqs. 15-17) ---
        l_vol = loss_volume(sample_occ, sample_weights, self.cfg.loss.vol_sigmoid_beta)
        l_bin = loss_binarization(sample_occ.reshape(-1))

        # --- Smoothness loss (L_smo, Eqs. 11-14) ---
        weights = self.scheduler.get_weights(epoch)
        if weights["smo"] > 0 and sample_occ.shape[0] > 0:
            l_smo = loss_smoothness(
                sample_points,
                sample_occ,
                img_width=self.dataset.img_size,
                theta=self.cfg.loss.grad_threshold,
                k1=self.cfg.loss.smo_k1,
                k2=self.cfg.loss.smo_k2,
                max_candidates=self.cfg.loss.n_smo_samples,
            )
        else:
            l_smo = torch.tensor(0.0, device=self.device)

        loss_terms = {
            "ren": l_ren,
            "coh": l_coh,
            "smo": l_smo,
            "vol": l_vol,
            "bin": l_bin,
        }
        total_loss = self.scheduler.compute_total_loss(epoch, loss_terms)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            **{k: v.item() for k, v in loss_terms.items()},
        }

    def _run_registration(self) -> None:
        """Re-register targets to the current rendered shadows (paper Sec. 3.3)."""
        rendered = self.renderer.render_all_views(
            self.model, self.ray_gen, self.dataset.n_views, self.dataset.img_size
        )
        originals = [
            self.dataset[i]["mask"].to(self.device)
            for i in range(self.dataset.n_views)
        ]
        self.targets = self.registration.update(originals, rendered)
        print("Registration: targets re-registered to rendered shadows (ICP).")

    def train(self, start_epoch: int = 0) -> None:
        """Run the full training loop."""
        cfg = self.cfg.train
        print(f"Training on device: {self.device}")
        print(f"Views: {self.dataset.n_views}  |  Image size: {self.dataset.img_size}")
        print(f"Epochs: {cfg.epochs}  |  Rays/step: {cfg.batch_size_rays}")
        if self.registration is not None:
            print("Rigid registration: enabled")

        for epoch in range(start_epoch, cfg.epochs):
            epoch_losses: list[dict[str, float]] = []
            t0 = time.time()

            # Determine steps per epoch: enough to see all pixels ~once,
            # times steps_per_epoch_mult to add iterations without perturbing
            # the epoch-indexed loss schedule.
            n_pixels_total = self.dataset.img_size ** 2 * self.dataset.n_views
            steps_per_epoch = max(1, n_pixels_total // cfg.batch_size_rays)
            steps_per_epoch *= max(1, cfg.steps_per_epoch_mult)

            with tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                leave=False,
            ) as pbar:
                for step in range(steps_per_epoch):
                    losses = self._train_step(epoch)
                    epoch_losses.append(losses)
                    if (step + 1) % cfg.log_every == 0:
                        pbar.set_postfix(
                            loss=f"{losses['total']:.4f}",
                            ren=f"{losses['ren']:.4f}",
                        )
                    pbar.update(1)

            avg = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0]
            }
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                f"loss={avg['total']:.4f}  ren={avg['ren']:.4f}  "
                f"coh={avg['coh']:.4f}  smo={avg['smo']:.4f}  "
                f"vol={avg['vol']:.4f}  bin={avg['bin']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if (
                self.registration is not None
                and (epoch + 1) % cfg.registration_every == 0
                and epoch + 1 < cfg.epochs
            ):
                self._run_registration()

            if (epoch + 1) % cfg.checkpoint_every == 0 or epoch + 1 == cfg.epochs:
                self.save_checkpoint(epoch + 1)

        print("Training complete.")

    def save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(
            self.output_dir, "checkpoints", f"epoch_{epoch:04d}.pt"
        )
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "n_views": self.dataset.n_views,
            "config": self.cfg,
        }
        if self.registration is not None:
            payload["registration_state"] = self.registration.state_dict()
        torch.save(payload, path)
        print(f"Checkpoint saved → {path}")
        return path

    @staticmethod
    def load_checkpoint(
        path: str, cfg: Config
    ) -> tuple[ShadowArtModel, int, dict]:
        """Restore a model from a checkpoint file.

        Returns:
            (model, epoch, extra) where extra contains optimizer state etc.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        n_views = payload["n_views"]
        model = ShadowArtModel(cfg, n_views)
        model.load_state_dict(payload["model_state"])
        return model, payload["epoch"], payload
