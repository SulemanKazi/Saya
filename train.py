#!/usr/bin/env python
"""Neural Shadow Art — training CLI.

Minimal example:
    python train.py --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png

With a GPU and full settings:
    python train.py \\
        --images shadow_front.png shadow_side.png \\
        --config configs/default.yaml \\
        --epochs 30 --device cuda --export-mesh
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from src.neural_shadow_art.config import Config, load_config, resolve_device
from src.neural_shadow_art.dataset import ShadowDataset
from src.neural_shadow_art.mesh_export import MarchingCubesMeshExporter
from src.neural_shadow_art.model import ShadowArtModel
from src.neural_shadow_art.registration import RigidRegistration
from src.neural_shadow_art.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Neural Shadow Art sculpture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Paths to binary shadow target images (one per view). "
             "White pixels = shadow; black = background.",
    )

    # Output
    p.add_argument("--output-dir", default="./output", metavar="DIR")
    p.add_argument(
        "--config", default=None, metavar="YAML",
        help="Path to a YAML config file. CLI flags override YAML values.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="MLP learning rate")
    p.add_argument("--lr-light", type=float, default=None,
                   help="Learning rate for light/screen parameters")
    p.add_argument("--batch-size-rays", type=int, default=None,
                   help="Total rays per gradient step")
    p.add_argument("--seed", type=int, default=None)

    # Rendering
    p.add_argument("--img-size", type=int, default=None,
                   help="Resize all input images to this square size")
    p.add_argument("--samples-per-ray", type=int, default=None,
                   help="Stratified samples along each ray "
                        "(default: image width, i.e. n = w as in the paper)")
    p.add_argument("--no-frustum-truncation", action="store_true",
                   help="Disable ray frustum truncation (ablation mode)")
    p.add_argument("--light-dirs", nargs="+", default=None, metavar="X,Y,Z",
                   help="Initial light directions (direction of travel), one "
                        "comma-separated triple per view, e.g. "
                        "--light-dirs 0,0,-1 -1,0,0. Default: axis-aligned.")

    # Registration
    p.add_argument("--use-registration", action="store_true",
                   help="Enable rigid registration for incompatible silhouettes")

    # Device
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)

    # Mesh export
    p.add_argument("--export-mesh", action="store_true",
                   help="Run Marching Cubes after training and export a mesh")
    p.add_argument(
        "--mesh-format", choices=["stl", "obj", "ply"], default=None,
        help="Output mesh format (default: from config)",
    )

    # Resume
    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="Path to a checkpoint to resume training from")

    # Image convention
    p.add_argument("--invert", action="store_true",
                   help="Invert images (black=shadow, white=background)")

    return p.parse_args()


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Overwrite config fields with any CLI flags the user explicitly set."""
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.lr_light is not None:
        cfg.train.lr_light = args.lr_light
    if args.batch_size_rays is not None:
        cfg.train.batch_size_rays = args.batch_size_rays
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.samples_per_ray is not None:
        cfg.render.n_samples_per_ray = args.samples_per_ray
    if args.no_frustum_truncation:
        cfg.render.frustum_truncation = False
    if args.use_registration:
        cfg.train.use_registration = True
    if args.device is not None:
        cfg.device = args.device
    if args.mesh_format is not None:
        cfg.mesh.output_format = args.mesh_format
    return cfg


def parse_light_dirs(specs: list[str] | None, n_views: int) -> torch.Tensor | None:
    """Parse --light-dirs values ('x,y,z' per view) into an (n_views, 3) tensor."""
    if specs is None:
        return None
    if len(specs) != n_views:
        raise SystemExit(
            f"--light-dirs needs one x,y,z triple per view "
            f"({n_views} views, got {len(specs)})"
        )
    try:
        dirs = [[float(c) for c in s.split(",")] for s in specs]
    except ValueError:
        raise SystemExit(f"Could not parse --light-dirs {specs}")
    if any(len(d) != 3 for d in dirs):
        raise SystemExit("Each --light-dirs entry must have exactly 3 components")
    return torch.tensor(dirs, dtype=torch.float32)


def main() -> None:
    args = parse_args()

    # --- Config ---
    payload = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)

    if args.config is not None:
        cfg = load_config(args.config)
    elif payload is not None and payload.get("config") is not None:
        # Resume with the exact config stored in the checkpoint
        cfg = payload["config"]
    else:
        cfg = load_config(None)
    cfg = apply_cli_overrides(cfg, args)

    # --- Reproducibility ---
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Dataset ---
    img_size = args.img_size or cfg.train.img_size
    dataset = ShadowDataset(args.images, img_size=img_size, invert=args.invert)
    print(f"Loaded {dataset.n_views} view(s) at {img_size}×{img_size}")

    # Warn if silhouettes may need registration
    if dataset.n_views > 1:
        masks = [dataset[i]["mask"] for i in range(dataset.n_views)]
        if not RigidRegistration.is_compatible(masks):
            print(
                "[WARNING] Input silhouettes have low bounding-box overlap. "
                "Consider using --use-registration."
            )

    # --- Model ---
    start_epoch = 0
    if payload is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        model = ShadowArtModel(cfg, n_views=payload["n_views"])
        model.load_state_dict(payload["model_state"])
        start_epoch = payload["epoch"]
        print(f"  → resuming at epoch {start_epoch}")
    else:
        init_dirs = parse_light_dirs(args.light_dirs, dataset.n_views)
        model = ShadowArtModel(
            cfg, n_views=dataset.n_views, init_light_dirs=init_dirs
        )

    # --- Registration ---
    registration = (
        RigidRegistration(n_views=dataset.n_views)
        if cfg.train.use_registration
        else None
    )
    if (
        registration is not None
        and payload is not None
        and "registration_state" in payload
    ):
        registration.load_state_dict(payload["registration_state"])

    # --- Trainer ---
    trainer = Trainer(
        cfg=cfg,
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        registration=registration,
    )

    if payload is not None and "optimizer_state" in payload:
        trainer.optimizer.load_state_dict(payload["optimizer_state"])

    trainer.train(start_epoch=start_epoch)

    # --- Mesh Export ---
    if args.export_mesh:
        device = resolve_device(cfg)
        exporter = MarchingCubesMeshExporter(cfg)
        mesh_path = os.path.join(args.output_dir, "sculpture." + cfg.mesh.output_format)
        exporter.export(model, device, mesh_path)


if __name__ == "__main__":
    main()
