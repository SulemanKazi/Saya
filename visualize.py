#!/usr/bin/env python
"""Neural Shadow Art — visualization and evaluation CLI.

Re-renders shadow maps from a trained checkpoint and compares with targets.

Example:
    python visualize.py \\
        --checkpoint output/checkpoints/epoch_0030.pt \\
        --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \\
        --iou-report --save-comparison
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.neural_shadow_art.config import load_config, resolve_device
from src.neural_shadow_art.dataset import ShadowDataset
from src.neural_shadow_art.mesh_export import MarchingCubesMeshExporter
from src.neural_shadow_art.renderer import DifferentiableRenderer, RayGenerator
from src.neural_shadow_art.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize shadow art results from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Path to a saved checkpoint (.pt file)")
    p.add_argument("--images", nargs="+", required=True, metavar="PATH",
                   help="Target shadow images (same order as during training)")
    p.add_argument("--output-dir", default="./output/viz", metavar="DIR")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--samples-per-ray", type=int, default=None,
                   help="Samples along each ray (default: image width, n = w)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    p.add_argument("--iou-report", action="store_true",
                   help="Print IoU and Dice score for each view")
    p.add_argument("--save-comparison", action="store_true",
                   help="Save side-by-side [target | predicted | diff] images")
    p.add_argument("--export-mesh", action="store_true",
                   help="Run Marching Cubes on the checkpoint and export a mesh")
    p.add_argument("--mesh-format", choices=["stl", "obj", "ply"], default=None,
                   help="Output mesh format (default: from config)")
    p.add_argument("--invert", action="store_true")
    return p.parse_args()


def compute_iou_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
    p = pred >= threshold
    t = target >= threshold
    intersection = (p & t).sum()
    union = (p | t).sum()
    iou = intersection / (union + 1e-7)
    dice = 2 * intersection / (p.sum() + t.sum() + 1e-7)
    return float(iou), float(dice)


def save_comparison(
    target: np.ndarray,
    predicted: np.ndarray,
    view_idx: int,
    output_dir: str,
) -> None:
    diff = predicted - target
    false_pos = np.clip(diff, 0, 1)
    false_neg = np.clip(-diff, 0, 1)
    diff_rgb = np.stack([
        false_pos,                            # red channel: false positives
        np.zeros_like(diff),
        false_neg,                            # blue channel: false negatives
    ], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [target, predicted, diff_rgb],
        ["Target", "Predicted", "Diff (red=FP, blue=FN)"],
    ):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img.clip(0, 1))
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    path = os.path.join(output_dir, f"view_{view_idx:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison saved → {path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint and config
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config")
    if cfg is None:
        cfg = load_config()

    if args.device is not None:
        cfg.device = args.device
    if args.mesh_format is not None:
        cfg.mesh.output_format = args.mesh_format
    device = resolve_device(cfg)

    n_samples = args.samples_per_ray or cfg.render.n_samples_per_ray or args.img_size

    model, epoch, _ = Trainer.load_checkpoint(args.checkpoint, cfg)
    model.to(device)
    model.eval()

    dataset = ShadowDataset(args.images, img_size=args.img_size, invert=args.invert)

    ray_gen = RayGenerator(cfg)
    renderer = DifferentiableRenderer(cfg)

    print(f"Rendering {dataset.n_views} view(s) from epoch {epoch} checkpoint …")
    predicted_maps = renderer.render_all_views(
        model, ray_gen, dataset.n_views, dataset.img_size, n_samples
    )

    if args.iou_report:
        print(f"\n{'View':<6} {'IoU':>8} {'Dice':>8}")
        print("-" * 24)

    for i in range(dataset.n_views):
        pred_np = predicted_maps[i].cpu().numpy()
        target_np = dataset[i]["mask"].numpy()

        if args.iou_report:
            iou, dice = compute_iou_dice(pred_np, target_np)
            print(f"{i:<6} {iou:>8.4f} {dice:>8.4f}")

        if args.save_comparison:
            save_comparison(target_np, pred_np, i, args.output_dir)

    if args.iou_report:
        print()

    if args.export_mesh:
        exporter = MarchingCubesMeshExporter(cfg)
        mesh_path = os.path.join(args.output_dir, "sculpture." + cfg.mesh.output_format)
        exporter.export(model, device, mesh_path)


if __name__ == "__main__":
    main()
