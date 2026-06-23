# Saya — Neural Shadow Art

Implementation of the paper:

> **Neural Shadow Art**  
> Caoliwen Wang, Bailin Deng, Juyong Zhang  
> Pacific Graphics 2025 — [arXiv:2411.19161](https://arxiv.org/abs/2411.19161)

Given a set of binary silhouette images (one per view), this system optimizes a neural implicit occupancy field to produce a 3D sculpture whose shadows match the targets. The result is exported as a watertight mesh ready for 3D printing.

See [`neural_shadow_art.md`](neural_shadow_art.md) for a full technical breakdown of the paper.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) (fast Python package manager):

```bash
# Install uv (no pip required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a Python 3.12 venv and install all dependencies
uv venv --python 3.12
uv pip install -e .
```

The project runs on **CPU, CUDA, or Apple MPS** — device is auto-detected.

---

## Quick Start

**1. Generate example input images:**
```bash
python examples/generate_examples.py
```

**2. Train (CPU, fast test):**
```bash
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/cpu_fast.yaml \
    --export-mesh
```

**3. Train (GPU, full quality):**
```bash
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/default.yaml \
    --device cuda \
    --export-mesh
```

**4. Visualize results:**
```bash
python visualize.py \
    --checkpoint output/checkpoints/epoch_0030.pt \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --save-comparison --iou-report
```

---

## Custom Images

Provide any PNG/JPG binary silhouettes:

```bash
python train.py \
    --images front_view.png side_view.png top_view.png \
    --epochs 30 --export-mesh
```

- White pixels = shadow (cast by object), black = background  
- Pass `--invert` if your convention is reversed  
- Images are auto-thresholded (Otsu) and resized — no manual pre-processing needed  
- Add `--use-registration` if input silhouettes are geometrically incompatible

---

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--images` | required | Shadow target images (one per view) |
| `--config` | — | YAML config file (see `configs/`) |
| `--epochs` | 30 | Training epochs |
| `--device` | auto | `auto` \| `cuda` \| `mps` \| `cpu` |
| `--export-mesh` | off | Run Marching Cubes and export mesh |
| `--mesh-format` | stl | `stl` \| `obj` \| `ply` |
| `--use-registration` | off | Enable rigid registration for incompatible silhouettes |
| `--resume` | — | Resume training from a checkpoint |

---

## Project Structure

```
Saya/
├── src/neural_shadow_art/
│   ├── config.py          # All hyperparameters (dataclasses + YAML loader)
│   ├── model.py           # Positional encoding + 8-layer MLP + ShadowArtModel
│   ├── dataset.py         # Image loading + Otsu binarization
│   ├── renderer.py        # Parallel-projection ray generation + differentiable rendering
│   ├── losses.py          # 5 loss functions + LossScheduler
│   ├── trainer.py         # Training loop + checkpointing
│   ├── mesh_export.py     # Marching Cubes mesh export
│   └── registration.py   # Optional rigid registration for incompatible silhouettes
├── train.py               # Training CLI
├── visualize.py           # Visualization + IoU/Dice evaluation CLI
├── configs/
│   ├── default.yaml       # Full-quality settings (GPU recommended)
│   └── cpu_fast.yaml      # Reduced settings for CPU testing
├── examples/              # Synthetic example inputs
└── tests/                 # Unit + integration tests
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

The slow end-to-end test requires the `-m slow` mark to run:

```bash
uv run pytest tests/ -v -m slow
```

---

## Expected Performance

| Device | Epochs | Approximate time |
|---|---|---|
| NVIDIA RTX 3090 | 30 | ~1.5 hours (paper) |
| Any modern GPU | 30 | 1–3 hours |
| CPU (full config) | 30 | Several hours |
| CPU (`cpu_fast.yaml`) | 5 | 5–15 minutes |
