# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Neural Shadow Art — optimizes a neural implicit occupancy field so its rendered shadows match target binary silhouette images, then exports the result as a printable mesh. Implementation of Wang et al., *Pacific Graphics 2025* ([arXiv:2411.19161](https://arxiv.org/abs/2411.19161)).

## Environment

A virtual environment is at `.venv/` (Python 3.14, created because Python 3.12 is unavailable on this machine). `pyproject.toml` has been relaxed to `requires-python = ">=3.12"`.

```bash
source .venv/bin/activate
```

## Commands

```bash
# Tests (fast — all except @pytest.mark.slow)
pytest tests/ -v

# Full end-to-end sanity test (runs a tiny training loop)
pytest tests/ -v -m slow

# Single test file
pytest tests/test_losses.py -v

# Generate synthetic example inputs
python examples/generate_examples.py

# Train — CPU quick test
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/cpu_fast.yaml --export-mesh

# Train — GPU full quality
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/default.yaml --device cuda --export-mesh

# Visualize / evaluate a checkpoint
python visualize.py \
    --checkpoint output/checkpoints/epoch_0030.pt \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --save-comparison --iou-report
```

## Architecture

The training pipeline is: **dataset → ray generation → differentiable rendering → losses → optimizer step → (mesh export)**.

### Data flow

| Module | Role |
|---|---|
| `dataset.py` | Loads PNG/JPG silhouettes, Otsu-binarizes them, resizes to `img_size`. Returns per-view binary masks and shadow area ratios (used to reweight `L_ren`). |
| `model.py` | `OccupancyMLP` — NeRF-style positional encoding → 8-layer ReLU MLP → sigmoid → scalar in [0,1]. `ShadowArtModel` wraps the MLP and adds `light_dirs` and `screen_normals` as learnable `nn.Parameter`s (jointly optimized with the geometry). |
| `renderer.py` | `RayGenerator` builds parallel-projection ray bundles: screen pixels (u,v) ∈ [−1,1]² map to 3D positions on a screen plane placed beyond the scene in the light direction; rays travel in −light_dir. `DifferentiableRenderer` stratified-samples K points per ray and aggregates occupancy as O = 1 − ∏(1 − f_k) in log-space. |
| `losses.py` | Five terms: `L_ren` (MSE on shadow map), `L_coh` (penalizes occupancy jumps along rays), `L_smo` (surface normal consistency via second-order gradients), `L_vol` (minimize occupied volume), `L_bin` (push occupancy to 0/1). `LossScheduler` manages their scheduling. |
| `trainer.py` | Adam with **three parameter groups**: MLP weights (`lr`), light+screen directions (`lr_light`, 10× faster), optional registration transforms (`lr=1e-3`). Steps per epoch = `img_size² × n_views / batch_size_rays` to cover all pixels ~once. |
| `mesh_export.py` | Marching Cubes on a `grid_resolution³` grid at `iso_threshold=0.5`, exported via trimesh. |
| `registration.py` | Optional rigid alignment of incompatible silhouettes, enabled with `--use-registration`. |
| `config.py` | Typed dataclasses (`ModelConfig`, `RenderConfig`, `LossConfig`, `TrainConfig`, `MeshConfig`). `load_config(path)` merges YAML over defaults; unknown keys raise `ValueError`. |

### Key non-obvious design points

- **Light directions and screen normals are learnable** — the model discovers both the 3D shape *and* the light/view configuration simultaneously, not from fixed camera positions.
- **Frustum truncation** (`cfg.render.frustum_truncation`) clips ray intervals to only the region visible from all other views, preventing geometry from growing in unseen regions that would corrupt shadow predictions.
- **`L_smo` requires second-order gradients** (it calls `torch.autograd.grad(..., create_graph=True)` then backpropagates through those gradients). This is why `cpu_fast.yaml` sets `beta_smo: 0.0` — it's numerically unstable on small CPU models.
- **Loss scheduling** (`LossScheduler`): cohesion and binarization weights ramp as `2^min(epoch, 3)` for the first 3 epochs; smoothness and volume are zeroed until epoch 4.
- **Checkpoint payload** stores `model_state`, `optimizer_state`, `n_views`, and the full `Config` object so training can be resumed exactly with `--resume`.

### Config system

YAML files in `configs/` override dataclass defaults field-by-field. CLI flags (e.g., `--epochs`, `--device`) take final precedence. To add a new hyperparameter, add it to the relevant dataclass in `config.py` and optionally expose it in `train.py`'s argparse.
