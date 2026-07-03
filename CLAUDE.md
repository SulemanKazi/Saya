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
| `renderer.py` | `RayGenerator` builds parallel-projection ray bundles: screen pixels (u,v) ∈ [−1,1]² map to 3D positions on a screen plane beyond the scene in the light direction; the image spans exactly the normalized cube (half-extent 0.5, paper Eq. 4) so every pixel is reachable by in-bbox geometry. Rays travel in −light_dir. `DifferentiableRenderer` stratified-samples K points per ray (default K = image width, paper n = w) and returns `RenderResult` (aggregated occupancy O = 1 − ∏(1 − f_k) in log-space, per-sample occupancies, sample positions, trapezoid segment weights ω per Eq. 16). |
| `losses.py` | Five terms: `L_ren` (MSE on shadow map, weighted by the max image/shadow-bbox area ratio α), `L_coh` (penalizes occupancy jumps along rays), `L_smo` (surface normal consistency; gradients estimated via least-squares finite differences over k₁=26 sample neighbors, Eqs. 11–14 — **not** autograd, which the paper notes is unstable), `L_vol` (ω-weighted soft volume integral, Eqs. 15–16), `L_bin` (push occupancy to 0/1). `LossScheduler` manages their scheduling. |
| `trainer.py` | Adam with **two parameter groups**: MLP weights (`lr`) and light+screen directions (`lr_light`, 10× faster). Steps per epoch = `img_size² × n_views / batch_size_rays` to cover all pixels ~once. When registration is enabled, targets are re-registered by ICP every `registration_every` epochs. |
| `mesh_export.py` | Marching Cubes on a `grid_resolution³` grid at `iso_threshold=0.5`, exported via trimesh. The occupancy grid is truncated to the intersection of all view frustums first (paper Sec. 3.4), so stray geometry cannot cast shadows outside the targets. Before Marching Cubes, disconnected components are joined for 3D-printability by struts Dijkstra-routed through the *target visual hull* (`connect_grid_components`) — provably shadow-safe because the render is a union (adding material only darkens already-dark pixels). Components in different hull islands cannot be joined without changing shadows and are left as separate pieces. Requires target masks at export (`export(..., target_masks=…)`); config knobs in `MeshConfig` (`connect_components`, `strut_radius_mm`, `model_size_mm`, `hull_erosion_px`). |
| `registration.py` | Optional ICP alignment of incompatible silhouettes (paper Sec. 3.3), enabled with `--use-registration`: every 5 epochs the target boundary point cloud is rigidly registered to the rendered shadow boundary and the training targets are re-warped from the originals. Transforms are buffers (checkpointed state), not learnable parameters. |
| `config.py` | Typed dataclasses (`ModelConfig`, `RenderConfig`, `LossConfig`, `TrainConfig`, `MeshConfig`). `load_config(path)` merges YAML over defaults; unknown keys raise `ValueError`. |

### Key non-obvious design points

- **Light directions and screen normals are learnable** — the model discovers both the 3D shape *and* the light/view configuration simultaneously. Initial directions default to axis-aligned (override with `--light-dirs x,y,z ...`); screen normals initialize to −light_dir (paper convention ⟨l, s⟩ < 0).
- **Frustum truncation** (`cfg.render.frustum_truncation`) clips ray intervals to only the region visible from all other views, preventing geometry from growing in unseen regions that would corrupt shadow predictions. The same truncation is applied to the occupancy grid at mesh-export time. The screen-coordinate intercept is `u = (x−c)·e − ((x−c)·s)·(l·e)/⟨l,s⟩` — the minus sign on the second term matters only for oblique light/screen pairs, so a sign bug there is invisible in perpendicular-view tests (see `test_frustum_truncation_oblique_no_overclip`).
- **`L_smo` uses least-squares finite-difference gradients** over batch ray samples (paper Eqs. 12–13), not `torch.autograd.grad` — the paper explicitly calls network backprop unstable for this. The surface threshold is θ·w (θ=`grad_threshold`, w=image width). It is stable on CPU and enabled in `cpu_fast.yaml`.
- **Loss scheduling** (`LossScheduler`): cohesion and binarization weights ramp as `2^min(epoch, 3)` for the first 3 epochs; smoothness and volume are zeroed until epoch 4.
- **Checkpoint payload** stores `model_state`, `optimizer_state`, `n_views`, registration state, and the full `Config` object; `--resume` uses the stored config unless `--config` is passed explicitly.
- **NaN-safe slab intersection**: ray/bbox division clamps near-zero direction components *before* dividing. A `torch.where(cond, 1/d, big)` still evaluates `1/0 = inf` in the forward pass and its backward emits `0·inf = NaN` into the light-direction gradients — this bites immediately with axis-aligned initial lights.
- **The occupancy field must initialize near-empty** (`model.init_bias`, default −6). With O = 1 − ∏(1 − f_k), a field starting at f ≈ 0.5 saturates every ray (O ≈ 1) and the per-sample rendering gradient ∏_{j≠k}(1 − f_j) = 0.5^(K−1) vanishes — it underflows float32 at the paper's n = w = 256 — while the ramping binarization loss freezes the field at all-ones. The paper doesn't state this, but its formulation cannot train without it.

### Config system

YAML files in `configs/` override dataclass defaults field-by-field. CLI flags (e.g., `--epochs`, `--device`) take final precedence. To add a new hyperparameter, add it to the relevant dataclass in `config.py` and optionally expose it in `train.py`'s argparse.
