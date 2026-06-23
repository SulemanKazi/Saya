# Example Input Images

## Format specification

| Property | Requirement |
|---|---|
| Format | PNG or JPG |
| Color | Grayscale or RGB (converted internally) |
| Shadow pixels | **White** (value = 255 / 1.0) |
| Background | **Black** (value = 0) |
| Size | Any size; all images resized to `--img-size` (default 256×256) |
| Shape | Square recommended; non-square images are padded before resize |

If your images use the opposite convention (black shadow, white background), pass `--invert` to the training CLI.

## Generating example images

Run once to create all example PNGs:

```bash
python examples/generate_examples.py
```

This creates:

```
examples/
├── two_view/
│   ├── shadow_0.png   # circle silhouette
│   └── shadow_1.png   # square silhouette
└── three_view/
    ├── shadow_0.png   # triangle
    ├── shadow_1.png   # star
    └── shadow_2.png   # cross
```

## Quick-start training commands

**Two-view (CPU, fast test):**
```bash
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/cpu_fast.yaml \
    --export-mesh
```

**Two-view (GPU, full quality):**
```bash
python train.py \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --config configs/default.yaml \
    --device cuda \
    --export-mesh
```

**Three-view:**
```bash
python train.py \
    --images examples/three_view/shadow_0.png \
             examples/three_view/shadow_1.png \
             examples/three_view/shadow_2.png \
    --config configs/default.yaml \
    --device cuda \
    --export-mesh
```

**Custom images:**
```bash
python train.py \
    --images /path/to/front_shadow.png /path/to/side_shadow.png \
    --epochs 30 \
    --export-mesh
```

## Using the visualization tool

After training, compare predicted shadows with targets:

```bash
python visualize.py \
    --checkpoint output/checkpoints/epoch_0030.pt \
    --images examples/two_view/shadow_0.png examples/two_view/shadow_1.png \
    --save-comparison \
    --iou-report
```

Outputs side-by-side PNG comparisons (target | predicted | diff) in `output/viz/`.
