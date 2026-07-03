# Improving 3D-Printability of Neural Shadow Art Output

**Problem:** Trained models produce dispersed, disconnected blobs instead of a single
continuous sculpture. Unsuitable for 3D printing even with supports.

**Root cause:** Nothing in the current objective rewards 3D connectivity, and one term
actively fights it:

- `L_coh` (`losses.py`) penalizes occupancy jumps **along each ray** only — it makes
  individual blobs solid but says nothing about connectivity *across* rays.
- `L_vol` penalizes total volume. Connective material (necks, bridges) adds volume
  without improving any shadow, so the optimizer's cheapest solution is the minimal
  set of disconnected blobs that covers the silhouettes.
- `L_bin` then freezes the fragmented configuration into hard 0/1 before blobs can merge.

**Key structural fact (makes this fixable with a guarantee):** the rendered shadow is a
union, `O = 1 − ∏(1 − f_k)`. Adding material can only *darken* pixels, never lighten
them. Therefore any voxel inside the **visual hull of the targets** — the region that
projects into the dark part of the target silhouette in *every* view — can be filled
freely without changing any shadow. (This is the core insight of Mitra & Pauly,
*Shadow Art*, SIGGRAPH Asia 2009.) All three options below exploit or respect this.

**Caveat — don't try hull-full initialization:** initializing the field full at the
visual hull and carving down does *not* work with this renderer. The rendering gradient
at sample k is ∏_{j≠k}(1−f_j), which is zero whenever any other sample on the ray
saturates at 1 — a full field can't train (mirror image of the `init_bias = −6`
requirement documented in CLAUDE.md).

---

## Option 1 — Post-process strut routing at export time

**Status: ✅ IMPLEMENTED (2026-07-03).** `connect_grid_components` +
`compute_target_hull` in `mesh_export.py`, `RayGenerator.points_in_target_hull` in
`renderer.py`, config knobs in `MeshConfig`, wired into `train.py` / `visualize.py`,
tests in `tests/test_mesh_export.py`. On the `two_view_2` epoch-30 checkpoint:
69 components → 8 pieces with zero change in false-positive shadow pixels. The
remaining 8 pieces are a *proven* limit for those targets — the visual hull itself has
8 disconnected islands, so no shadow-preserving connection between them exists (the
exporter merges each island into one piece and warns about the rest; a real
installation hangs such pieces on threads). One learning vs. the plan below: merging
must run **per hull island**, not from the single largest component — otherwise all
islands except the first stay fragmented internally.

### Idea

After evaluating the occupancy grid in `mesh_export.py`, detect disconnected
components and connect them with struts routed entirely through the target visual
hull. Every added voxel projects inside the target silhouette in every view, so the
shadows are provably unchanged.

### Algorithm

1. **Evaluate grid** as today (`evaluate_on_grid`), threshold at `iso_threshold`.
2. **Compute the target visual hull on the same grid.** For each grid point, project
   into each view (same math as `RayGenerator.points_in_frustums`, but instead of
   only checking frustum membership, sample the *target mask* at the projected (u,v)):
   `hull(x) = ∏_views target_v(project_v(x))`. Needs the target masks and the trained
   `light_dirs` / `screen_normals` at export time — pass the dataset into the exporter
   (new parameter) or store masks in the checkpoint.
3. **Label components** of the thresholded occupancy with `scipy.ndimage.label`
   (26-connectivity).
4. **If > 1 component:** build a voxel graph over hull voxels. Edge cost
   `w = 1 − occ + ε` so paths prefer high-occupancy voxels and hug existing geometry.
   - Compute pairwise shortest paths between components (multi-source Dijkstra seeded
     from each component's voxels; `scipy.sparse.csgraph.dijkstra` or a small
     hand-rolled heap on the 26-neighborhood).
   - Take the **minimum spanning tree** over components using those path costs.
   - For each MST edge, mark the voxels along the path.
5. **Dilate each path** to the minimum printable strut diameter (e.g.
   `scipy.ndimage.binary_dilation` with a spherical structuring element of radius
   `r_min / voxel_size`), then **intersect with the hull** again (dilation may leak
   out), then OR into the occupancy grid (set those voxels to 1.0, or to
   `max(occ, 1.0)` pre-threshold).
6. Run Marching Cubes on the repaired grid as before.

### Config additions (`MeshConfig`)

```python
connect_components: bool = True      # enable strut routing
strut_radius_mm: float = 2.0         # physical radius; convert via model_size_mm
model_size_mm: float = 100.0         # physical size of the bbox for unit conversion
```

### Notes / edge cases

- If a component pair has **no hull path** between them (hull itself disconnected),
  fall back: report it, and either keep both (user adds supports) or drop the smaller
  component *only after verifying* every target pixel it covers is also covered by the
  remaining geometry (deleting is NOT shadow-safe in general — deletion lightens pixels).
- Struts are visible on the sculpture but invisible in the shadows. Cosmetic quality
  can be improved by smoothing the strut path (e.g. moving-average of voxel centers)
  before dilation.
- Cheap to test: `trimesh` `mesh.split()` on current exports tells you component
  count/sizes before and after.
- Also fixes the related printing issue where a component touches another only at a
  single voxel corner (26-connectivity artifact): use 6-connectivity in `label` so
  corner-touching counts as disconnected and gets a proper strut.

### Effort

Contained to `mesh_export.py` + plumbing target masks into the exporter. ~150 lines.
Testable in isolation with a synthetic two-blob grid.

---

## Option 2 — Differentiable connectivity loss during training

**Status: fixes the problem at the source; produces organic connections instead of
visible struts. Requires retraining.**

### Idea

Add `L_con`: a soft flood-fill reachability loss on a coarse occupancy grid. Occupied
mass that is not reachable from the main body gets penalized — stragglers must either
connect or vanish.

### Algorithm

Every `connectivity_every` steps (this is a global-grid loss, so it does not need to
run every batch):

1. Sample the field on a coarse grid `G` (32³–48³ is enough; forward pass only needs
   ~100k points, batched like `evaluate_on_grid`). Keep the autograd graph.
2. Seed reachability at the most-occupied voxel:
   `r₀ = 1` at `argmax(f)`, else 0. (Detach the argmax choice.)
3. Iterate a **soft flood fill**: `r ← maxpool3d(r, kernel=3, stride=1, pad=1) · f`,
   repeated `R ≈ grid diameter` times (e.g. R = 48 for a 48³ grid). Each step lets
   reachability spread one voxel through occupied space; multiplying by `f` gates the
   spread by occupancy, keeping everything differentiable.
4. Loss = occupied-but-unreachable fraction:

   ```
   L_con = Σ f · (1 − r) / (Σ f + ε)
   ```

Gradient behavior: for a disconnected blob, `∂L/∂f` is positive on the blob (shrink)
and negative along the highest-occupancy corridor toward the main body (grow a bridge)
— because raising `f` along the corridor raises `r` over the whole blob.

### Integration points

- `losses.py`: new `loss_connectivity(model, cfg, device)` — unlike the other losses
  it queries the model on its own grid rather than reusing batch samples.
- `LossConfig`: `beta_con: float = 0.01`, `con_grid: int = 48`,
  `con_every: int = 20` (steps).
- `LossScheduler.get_weights`: gate like `smo`/`vol` — zero for epochs 0–2, active
  from epoch 3 (the shape must exist before connectivity is meaningful). Consider
  ramping it *after* `L_vol` has done its shrinking, e.g. from epoch 5.
- `trainer.py`: add the term every `con_every` steps (scale by `con_every` to keep the
  effective weight comparable, or just tune `beta_con`).

### Tuning notes

- If `beta_con` is too strong too early it can collapse everything into one blob near
  the seed and hurt silhouette IoU; start at 1e-2 and watch `L_ren`.
- Seed stability: argmax can jump between blobs across steps. More stable: seed the
  entire largest connected component (compute components on the detached thresholded
  grid, seed `r₀ = 1` on all its voxels).
- Memory: R iterations of maxpool on 48³ with graph retained is ~R·48³ floats ≈ 40 MB
  — fine on GPU, acceptable on CPU at 32³.
- Optional targeted variant (cruder, cheaper): detect components periodically, sample
  points along hull-routed segments between component pairs, and apply BCE-toward-1 on
  the field at those points. Very direct, but produces strut-like bridges — at that
  point Option 1 is simpler.

### Effort

~100 lines in `losses.py` + scheduler/config/trainer plumbing. Needs a unit test:
two synthetic Gaussian blobs → loss decreases as a bridge is added.

---

## Option 3 — Rebalance the existing losses

**Status: cheapest experiment, try immediately. Reduces fragmentation but guarantees
nothing.**

### Rationale

`L_vol` is the main force amputating necks between blobs; the fast `L_bin` ramp
(2^min(epoch,3)) locks the amputation in before geometry can merge.

### Concrete changes (all YAML-only, no code)

```yaml
loss:
  beta_vol: 0.00001    # 10× down from 1e-4; or delay onset (needs 1-line scheduler change)
  beta_coh: 0.003      # 3× up — stronger along-ray solidity
  beta_bin: 0.02       # slower push to hard 0/1
```

Optional 1-line code change in `LossScheduler.get_weights`: make the vol gate epoch
configurable (`vol_start_epoch`, default 3 → try 8) so volume minimization only starts
after the shape has connected on its own.

### What to measure

- Component count and volume distribution of the exported mesh:
  `len(trimesh.load(path).split())` — track across runs.
- Silhouette IoU (via `visualize.py --iou-report`) to confirm shadow quality isn't
  sacrificed.

### Expected outcome

Often turns "many blobs" into "few blobs"; rarely into exactly one. Use as a
complement to Options 1/2, not a substitute.

---

## Suggested order of implementation

1. **Option 3** — config tweak, run today, establishes a baseline component count.
2. **Option 1** — guaranteed-connected output from any checkpoint, including old ones.
3. **Option 2** — for final quality: organic connections the optimizer shapes itself,
   with Option 1 kept as a safety net at export.

## Extra printability items (later, separate from connectivity)

- **Minimum feature size:** morphological closing then opening at ~nozzle diameter on
  the export grid, intersected with the visual hull (closing adds material → must stay
  in hull; opening removes → verify shadow coverage afterward).
- **Thin-shell / single-voxel walls:** a min-thickness loss on along-ray occupied
  segment lengths (the per-sample ω weights in `RenderResult` already give segment
  lengths) — penalize occupied runs shorter than a threshold.
- **Watertightness:** already handled (`trimesh.repair.fill_holes`), but check
  `mesh.is_watertight` after strut fusion and re-run repair.
