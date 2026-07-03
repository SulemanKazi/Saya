import numpy as np
import torch
import torch.nn.functional as F

from src.neural_shadow_art.config import Config
from src.neural_shadow_art.mesh_export import (
    MarchingCubesMeshExporter,
    connect_grid_components,
)
from src.neural_shadow_art.renderer import RayGenerator


def _label_count(grid, threshold=0.5):
    from scipy import ndimage

    _, n = ndimage.label(grid >= threshold)
    return n


def make_two_blob_grid(R=32):
    """Two solid cubes separated along z."""
    grid = np.zeros((R, R, R), dtype=np.float32)
    grid[12:18, 12:18, 4:10] = 1.0
    grid[12:18, 12:18, 22:28] = 1.0
    return grid


# ---------------------------------------------------------------------------
# connect_grid_components
# ---------------------------------------------------------------------------

def test_connects_two_blobs_into_one_component():
    grid = make_two_blob_grid()
    allowed = np.ones_like(grid, dtype=bool)

    repaired, info = connect_grid_components(grid, allowed, strut_radius_vox=1)

    assert info["n_components_before"] == 2
    assert info["n_components_after"] == 1
    assert _label_count(repaired) == 1
    assert info["n_voxels_added"] > 0
    # Original geometry is never removed.
    assert (repaired[grid >= 0.5] >= 0.5).all()


def test_added_material_stays_inside_allowed_region():
    grid = make_two_blob_grid()
    occ = grid >= 0.5
    # Corridor: a thin tube between the blobs, offset from the straight line.
    allowed = np.zeros_like(grid, dtype=bool)
    allowed[13:16, 20:24, 4:28] = True  # detour through y = 20..24
    allowed[13:16, 12:24, 8:11] = True  # connectors from each blob to the tube
    allowed[13:16, 12:24, 21:24] = True

    repaired, info = connect_grid_components(grid, allowed, strut_radius_vox=2)

    assert info["n_components_after"] == 1
    added = (repaired >= 0.5) & ~occ
    assert added.any()
    assert (allowed | occ)[added].all()


def test_unreachable_component_is_reported_not_forced():
    grid = make_two_blob_grid()
    # No corridor at all: material may only exist where it already does.
    allowed = np.zeros_like(grid, dtype=bool)

    repaired, info = connect_grid_components(grid, allowed, strut_radius_vox=1)

    assert info["n_components_before"] == 2
    assert info["n_components_after"] == 2
    assert info["n_unreachable"] == 1
    np.testing.assert_array_equal(repaired, grid)


def test_single_component_is_untouched():
    grid = np.zeros((16, 16, 16), dtype=np.float32)
    grid[4:12, 4:12, 4:12] = 1.0

    repaired, info = connect_grid_components(
        grid, np.ones_like(grid, dtype=bool), strut_radius_vox=1
    )

    assert info["n_components_before"] == 1
    assert info["n_voxels_added"] == 0
    np.testing.assert_array_equal(repaired, grid)


def test_corner_touching_blobs_count_as_disconnected():
    # Two voxels sharing only a corner: 6-connectivity sees two components,
    # which matches what a printed part can actually hold together.
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    grid[2, 2, 2] = 1.0
    grid[3, 3, 3] = 1.0

    repaired, info = connect_grid_components(
        grid, np.ones_like(grid, dtype=bool), strut_radius_vox=1
    )

    assert info["n_components_before"] == 2
    assert info["n_components_after"] == 1


def test_each_hull_island_is_merged_independently():
    # Two islands of the allowed region, two blobs in each: the result must be
    # exactly two pieces — merged within islands, never across (crossing would
    # change a shadow).
    grid = np.zeros((16, 16, 32), dtype=np.float32)
    grid[6:10, 6:10, 2:5] = 1.0    # island A, blob 1
    grid[6:10, 6:10, 8:11] = 1.0   # island A, blob 2
    grid[6:10, 6:10, 20:23] = 1.0  # island B, blob 1
    grid[6:10, 6:10, 26:29] = 1.0  # island B, blob 2
    allowed = np.zeros_like(grid, dtype=bool)
    allowed[5:11, 5:11, 1:12] = True   # island A
    allowed[5:11, 5:11, 19:30] = True  # island B

    repaired, info = connect_grid_components(grid, allowed, strut_radius_vox=1)

    assert info["n_components_before"] == 4
    assert info["n_components_after"] == 2
    assert info["n_unreachable"] == 1
    # No material in the gap between the islands.
    assert (repaired[:, :, 12:19] < 0.5).all()


def test_struts_prefer_high_occupancy_corridors():
    grid = np.zeros((16, 16, 32), dtype=np.float32)
    grid[6:10, 6:10, 2:6] = 1.0
    grid[6:10, 6:10, 26:30] = 1.0
    # Near-solid corridor at y=2 vs. empty space on the straight line: the
    # occupancy-hugging cost should route through the corridor.
    grid[7:9, 2:4, 6:26] = 0.45  # below threshold, but nearly free to traverse
    allowed = np.ones_like(grid, dtype=bool)

    repaired, info = connect_grid_components(
        grid, allowed, strut_radius_vox=0, path_eps=0.01
    )

    assert info["n_components_after"] == 1
    added = (repaired >= 0.5) & ~(grid >= 0.5)
    # The added path spends its interior in the corridor, not on the straight line.
    interior = added[:, :, 8:24]
    assert interior[:, :6, :].sum() > 0
    assert interior[:, 6:, :].sum() == 0


# ---------------------------------------------------------------------------
# RayGenerator.points_in_target_hull
# ---------------------------------------------------------------------------

def test_hull_respects_mask_orientation():
    # Light travels −z; screen normal +z. For this configuration the screen
    # basis gives u = −y and v = x (in units of h = 0.5), and the trainer maps
    # col = (u+1)/2·(W−1). A mask that is white on the left half (col < W/2,
    # i.e. u < 0) therefore admits points with y > 0 only.
    cfg = Config()
    rg = RayGenerator(cfg)
    light_dirs = torch.tensor([[0.0, 0.0, -1.0]])
    screen_normals = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.zeros(1, 16, 16)
    mask[:, :, :8] = 1.0

    points = torch.tensor(
        [
            [0.0, 0.25, 0.0],   # u = −0.5 → white half
            [0.0, -0.25, 0.0],  # u = +0.5 → black half
            [0.0, 0.6, 0.0],    # u = −1.2 → off the screen entirely
        ]
    )
    inside = rg.points_in_target_hull(points, light_dirs, screen_normals, mask)
    assert inside.tolist() == [True, False, False]


def test_hull_with_all_white_masks_equals_frustum_test():
    cfg = Config()
    rg = RayGenerator(cfg)
    torch.manual_seed(0)
    light_dirs = F.normalize(torch.randn(3, 3), dim=1)
    screen_normals = -light_dirs
    masks = torch.ones(3, 32, 32)
    points = (torch.rand(500, 3) * 2 - 1) * 0.7  # some outside the frustums

    hull = rg.points_in_target_hull(points, light_dirs, screen_normals, masks)
    frustum = rg.points_in_frustums(points, light_dirs, screen_normals)
    assert torch.equal(hull, frustum)


def test_hull_is_intersection_over_views():
    cfg = Config()
    rg = RayGenerator(cfg)
    light_dirs = torch.tensor([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    screen_normals = -light_dirs
    # View 0 admits y > 0 (left-half mask, see orientation test); view 1 all-white.
    masks = torch.ones(2, 16, 16)
    masks[0, :, 8:] = 0.0

    points = torch.tensor([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]])
    inside = rg.points_in_target_hull(points, light_dirs, screen_normals, masks)
    assert inside.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Mask erosion (conservative hull margin)
# ---------------------------------------------------------------------------

def test_mask_erosion_shrinks_silhouette():
    cfg = Config()
    cfg.mesh.hull_erosion_px = 1
    exporter = MarchingCubesMeshExporter(cfg)
    mask = torch.zeros(1, 12, 12)
    mask[:, 3:9, 3:9] = 1.0

    eroded = exporter._erode_masks(mask)
    assert eroded[0, 4:8, 4:8].min() == 1.0
    assert eroded[0, 3, :].max() == 0.0  # boundary ring removed
    assert eroded.sum() == 4 * 4


def test_mask_erosion_falls_back_when_target_vanishes():
    cfg = Config()
    cfg.mesh.hull_erosion_px = 1
    exporter = MarchingCubesMeshExporter(cfg)
    mask = torch.zeros(1, 12, 12)
    mask[:, 6, 6] = 1.0  # single pixel: erosion would erase it

    eroded = exporter._erode_masks(mask)
    torch.testing.assert_close(eroded, mask)
