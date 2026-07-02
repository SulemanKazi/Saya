import torch
import torch.nn.functional as F
import pytest
from src.neural_shadow_art.config import Config
from src.neural_shadow_art.model import ShadowArtModel
from src.neural_shadow_art.renderer import RayGenerator, DifferentiableRenderer, RayBundle


def make_small_cfg():
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    cfg.render.n_samples_per_ray = 4
    return cfg


def test_ray_generation_shapes():
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    light_dir = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    screen_normal = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    pixel_coords = torch.rand(16, 2) * 2 - 1
    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)
    assert bundle.origins.shape == (16, 3)
    assert bundle.directions.shape == (16, 3)
    assert bundle.t_near.shape == (16,)
    assert bundle.t_far.shape == (16,)
    assert bundle.valid.shape == (16,)


def test_ray_directions_normalized():
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    light_dir = F.normalize(torch.tensor([1.0, 0.5, 0.3]), dim=0)
    screen_normal = F.normalize(torch.tensor([1.0, 0.5, 0.3]), dim=0)
    pixel_coords = torch.zeros(8, 2)
    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)
    norms = bundle.directions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), "Ray directions must be unit vectors"


def test_rays_hit_bbox_for_center_pixels():
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    # Use axis-aligned light; center pixels should hit the bbox
    light_dir = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    screen_normal = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    pixel_coords = torch.zeros(4, 2)  # all at center
    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)
    assert bundle.valid.any(), "Center rays should hit the bbox"


def test_frustum_truncation_reduces_range():
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    light_dir0 = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    screen_normal0 = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    light_dir1 = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=0)
    screen_normal1 = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=0)

    all_light_dirs = torch.stack([light_dir0, light_dir1])
    all_screen_normals = torch.stack([screen_normal0, screen_normal1])

    pixel_coords = torch.zeros(8, 2)
    bundle = rg.generate_rays_for_view(light_dir0, screen_normal0, pixel_coords)
    t_span_before = (bundle.t_far - bundle.t_near).sum().item()

    truncated = rg.apply_frustum_truncation(bundle, all_light_dirs, all_screen_normals, 0)
    t_span_after = ((truncated.t_far - truncated.t_near).clamp(min=0)).sum().item()

    assert t_span_after <= t_span_before + 1e-4, "Frustum truncation should not increase t range"


def test_renderer_output_shapes():
    cfg = make_small_cfg()
    model = ShadowArtModel(cfg, n_views=2)
    renderer = DifferentiableRenderer(cfg)
    rg = RayGenerator(cfg)

    light_dir = model.get_light_dirs()[0]
    screen_normal = model.get_screen_normals()[0]
    pixel_coords = torch.rand(32, 2) * 2 - 1

    bundle = rg.generate_rays_for_view(light_dir.detach(), screen_normal.detach(), pixel_coords)
    result = renderer.render_view(model, bundle, n_samples=4)

    assert result.pred_occ.shape == (32,)
    assert result.sample_occ.shape == (32, 4)
    assert result.sample_points.shape == (32, 4, 3)
    assert result.sample_weights.shape == (32, 4)
    assert (result.pred_occ >= 0).all() and (result.pred_occ <= 1).all()
    # Trapezoid weights of a valid ray sum to ~ its truncated length: stratified
    # samples sit inside their K bins, so the sum lies within (K−2)/K·span
    # (samples at bin centers minus slack) and 2·span (end segments double-counted)
    v = result.valid
    K = result.sample_weights.shape[1]
    spans = (bundle.t_far[v] - bundle.t_near[v])
    sums = result.sample_weights[v].sum(dim=1)
    assert (sums > spans * (K - 2) / K - spans / K).all()
    assert (sums <= spans * 2.0 + 1e-5).all()


def test_renderer_occupancy_range():
    cfg = make_small_cfg()
    model = ShadowArtModel(cfg, n_views=1)
    renderer = DifferentiableRenderer(cfg)
    rg = RayGenerator(cfg)

    light_dir = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    screen_normal = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    pixel_coords = torch.rand(64, 2) * 2 - 1

    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)
    result = renderer.render_view(model, bundle, n_samples=4)
    assert (result.pred_occ >= 0).all() and (result.pred_occ <= 1 + 1e-5).all()


def test_screen_matches_paper_scale():
    """Paper Eq. 4: the image spans the normalized space, so pixel u=±1 maps
    to a ±0.5 world offset for the default unit-cube bbox."""
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    assert rg.screen_half_size == pytest.approx(0.5)

    light_dir = torch.tensor([0.0, 0.0, 1.0])
    screen_normal = torch.tensor([0.0, 0.0, -1.0])
    pixel_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)

    center = bundle.origins[0]
    offsets = (bundle.origins - center).norm(dim=-1)
    assert offsets[1].item() == pytest.approx(0.5, abs=1e-5)
    assert offsets[2].item() == pytest.approx((0.5**2 + 0.5**2) ** 0.5, abs=1e-5)

    # A near-edge pixel must still be able to hit the bbox with axis-aligned
    # light (this was impossible with the old oversized screen).
    edge = rg.generate_rays_for_view(
        light_dir, screen_normal, torch.tensor([[0.95, 0.0]])
    )
    assert edge.valid.all()


def test_frustum_truncation_oblique_no_overclip():
    """Regression test for the U0 sign bug: when the other view's frustum
    fully contains the bbox, truncation must leave ray intervals unchanged —
    including for oblique (non-perpendicular) light/screen pairs."""
    cfg = make_small_cfg()
    # Enlarge the screen so view j's frustum strictly contains the whole bbox.
    rg = RayGenerator(cfg)
    rg.screen_half_size = 2.0

    li = torch.tensor([0.0, 0.0, 1.0])
    si = torch.tensor([0.0, 0.0, -1.0])
    lj = F.normalize(torch.tensor([0.6, 0.2, -0.75]), dim=0)
    sj = F.normalize(torch.tensor([0.9, -0.1, -0.4]), dim=0)  # oblique: ⟨l,s⟩ ∉ {±1}

    all_lights = torch.stack([li, lj])
    all_screens = torch.stack([si, sj])

    pixel_coords = torch.tensor([[0.0, 0.0], [0.1, -0.05], [-0.15, 0.1]])
    bundle = rg.generate_rays_for_view(li, si, pixel_coords)
    assert bundle.valid.all()

    truncated = rg.apply_frustum_truncation(bundle, all_lights, all_screens, 0)
    assert truncated.valid.all(), "Rays inside a containing frustum must stay valid"
    assert torch.allclose(truncated.t_near, bundle.t_near, atol=1e-4)
    assert torch.allclose(truncated.t_far, bundle.t_far, atol=1e-4)


def test_points_in_frustums():
    cfg = make_small_cfg()
    rg = RayGenerator(cfg)
    lights = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    screens = -lights.clone()

    points = torch.tensor([
        [0.0, 0.0, 0.0],    # center: inside both frustums
        [0.2, 0.2, 0.2],    # inside
        [0.0, 0.8, 0.0],    # |y| > 0.5: outside both image extents
    ])
    inside = rg.points_in_frustums(points, lights, screens)
    assert inside.tolist() == [True, True, False]
