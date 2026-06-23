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
    cfg.render.rays_per_pixel = 4
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
    pred_occ, per_sample_occ = renderer.render_view(model, bundle, n_samples=4)

    assert pred_occ.shape == (32,)
    assert per_sample_occ.shape == (32, 4)
    assert (pred_occ >= 0).all() and (pred_occ <= 1).all()


def test_renderer_occupancy_range():
    cfg = make_small_cfg()
    model = ShadowArtModel(cfg, n_views=1)
    renderer = DifferentiableRenderer(cfg)
    rg = RayGenerator(cfg)

    light_dir = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    screen_normal = F.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    pixel_coords = torch.rand(64, 2) * 2 - 1

    bundle = rg.generate_rays_for_view(light_dir, screen_normal, pixel_coords)
    pred_occ, _ = renderer.render_view(model, bundle, n_samples=4)
    assert (pred_occ >= 0).all() and (pred_occ <= 1 + 1e-5).all()
