import torch
import pytest
from src.neural_shadow_art.config import Config, ModelConfig
from src.neural_shadow_art.model import PositionalEncoding, OccupancyMLP, ShadowArtModel


def test_positional_encoding_output_shape():
    enc = PositionalEncoding(n_levels=6)
    x = torch.randn(10, 3)
    out = enc(x)
    assert out.shape == (10, 3 * (1 + 2 * 6)), f"Expected (10, 39), got {out.shape}"


def test_positional_encoding_no_nan():
    enc = PositionalEncoding(n_levels=6)
    x = torch.randn(100, 3)
    out = enc(x)
    assert not torch.isnan(out).any()


def test_mlp_output_range():
    cfg = ModelConfig(n_layers=2, hidden_dim=16, pos_enc_levels=4)
    mlp = OccupancyMLP(cfg)
    x = torch.randn(50, 3)
    out = mlp(x)
    assert out.shape == (50, 1)
    assert (out >= 0).all() and (out <= 1).all(), "MLP output must be in [0, 1]"


def test_mlp_gradient_flow():
    cfg = ModelConfig(n_layers=2, hidden_dim=16, pos_enc_levels=4)
    mlp = OccupancyMLP(cfg)
    x = torch.randn(8, 3)
    out = mlp(x).sum()
    out.backward()
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_shadow_art_model_forward():
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    model = ShadowArtModel(cfg, n_views=3)
    pts = torch.randn(20, 3)
    occ = model.occupancy(pts)
    assert occ.shape == (20, 1)
    assert (occ >= 0).all() and (occ <= 1).all()


def test_shadow_art_model_normalized_dirs():
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    model = ShadowArtModel(cfg, n_views=2)
    ld = model.get_light_dirs()
    sn = model.get_screen_normals()
    norms_l = ld.norm(dim=-1)
    norms_s = sn.norm(dim=-1)
    assert torch.allclose(norms_l, torch.ones(2), atol=1e-5), "Light dirs must be unit vectors"
    assert torch.allclose(norms_s, torch.ones(2), atol=1e-5), "Screen normals must be unit vectors"


def test_mlp_layer_count_matches_paper():
    """Paper: n_layers counts all FC layers including the sigmoid output layer."""
    cfg = ModelConfig(n_layers=8, hidden_dim=32, pos_enc_levels=4)
    mlp = OccupancyMLP(cfg)
    n_linear = sum(1 for m in mlp.net if isinstance(m, torch.nn.Linear))
    assert n_linear == 8


def test_light_screen_convention():
    """Paper: screen normals point toward the object, ⟨l_i, s_i⟩ < 0."""
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    model = ShadowArtModel(cfg, n_views=3)
    ld = model.get_light_dirs()
    sn = model.get_screen_normals()
    dots = (ld * sn).sum(dim=-1)
    assert (dots < 0).all(), "Initial ⟨l, s⟩ must be negative (paper convention)"


def test_custom_light_dirs():
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    init = torch.tensor([[0.0, 0.0, -2.0], [3.0, 0.0, 0.0]])
    model = ShadowArtModel(cfg, n_views=2, init_light_dirs=init)
    ld = model.get_light_dirs()
    expected = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(ld, expected, atol=1e-6)

    with pytest.raises(ValueError):
        ShadowArtModel(cfg, n_views=2, init_light_dirs=torch.zeros(3, 3))


def test_shadow_art_model_checkpoint():
    cfg = Config()
    cfg.model.n_layers = 2
    cfg.model.hidden_dim = 16
    model = ShadowArtModel(cfg, n_views=2)
    sd = model.state_dict()
    assert "light_dirs" in sd
    assert "screen_normals" in sd
    assert "mlp.net.0.weight" in sd
