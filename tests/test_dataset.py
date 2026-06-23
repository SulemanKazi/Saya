import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.neural_shadow_art.dataset import ShadowDataset, _otsu_threshold


def make_test_image(path: str, size: int = 64, fill_value: int = 200):
    """Create a simple synthetic grayscale image."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = fill_value
    Image.fromarray(arr).save(path)


def test_dataset_loads_and_binarizes():
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "shadow_0.png")
        p1 = os.path.join(d, "shadow_1.png")
        make_test_image(p0, size=32, fill_value=220)
        make_test_image(p1, size=32, fill_value=200)

        ds = ShadowDataset([p0, p1], img_size=32)
        assert ds.n_views == 2
        assert ds.image_size == (32, 32)

        mask = ds[0]["mask"]
        assert mask.dtype == torch.float32
        # Binary: only 0 and 1
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Non-binary value {v.item()} in mask"


def test_dataset_invert():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "shadow.png")
        make_test_image(p, size=32, fill_value=220)
        ds_normal = ShadowDataset([p], img_size=32)
        ds_inverted = ShadowDataset([p], img_size=32, invert=True)
        mask_n = ds_normal[0]["mask"]
        mask_i = ds_inverted[0]["mask"]
        # Sum should complement
        total = mask_n.sum() + mask_i.sum()
        expected = mask_n.numel()
        assert abs(total.item() - expected) < 5, "Inverted mask should complement original"


def test_shadow_area_ratios_range():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "shadow.png")
        make_test_image(p, size=32, fill_value=220)
        ds = ShadowDataset([p], img_size=32)
        ratios = ds.shadow_area_ratios()
        assert ratios.shape == (1,)
        assert 0.0 <= ratios[0].item() <= 1.0


def test_otsu_threshold_uniform():
    # Fully uniform image → threshold should be somewhere in [0, 1]
    arr = np.full((32, 32), 0.5, dtype=np.float32)
    t = _otsu_threshold(arr)
    assert 0.0 <= t <= 1.0


def test_otsu_threshold_bimodal():
    # Clear bimodal distribution → threshold should split them
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[:32, :] = 0.1
    arr[32:, :] = 0.9
    t = _otsu_threshold(arr)
    assert 0.2 <= t <= 0.8, f"Threshold {t} should be between the two modes"
