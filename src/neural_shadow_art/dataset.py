from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class ShadowDataset:
    """Loads and preprocesses binary shadow target images.

    Convention: white pixels (value=1) represent cast shadow; black (0) is background.
    Pass ``invert=True`` if your images use the opposite convention.
    """

    def __init__(
        self,
        image_paths: list[str],
        img_size: int = 256,
        invert: bool = False,
    ):
        self.paths = [str(p) for p in image_paths]
        self.img_size = img_size
        self.invert = invert
        self._masks: list[Tensor] = []

        for path in self.paths:
            mask = self._load(path)
            self._masks.append(mask)

    def _load(self, path: str) -> Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Otsu threshold to binarize
        threshold = _otsu_threshold(arr)
        binary = (arr >= threshold).astype(np.float32)
        if self.invert:
            binary = 1.0 - binary
        return torch.from_numpy(binary)  # (H, W) float32, values in {0, 1}

    def __len__(self) -> int:
        return len(self._masks)

    def __getitem__(self, idx: int) -> dict:
        return {"mask": self._masks[idx], "path": self.paths[idx], "view_idx": idx}

    @property
    def n_views(self) -> int:
        return len(self._masks)

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.img_size, self.img_size)

    def shadow_area_ratios(self) -> Tensor:
        """α per view: image_area / shadow_bounding_box_area (paper Eq. 8), shape (n_views,)."""
        ratios = []
        for mask in self._masks:
            rows = mask.any(dim=1).nonzero(as_tuple=False)
            cols = mask.any(dim=0).nonzero(as_tuple=False)
            if rows.numel() == 0:
                ratios.append(1.0)
                continue
            h = rows[-1].item() - rows[0].item() + 1
            w = cols[-1].item() - cols[0].item() + 1
            ratios.append(float(mask.numel()) / (h * w))
        return torch.tensor(ratios, dtype=torch.float32)

    def get_all_masks(self) -> Tensor:
        """Stack all masks: (n_views, H, W)."""
        return torch.stack(self._masks, dim=0)


def _otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu's binarization threshold for a float image in [0,1]."""
    hist, bin_edges = np.histogram(arr.flatten(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_thresh = 0.5
    best_var = -1.0
    cumsum = 0.0
    cum_mean = 0.0
    total_mean = (hist * bin_centers).sum() / total

    for i in range(len(hist)):
        cumsum += hist[i]
        if cumsum == 0:
            continue
        cum_mean += hist[i] * bin_centers[i]
        w0 = cumsum / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean / cumsum
        mu1 = (total_mean * total - cum_mean) / (total - cumsum)
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            # Use the right edge of this bin as the threshold so that pixels
            # whose value equals the bin center are classified as background (< threshold).
            best_thresh = bin_edges[i + 1]

    return float(best_thresh)
