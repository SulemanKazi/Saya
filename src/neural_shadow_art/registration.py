from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


class RigidRegistration(nn.Module):
    """Per-view learnable 2D rigid transform applied to target masks.

    Addresses incompatible input silhouettes by allowing each target mask to
    be slightly rotated and translated before comparing with predictions.
    Applied via F.affine_grid + F.grid_sample, making it fully differentiable.

    A small L2 regularization on translation and rotation prevents degenerate
    solutions (large offsets that hide mismatches).
    """

    REG_WEIGHT = 0.1  # regularization weight for transform magnitude

    def __init__(self, n_views: int):
        super().__init__()
        # Per-view: [tx, ty, rotation_angle]
        self.transforms = nn.Parameter(torch.zeros(n_views, 3))

    def _build_theta(self, view_idx: int) -> Tensor:
        """Build the 2×3 affine matrix for F.affine_grid."""
        tx, ty, angle = self.transforms[view_idx].unbind()
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        # 2x3 affine matrix: rotation + translation
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, tx]),
            torch.stack([sin_a,  cos_a, ty]),
        ], dim=0)  # (2, 3)
        return theta

    def transform_mask(self, mask: Tensor, view_idx: int) -> Tensor:
        """Apply learned 2D rigid transform to a binary mask.

        Args:
            mask: (H, W) float tensor.
            view_idx: which view's transform to apply.
        Returns:
            Transformed (H, W) float tensor.
        """
        H, W = mask.shape
        theta = self._build_theta(view_idx).unsqueeze(0)  # (1, 2, 3)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
        out = F.grid_sample(
            mask.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return out.squeeze(0).squeeze(0)

    def regularization_loss(self) -> Tensor:
        """L2 penalty on transform magnitude to prevent degenerate solutions."""
        return self.REG_WEIGHT * (self.transforms ** 2).sum()

    def get_params(self) -> list[nn.Parameter]:
        return [self.transforms]

    @staticmethod
    def is_compatible(masks: list[Tensor], iou_threshold: float = 0.1) -> bool:
        """Heuristic: warn if any pair of masks has very low overlap.

        Low IoU between bounding boxes suggests incompatible silhouettes that
        may benefit from registration.
        """
        def bbox(m: Tensor):
            rows = m.any(dim=1).nonzero(as_tuple=False)
            cols = m.any(dim=0).nonzero(as_tuple=False)
            if rows.numel() == 0 or cols.numel() == 0:
                return None
            return rows[0].item(), cols[0].item(), rows[-1].item(), cols[-1].item()

        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                bi = bbox(masks[i] > 0.5)
                bj = bbox(masks[j] > 0.5)
                if bi is None or bj is None:
                    continue
                inter_r0 = max(bi[0], bj[0])
                inter_c0 = max(bi[1], bj[1])
                inter_r1 = min(bi[2], bj[2])
                inter_c1 = min(bi[3], bj[3])
                inter_area = max(0, inter_r1 - inter_r0) * max(0, inter_c1 - inter_c0)
                area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
                union_area = area_i + area_j - inter_area
                if union_area > 0 and inter_area / union_area < iou_threshold:
                    return False  # likely incompatible
        return True
