from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RigidRegistration(nn.Module):
    """Periodic ICP registration of target images to rendered shadows (paper Sec. 3.3).

    Every few epochs the current shadows are rendered, boundary point clouds
    are extracted from both the rendered and the target masks, and ICP rigidly
    registers the target boundary to the rendered boundary. The resulting
    transform is composed with the view's accumulated transform and applied to
    the *original* target image (a single warp, avoiding repeated resampling)
    to produce the updated training target. This lets the model converge to a
    solution consistent with the inputs up to rigid motions.

    Transforms are stored as buffers — state carried in checkpoints, not
    gradient-optimized parameters.
    """

    def __init__(
        self,
        n_views: int,
        max_icp_iters: int = 30,
        max_boundary_points: int = 2000,
    ):
        super().__init__()
        # Per-view rigid transform in normalized [-1, 1] image coords, stored
        # as [tx, ty, angle]: maps original-target points p to registered
        # points p' = R(angle) p + t.
        self.register_buffer("transforms", torch.zeros(n_views, 3))
        self.max_icp_iters = max_icp_iters
        self.max_boundary_points = max_boundary_points

    @staticmethod
    def _rotation(angle: Tensor) -> Tensor:
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        return torch.stack([
            torch.stack([cos_a, -sin_a]),
            torch.stack([sin_a, cos_a]),
        ])

    def is_identity(self) -> bool:
        return bool((self.transforms == 0).all())

    def warp_mask(self, mask: Tensor, view_idx: int) -> Tensor:
        """Warp an original target mask by the view's accumulated transform.

        Args:
            mask: (H, W) binary float tensor.
            view_idx: which view's transform to apply.
        Returns:
            Warped, re-binarized (H, W) float tensor.
        """
        tx, ty, angle = self.transforms[view_idx].unbind()
        if tx == 0 and ty == 0 and angle == 0:
            return mask

        H, W = mask.shape
        # grid_sample pulls input coords from output coords, so the warp
        # I'(x) = I(T⁻¹x) uses the inverse transform: R⁻¹ = Rᵀ, t⁻¹ = −Rᵀt.
        R_inv = self._rotation(angle).T
        t = torch.stack([tx, ty])
        theta = torch.cat([R_inv, (-R_inv @ t).unsqueeze(1)], dim=1)  # (2, 3)

        grid = F.affine_grid(
            theta.unsqueeze(0).to(mask.device),
            (1, 1, H, W),
            align_corners=False,
        )
        out = F.grid_sample(
            mask.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return (out.squeeze(0).squeeze(0) > 0.5).float()

    @torch.no_grad()
    def update(
        self,
        original_masks: list[Tensor],
        rendered_masks: list[Tensor],
    ) -> list[Tensor]:
        """Run one registration round and return the updated target masks.

        Args:
            original_masks: unmodified input target masks, (H, W) each.
            rendered_masks: current rendered shadow masks, (H, W) each.
        Returns:
            List of warped target masks to train against next.
        """
        updated = []
        for v, (orig, rendered) in enumerate(zip(original_masks, rendered_masks)):
            current = self.warp_mask(orig, v)
            src = self._boundary_points(current)           # target boundary
            dst = self._boundary_points(rendered > 0.5)    # rendered boundary
            if src.shape[0] >= 3 and dst.shape[0] >= 3:
                d_angle, d_t = self._icp(src, dst)
                # Compose: p' = ΔR (R p + t) + Δt
                tx, ty, angle = self.transforms[v].unbind()
                dR = self._rotation(d_angle)
                new_t = dR @ torch.stack([tx, ty]) + d_t
                self.transforms[v] = torch.stack(
                    [new_t[0], new_t[1], angle + d_angle]
                )
            updated.append(self.warp_mask(orig, v))
        return updated

    def _boundary_points(self, mask: Tensor) -> Tensor:
        """Extract boundary pixel coordinates in normalized [-1, 1] (x, y)."""
        binary = (mask > 0.5).float()
        eroded = 1.0 - F.max_pool2d(
            (1.0 - binary).unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
        ).squeeze(0).squeeze(0)
        boundary = (binary > 0.5) & (eroded < 0.5)

        idx = boundary.nonzero(as_tuple=False).float()  # (M, 2) as (row, col)
        M = idx.shape[0]
        if M == 0:
            return idx.new_zeros(0, 2)
        if M > self.max_boundary_points:
            sel = torch.randperm(M, device=idx.device)[: self.max_boundary_points]
            idx = idx[sel]

        H, W = mask.shape
        x = idx[:, 1] / (W - 1) * 2.0 - 1.0
        y = idx[:, 0] / (H - 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=1)  # (M, 2)

    def _icp(self, src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
        """2D point-to-point ICP registering src onto dst.

        Returns:
            (angle, translation) of the rigid transform p' = R(angle) p + t.
        """
        angle = src.new_zeros(())
        t = src.new_zeros(2)
        prev_err = float("inf")

        for _ in range(self.max_icp_iters):
            cur = src @ self._rotation(angle).T + t
            nn_idx = torch.cdist(cur, dst).argmin(dim=1)
            matched = dst[nn_idx]

            err = (cur - matched).norm(dim=1).mean().item()
            if prev_err - err < 1e-6:
                break
            prev_err = err

            # Kabsch: best rigid transform from src (not cur) to matched,
            # replacing the accumulated estimate each iteration.
            cs, cd = src.mean(dim=0), matched.mean(dim=0)
            Hm = (src - cs).T @ (matched - cd)
            U, _, Vt = torch.linalg.svd(Hm)
            d = torch.sign(torch.linalg.det(Vt.T @ U.T))
            D = torch.diag(torch.stack([torch.ones_like(d), d]))
            R = Vt.T @ D @ U.T
            angle = torch.atan2(R[1, 0], R[0, 0])
            t = cd - R @ cs

        return angle, t

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
