from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Config


@dataclass
class RayBundle:
    """A batch of parallel-projection rays."""

    origins: Tensor     # (N, 3)
    directions: Tensor  # (N, 3) unit vectors (all equal for parallel projection)
    t_near: Tensor      # (N,) minimum valid t
    t_far: Tensor       # (N,) maximum valid t
    valid: Tensor       # (N,) bool — False for rays that miss the scene bbox


def _orthonormal_basis(n: Tensor) -> tuple[Tensor, Tensor]:
    """Given a unit vector n, return (e1, e2) orthonormal and perpendicular to n."""
    device, dtype = n.device, n.dtype
    if abs(n[0].item()) < 0.9:
        up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    e1 = torch.linalg.cross(up, n)
    e1 = F.normalize(e1, dim=0)
    e2 = torch.linalg.cross(n, e1)
    return e1, e2


class RayGenerator:
    """Generates parallel-projection ray bundles and applies frustum truncation."""

    def __init__(self, cfg: Config):
        bbox_min = torch.tensor(cfg.render.bbox_min, dtype=torch.float32)
        bbox_max = torch.tensor(cfg.render.bbox_max, dtype=torch.float32)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.scene_center = (bbox_min + bbox_max) * 0.5
        self.scene_radius = (bbox_max - bbox_min).norm() * 0.5
        # Screen placed just beyond bbox in the light direction
        self.screen_dist = float(self.scene_radius) + 0.1
        # Conservative screen half-size: covers the full scene from any angle
        self.screen_half_size = float(self.scene_radius) * 1.2
        self.frustum_truncation = cfg.render.frustum_truncation

    def _ray_bbox_intersection(
        self, origins: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Slab method AABB intersection.

        Returns (t_near, t_far, valid) where valid means t_near < t_far.
        """
        bbox_min = self.bbox_min.to(origins.device)
        bbox_max = self.bbox_max.to(origins.device)
        inv_dir = 1.0 / directions.clamp(min=1e-7).abs() * directions.sign()
        # Handle zero directions
        safe_inv = torch.where(
            directions.abs() > 1e-7,
            1.0 / directions,
            torch.full_like(directions, 1e7),
        )
        t1 = (bbox_min - origins) * safe_inv  # (N, 3)
        t2 = (bbox_max - origins) * safe_inv  # (N, 3)
        t_min = torch.minimum(t1, t2).amax(dim=-1)  # (N,) - max of per-axis mins
        t_max = torch.maximum(t1, t2).amin(dim=-1)  # (N,) - min of per-axis maxs
        valid = t_min < t_max
        return t_min, t_max, valid

    def generate_rays_for_view(
        self,
        light_dir: Tensor,
        screen_normal: Tensor,
        pixel_coords: Tensor,
    ) -> RayBundle:
        """Generate a bundle of rays for one view.

        The shadow screen is placed beyond the scene in the light direction.
        Each pixel (u, v) ∈ [-1, 1]² maps to a 3D position on the screen,
        and the ray travels in the -light_dir direction (back toward the light
        source, through the scene).

        Args:
            light_dir: (3,) normalized; direction light travels (source → scene).
            screen_normal: (3,) normalized; normal of the shadow-casting screen.
            pixel_coords: (N, 2) in [-1, 1].

        Returns:
            RayBundle with t_near / t_far determined by bbox intersection.
        """
        device = pixel_coords.device
        light_dir = F.normalize(light_dir.to(device), dim=0)
        screen_normal = F.normalize(screen_normal.to(device), dim=0)

        e1, e2 = _orthonormal_basis(screen_normal)
        scene_center = self.scene_center.to(device)

        screen_center = scene_center + light_dir * self.screen_dist
        h = self.screen_half_size

        # 3D positions on the screen plane
        origins = (
            screen_center.unsqueeze(0)
            + pixel_coords[:, 0:1] * e1.unsqueeze(0) * h
            + pixel_coords[:, 1:2] * e2.unsqueeze(0) * h
        )  # (N, 3)

        # Rays go from the screen back toward the light source (through the scene)
        ray_dir = -light_dir  # (3,)
        directions = ray_dir.unsqueeze(0).expand(origins.shape[0], -1)  # (N, 3)

        t_near, t_far, valid = self._ray_bbox_intersection(origins, directions)

        return RayBundle(
            origins=origins,
            directions=directions,
            t_near=t_near,
            t_far=t_far,
            valid=valid,
        )

    def apply_frustum_truncation(
        self,
        bundle: RayBundle,
        all_light_dirs: Tensor,
        all_screen_normals: Tensor,
        view_idx: int,
    ) -> RayBundle:
        """Truncate rays so sample points lie within the frustums of all other views.

        For each view j ≠ i, a 3D point on the current ray must project onto
        view j's image plane within [-h, h]². This prevents geometry from growing
        in regions invisible to any view (which would cast incorrect shadows).

        The constraint per-axis is linear in t, giving a t-interval per view.
        """
        device = bundle.origins.device
        t_near = bundle.t_near.clone()
        t_far = bundle.t_far.clone()
        h = self.screen_half_size
        scene_center = self.scene_center.to(device)
        n_views = all_light_dirs.shape[0]
        eps = 1e-6

        for j in range(n_views):
            if j == view_idx:
                continue

            l_j = F.normalize(all_light_dirs[j].to(device), dim=0)
            s_j = F.normalize(all_screen_normals[j].to(device), dim=0)
            e1_j, e2_j = _orthonormal_basis(s_j)
            c_j = scene_center + l_j * self.screen_dist

            C_j = torch.dot(l_j, s_j)
            if abs(C_j.item()) < eps:
                continue

            # For each screen axis {e1_j, e2_j}, compute the t-interval
            # where the 3D point on the current ray projects within [-h, h].
            for e_j in (e1_j, e2_j):
                D_j = torch.dot(l_j, e_j)

                # Offset from screen center along this axis as a function of t:
                #   u(t) = U0 + t * U1
                # where U0 and U1 are derived from ray origin and direction.
                oc = bundle.origins - c_j  # (N, 3)
                A_j = (oc * s_j).sum(dim=-1)          # (N,)
                B_j = (bundle.directions[0] * s_j).sum()  # scalar
                E_j = (oc * e_j).sum(dim=-1)           # (N,)
                F_j = (bundle.directions[0] * e_j).sum()  # scalar

                U0 = E_j + A_j * (D_j / C_j)         # (N,)
                U1 = float(F_j - B_j * (D_j / C_j))  # scalar

                if abs(U1) > eps:
                    t_lo = (-h - U0) / U1  # (N,)
                    t_hi = (h - U0) / U1   # (N,)
                    if U1 < 0:
                        t_lo, t_hi = t_hi, t_lo
                    t_near = torch.maximum(t_near, t_lo)
                    t_far = torch.minimum(t_far, t_hi)
                else:
                    # Ray is nearly parallel to screen; outside rays are invalid
                    outside = U0.abs() > h + eps
                    t_far = torch.where(outside, torch.full_like(t_far, -1.0), t_far)

        valid = bundle.valid & (t_near < t_far)
        return RayBundle(
            origins=bundle.origins,
            directions=bundle.directions,
            t_near=t_near,
            t_far=t_far,
            valid=valid,
        )


class DifferentiableRenderer:
    """Renders predicted shadow maps from a neural occupancy field.

    Uses stratified sampling along rays and aggregates occupancy via the
    product formula O = 1 - ∏(1 - f_k), computed in log-space for stability.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def sample_points_along_rays(
        self,
        origins: Tensor,
        directions: Tensor,
        t_near: Tensor,
        t_far: Tensor,
        n_samples: int,
    ) -> tuple[Tensor, Tensor]:
        """Stratified sampling along ray segments.

        Returns:
            points: (N, K, 3) world-space positions.
            t_vals: (N, K) corresponding t values.
        """
        N = origins.shape[0]
        K = n_samples
        device = origins.device

        # Uniform bin edges in [0, 1]
        edges = torch.linspace(0.0, 1.0, K + 1, device=device)  # (K+1,)
        # Random offset within each bin for stratification
        rand = torch.rand(N, K, device=device)
        t_norm = edges[:K].unsqueeze(0) + rand * (edges[1:] - edges[:K]).unsqueeze(0)

        # Map to [t_near, t_far]
        span = (t_far - t_near).unsqueeze(1)  # (N, 1)
        t_vals = t_near.unsqueeze(1) + t_norm * span  # (N, K)

        points = origins.unsqueeze(1) + t_vals.unsqueeze(2) * directions.unsqueeze(1)
        return points, t_vals

    def render_view(
        self,
        model: "ShadowArtModel",  # noqa: F821
        bundle: RayBundle,
        n_samples: int,
    ) -> tuple[Tensor, Tensor]:
        """Render predicted occupancy for a batch of rays.

        Returns:
            pred_occ: (N,) aggregated occupancy per ray in [0, 1].
                      Invalid rays (outside bbox) contribute 0.
            per_sample_occ: (N, K) per-sample occupancy (needed for L_coh).
        """
        device = bundle.origins.device
        N = bundle.origins.shape[0]
        K = n_samples

        pred_occ = torch.zeros(N, device=device)
        per_sample_occ = torch.zeros(N, K, device=device)

        valid = bundle.valid & (bundle.t_near < bundle.t_far)
        if valid.sum() == 0:
            return pred_occ, per_sample_occ

        v_origins = bundle.origins[valid]
        v_directions = bundle.directions[valid]
        v_t_near = bundle.t_near[valid]
        v_t_far = bundle.t_far[valid]

        points, _ = self.sample_points_along_rays(
            v_origins, v_directions, v_t_near, v_t_far, K
        )  # (N_v, K, 3)

        N_v = v_origins.shape[0]
        points_flat = points.reshape(N_v * K, 3)
        occ_flat = model.occupancy(points_flat).squeeze(-1)  # (N_v*K,)
        occ = occ_flat.reshape(N_v, K)  # (N_v, K)

        # Aggregate: O = 1 - exp(sum_k log(1 - f_k))   [log-space product]
        log_trans = torch.log(1.0 - occ.clamp(0.0, 1.0 - 1e-7) + 1e-7).sum(dim=-1)
        occ_agg = 1.0 - torch.exp(log_trans)  # (N_v,)

        pred_occ[valid] = occ_agg
        per_sample_occ[valid] = occ

        return pred_occ, per_sample_occ

    def render_all_views(
        self,
        model: "ShadowArtModel",  # noqa: F821
        ray_gen: RayGenerator,
        n_views: int,
        img_size: int,
        n_samples: int | None = None,
    ) -> list[Tensor]:
        """Render full shadow maps for all views.

        Returns a list of (H, W) tensors, one per view.
        """
        if n_samples is None:
            n_samples = self.cfg.render.rays_per_pixel

        device = next(model.parameters()).device
        H = W = img_size

        u = torch.linspace(-1.0, 1.0, W, device=device)
        v = torch.linspace(-1.0, 1.0, H, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        pixel_coords = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=1)

        light_dirs = model.get_light_dirs()
        screen_normals = model.get_screen_normals()
        results = []

        model.eval()
        with torch.no_grad():
            for i in range(n_views):
                bundle = ray_gen.generate_rays_for_view(
                    light_dirs[i], screen_normals[i], pixel_coords
                )
                if ray_gen.frustum_truncation and n_views > 1:
                    bundle = ray_gen.apply_frustum_truncation(
                        bundle, light_dirs, screen_normals, i
                    )
                pred_occ, _ = self.render_view(model, bundle, n_samples)
                results.append(pred_occ.reshape(H, W))

        model.train()
        return results
