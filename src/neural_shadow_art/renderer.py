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


@dataclass
class RenderResult:
    """Per-ray render outputs; invalid rays have zeroed sample rows."""

    pred_occ: Tensor        # (N,) aggregated ray occupancy in [0, 1]
    sample_occ: Tensor      # (N, K) occupancy at each sample
    sample_points: Tensor   # (N, K, 3) world-space sample positions (detached)
    sample_weights: Tensor  # (N, K) trapezoid segment lengths ω (paper Eq. 16, detached)
    valid: Tensor           # (N,) bool


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
        # Screen placed just beyond bbox in the light direction. For parallel
        # projection the distance only positions the ray origins, never the
        # pixel→ray mapping, so any value beyond the bbox works.
        self.screen_dist = float(self.scene_radius) + 0.1
        # Paper Eq. 4: the image spans the normalized space, i.e. the image
        # half-extent equals half the bbox extent (0.5 for the unit cube).
        # A larger screen would leave border pixels that no geometry inside
        # the bbox can ever shadow.
        self.screen_half_size = float((bbox_max - bbox_min).max()) * 0.5
        self.frustum_truncation = cfg.render.frustum_truncation

    def _ray_bbox_intersection(
        self, origins: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Slab method AABB intersection.

        Returns (t_near, t_far, valid) where valid means t_near < t_far.
        """
        bbox_min = self.bbox_min.to(origins.device)
        bbox_max = self.bbox_max.to(origins.device)
        # Clamp near-zero components BEFORE dividing: a `where(cond, 1/d, big)`
        # would still compute 1/0 = inf in the forward pass, whose backward
        # yields 0·inf = NaN gradients for the light directions.
        sign = torch.where(directions >= 0, 1.0, -1.0)
        safe_dir = torch.where(
            directions.abs() > 1e-7, directions, sign * 1e-7
        )
        safe_inv = 1.0 / safe_dir
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

                # Screen coordinate of a ray point x(t) projected along l_j
                # onto screen j:  u = (x−c)·e − ((x−c)·s)·(l·e)/⟨l,s⟩,
                # linear in t:  u(t) = U0 + t · U1.
                oc = bundle.origins - c_j  # (N, 3)
                A_j = (oc * s_j).sum(dim=-1)          # (N,)
                B_j = (bundle.directions[0] * s_j).sum()  # scalar
                E_j = (oc * e_j).sum(dim=-1)           # (N,)
                F_j = (bundle.directions[0] * e_j).sum()  # scalar

                U0 = E_j - A_j * (D_j / C_j)                        # (N,)
                U1 = (F_j - B_j * (D_j / C_j)).detach().item()  # scalar

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

    def points_in_frustums(
        self,
        points: Tensor,
        all_light_dirs: Tensor,
        all_screen_normals: Tensor,
    ) -> Tensor:
        """Boolean mask of points inside the intersection of all view frustums.

        Each frustum is the prism obtained by translating the image rectangle
        along its light direction (paper Sec. 3.2). Used at reconstruction time
        so the exported mesh casts no shadows outside the target images.

        Args:
            points: (P, 3) world-space points.
            all_light_dirs / all_screen_normals: (n_views, 3).
        Returns:
            (P,) bool mask.
        """
        device = points.device
        h = self.screen_half_size
        scene_center = self.scene_center.to(device)
        inside = torch.ones(points.shape[0], dtype=torch.bool, device=device)
        eps = 1e-6

        for j in range(all_light_dirs.shape[0]):
            l_j = F.normalize(all_light_dirs[j].to(device), dim=0)
            s_j = F.normalize(all_screen_normals[j].to(device), dim=0)
            C_j = torch.dot(l_j, s_j)
            if abs(C_j.item()) < eps:
                continue
            c_j = scene_center + l_j * self.screen_dist
            e1_j, e2_j = _orthonormal_basis(s_j)

            pc = points - c_j  # (P, 3)
            a = (pc * s_j).sum(dim=-1)  # (P,)
            for e_j in (e1_j, e2_j):
                u = (pc * e_j).sum(dim=-1) - a * (torch.dot(l_j, e_j) / C_j)
                inside &= u.abs() <= h

        return inside

    def points_in_target_hull(
        self,
        points: Tensor,
        all_light_dirs: Tensor,
        all_screen_normals: Tensor,
        target_masks: Tensor,
    ) -> Tensor:
        """Boolean mask of points inside the visual hull of the target shadows.

        A point is in the hull iff its projection along every view's light
        direction lands on a shadow pixel of that view's target mask. Since
        the rendered shadow is a union (O = 1 − ∏(1 − f_k)), filling any hull
        point with material only darkens pixels that are already dark in all
        targets — i.e. it cannot change any shadow. Used at export time to
        route connectivity struts (mesh_export.connect_grid_components).

        Args:
            points: (P, 3) world-space points.
            all_light_dirs / all_screen_normals: (n_views, 3).
            target_masks: (n_views, H, W) binary masks in the same pixel
                convention as training targets (row ↔ v, col ↔ u in [-1, 1]).
        Returns:
            (P,) bool mask.
        """
        device = points.device
        h = self.screen_half_size
        scene_center = self.scene_center.to(device)
        inside = torch.ones(points.shape[0], dtype=torch.bool, device=device)
        eps = 1e-6

        for j in range(all_light_dirs.shape[0]):
            l_j = F.normalize(all_light_dirs[j].to(device), dim=0)
            s_j = F.normalize(all_screen_normals[j].to(device), dim=0)
            C_j = torch.dot(l_j, s_j)
            if abs(C_j.item()) < eps:
                continue
            c_j = scene_center + l_j * self.screen_dist
            e1_j, e2_j = _orthonormal_basis(s_j)

            pc = points - c_j  # (P, 3)
            a = (pc * s_j).sum(dim=-1)  # (P,)
            u = (pc * e1_j).sum(dim=-1) - a * (torch.dot(l_j, e1_j) / C_j)
            v = (pc * e2_j).sum(dim=-1) - a * (torch.dot(l_j, e2_j) / C_j)
            un, vn = u / h, v / h  # normalized screen coords in [-1, 1]
            inside &= (un.abs() <= 1.0) & (vn.abs() <= 1.0)

            # Inverse of the trainer's pixel→coord mapping:
            # u = col/(W−1)·2 − 1, v = row/(H−1)·2 − 1.
            mask = target_masks[j].to(device)
            H, W = mask.shape
            col = ((un + 1.0) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
            row = ((vn + 1.0) * 0.5 * (H - 1)).round().long().clamp(0, H - 1)
            inside &= mask[row, col] > 0.5

        return inside


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
    ) -> RenderResult:
        """Render predicted occupancy for a batch of rays.

        Invalid rays (outside bbox / fully truncated) get pred_occ 0 and
        zeroed sample rows (their sample_weights are 0 so they drop out of
        the sample-based losses).
        """
        device = bundle.origins.device
        N = bundle.origins.shape[0]
        K = n_samples

        pred_occ = torch.zeros(N, device=device)
        per_sample_occ = torch.zeros(N, K, device=device)
        sample_points = torch.zeros(N, K, 3, device=device)
        sample_weights = torch.zeros(N, K, device=device)

        valid = bundle.valid & (bundle.t_near < bundle.t_far)
        if valid.sum() == 0:
            return RenderResult(
                pred_occ, per_sample_occ, sample_points, sample_weights, valid
            )

        v_origins = bundle.origins[valid]
        v_directions = bundle.directions[valid]
        v_t_near = bundle.t_near[valid]
        v_t_far = bundle.t_far[valid]

        points, t_vals = self.sample_points_along_rays(
            v_origins, v_directions, v_t_near, v_t_far, K
        )  # (N_v, K, 3), (N_v, K)

        N_v = v_origins.shape[0]
        points_flat = points.reshape(N_v * K, 3)
        occ_flat = model.occupancy(points_flat).squeeze(-1)  # (N_v*K,)
        occ = occ_flat.reshape(N_v, K)  # (N_v, K)

        # Aggregate: O = 1 - exp(sum_k log(1 - f_k))   [log-space product]
        log_trans = torch.log(1.0 - occ.clamp(0.0, 1.0 - 1e-7) + 1e-7).sum(dim=-1)
        occ_agg = 1.0 - torch.exp(log_trans)  # (N_v,)

        # Trapezoid segment lengths ω (paper Eq. 16). Directions are unit
        # vectors, so |t_{k+1} − t_k| equals the world-space spacing.
        weights = torch.zeros(N_v, K, device=device)
        if K >= 2:
            dt = (t_vals[:, 1:] - t_vals[:, :-1]).abs()  # (N_v, K-1)
            weights[:, 0] = dt[:, 0]
            weights[:, -1] = dt[:, -1]
            if K > 2:
                weights[:, 1:-1] = 0.5 * (dt[:, :-1] + dt[:, 1:])
        else:
            weights[:, 0] = v_t_far - v_t_near

        pred_occ[valid] = occ_agg
        per_sample_occ[valid] = occ
        sample_points[valid] = points.detach()
        sample_weights[valid] = weights.detach()

        return RenderResult(
            pred_occ, per_sample_occ, sample_points, sample_weights, valid
        )

    def render_all_views(
        self,
        model: "ShadowArtModel",  # noqa: F821
        ray_gen: RayGenerator,
        n_views: int,
        img_size: int,
        n_samples: int | None = None,
        chunk_size: int | None = None,
    ) -> list[Tensor]:
        """Render full shadow maps for all views.

        Rays are rendered in blocks of ``chunk_size`` pixels so peak memory
        stays bounded regardless of ``img_size``. A full view is H*W rays ×
        n_samples points fed through the MLP at once — at the defaults that is
        ~16.8M points and tens of GB of activations, which OOMs on CPU. The
        training loop avoids this by sampling ``batch_size_rays`` per step;
        this path did not, so it chunks explicitly. Rays are independent, so
        chunking is statistically equivalent to a single-pass render (not
        bit-identical only because stratified sampling draws fresh per chunk).

        Returns a list of (H, W) tensors, one per view.
        """
        if n_samples is None:
            # Paper: n = w samples per ray (image width)
            n_samples = self.cfg.render.n_samples_per_ray or img_size
        if chunk_size is None:
            chunk_size = self.cfg.train.batch_size_rays

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
                pred_chunks = []
                for start in range(0, pixel_coords.shape[0], chunk_size):
                    coords = pixel_coords[start:start + chunk_size]
                    bundle = ray_gen.generate_rays_for_view(
                        light_dirs[i], screen_normals[i], coords
                    )
                    if ray_gen.frustum_truncation and n_views > 1:
                        bundle = ray_gen.apply_frustum_truncation(
                            bundle, light_dirs, screen_normals, i
                        )
                    result = self.render_view(model, bundle, n_samples)
                    pred_chunks.append(result.pred_occ)
                results.append(torch.cat(pred_chunks).reshape(H, W))

        model.train()
        return results
