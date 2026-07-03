from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Config
from .renderer import RayGenerator


def connect_grid_components(
    grid: np.ndarray,
    allowed: np.ndarray,
    threshold: float = 0.5,
    strut_radius_vox: int = 1,
    path_eps: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Connect disconnected occupancy components with struts inside ``allowed``.

    Components of ``grid >= threshold`` (6-connectivity, so corner-touching
    blobs count as disconnected and get a proper strut) are joined by
    minimum-cost voxel paths. Paths may only pass through ``allowed | occupied``
    voxels — the caller passes the target visual hull as ``allowed``, so added
    material provably cannot change any shadow. Edge cost per voxel is
    (1 − occupancy) + path_eps: struts hug existing high-occupancy geometry,
    with a small per-step cost to keep them short.

    Components sharing a connected island of the routing domain are merged
    Prim-style: multi-source Dijkstra from the growing tree (seeded at the
    island's largest component) to the nearest unmerged component, until the
    island is one piece. Components in *different* islands are left apart —
    the hull itself is disconnected between them, so no shadow-preserving
    connection exists (info["n_unreachable"] counts the resulting extra
    pieces). Each path is dilated to ``strut_radius_vox`` (clipped back to the
    allowed region — the 1-voxel core path is always kept, so dilation
    clipping cannot re-disconnect anything) and written into the grid as
    occupancy 1.0.

    Args:
        grid: (X, Y, Z) float occupancy field.
        allowed: (X, Y, Z) bool — voxels where material may be added.
        threshold: occupancy threshold defining solid voxels.
        strut_radius_vox: strut radius in voxels (0 → 1-voxel-wide path).
        path_eps: per-voxel base cost; higher values favor shorter struts over
            occupancy-hugging ones.
    Returns:
        (repaired grid copy, info dict with component/voxel counts).
    """
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    occ = grid >= threshold
    labels, n_before = ndimage.label(occ)  # default structure = 6-connectivity
    info = {
        "n_components_before": int(n_before),
        "n_components_after": int(n_before),
        "n_voxels_added": 0,
        "n_unreachable": 0,
    }
    if n_before <= 1:
        return grid, info

    # Routing domain: hull voxels plus already-occupied voxels (paths through
    # existing geometry add nothing, so they are trivially shadow-safe).
    domain = allowed | occ
    flat = np.flatnonzero(domain.ravel())
    n_nodes = flat.size
    node_id = np.full(grid.size, -1, dtype=np.int64)
    node_id[flat] = np.arange(n_nodes)
    nid = node_id.reshape(grid.shape)

    cost = (1.0 - grid.ravel()[flat]).clip(0.0, 1.0) + path_eps

    rows_list, cols_list = [], []
    for axis in range(3):
        a = np.moveaxis(nid, axis, 0)[:-1].ravel()
        b = np.moveaxis(nid, axis, 0)[1:].ravel()
        adj = (a >= 0) & (b >= 0)
        rows_list += [a[adj], b[adj]]
        cols_list += [b[adj], a[adj]]
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    graph = coo_matrix(
        (cost[cols], (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()  # edge weight = cost of entering the head voxel

    node_labels = labels.ravel()[flat]
    sizes = ndimage.sum_labels(occ, labels, index=np.arange(1, n_before + 1))

    # Components in different connected islands of the routing domain can
    # never be joined (the hull itself is disconnected between them — any
    # connection would have to leave ``allowed`` and change a shadow), so each
    # island is merged independently into one piece.
    dom_labels, _ = ndimage.label(domain)
    node_island = dom_labels.ravel()[flat]
    comp_island = np.zeros(n_before + 1, dtype=np.int64)
    comp_island[node_labels] = node_island  # all nodes of a component agree

    island_comps: dict[int, list[int]] = {}
    for lbl in range(1, n_before + 1):
        island_comps.setdefault(int(comp_island[lbl]), []).append(lbl)

    path_nodes_all: list[np.ndarray] = []
    n_pieces = 0

    for comps in island_comps.values():
        n_pieces += 1
        if len(comps) == 1:
            continue
        comps = sorted(comps, key=lambda l: sizes[l - 1], reverse=True)
        tree_mask = node_labels == comps[0]
        remaining = set(comps[1:])

        while remaining:
            dist, pred, _ = dijkstra(
                graph,
                directed=True,
                indices=np.flatnonzero(tree_mask),
                min_only=True,
                return_predecessors=True,
            )
            cand_ids = np.flatnonzero(
                ~tree_mask & np.isin(node_labels, list(remaining))
            )
            d = dist[cand_ids]
            best = int(np.argmin(d))
            if not np.isfinite(d[best]):
                # Same island ⇒ always reachable; guard against surprises.
                n_pieces += len(remaining)
                break

            node = int(cand_ids[best])
            path = []
            while node >= 0:
                path.append(node)
                node = int(pred[node])  # sources have predecessor -9999
            path = np.array(path)
            path_nodes_all.append(path)

            merged_label = int(node_labels[cand_ids[best]])
            tree_mask[path] = True
            tree_mask |= node_labels == merged_label
            remaining.discard(merged_label)

    info["n_unreachable"] = n_pieces - 1
    if not path_nodes_all:
        return grid, info

    path_mask = np.zeros(grid.shape, dtype=bool)
    path_mask.ravel()[flat[np.concatenate(path_nodes_all)]] = True

    r = max(0, int(strut_radius_vox))
    if r > 0:
        ox, oy, oz = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = ox**2 + oy**2 + oz**2 <= r * r
        strut = ndimage.binary_dilation(path_mask, structure=ball) & domain
    else:
        strut = path_mask

    repaired = grid.copy()
    repaired[strut] = 1.0

    _, n_after = ndimage.label(repaired >= threshold)
    info["n_components_after"] = int(n_after)
    info["n_voxels_added"] = int((strut & ~occ).sum())
    return repaired, info


class MarchingCubesMeshExporter:
    """Extracts a watertight mesh from the neural occupancy field.

    Evaluates the MLP on a 3D grid, truncates it to the intersection of all
    viewing frustums (paper Sec. 3.2/3.4 — so the mesh cannot cast shadows
    outside the target images), applies Marching Cubes at the iso-threshold,
    and exports the mesh for 3D printing.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _grid_coords(self) -> np.ndarray:
        """World-space coordinates of the export grid, shape (R³, 3)."""
        R = self.cfg.mesh.grid_resolution
        bmin = self.cfg.render.bbox_min
        bmax = self.cfg.render.bbox_max

        xs = np.linspace(bmin[0], bmax[0], R, dtype=np.float32)
        ys = np.linspace(bmin[1], bmax[1], R, dtype=np.float32)
        zs = np.linspace(bmin[2], bmax[2], R, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    def evaluate_on_grid(
        self,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
    ) -> np.ndarray:
        """Evaluate occupancy on a regular 3D grid, frustum-truncated.

        Returns:
            (R, R, R) float32 numpy array of occupancy values.
        """
        R = self.cfg.mesh.grid_resolution
        batch = self.cfg.mesh.eval_batch_size
        coords = self._grid_coords()

        occ_vals = np.zeros(len(coords), dtype=np.float32)

        truncate = self.cfg.render.frustum_truncation
        if truncate:
            ray_gen = RayGenerator(self.cfg)
            light_dirs = model.get_light_dirs().detach()
            screen_normals = model.get_screen_normals().detach()

        model.eval()
        with torch.no_grad():
            for start in range(0, len(coords), batch):
                chunk = torch.from_numpy(coords[start : start + batch]).to(device)
                out = model.occupancy(chunk).squeeze(-1)
                if truncate:
                    inside = ray_gen.points_in_frustums(
                        chunk, light_dirs, screen_normals
                    )
                    out = out * inside.float()
                occ_vals[start : start + batch] = out.cpu().numpy()

        model.train()
        return occ_vals.reshape(R, R, R)

    def _erode_masks(self, target_masks: Tensor) -> Tensor:
        """Erode binary masks by hull_erosion_px (min-pool) for a conservative hull."""
        px = self.cfg.mesh.hull_erosion_px
        if px <= 0:
            return target_masks
        inv = 1.0 - target_masks.float().unsqueeze(1)  # (V, 1, H, W)
        eroded = 1.0 - F.max_pool2d(inv, kernel_size=2 * px + 1, stride=1, padding=px)
        eroded = eroded.squeeze(1)
        for i in range(eroded.shape[0]):
            if eroded[i].sum() == 0:
                print(
                    f"[WARNING] View {i}: target vanished after {px}px hull "
                    "erosion (very thin silhouette) — using the uneroded mask."
                )
                eroded[i] = target_masks[i].float()
        return eroded

    def compute_target_hull(
        self,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
        target_masks: Tensor,
    ) -> np.ndarray:
        """Visual hull of the targets on the export grid: (R, R, R) bool.

        A grid point is in the hull iff it projects onto a shadow pixel in
        every view (using the trained light/screen directions). Filling hull
        points cannot change any shadow — see connect_grid_components.
        """
        R = self.cfg.mesh.grid_resolution
        batch = self.cfg.mesh.eval_batch_size
        coords = self._grid_coords()

        masks = self._erode_masks(target_masks.to(device))
        ray_gen = RayGenerator(self.cfg)
        light_dirs = model.get_light_dirs().detach()
        screen_normals = model.get_screen_normals().detach()

        hull = np.zeros(len(coords), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(coords), batch):
                chunk = torch.from_numpy(coords[start : start + batch]).to(device)
                inside = ray_gen.points_in_target_hull(
                    chunk, light_dirs, screen_normals, masks
                )
                hull[start : start + batch] = inside.cpu().numpy()
        return hull.reshape(R, R, R)

    def _repair_connectivity(
        self,
        grid: np.ndarray,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
        target_masks: Tensor,
        threshold: float,
    ) -> np.ndarray:
        """Join disconnected components with hull-routed struts (shadow-safe)."""
        mcfg = self.cfg.mesh
        print("Computing target visual hull for connectivity repair …")
        hull = self.compute_target_hull(model, device, target_masks)

        # The bbox's longest extent spans model_size_mm when printed, and the
        # grid has R−1 voxel steps across it.
        voxel_mm = mcfg.model_size_mm / (mcfg.grid_resolution - 1)
        r_vox = max(1, round(mcfg.strut_radius_mm / voxel_mm))

        repaired, info = connect_grid_components(
            grid, hull, threshold=threshold, strut_radius_vox=r_vox
        )
        if info["n_components_before"] <= 1:
            print("Connectivity: mesh already a single component.")
        else:
            print(
                f"Connectivity repair: {info['n_components_before']} components "
                f"→ {info['n_components_after']} "
                f"({info['n_voxels_added']} strut voxels added inside the "
                f"target hull, strut radius {r_vox} vox ≈ {mcfg.strut_radius_mm} mm)."
            )
        if info["n_unreachable"] > 0:
            print(
                f"[WARNING] The sculpture still has "
                f"{info['n_components_after']} pieces: the target visual hull "
                "itself is disconnected, so joining them is impossible without "
                "changing the shadows. Print them as separate parts (shadow "
                "artists typically hang pieces on thin threads), or relax the "
                "target silhouettes."
            )
        return repaired

    def export(
        self,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
        output_path: str,
        target_masks: Optional[Tensor] = None,
    ) -> str:
        """Run Marching Cubes and export a mesh file.

        Args:
            model: trained ShadowArtModel.
            device: device where model lives.
            output_path: destination path (extension can be .stl, .obj, or .ply).
                         If it has no extension, one is appended from the config.
            target_masks: (n_views, H, W) binary target masks (the registered
                         training targets if registration was used). Required
                         for connectivity repair (cfg.mesh.connect_components);
                         omitted → repair is skipped with a note.
        Returns:
            Absolute path to the exported mesh file.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as e:
            raise ImportError("scikit-image is required for mesh export.") from e

        try:
            import trimesh
        except ImportError as e:
            raise ImportError("trimesh is required for mesh export.") from e

        fmt = self.cfg.mesh.output_format.lower()
        if not output_path.endswith(("." + fmt,)):
            output_path = output_path.rstrip(".") + "." + fmt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        print(f"Evaluating occupancy on {self.cfg.mesh.grid_resolution}³ grid …")
        grid = self.evaluate_on_grid(model, device)

        g_min, g_max = float(grid.min()), float(grid.max())
        print(f"Occupancy range: [{g_min:.4f}, {g_max:.4f}]")

        threshold = self.cfg.mesh.iso_threshold
        if not (g_min < threshold < g_max):
            # Fall back to the midpoint of the actual occupancy range so
            # Marching Cubes always has a valid surface to extract.
            threshold = (g_min + g_max) / 2.0
            print(
                f"[WARNING] iso_threshold={self.cfg.mesh.iso_threshold} outside data range. "
                f"Using adaptive threshold={threshold:.4f} instead. "
                "Train for more epochs to get a properly binarized field."
            )

        if self.cfg.mesh.connect_components:
            if target_masks is None:
                print(
                    "[NOTE] connect_components is enabled but no target masks "
                    "were provided — skipping connectivity repair."
                )
            else:
                grid = self._repair_connectivity(
                    grid, model, device, target_masks, threshold
                )

        print(f"Running Marching Cubes at threshold {threshold:.4f} …")
        try:
            verts, faces, normals, _ = marching_cubes(grid, level=threshold)
        except ValueError as e:
            raise ValueError(
                "Marching Cubes found no surface. "
                "Try training longer or lowering iso_threshold."
            ) from e

        # Map voxel indices back to world coordinates
        bmin = np.array(self.cfg.render.bbox_min, dtype=np.float32)
        bmax = np.array(self.cfg.render.bbox_max, dtype=np.float32)
        R = self.cfg.mesh.grid_resolution
        scale = (bmax - bmin) / (R - 1)
        verts = verts * scale + bmin

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Basic mesh repair for watertight output
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

        abs_path = os.path.abspath(output_path)
        mesh.export(abs_path)
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        print(f"Mesh exported → {abs_path}  ({n_verts} vertices, {n_faces} faces)")
        return abs_path
