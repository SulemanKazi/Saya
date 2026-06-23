from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from .config import Config


class MarchingCubesMeshExporter:
    """Extracts a watertight mesh from the neural occupancy field.

    Evaluates the MLP on a 3D grid, applies Marching Cubes at the iso-threshold,
    and exports the mesh for 3D printing.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate_on_grid(
        self,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
    ) -> np.ndarray:
        """Evaluate occupancy on a regular 3D grid.

        Returns:
            (R, R, R) float32 numpy array of occupancy values.
        """
        R = self.cfg.mesh.grid_resolution
        batch = self.cfg.mesh.eval_batch_size
        bmin = self.cfg.render.bbox_min
        bmax = self.cfg.render.bbox_max

        xs = np.linspace(bmin[0], bmax[0], R, dtype=np.float32)
        ys = np.linspace(bmin[1], bmax[1], R, dtype=np.float32)
        zs = np.linspace(bmin[2], bmax[2], R, dtype=np.float32)

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

        occ_vals = np.zeros(len(coords), dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for start in range(0, len(coords), batch):
                chunk = torch.from_numpy(coords[start : start + batch]).to(device)
                out = model.occupancy(chunk).squeeze(-1).cpu().numpy()
                occ_vals[start : start + batch] = out

        model.train()
        return occ_vals.reshape(R, R, R)

    def export(
        self,
        model: "ShadowArtModel",  # noqa: F821
        device: torch.device,
        output_path: str,
    ) -> str:
        """Run Marching Cubes and export a mesh file.

        Args:
            model: trained ShadowArtModel.
            device: device where model lives.
            output_path: destination path (extension can be .stl, .obj, or .ply).
                         If it has no extension, one is appended from the config.
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

        print(f"Running Marching Cubes at threshold {self.cfg.mesh.iso_threshold} …")
        try:
            verts, faces, normals, _ = marching_cubes(
                grid, level=self.cfg.mesh.iso_threshold
            )
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
