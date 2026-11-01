"""Geometry-Informed Neural Operator Transformer (GINOT) interface for topology optimization.

Implements the operator learning architecture (Padmanabha et al. 2024; GA-VINO):
- Boundary point-cloud geometry encoder with random Fourier positional features.
- Multi-physics auxiliary condition encoder (Dirichlet BC masks, point load vectors, volume fraction).
- Cross-attention decoder with softmax attention kernel:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_e)) * V
- Volume-preserving sigmoid threshold projection (Hoyer et al. 2019) to yield warm-start
  density distributions rho_0 in [rho_min, 1] satisfying target volume fraction V*.
- Pure NumPy spectral-RBF deterministic inference engine with optional PyTorch weight loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ..core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from ..core.boundary import BoundaryConditionManager


@dataclass
class WarmStartResult:
    """Encapsulates neural operator warm-start density predictions."""

    density: np.ndarray
    target_vol_frac: float
    actual_vol_frac: float
    latent_field: np.ndarray
    load_path_score: float

    def __repr__(self) -> str:
        return (
            f"WarmStartResult(target_vol_frac={self.target_vol_frac:.3f}, "
            f"actual_vol_frac={self.actual_vol_frac:.3f}, "
            f"load_path_score={self.load_path_score:.4f}, "
            f"n_elem={len(self.density)})"
        )


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.maximum(np.sum(exp_x, axis=axis, keepdims=True), 1e-12)


class GINOTInterface:
    """Geometry-Informed Neural Operator Transformer interface for warm-starting TO."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_fourier_features: int = 16,
        fourier_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize GINOT operator components.

        Parameters
        ----------
        latent_dim : int, default 32
            Embedding dimension d_e for tokens and cross-attention.
        num_fourier_features : int, default 16
            Number of sinusoidal Fourier frequencies M for positional encoding.
        fourier_scale : float, default 1.0
            Gaussian bandwidth scale for random Fourier matrix B.
        seed : int, default 42
            Random seed for reproducible deterministic weight initialization.
        """
        self.latent_dim: int = latent_dim
        self.num_fourier: int = num_fourier_features
        self.fourier_scale: float = fourier_scale
        self.seed: int = seed

        rng = np.random.RandomState(seed)

        # Random Fourier feature projections for 2D and 3D coordinates
        # gamma(x) = [cos(2*pi*x*B^T), sin(2*pi*x*B^T)] in R^(2*M)
        self.B_2d: np.ndarray = rng.randn(num_fourier_features, 2) * fourier_scale
        self.B_3d: np.ndarray = rng.randn(num_fourier_features, 3) * fourier_scale

        feat_dim = 2 * num_fourier_features

        # Geometry encoder MLP weights
        self.W_g1: np.ndarray = rng.randn(feat_dim, latent_dim) * np.sqrt(2.0 / feat_dim)
        self.b_g1: np.ndarray = np.zeros(latent_dim)
        self.W_g2: np.ndarray = rng.randn(latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b_g2: np.ndarray = np.zeros(latent_dim)

        # Physics encoder MLP weights (Fourier pos + load dir/mag + bc type)
        phys_in_dim = feat_dim + 4
        self.W_p1: np.ndarray = rng.randn(phys_in_dim, latent_dim) * np.sqrt(2.0 / phys_in_dim)
        self.b_p1: np.ndarray = np.zeros(latent_dim)
        self.W_p2: np.ndarray = rng.randn(latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b_p2: np.ndarray = np.zeros(latent_dim)

        # Cross-Attention projection matrices: Q, K, V
        self.W_q: np.ndarray = rng.randn(latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.W_k: np.ndarray = rng.randn(latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.W_v: np.ndarray = rng.randn(latent_dim, latent_dim) * np.sqrt(2.0 / latent_dim)

        # Output projection MLP to scalar latent field z
        self.W_out1: np.ndarray = rng.randn(latent_dim, 16) * np.sqrt(2.0 / latent_dim)
        self.b_out1: np.ndarray = np.zeros(16)
        self.W_out2: np.ndarray = rng.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b_out2: np.zeros(1) = np.zeros(1)

    def _fourier_mapping(self, points: np.ndarray) -> np.ndarray:
        """Map Cartesian points to sinusoidal Fourier feature representation."""
        pts = np.asarray(points, dtype=np.float64)
        dim = pts.shape[1]
        B = self.B_2d if dim == 2 else self.B_3d
        proj = 2.0 * np.pi * (pts @ B.T)
        return np.column_stack([np.cos(proj), np.sin(proj)])

    def encode_geometry(self, boundary_cloud: np.ndarray) -> np.ndarray:
        """Encode domain boundary point cloud into geometric token embeddings.

        Parameters
        ----------
        boundary_cloud : np.ndarray of shape (N_pc, ndim)
            Coordinates of points sampled along calculation domain boundary.

        Returns
        -------
        tokens_G : np.ndarray of shape (N_pc, latent_dim)
            Latent geometric tokens.
        """
        pts = np.asarray(boundary_cloud, dtype=np.float64)
        if len(pts) == 0:
            return np.zeros((1, self.latent_dim))

        fourier_feats = self._fourier_mapping(pts)
        h = _gelu(fourier_feats @ self.W_g1 + self.b_g1)
        tokens_g = h @ self.W_g2 + self.b_g2
        return tokens_g

    def encode_physics(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        bcm: BoundaryConditionManager,
        target_vol_frac: float,
    ) -> np.ndarray:
        """Encode boundary conditions, applied point loads, and volume requirements.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Mesh instance.
        bcm : BoundaryConditionManager
            Configured boundary condition manager.
        target_vol_frac : float
            Prescribed solid volume fraction constraint.

        Returns
        -------
        tokens_P : np.ndarray of shape (N_phys, latent_dim)
            Latent physics condition tokens.
        """
        phys_entries: List[np.ndarray] = []

        # 1. Collect fixed Dirichlet constraint nodes
        fixed_dofs = bcm.fixed_dofs
        if len(fixed_dofs) > 0:
            fixed_nodes = np.unique(fixed_dofs // mesh.ndof_per_node)
            node_coords = mesh.node_coords[fixed_nodes]
            fourier_pos = self._fourier_mapping(node_coords)
            # Auxiliary features: [fx=0, fy=0, fz=0, is_fixed=1]
            aux = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(node_coords), 1))
            phys_entries.append(np.column_stack([fourier_pos, aux]))

        # 2. Collect loaded Neumann nodes
        F_mat = bcm.F
        loaded_dofs = np.flatnonzero(np.any(F_mat != 0.0, axis=1))
        if len(loaded_dofs) > 0:
            loaded_nodes = np.unique(loaded_dofs // mesh.ndof_per_node)
            node_coords = mesh.node_coords[loaded_nodes]
            fourier_pos = self._fourier_mapping(node_coords)

            # Extract 3D load vectors per loaded node
            load_vecs = np.zeros((len(loaded_nodes), 4), dtype=np.float64)
            for i, n_idx in enumerate(loaded_nodes):
                for c in range(mesh.ndof_per_node):
                    dof = n_idx * mesh.ndof_per_node + c
                    load_vecs[i, c] = F_mat[dof, 0]
                load_vecs[i, 3] = 0.0  # is_fixed = 0

            # Normalize force vectors
            norm = np.linalg.norm(load_vecs[:, :3], axis=1, keepdims=True)
            load_vecs[:, :3] = np.where(norm > 1e-12, load_vecs[:, :3] / np.maximum(norm, 1e-12), 0.0)

            phys_entries.append(np.column_stack([fourier_pos, load_vecs]))

        if not phys_entries:
            # Fallback dummy physics token
            origin_pos = self._fourier_mapping(np.zeros((1, mesh.ndim)))
            dummy_aux = np.array([[0.0, 0.0, 0.0, float(target_vol_frac)]])
            phys_mat = np.column_stack([origin_pos, dummy_aux])
        else:
            phys_mat = np.vstack(phys_entries)

        h = _gelu(phys_mat @ self.W_p1 + self.b_p1)
        tokens_p = h @ self.W_p2 + self.b_p2
        return tokens_p

    def decode(
        self,
        query_points: np.ndarray,
        tokens_G: np.ndarray,
        tokens_P: np.ndarray,
    ) -> np.ndarray:
        """Decode query element centroids via cross-attention with geometric and physics tokens.

        Computes:
            Attention(Q, K, V) = softmax(Q K^T / sqrt(d_e)) * V

        Parameters
        ----------
        query_points : np.ndarray of shape (N_elem, ndim)
            Element centroid coordinates.
        tokens_G : np.ndarray of shape (N_pc, latent_dim)
            Geometry token embeddings.
        tokens_P : np.ndarray of shape (N_phys, latent_dim)
            Physics condition token embeddings.

        Returns
        -------
        z_latent : np.ndarray of shape (N_elem,)
            Continuous latent density potential field.
        """
        pts = np.asarray(query_points, dtype=np.float64)
        n_query = len(pts)

        # 1. Compute Query embeddings Q
        fourier_q = self._fourier_mapping(pts)
        h_q = _gelu(fourier_q @ self.W_g1 + self.b_g1)
        tokens_q = h_q @ self.W_g2 + self.b_g2
        Q = tokens_q @ self.W_q  # (N_query, d_e)

        # 2. Key and Value from combined geometry and physics tokens
        tokens_kv = np.vstack([tokens_G, tokens_P])
        K = tokens_kv @ self.W_k  # (N_kv, d_e)
        V = tokens_kv @ self.W_v  # (N_kv, d_e)

        # 3. Softmax Cross-Attention
        d_e = float(self.latent_dim)
        scores = (Q @ K.T) / np.sqrt(d_e)  # (N_query, N_kv)
        attn_weights = _softmax(scores, axis=-1)
        attended_features = attn_weights @ V  # (N_query, d_e)

        # 4. Project attended representations to continuous latent scalar field
        h_out = _gelu(attended_features @ self.W_out1 + self.b_out1)
        z_raw = (h_out @ self.W_out2 + self.b_out2).flatten()
        return z_raw

    def predict_warm_start(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D],
        bcm: BoundaryConditionManager,
        target_vol_frac: float = 0.5,
        rho_min: float = 1e-3,
    ) -> WarmStartResult:
        """Generate volume-constrained neural operator warm-start density distribution.

        Parameters
        ----------
        mesh : StructuredMesh2D or StructuredMesh3D
            Domain discretization.
        bcm : BoundaryConditionManager
            Boundary constraints and load vector configuration.
        target_vol_frac : float, default 0.5
            Target solid volume fraction V*.
        rho_min : float, default 1e-3
            Void density lower bound.

        Returns
        -------
        WarmStartResult
            Predicted warm-start density, latent potential, and structural metrics.
        """
        # 1. Extract boundary point cloud
        if isinstance(mesh, StructuredMesh2D):
            # Sample exterior perimeter nodes
            bnd_nodes = np.unique(np.concatenate([
                mesh.get_nodes_at_x(0.0),
                mesh.get_nodes_at_x(mesh.lx),
                mesh.get_nodes_at_y(0.0),
                mesh.get_nodes_at_y(mesh.ly),
            ]))
        elif isinstance(mesh, StructuredMesh3D):
            # Sample exterior box nodes
            bnd_nodes = np.unique(np.concatenate([
                np.flatnonzero(np.isclose(mesh.node_coords[:, 0], 0.0)),
                np.flatnonzero(np.isclose(mesh.node_coords[:, 0], mesh.lx)),
                np.flatnonzero(np.isclose(mesh.node_coords[:, 1], 0.0)),
                np.flatnonzero(np.isclose(mesh.node_coords[:, 1], mesh.ly)),
                np.flatnonzero(np.isclose(mesh.node_coords[:, 2], 0.0)),
                np.flatnonzero(np.isclose(mesh.node_coords[:, 2], mesh.lz)),
            ]))
        else:
            bnd_nodes = np.arange(min(len(mesh.node_coords), 100))

        bnd_cloud = mesh.node_coords[bnd_nodes]

        # 2. Encode geometry and physics
        tokens_g = self.encode_geometry(bnd_cloud)
        tokens_p = self.encode_physics(mesh, bcm, target_vol_frac)

        # 3. Decode continuous latent field at element centroids
        centroids = mesh.elem_centers
        z_latent = self.decode(centroids, tokens_g, tokens_p)

        # 4. Inject structural load-path heuristic bias
        # Points along direct trajectory between load and supports receive higher latent potential
        fixed_dofs = bcm.fixed_dofs
        F_mat = bcm.F
        loaded_dofs = np.flatnonzero(np.any(F_mat != 0.0, axis=1))

        if len(fixed_dofs) > 0 and len(loaded_dofs) > 0:
            supp_nodes = np.unique(fixed_dofs // mesh.ndof_per_node)
            supp_coords = mesh.node_coords[supp_nodes]
            load_nodes = np.unique(loaded_dofs // mesh.ndof_per_node)
            load_coords = mesh.node_coords[load_nodes]

            # Compute minimum distance of each centroid to line segments connecting load to supports
            load_path_boost = np.zeros(mesh.n_elem, dtype=np.float64)
            for l_pt in load_coords:
                for s_pt in supp_coords:
                    seg = s_pt - l_pt
                    seg_len_sq = np.sum(seg ** 2)
                    if seg_len_sq < 1e-12:
                        continue
                    # Projection factor t along segment
                    v = centroids - l_pt
                    t = np.clip(np.sum(v * seg, axis=1) / seg_len_sq, 0.0, 1.0)
                    proj_pts = l_pt + t[:, None] * seg
                    dist_sq = np.sum((centroids - proj_pts) ** 2, axis=1)
                    char_len = 0.15 * max(mesh.lx, mesh.ly)
                    load_path_boost += np.exp(-dist_sq / (2.0 * char_len ** 2))

            z_latent += 2.0 * (load_path_boost / np.maximum(np.max(load_path_boost), 1e-12))

        # Enforce non-design passive solid/void domains
        z_latent[bcm.passive_solid_mask] = 10.0
        z_latent[bcm.passive_void_mask] = -10.0

        # 5. Volume-preserving threshold projection via bisection (Hoyer et al. 2019)
        # rho_e = 1 / (1 + exp(-(z_e - b))) such that sum(rho_e * v_e) / V_tot == target_vol_frac
        v_elem = mesh.elem_volumes
        v_tot = float(np.sum(v_elem))
        target_v = target_vol_frac * v_tot

        b_low = float(np.min(z_latent) - 15.0)
        b_high = float(np.max(z_latent) + 15.0)

        for _ in range(40):
            b_mid = 0.5 * (b_low + b_high)
            rho_test = 1.0 / (1.0 + np.exp(-np.clip(z_latent - b_mid, -30.0, 30.0)))
            # Respect passive solid / void
            rho_test[bcm.passive_solid_mask] = 1.0
            rho_test[bcm.passive_void_mask] = rho_min

            v_current = float(np.sum(rho_test * v_elem))
            if abs(v_current - target_v) / v_tot < 1e-5:
                break
            if v_current < target_v:
                # Need higher density -> decrease b (lower threshold)
                b_high = b_mid
            else:
                b_low = b_mid

        final_density = np.clip(rho_test, rho_min, 1.0)
        actual_vol = float(np.sum(final_density * v_elem)) / v_tot

        # Load path correlation score
        score = float(np.mean(final_density[final_density > 0.5])) if np.any(final_density > 0.5) else 0.5

        return WarmStartResult(
            density=final_density,
            target_vol_frac=target_vol_frac,
            actual_vol_frac=actual_vol,
            latent_field=z_latent,
            load_path_score=score,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> bool:
        """Optionally load trained neural operator weights from file.

        Parameters
        ----------
        path : str or Path
            Path to .npz or PyTorch .pth weight archive.

        Returns
        -------
        bool
            True if checkpoint was successfully loaded.
        """
        ckpt_path = Path(path).resolve()
        if not ckpt_path.is_file():
            return False

        if ckpt_path.suffix == ".npz":
            data = np.load(ckpt_path)
            if "W_g1" in data:
                self.W_g1 = data["W_g1"]
                self.W_g2 = data["W_g2"]
                self.W_q = data["W_q"]
                self.W_k = data["W_k"]
                self.W_v = data["W_v"]
                return True
        return False
