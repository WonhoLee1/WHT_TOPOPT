"""TONIC: Hybrid neural-physics warm-start & implicit coordinate-MLP density representation.

Implements implicit coordinate network topology optimization (Hoyer et al. 2019;
Chandrasekhar & Suresh 2021; Zehnder et al. 2021; METHODS_SURVEY_2026 §5(E)):
- Sinusoidal Fourier feature mapping gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)] to eliminate spectral bias.
- Swish coordinate MLP rho_theta(x) = sigma(MLP_theta(gamma(x)) - b_vol) with volume-preserving offset b_vol.
- Analytical vector-Jacobian product projecting objective sensitivities df/drho to network weight gradients.
- Integrated Adam optimizer for end-to-end neural reparameterization topology optimization.
- Physics-based strain energy trajectory warm-start synthesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ..core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from ..core.boundary import BoundaryConditionManager


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function: f(x) = x * sigmoid(x)."""
    return x * _sigmoid(x)


def _swish_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of Swish: f'(x) = sigma(x) * (1 + x * (1 - sigma(x)))."""
    sig = _sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


class TONICInitializer:
    """TONIC implicit coordinate-MLP density parameterization and warm-start engine."""

    def __init__(
        self,
        ndim: int = 2,
        hidden_dim: int = 64,
        num_fourier_features: int = 32,
        fourier_scale: float = 2.0,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        """Initialize coordinate MLP architecture and optimizer state.

        Parameters
        ----------
        ndim : int, default 2
            Spatial coordinate dimension (2 or 3).
        hidden_dim : int, default 64
            Hidden layer neuron count.
        num_fourier_features : int, default 32
            Number of random Fourier frequencies M.
        fourier_scale : float, default 2.0
            Standard deviation for random Fourier projection matrix B.
        learning_rate : float, default 1e-3
            Adam optimization learning rate.
        seed : int, default 42
            Random seed for deterministic initialization.
        """
        self.ndim: int = ndim
        self.hidden_dim: int = hidden_dim
        self.num_fourier: int = num_fourier_features
        self.lr: float = learning_rate
        self.b_vol: float = 0.0

        rng = np.random.RandomState(seed)

        # Random Gaussian Fourier projection matrix B in R^(M x ndim)
        self.B: np.ndarray = rng.randn(num_fourier_features, ndim) * fourier_scale
        in_dim = 2 * num_fourier_features

        # Layer 1: in_dim -> hidden_dim
        self.W1: np.ndarray = rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1: np.ndarray = np.zeros(hidden_dim)

        # Layer 2: hidden_dim -> hidden_dim
        self.W2: np.ndarray = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2: np.ndarray = np.zeros(hidden_dim)

        # Layer 3: hidden_dim -> 1
        self.W3: np.ndarray = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b3: np.ndarray = np.zeros(1)

        # Adam optimizer state variables
        self._step_count: int = 0
        self._m: Dict[str, np.ndarray] = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
            "W3": np.zeros_like(self.W3), "b3": np.zeros_like(self.b3),
        }
        self._v: Dict[str, np.ndarray] = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
            "W3": np.zeros_like(self.W3), "b3": np.zeros_like(self.b3),
        }

        # Cache for backpropagation
        self._last_coords: Optional[np.ndarray] = None
        self._last_fourier: Optional[np.ndarray] = None
        self._last_a1: Optional[np.ndarray] = None
        self._last_h1: Optional[np.ndarray] = None
        self._last_a2: Optional[np.ndarray] = None
        self._last_h2: Optional[np.ndarray] = None
        self._last_z: Optional[np.ndarray] = None
        self._last_rho: Optional[np.ndarray] = None

    def fourier_features(self, coords: np.ndarray) -> np.ndarray:
        """Map physical coordinates x to Fourier feature representations.

        gamma(x) = [cos(2*pi*x*B^T), sin(2*pi*x*B^T)] in R^(2*M)

        Parameters
        ----------
        coords : np.ndarray of shape (N, ndim)
            Physical query coordinates.

        Returns
        -------
        gamma : np.ndarray of shape (N, 2*num_fourier_features)
            Fourier feature representation.
        """
        pts = np.asarray(coords, dtype=np.float64)
        proj = 2.0 * np.pi * (pts @ self.B.T)
        return np.column_stack([np.cos(proj), np.sin(proj)])

    def forward(
        self,
        coords: np.ndarray,
        b_offset: Optional[float] = None,
        cache_activations: bool = True,
    ) -> np.ndarray:
        """Evaluate continuous implicit density field at arbitrary coordinate points.

        Parameters
        ----------
        coords : np.ndarray of shape (N, ndim)
            Spatial query points.
        b_offset : float, optional
            Threshold bias offset for volume fraction matching. Defaults to self.b_vol.
        cache_activations : bool, default True
            If True, stores forward activations for subsequent analytical backpropagation.

        Returns
        -------
        rho : np.ndarray of shape (N,)
            Predicted continuous physical densities in [0, 1].
        """
        pts = np.asarray(coords, dtype=np.float64)
        offset = self.b_vol if b_offset is None else float(b_offset)

        # 1. Fourier positional mapping
        gamma = self.fourier_features(pts)

        # 2. Hidden layer 1
        a1 = gamma @ self.W1 + self.b1
        h1 = _swish(a1)

        # 3. Hidden layer 2
        a2 = h1 @ self.W2 + self.b2
        h2 = _swish(a2)

        # 4. Output projection
        z = (h2 @ self.W3 + self.b3).flatten()

        # 5. Volume-preserving sigmoid
        rho = _sigmoid(z - offset)

        if cache_activations:
            self._last_coords = pts
            self._last_fourier = gamma
            self._last_a1 = a1
            self._last_h1 = h1
            self._last_a2 = a2
            self._last_h2 = h2
            self._last_z = z
            self._last_rho = rho

        return rho

    def compute_volume_offset(
        self,
        coords: np.ndarray,
        elem_volumes: np.ndarray,
        target_vol_frac: float,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
        rho_min: float = 1e-3,
    ) -> float:
        """Find scalar offset b_vol via bisection to satisfy target volume fraction.

        Parameters
        ----------
        coords : np.ndarray of shape (N_elem, ndim)
            Element centroids.
        elem_volumes : np.ndarray of shape (N_elem,)
            Volume of each element.
        target_vol_frac : float
            Target solid volume fraction.
        passive_solid : np.ndarray of bool, optional
            Mask of non-design solid elements.
        passive_void : np.ndarray of bool, optional
            Mask of non-design void elements.
        rho_min : float, default 1e-3
            Void lower bound.

        Returns
        -------
        float
            Solved volume offset b_vol.
        """
        pts = np.asarray(coords, dtype=np.float64)
        gamma = self.fourier_features(pts)
        h1 = _swish(gamma @ self.W1 + self.b1)
        h2 = _swish(h1 @ self.W2 + self.b2)
        z = (h2 @ self.W3 + self.b3).flatten()

        v_tot = float(np.sum(elem_volumes))
        target_v = target_vol_frac * v_tot

        b_low = float(np.min(z) - 20.0)
        b_high = float(np.max(z) + 20.0)

        for _ in range(40):
            b_mid = 0.5 * (b_low + b_high)
            rho_test = _sigmoid(z - b_mid)
            if passive_solid is not None:
                rho_test[passive_solid] = 1.0
            if passive_void is not None:
                rho_test[passive_void] = rho_min

            v_cur = float(np.sum(rho_test * elem_volumes))
            if abs(v_cur - target_v) / v_tot < 1e-5:
                break
            if v_cur < target_v:
                b_high = b_mid
            else:
                b_low = b_mid

        self.b_vol = b_mid
        return self.b_vol

    def project_sensitivities_to_weights(
        self,
        df_drho: np.ndarray,
        coords: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Backpropagate objective sensitivities to network parameter gradients via vector-Jacobian product.

        Computes:
            grad_theta = sum_e (df / drho_e) * (drho_e / d_theta)

        Parameters
        ----------
        df_drho : np.ndarray of shape (N_elem,)
            Physical sensitivities computed by the FE solver (e.g. compliance or stress).
        coords : np.ndarray of shape (N_elem, ndim), optional
            Element centroid coordinates (if different from cached forward evaluation).

        Returns
        -------
        grads : dict of str -> np.ndarray
            Parameter gradients {'W1', 'b1', 'W2', 'b2', 'W3', 'b3'}.
        """
        sens = np.asarray(df_drho, dtype=np.float64).flatten()

        if coords is not None or self._last_rho is None:
            c = coords if coords is not None else self._last_coords
            if c is None:
                raise ValueError("Coordinates must be supplied if forward() was not called.")
            self.forward(c, cache_activations=True)

        assert self._last_rho is not None
        assert self._last_h2 is not None
        assert self._last_a2 is not None
        assert self._last_h1 is not None
        assert self._last_a1 is not None
        assert self._last_fourier is not None

        # 1. drho / dz = rho * (1 - rho)
        # dL / dz = (dL / drho) * (drho / dz)
        drho_dz = self._last_rho * (1.0 - self._last_rho)
        delta_z = (sens * drho_dz)[:, None]  # (N, 1)

        # 2. Gradients for Layer 3 (W3: hidden x 1, b3: 1)
        grad_W3 = self._last_h2.T @ delta_z
        grad_b3 = np.sum(delta_z, axis=0)

        # 3. Backpropagate to Layer 2
        # delta_h2 = delta_z @ W3.T * swish'(a2)
        delta_h2 = (delta_z @ self.W3.T) * _swish_grad(self._last_a2)
        grad_W2 = self._last_h1.T @ delta_h2
        grad_b2 = np.sum(delta_h2, axis=0)

        # 4. Backpropagate to Layer 1
        # delta_h1 = delta_h2 @ W2.T * swish'(a1)
        delta_h1 = (delta_h2 @ self.W2.T) * _swish_grad(self._last_a1)
        grad_W1 = self._last_fourier.T @ delta_h1
        grad_b1 = np.sum(delta_h1, axis=0)

        return {
            "W1": grad_W1, "b1": grad_b1,
            "W2": grad_W2, "b2": grad_b2,
            "W3": grad_W3, "b3": grad_b3,
        }

    def step(
        self,
        grads: Dict[str, np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Apply an Adam optimizer update step to all network parameters.

        Parameters
        ----------
        grads : dict of str -> np.ndarray
            Parameter gradients returned by project_sensitivities_to_weights.
        beta1 : float, default 0.9
            First moment decay coefficient.
        beta2 : float, default 0.999
            Second moment decay coefficient.
        eps : float, default 1e-8
            Small constant for numerical stability.
        """
        self._step_count += 1
        t = self._step_count

        param_dict = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

        for key, p in param_dict.items():
            g = grads[key]
            # Update biased first moment estimate
            self._m[key] = beta1 * self._m[key] + (1.0 - beta1) * g
            # Update biased second raw moment estimate
            self._v[key] = beta2 * self._v[key] + (1.0 - beta2) * (g ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self._m[key] / (1.0 - beta1 ** t)
            v_hat = self._v[key] / (1.0 - beta2 ** t)

            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def warm_start(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D],
        bcm: BoundaryConditionManager,
        target_vol_frac: float = 0.5,
        num_pretrain_steps: int = 40,
        rho_min: float = 1e-3,
    ) -> np.ndarray:
        """Pretrain coordinate network to a physics-informed strain energy trajectory proxy.

        Synthesizes a continuous warm-start density field using geometric distance
        decay along stress transmission lines between point loads and fixed constraints,
        and fits the coordinate MLP to match this distribution prior to SIMP/MMA iterations.

        Parameters
        ----------
        mesh : StructuredMesh2D or StructuredMesh3D
            Computational mesh instance.
        bcm : BoundaryConditionManager
            Boundary constraints and external loads.
        target_vol_frac : float, default 0.5
            Prescribed target volume fraction.
        num_pretrain_steps : int, default 40
            Number of Adam optimization steps for neural fitting.
        rho_min : float, default 1e-3
            Void density lower bound.

        Returns
        -------
        initial_density : np.ndarray of shape (N_elem,)
            Volume-constrained warm-start density field evaluated from the trained coordinate MLP.
        """
        centroids = mesh.elem_centers
        n_elem = len(centroids)

        # 1. Synthesize target physics-informed load path field
        fixed_dofs = bcm.fixed_dofs
        F_mat = bcm.F
        loaded_dofs = np.flatnonzero(np.any(F_mat != 0.0, axis=1))

        heuristic_field = np.full(n_elem, target_vol_frac, dtype=np.float64)

        if len(fixed_dofs) > 0 and len(loaded_dofs) > 0:
            supp_nodes = np.unique(fixed_dofs // mesh.ndof_per_node)
            supp_coords = mesh.node_coords[supp_nodes]
            load_nodes = np.unique(loaded_dofs // mesh.ndof_per_node)
            load_coords = mesh.node_coords[load_nodes]

            load_path_boost = np.zeros(n_elem, dtype=np.float64)
            for l_pt in load_coords:
                for s_pt in supp_coords:
                    seg = s_pt - l_pt
                    seg_len_sq = np.sum(seg ** 2)
                    if seg_len_sq < 1e-12:
                        continue
                    v = centroids - l_pt
                    t = np.clip(np.sum(v * seg, axis=1) / seg_len_sq, 0.0, 1.0)
                    proj_pts = l_pt + t[:, None] * seg
                    dist_sq = np.sum((centroids - proj_pts) ** 2, axis=1)
                    char_len = 0.20 * max(mesh.lx, mesh.ly)
                    load_path_boost += np.exp(-dist_sq / (2.0 * char_len ** 2))

            norm_boost = load_path_boost / np.maximum(np.max(load_path_boost), 1e-12)
            heuristic_field = 0.1 + 0.8 * norm_boost

        heuristic_field[bcm.passive_solid_mask] = 1.0
        heuristic_field[bcm.passive_void_mask] = rho_min

        # 2. Pretrain network to match heuristic field via MSE loss
        lr_orig = self.lr
        self.lr = 5e-3

        for _ in range(num_pretrain_steps):
            rho_pred = self.forward(centroids, cache_activations=True)
            # MSE loss: L = 1/2 * ||rho_pred - heuristic_field||^2
            # dL / drho_pred = rho_pred - heuristic_field
            dL_drho = (rho_pred - heuristic_field)
            grads = self.project_sensitivities_to_weights(dL_drho, centroids)
            self.step(grads)

        self.lr = lr_orig

        # 3. Find volume-preserving offset b_vol
        self.compute_volume_offset(
            centroids,
            mesh.elem_volumes,
            target_vol_frac,
            passive_solid=bcm.passive_solid_mask,
            passive_void=bcm.passive_void_mask,
            rho_min=rho_min,
        )

        final_rho = self.forward(centroids, cache_activations=True)
        final_rho[bcm.passive_solid_mask] = 1.0
        final_rho[bcm.passive_void_mask] = rho_min
        return np.clip(final_rho, rho_min, 1.0)
