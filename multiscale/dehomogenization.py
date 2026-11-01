"""Spinodal Gaussian random field de-homogenization on fine multi-scale grids.

Implements Kumar et al. (2020) and Groen & Sigmund (2018) multiscale de-homogenization:
1. Synthesizes an isotropic or anisotropic Gaussian Random Field (GRF):
   G(x) = sqrt(2 / N_w) * sum_{m=1}^{N_w} cos(beta_0 * n_m . x + phi_m)
   with unit wave vectors n_m and random phase angles phi_m in [0, 2*pi).
2. Computes the exact Gaussian quantile / inverse error function threshold:
   tau(x) = sqrt(2) * erfinv(1 - 2 * rho(x))
   so that P(G(x) > tau(x)) = rho(x) matches the macroscopic target volume fraction.
3. Applies smooth Heaviside or crisp indicator projection to generate the physical
   fine-scale composite/lattice geometry.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfinv

from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D


class SpinodalDehomogenizer:
    """Multiscale spinodal de-homogenizer using Gaussian Random Field synthesis."""

    def __init__(
        self,
        beta_0: float = 20.0,
        n_waves: int = 120,
        dim: int = 2,
        anisotropy_angle: Optional[float] = None,
        cone_half_angle: Optional[float] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize SpinodalDehomogenizer.

        Parameters
        ----------
        beta_0 : float
            Characteristic wave number beta_0 = 2*pi / lambda_0 (controls pore/strut size).
        n_waves : int
            Number of plane wave modes N_w (>= 20 recommended for Gaussianity).
        dim : int
            Spatial dimension (2 or 3).
        anisotropy_angle : Optional[float]
            Preferred orientation angle in radians (2D) or azimuth (3D).
        cone_half_angle : Optional[float]
            Half-angle in radians of the directional wave cone around anisotropy_angle.
        seed : Optional[int]
            Random seed for reproducible wave vector and phase generation.
        """
        if beta_0 <= 0.0:
            raise ValueError(f"beta_0 must be positive, got {beta_0}")
        if n_waves < 10:
            raise ValueError(f"n_waves must be >= 10, got {n_waves}")
        if dim not in (2, 3):
            raise ValueError(f"Spatial dimension dim must be 2 or 3, got {dim}")

        self.beta_0: float = float(beta_0)
        self.n_waves: int = int(n_waves)
        self.dim: int = int(dim)
        self.anisotropy_angle: Optional[float] = anisotropy_angle
        self.cone_half_angle: Optional[float] = cone_half_angle
        self.seed: Optional[int] = seed

        # Generate wave vectors and phases
        self.wave_vectors, self.phases = self._generate_wave_modes()

    def _generate_wave_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate unit wave direction vectors n_m and phases phi_m.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            wave_vectors of shape (N_w, dim) and phases of shape (N_w,).
        """
        rng = np.random.RandomState(self.seed)

        if self.dim == 2:
            if self.anisotropy_angle is not None and self.cone_half_angle is not None:
                # Sample angles within cone [theta - d_theta, theta + d_theta] and opposite cone
                d_th = float(self.cone_half_angle)
                th0 = float(self.anisotropy_angle)
                half_nw = self.n_waves // 2
                rem = self.n_waves - half_nw

                a1 = rng.uniform(th0 - d_th, th0 + d_th, half_nw)
                a2 = rng.uniform(th0 + np.pi - d_th, th0 + np.pi + d_th, rem)
                angles = np.concatenate([a1, a2])
            else:
                # Isotropic uniform distribution on unit circle
                angles = rng.uniform(0.0, 2.0 * np.pi, self.n_waves)

            wave_vectors = np.column_stack([np.cos(angles), np.sin(angles)])

        elif self.dim == 3:
            if self.anisotropy_angle is not None and self.cone_half_angle is not None:
                # 3D cone around unit direction vector
                th0 = float(self.anisotropy_angle)
                d_th = float(self.cone_half_angle)
                # Primary axis in XY plane at th0, z = 0
                v_axis = np.array([np.cos(th0), np.sin(th0), 0.0])

                # Sample random unit vectors with rejection sampling or spherical cap
                vecs = []
                while len(vecs) < self.n_waves:
                    cand = rng.normal(size=3)
                    norm = np.linalg.norm(cand)
                    if norm < 1e-12:
                        continue
                    cand /= norm
                    cos_angle = abs(np.dot(cand, v_axis))
                    if cos_angle >= np.cos(d_th):
                        vecs.append(cand)
                wave_vectors = np.array(vecs, dtype=np.float64)
            else:
                # Isotropic 3D uniform distribution on unit sphere
                gauss = rng.normal(size=(self.n_waves, 3))
                norms = np.linalg.norm(gauss, axis=1, keepdims=True)
                wave_vectors = gauss / norms

        phases = rng.uniform(0.0, 2.0 * np.pi, self.n_waves)
        return wave_vectors, phases

    def evaluate_grf(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate normalized Gaussian Random Field G(x) at coordinate points.

        G(x) = sqrt(2 / N_w) * sum_{m=1}^{N_w} cos(beta_0 * n_m . x + phi_m).

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N_pts, dim) or (dim,).

        Returns
        -------
        np.ndarray
            G array of shape (N_pts,).
        """
        pts = np.asarray(coords, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] != self.dim:
            raise ValueError(f"coords dim {pts.shape[1]} does not match self.dim {self.dim}")

        # Projection: (N_pts, N_w) = pts @ wave_vectors.T
        proj = pts @ self.wave_vectors.T
        # Sum cosines: G = sqrt(2 / N_w) * sum_m cos(beta_0 * proj + phi_m)
        arg = self.beta_0 * proj + self.phases[None, :]
        G = np.sqrt(2.0 / self.n_waves) * np.sum(np.cos(arg), axis=1)
        return G

    @staticmethod
    def compute_threshold(rho: Union[float, np.ndarray], eps: float = 1e-6) -> Union[float, np.ndarray]:
        """Compute exact Gaussian quantile threshold tau(rho).

        tau = sqrt(2) * erfinv(1 - 2 * rho)

        Parameters
        ----------
        rho : float or np.ndarray
            Relative density in (0, 1).
        eps : float
            Boundary clipping tolerance to prevent infinite thresholds.

        Returns
        -------
        float or np.ndarray
            Threshold value tau matching P(G > tau) = rho.
        """
        rho_clipped = np.clip(np.asarray(rho, dtype=np.float64), eps, 1.0 - eps)
        tau = np.sqrt(2.0) * erfinv(1.0 - 2.0 * rho_clipped)
        if isinstance(rho, (float, int)):
            return float(tau)
        return tau

    def dehomogenize(
        self,
        macro_rho: np.ndarray,
        coarse_mesh: Union[StructuredMesh2D, StructuredMesh3D],
        fine_mesh_or_coords: Union[StructuredMesh2D, StructuredMesh3D, np.ndarray],
        beta: float = 32.0,
        projection_type: str = "smooth",
    ) -> np.ndarray:
        """Synthesize fine-scale de-homogenized density field.

        Parameters
        ----------
        macro_rho : np.ndarray
            Macroscopic density array on coarse_mesh (N_coarse_elem,).
        coarse_mesh : StructuredMesh2D or StructuredMesh3D
            Coarse macroscopic finite element mesh.
        fine_mesh_or_coords : StructuredMesh2D, StructuredMesh3D, or np.ndarray
            Target high-resolution mesh or coordinate array (N_fine, dim).
        beta : float
            Heaviside smoothness parameter beta >= 1.0.
        projection_type : str
            'smooth' for sigmoid H_beta, or 'crisp' for binary 0/1 indicator.

        Returns
        -------
        np.ndarray
            High-resolution physical density field rho_fine (N_fine,).
        """
        macro_rho_arr = np.asarray(macro_rho, dtype=np.float64)
        if macro_rho_arr.shape[0] != coarse_mesh.n_elem:
            raise ValueError(
                f"macro_rho length {macro_rho_arr.shape[0]} does not match coarse_mesh.n_elem {coarse_mesh.n_elem}"
            )

        # 1. Extract fine evaluation coordinates
        if isinstance(fine_mesh_or_coords, (StructuredMesh2D, StructuredMesh3D)):
            fine_coords = fine_mesh_or_coords.elem_centers
        else:
            fine_coords = np.asarray(fine_mesh_or_coords, dtype=np.float64)

        # 2. Interpolate coarse density to fine coordinates
        if isinstance(coarse_mesh, StructuredMesh2D):
            # Coarse element center grids
            dx, dy = coarse_mesh.dx, coarse_mesh.dy
            x_coarse = (np.arange(coarse_mesh.nelx) + 0.5) * dx
            y_coarse = (np.arange(coarse_mesh.nely) + 0.5) * dy
            rho_grid = macro_rho_arr.reshape((coarse_mesh.nely, coarse_mesh.nelx))

            interp = RegularGridInterpolator(
                (y_coarse, x_coarse),
                rho_grid,
                bounds_error=False,
                fill_value=None,  # extrapolate nearest
            )
            pts_eval = np.column_stack([fine_coords[:, 1], fine_coords[:, 0]])  # (y, x)
            rho_interp = interp(pts_eval)

        elif isinstance(coarse_mesh, StructuredMesh3D):
            dx, dy, dz = coarse_mesh.dx, coarse_mesh.dy, coarse_mesh.dz
            x_coarse = (np.arange(coarse_mesh.nelx) + 0.5) * dx
            y_coarse = (np.arange(coarse_mesh.nely) + 0.5) * dy
            z_coarse = (np.arange(coarse_mesh.nelz) + 0.5) * dz
            rho_grid = macro_rho_arr.reshape((coarse_mesh.nelz, coarse_mesh.nely, coarse_mesh.nelx))

            interp = RegularGridInterpolator(
                (z_coarse, y_coarse, x_coarse),
                rho_grid,
                bounds_error=False,
                fill_value=None,
            )
            pts_eval = np.column_stack([fine_coords[:, 2], fine_coords[:, 1], fine_coords[:, 0]])
            rho_interp = interp(pts_eval)
        else:
            raise TypeError(f"Unsupported coarse mesh type: {type(coarse_mesh)}")

        # Clip interpolated density to physical bounds
        rho_interp = np.clip(rho_interp, 1e-4, 1.0 - 1e-4)

        # 3. Compute local quantile threshold tau(x) and evaluate GRF G(x)
        tau = self.compute_threshold(rho_interp)
        G = self.evaluate_grf(fine_coords)

        # 4. Projection
        diff = G - tau
        if projection_type.lower() == "crisp":
            rho_fine = (diff > 0.0).astype(np.float64)
        elif projection_type.lower() == "smooth":
            # Logistic sigmoid with sharpness beta: 1 / (1 + exp(-2 * beta * (G - tau)))
            # Clip argument to avoid overflow
            arg = np.clip(-2.0 * beta * diff, -60.0, 60.0)
            rho_fine = 1.0 / (1.0 + np.exp(arg))
        else:
            raise ValueError(f"Unknown projection_type '{projection_type}'. Must be 'smooth' or 'crisp'")

        return rho_fine
