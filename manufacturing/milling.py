"""Langelaar (2018) multi-axis CNC milling filter for subtractive manufacturing.

Implements directional ray-based shadow/undercut propagation from tool approach
faces and smooth multi-axis minimum aggregation with exact reverse-sweep adjoint
sensitivities for 2D and 3D regular structured meshes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from wht_topopt.core.filter import FilterOperator
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D


class MillingFilter(FilterOperator):
    """Multi-axis CNC milling filter (Langelaar 2018).

    Enforces that internal cavities and undercuts not reachable by the specified
    milling tool directions are filled, preventing unmachinable geometries.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D],
        directions: Optional[Sequence[str]] = None,
        q: float = 20.0,
        epsilon_max: float = 1e-4,
    ) -> None:
        """Initialize MillingFilter.

        Parameters
        ----------
        mesh : StructuredMesh2D or StructuredMesh3D
            Computational mesh.
        directions : Sequence[str], optional
            Tool approach directions. Default in 2D: ['+y', '-y', '+x', '-x'].
            Default in 3D: ['+z', '-z', '+y', '-y', '+x', '-x'].
        q : float
            Smooth minimum scaling parameter for multi-direction aggregation (default 20.0).
        epsilon_max : float
            Regularization constant for smooth maximum along rays (default 1e-4).
        """
        if not isinstance(mesh, (StructuredMesh2D, StructuredMesh3D)):
            raise TypeError("MillingFilter requires StructuredMesh2D or StructuredMesh3D")

        self.mesh = mesh
        self.q: float = float(q)
        self.epsilon_max: float = float(epsilon_max)
        self.n_elem: int = mesh.n_elem
        self._last_rho: Optional[np.ndarray] = None

        if directions is None:
            if isinstance(mesh, StructuredMesh2D):
                directions = ["+y", "-y", "+x", "-x"]
            else:
                directions = ["+z", "-z", "+y", "-y", "+x", "-x"]

        self.directions: List[str] = [d.lower() for d in directions]
        self.n_dirs: int = len(self.directions)
        if self.n_dirs == 0:
            raise ValueError("MillingFilter requires at least one approach direction")

        # Precompute 2D ray index arrays for each direction: (N_rays, Ray_length)
        self._build_ray_indices()

    def _build_ray_indices(self) -> None:
        """Assemble ray indices for each approach direction."""
        self.ray_indices: Dict[str, np.ndarray] = {}

        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            for d in self.directions:
                if d == "+y":
                    # Tool enters at y = ny - 1, travels down to 0
                    rays = np.empty((nx, ny), dtype=np.int64)
                    for ex in range(nx):
                        rays[ex, :] = [k * nx + ex for k in range(ny - 1, -1, -1)]
                    self.ray_indices[d] = rays
                elif d == "-y":
                    # Tool enters at y = 0, travels up to ny - 1
                    rays = np.empty((nx, ny), dtype=np.int64)
                    for ex in range(nx):
                        rays[ex, :] = [k * nx + ex for k in range(ny)]
                    self.ray_indices[d] = rays
                elif d == "+x":
                    # Tool enters at x = nx - 1, travels down to 0
                    rays = np.empty((ny, nx), dtype=np.int64)
                    for ey in range(ny):
                        rays[ey, :] = [ey * nx + k for k in range(nx - 1, -1, -1)]
                    self.ray_indices[d] = rays
                elif d == "-x":
                    # Tool enters at x = 0, travels up to nx - 1
                    rays = np.empty((ny, nx), dtype=np.int64)
                    for ey in range(ny):
                        rays[ey, :] = [ey * nx + k for k in range(nx)]
                    self.ray_indices[d] = rays
                else:
                    raise ValueError(f"Unsupported 2D tool direction '{d}'. Use '+y', '-y', '+x', '-x'.")

        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            # elem_id = (ez * ny + ey) * nx + ex
            for d in self.directions:
                if d == "+z":
                    rays = np.empty((nx * ny, nz), dtype=np.int64)
                    idx = 0
                    for ey in range(ny):
                        for ex in range(nx):
                            rays[idx, :] = [((k * ny + ey) * nx + ex) for k in range(nz - 1, -1, -1)]
                            idx += 1
                    self.ray_indices[d] = rays
                elif d == "-z":
                    rays = np.empty((nx * ny, nz), dtype=np.int64)
                    idx = 0
                    for ey in range(ny):
                        for ex in range(nx):
                            rays[idx, :] = [((k * ny + ey) * nx + ex) for k in range(nz)]
                            idx += 1
                    self.ray_indices[d] = rays
                elif d == "+y":
                    rays = np.empty((nx * nz, ny), dtype=np.int64)
                    idx = 0
                    for ez in range(nz):
                        for ex in range(nx):
                            rays[idx, :] = [((ez * ny + k) * nx + ex) for k in range(ny - 1, -1, -1)]
                            idx += 1
                    self.ray_indices[d] = rays
                elif d == "-y":
                    rays = np.empty((nx * nz, ny), dtype=np.int64)
                    idx = 0
                    for ez in range(nz):
                        for ex in range(nx):
                            rays[idx, :] = [((ez * ny + k) * nx + ex) for k in range(ny)]
                            idx += 1
                    self.ray_indices[d] = rays
                elif d == "+x":
                    rays = np.empty((ny * nz, nx), dtype=np.int64)
                    idx = 0
                    for ez in range(nz):
                        for ey in range(ny):
                            rays[idx, :] = [((ez * ny + ey) * nx + k) for k in range(nx - 1, -1, -1)]
                            idx += 1
                    self.ray_indices[d] = rays
                elif d == "-x":
                    rays = np.empty((ny * nz, nx), dtype=np.int64)
                    idx = 0
                    for ez in range(nz):
                        for ey in range(ny):
                            rays[idx, :] = [((ez * ny + ey) * nx + k) for k in range(nx)]
                            idx += 1
                    self.ray_indices[d] = rays
                else:
                    raise ValueError(f"Unsupported 3D tool direction '{d}'. Use '+z', '-z', '+y', '-y', '+x', '-x'.")

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward multi-axis milling filter sweep.

        Parameters
        ----------
        rho : np.ndarray
            Blueprint density field of shape (N_elem,).

        Returns
        -------
        np.ndarray
            Machinable physical density array of shape (N_elem,).
        """
        rho = np.asarray(rho, dtype=np.float64)
        self._last_rho = rho.copy()
        M = self.n_dirs

        # Compute running smooth maximum along rays for each direction
        y_mach = np.empty((M, self.n_elem), dtype=np.float64)

        for m_idx, d in enumerate(self.directions):
            ray_idx = self.ray_indices[d]  # shape (N_rays, Ray_length)
            rho_ray = rho[ray_idx]
            Y = np.empty_like(rho_ray)

            Y[:, 0] = rho_ray[:, 0]
            ray_len = ray_idx.shape[1]
            for k in range(1, ray_len):
                a = rho_ray[:, k]
                b = Y[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_max)
                Y[:, k] = 0.5 * (a + b + rad - np.sqrt(self.epsilon_max))

            y_mach[m_idx, ray_idx] = Y

        # Multi-direction smooth minimum aggregation via log-sum-exp
        y_min = np.min(y_mach, axis=0)  # (N_elem,)
        shift = y_mach - y_min[None, :]  # (M, N_elem) >= 0
        exp_terms = np.exp(-self.q * shift)  # (M, N_elem)
        sum_exp = np.sum(exp_terms, axis=0)  # (N_elem,)

        # smin = -1/Q * ln(1/M * sum exp(-Q * y)) = y_min - 1/Q * ln(sum_exp / M)
        xi = y_min - (1.0 / self.q) * np.log(sum_exp / float(M))
        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Adjoint backward sensitivity sweep across all milling directions.

        Parameters
        ----------
        df_dxi : np.ndarray
            Sensitivity vector dJ/dxi of shape (N_elem,).
        rho : Optional[np.ndarray]
            Blueprint density array of shape (N_elem,).

        Returns
        -------
        np.ndarray
            Sensitivity vector dJ/drho of shape (N_elem,).
        """
        df_dxi = np.asarray(df_dxi, dtype=np.float64)
        if rho is None:
            if self._last_rho is None:
                raise ValueError("rho is required for backward sensitivity evaluation")
            rho = self._last_rho
        else:
            rho = np.asarray(rho, dtype=np.float64)

        M = self.n_dirs
        y_mach = np.empty((M, self.n_elem), dtype=np.float64)
        Y_rays: List[np.ndarray] = []

        for m_idx, d in enumerate(self.directions):
            ray_idx = self.ray_indices[d]
            rho_ray = rho[ray_idx]
            Y = np.empty_like(rho_ray)

            Y[:, 0] = rho_ray[:, 0]
            ray_len = ray_idx.shape[1]
            for k in range(1, ray_len):
                a = rho_ray[:, k]
                b = Y[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_max)
                Y[:, k] = 0.5 * (a + b + rad - np.sqrt(self.epsilon_max))

            y_mach[m_idx, ray_idx] = Y
            Y_rays.append(Y)

        # Multi-direction smooth minimum sensitivities
        y_min = np.min(y_mach, axis=0)
        shift = y_mach - y_min[None, :]
        exp_terms = np.exp(-self.q * shift)
        sum_exp = np.sum(exp_terms, axis=0)  # (N_elem,)

        # dxi / d(y_m) = exp(-Q * shift_m) / sum_exp
        weights_m = exp_terms / (sum_exp[None, :] + 1e-30)  # (M, N_elem)
        w_all = df_dxi[None, :] * weights_m  # (M, N_elem)

        dJ_drho = np.zeros(self.n_elem, dtype=np.float64)

        # Reverse sweep along each direction
        for m_idx, d in enumerate(self.directions):
            ray_idx = self.ray_indices[d]
            rho_ray = rho[ray_idx]
            Y = Y_rays[m_idx]
            ray_len = ray_idx.shape[1]

            W = w_all[m_idx, ray_idx]  # (N_rays, Ray_length)
            Lambda = np.empty_like(W)
            Lambda[:, -1] = W[:, -1]

            drho_ray = np.zeros_like(rho_ray)

            for k in range(ray_len - 1, 0, -1):
                a = rho_ray[:, k]
                b = Y[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_max)

                dsmax_da = 0.5 * (1.0 + diff / rad)
                dsmax_db = 0.5 * (1.0 - diff / rad)

                lam_k = Lambda[:, k]
                drho_ray[:, k] = lam_k * dsmax_da
                Lambda[:, k - 1] = W[:, k - 1] + lam_k * dsmax_db

            drho_ray[:, 0] = Lambda[:, 0]

            # Accumulate sensitivities
            np.add.at(dJ_drho, ray_idx.ravel(), drho_ray.ravel())

        return dJ_drho

    def jacobian_matvec(self, v: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Directional derivative J @ v."""
        if rho is None:
            return self.forward(v)
        h = 1e-7
        return (self.forward(rho + h * v) - self.forward(rho - h * v)) / (2.0 * h)
