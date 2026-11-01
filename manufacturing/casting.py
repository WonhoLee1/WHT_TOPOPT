"""Casting and mold-ability constraints for die casting and split-mold manufacturing.

Implements single-draw and parting-plane split mold casting filters (Gersborg & Sigmund 2006)
using smooth cumulative minimum projections along draw columns with exact reverse-mode
adjoint sensitivities for 2D and 3D regular structured meshes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from wht_topopt.core.filter import FilterOperator
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D


class CastingFilter(FilterOperator):
    """Die casting and split-mold filter.

    Enforces monotonicity of densities along the mold draw direction so that
    no internal undercuts or trapped voids exist when molds are released.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D],
        draw_axis: str = "y",
        mode: str = "single_draw",
        parting_coord: Optional[float] = None,
        epsilon_c: float = 1e-4,
    ) -> None:
        """Initialize CastingFilter.

        Parameters
        ----------
        mesh : StructuredMesh2D or StructuredMesh3D
            Computational mesh.
        draw_axis : str
            Mold opening direction axis ('y' or 'x' in 2D; 'z', 'y', or 'x' in 3D).
        mode : str
            Casting mode: 'single_draw' for single-die pull, or 'parting_plane'
            for two-piece split mold opening outward from a parting surface.
        parting_coord : Optional[float]
            Physical coordinate of the parting plane along draw_axis (used if mode == 'parting_plane').
            Defaults to the domain midpoint if None.
        epsilon_c : float
            Regularization parameter for smooth minimum (default 1e-4).
        """
        if not isinstance(mesh, (StructuredMesh2D, StructuredMesh3D)):
            raise TypeError("CastingFilter requires StructuredMesh2D or StructuredMesh3D")

        mode = mode.lower()
        if mode not in {"single_draw", "parting_plane"}:
            raise ValueError(f"mode must be 'single_draw' or 'parting_plane', got '{mode}'")

        self.mesh = mesh
        self.draw_axis: str = draw_axis.lower()
        self.mode: str = mode
        self.parting_coord: Optional[float] = parting_coord
        self.epsilon_c: float = float(epsilon_c)
        self.n_elem: int = mesh.n_elem
        self._last_rho: Optional[np.ndarray] = None

        # Build column index array: shape (N_cols, K_layers)
        self._build_column_structure()

    def _build_column_structure(self) -> None:
        """Assemble 2D column indices mapping elements along draw axis."""
        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            if self.draw_axis == "y":
                # K = ny layers, nx columns
                self.n_layers = ny
                self.n_cols = nx
                cols = np.empty((nx, ny), dtype=np.int64)
                for ex in range(nx):
                    cols[ex, :] = [k * nx + ex for k in range(ny)]
                self.col_indices: np.ndarray = cols
                L_draw = self.mesh.ly
            elif self.draw_axis == "x":
                # K = nx layers, ny columns
                self.n_layers = nx
                self.n_cols = ny
                cols = np.empty((ny, nx), dtype=np.int64)
                for ey in range(ny):
                    cols[ey, :] = [ey * nx + k for k in range(nx)]
                self.col_indices = cols
                L_draw = self.mesh.lx
            else:
                raise ValueError(f"Invalid draw_axis '{self.draw_axis}' for 2D mesh (must be 'x' or 'y')")

        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            # elem_id = (ez * ny + ey) * nx + ex
            if self.draw_axis == "z":
                self.n_layers = nz
                self.n_cols = nx * ny
                cols = np.empty((nx * ny, nz), dtype=np.int64)
                idx = 0
                for ey in range(ny):
                    for ex in range(nx):
                        cols[idx, :] = [((k * ny + ey) * nx + ex) for k in range(nz)]
                        idx += 1
                self.col_indices = cols
                L_draw = self.mesh.lz
            elif self.draw_axis == "y":
                self.n_layers = ny
                self.n_cols = nx * nz
                cols = np.empty((nx * nz, ny), dtype=np.int64)
                idx = 0
                for ez in range(nz):
                    for ex in range(nx):
                        cols[idx, :] = [((ez * ny + k) * nx + ex) for k in range(ny)]
                        idx += 1
                self.col_indices = cols
                L_draw = self.mesh.ly
            elif self.draw_axis == "x":
                self.n_layers = nx
                self.n_cols = ny * nz
                cols = np.empty((ny * nz, nx), dtype=np.int64)
                idx = 0
                for ez in range(nz):
                    for ey in range(ny):
                        cols[idx, :] = [((ez * ny + ey) * nx + k) for k in range(nx)]
                        idx += 1
                self.col_indices = cols
                L_draw = self.mesh.lx
            else:
                raise ValueError(f"Invalid draw_axis '{self.draw_axis}' for 3D mesh (must be 'x', 'y', or 'z')")

        # Determine parting plane index k0 if parting_plane mode
        if self.mode == "parting_plane":
            if self.parting_coord is None:
                coord = 0.5 * L_draw
            else:
                coord = float(self.parting_coord)
            delta = L_draw / float(self.n_layers)
            self.k0: int = int(np.clip(int(np.floor(coord / delta)), 0, self.n_layers - 1))
        else:
            self.k0 = 0

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward casting filter projection.

        Parameters
        ----------
        rho : np.ndarray
            Blueprint density field of shape (N_elem,).

        Returns
        -------
        np.ndarray
            Castable physical density array xi of shape (N_elem,).
        """
        rho = np.asarray(rho, dtype=np.float64)
        self._last_rho = rho.copy()
        cols = self.col_indices  # (N_cols, K)
        rho_col = rho[cols]
        K = self.n_layers

        xi_col = np.empty_like(rho_col)

        if self.mode == "single_draw":
            # Monotone non-increasing from base (k=0) to exit (k=K-1)
            xi_col[:, 0] = rho_col[:, 0]
            for k in range(1, K):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))
        else:
            # Parting plane at k0: split outward toward both mold halves
            k0 = self.k0
            xi_col[:, k0] = rho_col[:, k0]

            # Positive half: sweep outward from k0+1 to K-1
            for k in range(k0 + 1, K):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))

            # Negative half: sweep outward from k0-1 down to 0
            for k in range(k0 - 1, -1, -1):
                a = rho_col[:, k]
                b = xi_col[:, k + 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))

        xi = np.empty_like(rho)
        xi[cols] = xi_col
        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Adjoint backward sensitivity sweep along mold columns.

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

        cols = self.col_indices
        rho_col = rho[cols]
        K = self.n_layers
        V = df_dxi[cols]  # (N_cols, K)

        # Recompute xi_col
        xi_col = np.empty_like(rho_col)
        if self.mode == "single_draw":
            xi_col[:, 0] = rho_col[:, 0]
            for k in range(1, K):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))

            # Backward pass from K-1 down to 0
            Lambda = V.copy()
            drho_col = np.empty_like(rho_col)

            for k in range(K - 1, 0, -1):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)

                # dsmin / da and dsmin / db
                dsmin_da = 0.5 * (1.0 - diff / rad)
                dsmin_db = 0.5 * (1.0 + diff / rad)

                lam_k = Lambda[:, k]
                drho_col[:, k] = lam_k * dsmin_da
                Lambda[:, k - 1] += lam_k * dsmin_db

            drho_col[:, 0] = Lambda[:, 0]

        else:
            k0 = self.k0
            xi_col[:, k0] = rho_col[:, k0]

            for k in range(k0 + 1, K):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))

            for k in range(k0 - 1, -1, -1):
                a = rho_col[:, k]
                b = xi_col[:, k + 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)
                xi_col[:, k] = 0.5 * (a + b - rad + np.sqrt(self.epsilon_c))

            Lambda = V.copy()
            drho_col = np.empty_like(rho_col)

            # Positive half sweep from K-1 down to k0+1
            for k in range(K - 1, k0, -1):
                a = rho_col[:, k]
                b = xi_col[:, k - 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)

                dsmin_da = 0.5 * (1.0 - diff / rad)
                dsmin_db = 0.5 * (1.0 + diff / rad)

                lam_k = Lambda[:, k]
                drho_col[:, k] = lam_k * dsmin_da
                Lambda[:, k - 1] += lam_k * dsmin_db

            # Negative half sweep from 0 up to k0-1
            for k in range(0, k0):
                a = rho_col[:, k]
                b = xi_col[:, k + 1]
                diff = a - b
                rad = np.sqrt(diff**2 + self.epsilon_c)

                dsmin_da = 0.5 * (1.0 - diff / rad)
                dsmin_db = 0.5 * (1.0 + diff / rad)

                lam_k = Lambda[:, k]
                drho_col[:, k] = lam_k * dsmin_da
                Lambda[:, k + 1] += lam_k * dsmin_db

            # Parting plane receives accumulated sensitivity
            drho_col[:, k0] = Lambda[:, k0]

        df_drho = np.empty_like(rho)
        df_drho[cols] = drho_col
        return df_drho

    def jacobian_matvec(self, v: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Directional derivative J @ v."""
        if rho is None:
            return self.forward(v)
        h = 1e-7
        return (self.forward(rho + h * v) - self.forward(rho - h * v)) / (2.0 * h)
