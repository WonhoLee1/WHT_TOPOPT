"""Langelaar (2016, 2017) Additive Manufacturing self-supporting overhang filter.

Implements bottom-to-top layer-by-layer forward propagation with smooth max-min
support aggregation and exact reverse-layer adjoint sensitivity backward sweep
for 2D and 3D regular structured meshes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from wht_topopt.core.filter import FilterOperator
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D
from wht_topopt.core.numba_kernels import (
    HAS_NUMBA,
    numba_overhang_backward,
    numba_overhang_forward,
)


class OverhangFilter(FilterOperator):
    """Langelaar (2016/2017) self-supporting AM overhang filter.

    Ensures that fabricated structures have sufficient local support within
    the 45-degree self-supporting cone along the chosen build direction.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D],
        build_axis: str = "y",
        p: float = 20.0,
        epsilon_0: float = 1e-4,
        epsilon_s: float = 1e-4,
    ) -> None:
        """Initialize OverhangFilter.

        Parameters
        ----------
        mesh : StructuredMesh2D or StructuredMesh3D
            Computational mesh.
        build_axis : str
            Build direction axis ('y' or 'x' in 2D; 'z', 'y', or 'x' in 3D).
        p : float
            Smooth P-norm parameter for support maximum (default 20.0).
        epsilon_0 : float
            Regularization constant for smooth maximum (default 1e-4).
        epsilon_s : float
            Regularization constant for smooth minimum (default 1e-4).
        """
        if not isinstance(mesh, (StructuredMesh2D, StructuredMesh3D)):
            raise TypeError("OverhangFilter requires StructuredMesh2D or StructuredMesh3D")

        self.mesh = mesh
        self.build_axis = build_axis.lower()
        self.p: float = float(p)
        self.epsilon_0: float = float(epsilon_0)
        self.epsilon_s: float = float(epsilon_s)
        self.n_elem: int = mesh.n_elem
        self._last_rho: Optional[np.ndarray] = None

        # Precompute layer structure and 45-degree support neighborhood
        self.use_numba: bool = bool(HAS_NUMBA)
        self._build_layer_structure()
        self._pack_numba_arrays()

    def _pack_numba_arrays(self) -> None:
        """Pack layer elements and support neighborhoods into flat contiguous arrays for Numba."""
        n_cols = 3 if isinstance(self.mesh, StructuredMesh2D) else 9
        self._n_supp_cols: int = n_cols
        self._layer_ptr = np.zeros(self.n_layers + 1, dtype=np.int64)
        for j in range(self.n_layers):
            self._layer_ptr[j + 1] = self._layer_ptr[j] + len(self.layer_elems[j])
        self._layer_elems_flat = np.concatenate(self.layer_elems).astype(np.int64)

        n_base = len(self.layer_elems[0])
        base_supp = np.zeros((n_base, n_cols), dtype=np.int64)
        base_mask = np.zeros((n_base, n_cols), dtype=bool)
        if self.n_layers > 1:
            self._supp_ids_flat = np.vstack([base_supp] + self.support_elem_ids[1:]).astype(np.int64)
            self._supp_mask_flat = np.vstack([base_mask] + self.support_mask[1:]).astype(bool)
        else:
            self._supp_ids_flat = base_supp
            self._supp_mask_flat = base_mask

    def _build_layer_structure(self) -> None:
        """Identify layers along build axis and compute supporting elements in layer below."""
        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            if self.build_axis == "y":
                self.n_layers = ny
                self.layer_elems: List[np.ndarray] = [
                    np.arange(j * nx, (j + 1) * nx, dtype=np.int64) for j in range(ny)
                ]
                # Supporting elements for (ex, j) in layer j-1: ex-1, ex, ex+1
                self.support_elem_ids: List[np.ndarray] = []
                self.support_mask: List[np.ndarray] = []
                self.support_counts: List[np.ndarray] = []

                # Dummy index 0 for base layer
                self.support_elem_ids.append(np.empty((0, 3), dtype=np.int64))
                self.support_mask.append(np.empty((0, 3), dtype=bool))
                self.support_counts.append(np.empty(0, dtype=np.int64))

                for j in range(1, ny):
                    supp_ids = np.zeros((nx, 3), dtype=np.int64)
                    mask = np.zeros((nx, 3), dtype=bool)
                    base_prev = (j - 1) * nx

                    for ex in range(nx):
                        col = 0
                        for dx in (-1, 0, 1):
                            nx_val = ex + dx
                            if 0 <= nx_val < nx:
                                supp_ids[ex, col] = base_prev + nx_val
                                mask[ex, col] = True
                            col += 1

                    self.support_elem_ids.append(supp_ids)
                    self.support_mask.append(mask)
                    self.support_counts.append(np.sum(mask, axis=1))

            elif self.build_axis == "x":
                self.n_layers = nx
                self.layer_elems = [
                    (np.arange(ny, dtype=np.int64) * nx + j) for j in range(nx)
                ]
                self.support_elem_ids = [np.empty((0, 3), dtype=np.int64)]
                self.support_mask = [np.empty((0, 3), dtype=bool)]
                self.support_counts = [np.empty(0, dtype=np.int64)]

                for j in range(1, nx):
                    supp_ids = np.zeros((ny, 3), dtype=np.int64)
                    mask = np.zeros((ny, 3), dtype=bool)
                    for ey in range(ny):
                        col = 0
                        for dy in (-1, 0, 1):
                            ny_val = ey + dy
                            if 0 <= ny_val < ny:
                                supp_ids[ey, col] = (j - 1) + ny_val * nx
                                mask[ey, col] = True
                            col += 1
                    self.support_elem_ids.append(supp_ids)
                    self.support_mask.append(mask)
                    self.support_counts.append(np.sum(mask, axis=1))
            else:
                raise ValueError(f"Invalid build_axis '{self.build_axis}' for 2D mesh (must be 'x' or 'y')")

        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            # elem_id = (ez * ny + ey) * nx + ex
            stride_y = nx
            stride_z = ny * nx

            if self.build_axis == "z":
                self.n_layers = nz
                self.layer_elems = []
                for ez in range(nz):
                    elems = []
                    for ey in range(ny):
                        for ex in range(nx):
                            elems.append((ez * ny + ey) * nx + ex)
                    self.layer_elems.append(np.array(elems, dtype=np.int64))

                self.support_elem_ids = [np.empty((0, 9), dtype=np.int64)]
                self.support_mask = [np.empty((0, 9), dtype=bool)]
                self.support_counts = [np.empty(0, dtype=np.int64)]

                n_per_layer = ny * nx
                for j in range(1, nz):
                    supp_ids = np.zeros((n_per_layer, 9), dtype=np.int64)
                    mask = np.zeros((n_per_layer, 9), dtype=bool)
                    idx = 0
                    for ey in range(ny):
                        for ex in range(nx):
                            col = 0
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    nx_val = ex + dx
                                    ny_val = ey + dy
                                    if 0 <= nx_val < nx and 0 <= ny_val < ny:
                                        supp_ids[idx, col] = ((j - 1) * ny + ny_val) * nx + nx_val
                                        mask[idx, col] = True
                                    col += 1
                            idx += 1
                    self.support_elem_ids.append(supp_ids)
                    self.support_mask.append(mask)
                    self.support_counts.append(np.sum(mask, axis=1))

            elif self.build_axis == "y":
                self.n_layers = ny
                self.layer_elems = []
                for ey in range(ny):
                    elems = []
                    for ez in range(nz):
                        for ex in range(nx):
                            elems.append((ez * ny + ey) * nx + ex)
                    self.layer_elems.append(np.array(elems, dtype=np.int64))

                self.support_elem_ids = [np.empty((0, 9), dtype=np.int64)]
                self.support_mask = [np.empty((0, 9), dtype=bool)]
                self.support_counts = [np.empty(0, dtype=np.int64)]

                n_per_layer = nz * nx
                for j in range(1, ny):
                    supp_ids = np.zeros((n_per_layer, 9), dtype=np.int64)
                    mask = np.zeros((n_per_layer, 9), dtype=bool)
                    idx = 0
                    for ez in range(nz):
                        for ex in range(nx):
                            col = 0
                            for dz in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    nx_val = ex + dx
                                    nz_val = ez + dz
                                    if 0 <= nx_val < nx and 0 <= nz_val < nz:
                                        supp_ids[idx, col] = (nz_val * ny + (j - 1)) * nx + nx_val
                                        mask[idx, col] = True
                                    col += 1
                            idx += 1
                    self.support_elem_ids.append(supp_ids)
                    self.support_mask.append(mask)
                    self.support_counts.append(np.sum(mask, axis=1))

            elif self.build_axis == "x":
                self.n_layers = nx
                self.layer_elems = []
                for ex in range(nx):
                    elems = []
                    for ez in range(nz):
                        for ey in range(ny):
                            elems.append((ez * ny + ey) * nx + ex)
                    self.layer_elems.append(np.array(elems, dtype=np.int64))

                self.support_elem_ids = [np.empty((0, 9), dtype=np.int64)]
                self.support_mask = [np.empty((0, 9), dtype=bool)]
                self.support_counts = [np.empty(0, dtype=np.int64)]

                n_per_layer = nz * ny
                for j in range(1, nx):
                    supp_ids = np.zeros((n_per_layer, 9), dtype=np.int64)
                    mask = np.zeros((n_per_layer, 9), dtype=bool)
                    idx = 0
                    for ez in range(nz):
                        for ey in range(ny):
                            col = 0
                            for dz in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    ny_val = ey + dy
                                    nz_val = ez + dz
                                    if 0 <= ny_val < ny and 0 <= nz_val < nz:
                                        supp_ids[idx, col] = (nz_val * ny + ny_val) * nx + (j - 1)
                                        mask[idx, col] = True
                                    col += 1
                            idx += 1
                    self.support_elem_ids.append(supp_ids)
                    self.support_mask.append(mask)
                    self.support_counts.append(np.sum(mask, axis=1))
            else:
                raise ValueError(f"Invalid build_axis '{self.build_axis}' for 3D mesh (must be 'x', 'y', or 'z')")

    def forward(self, rho: np.ndarray, use_numba: Optional[bool] = None) -> np.ndarray:
        """Forward self-supporting filter sweep."""
        rho = np.ascontiguousarray(rho, dtype=np.float64)
        self._last_rho = rho.copy()

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel:
            xi = numba_overhang_forward(
                rho,
                self._layer_ptr,
                self._layer_elems_flat,
                self._supp_ids_flat,
                self._supp_mask_flat,
                self._n_supp_cols,
                self.p,
                self.epsilon_0,
                self.epsilon_s,
            )
            self._last_xi = xi
            return xi

        xi = np.empty_like(rho)

        # Baseplate layer j = 0: fully supported by build plate
        base_ids = self.layer_elems[0]
        xi[base_ids] = rho[base_ids]

        # Bottom-to-top sweep
        for j in range(1, self.n_layers):
            cur_ids = self.layer_elems[j]
            supp_ids = self.support_elem_ids[j]
            mask = self.support_mask[j]
            counts = self.support_counts[j]

            xi_supp = xi[supp_ids]
            base = np.where(mask, xi_supp + self.epsilon_0, 0.0)

            # Smooth max: S = (sum (xi + eps_0)^p)^(1/p) - |S|^(1/p) * eps_0
            U = np.sum(base**self.p, axis=1)
            U_pow = U ** (1.0 / self.p)
            S = U_pow - (counts ** (1.0 / self.p)) * self.epsilon_0

            # Smooth min: xi = smin(rho, S)
            rho_cur = rho[cur_ids]
            diff = rho_cur - S
            rad = np.sqrt(diff**2 + self.epsilon_s)
            xi[cur_ids] = 0.5 * (rho_cur + S - rad + np.sqrt(self.epsilon_s))

        return xi

    def backward(
        self,
        df_dxi: np.ndarray,
        rho: Optional[np.ndarray] = None,
        use_numba: Optional[bool] = None,
    ) -> np.ndarray:
        """Top-to-bottom adjoint sensitivity reverse sweep."""
        df_dxi = np.ascontiguousarray(df_dxi, dtype=np.float64)
        if rho is None:
            if self._last_rho is None:
                raise ValueError("rho is required for backward sensitivity evaluation")
            rho = self._last_rho
        else:
            rho = np.ascontiguousarray(rho, dtype=np.float64)

        accel = self.use_numba if use_numba is None else (bool(use_numba) and HAS_NUMBA)
        if accel:
            xi = numba_overhang_forward(
                rho,
                self._layer_ptr,
                self._layer_elems_flat,
                self._supp_ids_flat,
                self._supp_mask_flat,
                self._n_supp_cols,
                self.p,
                self.epsilon_0,
                self.epsilon_s,
            )
            return numba_overhang_backward(
                df_dxi,
                rho,
                xi,
                self._layer_ptr,
                self._layer_elems_flat,
                self._supp_ids_flat,
                self._supp_mask_flat,
                self._n_supp_cols,
                self.p,
                self.epsilon_0,
                self.epsilon_s,
            )

        # We first do a forward pass to have intermediate xi and S values
        xi = np.empty_like(rho)
        xi[self.layer_elems[0]] = rho[self.layer_elems[0]]

        S_all: List[np.ndarray] = [np.empty(0)]
        U_pow_all: List[np.ndarray] = [np.empty(0)]

        for j in range(1, self.n_layers):
            cur_ids = self.layer_elems[j]
            supp_ids = self.support_elem_ids[j]
            mask = self.support_mask[j]
            counts = self.support_counts[j]

            xi_supp = xi[supp_ids]
            base = np.where(mask, xi_supp + self.epsilon_0, 0.0)

            U = np.sum(base**self.p, axis=1)
            U_pow = U ** (1.0 / self.p)
            S = U_pow - (counts ** (1.0 / self.p)) * self.epsilon_0

            S_all.append(S)
            U_pow_all.append(U_pow)

            rho_cur = rho[cur_ids]
            diff = rho_cur - S
            rad = np.sqrt(diff**2 + self.epsilon_s)
            xi[cur_ids] = 0.5 * (rho_cur + S - rad + np.sqrt(self.epsilon_s))

        # Adjoint sensitivity back-propagation
        lambda_vec = df_dxi.copy()
        df_drho = np.empty_like(df_dxi)

        for j in range(self.n_layers - 1, 0, -1):
            cur_ids = self.layer_elems[j]
            supp_ids = self.support_elem_ids[j]
            mask = self.support_mask[j]

            S = S_all[j]
            U_pow = U_pow_all[j]

            rho_cur = rho[cur_ids]
            diff = rho_cur - S
            rad = np.sqrt(diff**2 + self.epsilon_s)

            # dsmin / da and dsmin / db
            dxi_drho = 0.5 * (1.0 - diff / rad)
            dxi_dS = 0.5 * (1.0 + diff / rad)

            lam_cur = lambda_vec[cur_ids]
            df_drho[cur_ids] = lam_cur * dxi_drho
            mu = lam_cur * dxi_dS

            # dS / d(xi_supp) = (base / U_pow)^(p - 1)
            xi_supp = xi[supp_ids]
            base = np.where(mask, xi_supp + self.epsilon_0, 0.0)
            ratio = np.where(mask, base / (U_pow[:, None] + 1e-30), 0.0)
            dS_dsupp = np.where(mask, ratio ** (self.p - 1), 0.0)

            # Propagate back to supporting elements in layer j-1
            contrib = mu[:, None] * dS_dsupp
            np.add.at(lambda_vec, supp_ids[mask], contrib[mask])

        # Base layer sensitivities
        base_ids = self.layer_elems[0]
        df_drho[base_ids] = lambda_vec[base_ids]

        return df_drho

    def jacobian_matvec(self, v: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Directional derivative J @ v via finite-difference approximation or forward pass."""
        if rho is None:
            return self.forward(v)
        h = 1e-7
        return (self.forward(rho + h * v) - self.forward(rho - h * v)) / (2.0 * h)
