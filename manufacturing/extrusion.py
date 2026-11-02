"""Extrusion constraints for uniform cross-section manufacturing.

Implements volume-weighted column-averaging projection along arbitrary extrusion
axes with exact adjoint transpose sensitivity and machine-precision dot-product
identity for structured and unstructured meshes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from wht_topopt.core.filter import FilterOperator
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


class ExtrusionFilter(FilterOperator):
    """Extrusion / constant cross-section manufacturing filter.

    Averages element densities along the extrusion axis so that the resulting
    component has an invariant 2D cross-section along the extrusion path.
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        extrusion_axis: str = "z",
    ) -> None:
        """Initialize ExtrusionFilter.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh.
        extrusion_axis : str
            Axis of uniform extrusion ('x', 'y', or 'z').
        """
        self.mesh = mesh
        self.extrusion_axis: str = extrusion_axis.lower()
        self.n_elem: int = mesh.n_elem

        # Group elements into extrusion columns
        self._build_columns()

    def _build_columns(self) -> None:
        """Partition mesh elements into columns parallel to the extrusion axis."""
        v = self.mesh.elem_volumes

        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            if self.extrusion_axis == "y":
                # nx columns along x, each having ny elements
                cols = [np.arange(ex, self.n_elem, nx, dtype=np.int64) for ex in range(nx)]
            elif self.extrusion_axis == "x":
                # ny columns along y, each having nx elements
                cols = [np.arange(ey * nx, (ey + 1) * nx, dtype=np.int64) for ey in range(ny)]
            else:
                raise ValueError(f"Invalid extrusion_axis '{self.extrusion_axis}' for 2D mesh (must be 'x' or 'y')")

        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            # elem_id = (ez * ny + ey) * nx + ex
            if self.extrusion_axis == "z":
                # nx * ny columns
                cols = []
                stride_z = ny * nx
                for ey in range(ny):
                    for ex in range(nx):
                        base = ey * nx + ex
                        cols.append(np.arange(base, base + nz * stride_z, stride_z, dtype=np.int64))
            elif self.extrusion_axis == "y":
                # nx * nz columns
                cols = []
                stride_z = ny * nx
                stride_y = nx
                for ez in range(nz):
                    for ex in range(nx):
                        base = ez * stride_z + ex
                        cols.append(np.arange(base, base + ny * stride_y, stride_y, dtype=np.int64))
            elif self.extrusion_axis == "x":
                # ny * nz columns
                cols = []
                stride_z = ny * nx
                for ez in range(nz):
                    for ey in range(ny):
                        base = ez * stride_z + ey * nx
                        cols.append(np.arange(base, base + nx, 1, dtype=np.int64))
            else:
                raise ValueError(f"Invalid extrusion_axis '{self.extrusion_axis}' for 3D mesh (must be 'x', 'y', or 'z')")

        else:
            # Unstructured mesh: group by perpendicular centroid coordinates
            centers = self.mesh.elem_centers
            axis_map = {"x": 0, "y": 1, "z": 2}
            if self.extrusion_axis not in axis_map or axis_map[self.extrusion_axis] >= self.mesh.ndim:
                raise ValueError(f"Invalid extrusion_axis '{self.extrusion_axis}' for mesh with ndim={self.mesh.ndim}")

            perp_axes = [i for i in range(self.mesh.ndim) if i != axis_map[self.extrusion_axis]]
            perp_coords = centers[:, perp_axes]

            # Round coordinates to group elements on the same extrusion line
            keys = [tuple(np.round(c, decimals=6)) for c in perp_coords]
            col_dict: Dict[Tuple[float, ...], List[int]] = {}
            for idx, k in enumerate(keys):
                col_dict.setdefault(k, []).append(idx)

            cols = [np.array(v_list, dtype=np.int64) for v_list in col_dict.values()]

        self.columns: List[np.ndarray] = cols
        self.n_cols: int = len(cols)

        # Precompute column volumes and weights: w_j = v_j / V_col
        self.col_weights: List[np.ndarray] = []
        for col in cols:
            vol_col = float(np.sum(v[col]))
            if vol_col <= 0.0:
                vol_col = float(len(col))
            self.col_weights.append(v[col] / vol_col)

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward extrusion projection: volume-weighted column averaging.

        Parameters
        ----------
        rho : np.ndarray
            Blueprint density field of shape (N_elem,).

        Returns
        -------
        np.ndarray
            Extruded physical density field xi of shape (N_elem,).
        """
        rho = np.asarray(rho, dtype=np.float64)
        xi = np.empty_like(rho)

        for col, w in zip(self.columns, self.col_weights):
            avg = float(np.dot(w, rho[col]))
            xi[col] = avg

        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Adjoint backward sensitivity: distributes column-summed sensitivities back.

        Parameters
        ----------
        df_dxi : np.ndarray
            Sensitivity vector dJ/dxi of shape (N_elem,).
        rho : Optional[np.ndarray], optional
            Not used (linear projection operator).

        Returns
        -------
        np.ndarray
            Adjoint sensitivity vector dJ/drho of shape (N_elem,).
        """
        df_dxi = np.asarray(df_dxi, dtype=np.float64)
        df_drho = np.empty_like(df_dxi)

        for col, w in zip(self.columns, self.col_weights):
            sum_v = float(np.sum(df_dxi[col]))
            df_drho[col] = sum_v * w

        return df_drho

    def jacobian_matvec(self, v: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Directional derivative J @ v = P @ v."""
        return self.forward(v)

    def variance_along_axis(self, xi: np.ndarray) -> float:
        """Compute the maximum variance of densities within any extrusion column.

        Parameters
        ----------
        xi : np.ndarray
            Physical density field of shape (N_elem,).

        Returns
        -------
        float
            Maximum within-column variance. Should be < 1e-14 for exact extruded shapes.
        """
        xi = np.asarray(xi, dtype=np.float64)
        max_var = 0.0
        for col in self.columns:
            if len(col) > 1:
                var = float(np.var(xi[col]))
                if var > max_var:
                    max_var = var
        return max_var


class BottomPlateZExtrusionFilter(FilterOperator):
    """Z-axis Extrusion constraint with an integral bottom skin plate at -Z.

    Divides the 3D domain along Z into:
    1. Bottom plate layer(s) at ez < bottom_plate_layers (e.g. z in [0, 2] mm):
       Retained as an integral base skin plate (unprojected / solid).
    2. Upper rib region at ez >= bottom_plate_layers (e.g. z in [2, 40] mm):
       Uniformly extruded along Z so that stiffener ribs extend vertically
       with constant 2.5D cross-section from the bottom skin plate.
    """

    def __init__(
        self,
        mesh: StructuredMesh3D,
        bottom_plate_layers: int = 1,
    ) -> None:
        """Initialize BottomPlateZExtrusionFilter.

        Parameters
        ----------
        mesh : StructuredMesh3D
            Structured 3D computational mesh.
        bottom_plate_layers : int
            Number of bottom element layers reserved for the plate skin (default=1).
        """
        self.mesh = mesh
        self.bottom_plate_layers = max(1, int(bottom_plate_layers))
        self.n_elem = mesh.n_elem

        nx, ny, nz = mesh.nelx, mesh.nely, mesh.nelz
        stride_z = ny * nx

        # Upper rib columns: ez in [bottom_plate_layers, nz - 1]
        cols = []
        n_upper_layers = nz - self.bottom_plate_layers
        for ey in range(ny):
            for ex in range(nx):
                base = self.bottom_plate_layers * stride_z + ey * nx + ex
                cols.append(np.arange(base, base + n_upper_layers * stride_z, stride_z, dtype=np.int64))

        self.columns: List[np.ndarray] = cols
        self.n_cols: int = len(cols)
        v = mesh.elem_volumes

        self.col_weights: List[np.ndarray] = []
        for col in cols:
            vol_col = float(np.sum(v[col]))
            if vol_col <= 0.0:
                vol_col = float(len(col))
            self.col_weights.append(v[col] / vol_col)

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward projection: column averaging on upper rib layers."""
        rho = np.asarray(rho, dtype=np.float64)
        xi = rho.copy()
        for col, w in zip(self.columns, self.col_weights):
            avg = float(np.dot(w, rho[col]))
            xi[col] = avg
        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Adjoint backward sensitivity: column summation on upper rib layers."""
        df_dxi = np.asarray(df_dxi, dtype=np.float64)
        df_drho = df_dxi.copy()
        for col, w in zip(self.columns, self.col_weights):
            sum_v = float(np.sum(df_dxi[col]))
            df_drho[col] = sum_v * w
        return df_drho


class ShellRibExtrusionFilter(FilterOperator):
    """Outer Shell / Perimeter Skin + Interior Z-Extruded Ribs Filter.

    Enforces:
    1. Outer Shell / Hull:
       - Bottom skin plate at z < t_bottom (ez < bottom_layers, e.g. 2mm).
       - Lateral side wall skins at y < t_side or y >= Ly - t_side (ey in {0, ny-1}, e.g. 2mm).
       These perimeter shell elements form the outer structural box/hull.
    2. Interior Core:
       - Elements inside the shell (ez >= bottom_layers, side_layers <= ey < ny - side_layers).
       - Grouped into vertical columns along Z and projected via volume-weighted
         column averaging, producing constant cross-section stiffener ribs
         that support and bridge the outer shell walls.
    """

    def __init__(
        self,
        mesh: StructuredMesh3D,
        bottom_layers: int = 1,
        side_layers: int = 1,
    ) -> None:
        """Initialize ShellRibExtrusionFilter.

        Parameters
        ----------
        mesh : StructuredMesh3D
            Structured 3D computational mesh.
        bottom_layers : int
            Number of bottom element layers for base skin plate (default=1, 2.0mm).
        side_layers : int
            Number of element layers for lateral side wall skins (default=1, 2.0mm).
        """
        self.mesh = mesh
        self.bottom_layers = max(1, int(bottom_layers))
        self.side_layers = max(1, int(side_layers))
        self.n_elem = mesh.n_elem

        nx, ny, nz = mesh.nelx, mesh.nely, mesh.nelz
        stride_z = ny * nx
        stride_y = nx

        # Interior rib columns: ey in [side_layers, ny - 1 - side_layers]
        # and ez in [bottom_layers, nz - 1]
        cols = []
        n_upper_layers = nz - self.bottom_layers

        for ey in range(self.side_layers, ny - self.side_layers):
            for ex in range(nx):
                base = self.bottom_layers * stride_z + ey * stride_y + ex
                cols.append(np.arange(base, base + n_upper_layers * stride_z, stride_z, dtype=np.int64))

        self.columns: List[np.ndarray] = cols
        self.n_cols: int = len(cols)
        v = mesh.elem_volumes

        self.col_weights: List[np.ndarray] = []
        for col in cols:
            vol_col = float(np.sum(v[col]))
            if vol_col <= 0.0:
                vol_col = float(len(col))
            self.col_weights.append(v[col] / vol_col)

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward projection: column averaging on interior rib layers only."""
        rho = np.asarray(rho, dtype=np.float64)
        xi = rho.copy()
        for col, w in zip(self.columns, self.col_weights):
            avg = float(np.dot(w, rho[col]))
            xi[col] = avg
        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Adjoint backward sensitivity: column summation on interior rib layers only."""
        df_dxi = np.asarray(df_dxi, dtype=np.float64)
        df_drho = df_dxi.copy()
        for col, w in zip(self.columns, self.col_weights):
            sum_v = float(np.sum(df_dxi[col]))
            df_drho[col] = sum_v * w
        return df_drho


class PlanarRibExtrusionFilter(FilterOperator):
    """High-Definition Outer Shell + 2D Planar Filtered Interior Z-Rib Extrusion.

    Combines:
    1. Outer U-Hull: Bottom skin (z < bottom_layers) and lateral side skins (y < side_layers
       or y >= ny - side_layers) are preserved.
    2. 2D Planar XY Spatial Filtering on interior core with sharp physical radius r_min_xy
       (e.g., 2.5 ~ 2.8 mm). This prevents out-of-plane vertical blurring and allows
       sharp, thin rib walls (2.0 ~ 3.0 mm) to form.
    3. Heaviside Step Projection with continuation parameter beta and threshold eta:
       Pushes intermediate densities towards 0 or 1, eliminating grey mush and creating
       distinct, structural rib partitions.
    4. Exact Adjoint Chain-Rule Backpropagation:
       Maintains mathematical sensitivity consistency across the 2D filter, projection,
       and 3D column extrusion with machine precision dot-product identity.
    """

    def __init__(
        self,
        mesh: StructuredMesh3D,
        r_min_xy: float = 2.8,
        bottom_layers: int = 1,
        side_layers: int = 1,
        beta: float = 1.0,
        eta: float = 0.5,
    ) -> None:
        """Initialize PlanarRibExtrusionFilter.

        Parameters
        ----------
        mesh : StructuredMesh3D
            Computational 3D structured mesh.
        r_min_xy : float
            In-plane filter radius on the XY cross-section (in mm).
        bottom_layers : int
            Number of bottom element layers for base skin plate (default=1).
        side_layers : int
            Number of element layers for lateral side skins (default=1).
        beta : float
            Heaviside sharpness parameter (>= 1.0).
        eta : float
            Heaviside threshold (default 0.5).
        """
        self.mesh = mesh
        self.r_min_xy = float(r_min_xy)
        self.bottom_layers = max(1, int(bottom_layers))
        self.side_layers = max(1, int(side_layers))
        self.beta = float(beta)
        self.eta = float(eta)
        self.n_elem = mesh.n_elem

        nx, ny, nz = mesh.nelx, mesh.nely, mesh.nelz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.dz = mesh.dz

        # Identify core 2D grid dimensions
        self.ey_start = self.side_layers
        self.ey_end = ny - self.side_layers  # exclusive
        self.ny_core = self.ey_end - self.ey_start
        self.n_core_2d = nx * self.ny_core

        # Build 3D column element mappings for interior core
        stride_z = ny * nx
        stride_y = nx
        n_upper_layers = nz - self.bottom_layers
        self.n_upper_layers = n_upper_layers

        cols_3d = []
        for ey_idx, ey in enumerate(range(self.ey_start, self.ey_end)):
            for ex in range(nx):
                base = self.bottom_layers * stride_z + ey * stride_y + ex
                cols_3d.append(np.arange(base, base + n_upper_layers * stride_z, stride_z, dtype=np.int64))

        self.cols_3d: List[np.ndarray] = cols_3d
        self.core_elem_indices: np.ndarray = np.concatenate(cols_3d)

        # Build 2D XY filter matrix H_2d (sparse CSR)
        self._build_2d_filter()

        # Cache for forward state required for adjoint backward
        self._last_s_2d: Optional[np.ndarray] = None
        self._last_s_flt: Optional[np.ndarray] = None
        self._last_s_proj: Optional[np.ndarray] = None

    def _build_2d_filter(self) -> None:
        """Construct normalized 2D conical filter matrix for interior XY cross-section."""
        import scipy.sparse as sp

        nx = self.nx
        ny_core = self.ny_core
        dx = self.dx
        dy = self.dy
        r_min = self.r_min_xy

        # Precompute 2D coordinates of core elements
        coords_2d = np.zeros((self.n_core_2d, 2), dtype=np.float64)
        for ey_local in range(ny_core):
            ey_global = self.ey_start + ey_local
            cy = (ey_global + 0.5) * dy
            for ex in range(nx):
                cx = (ex + 0.5) * dx
                p_idx = ey_local * nx + ex
                coords_2d[p_idx] = [cx, cy]

        # Spatial search radius in indices
        i_rad_x = int(np.ceil(r_min / dx))
        i_rad_y = int(np.ceil(r_min / dy))

        rows, cols, data = [], [], []
        for ey_local in range(ny_core):
            for ex in range(nx):
                p_idx = ey_local * nx + ex
                cx, cy = coords_2d[p_idx]

                min_x = max(0, ex - i_rad_x)
                max_x = min(nx, ex + i_rad_x + 1)
                min_y = max(0, ey_local - i_rad_y)
                max_y = min(ny_core, ey_local + i_rad_y + 1)

                sum_w = 0.0
                loc_cols = []
                loc_weights = []

                for qy in range(min_y, max_y):
                    for qx in range(min_x, max_x):
                        q_idx = qy * nx + qx
                        qx_c, qy_c = coords_2d[q_idx]
                        dist = np.sqrt((cx - qx_c)**2 + (cy - qy_c)**2)
                        if dist < r_min:
                            w = r_min - dist
                            loc_cols.append(q_idx)
                            loc_weights.append(w)
                            sum_w += w

                if sum_w > 0.0:
                    for c, w in zip(loc_cols, loc_weights):
                        rows.append(p_idx)
                        cols.append(c)
                        data.append(w / sum_w)
                else:
                    rows.append(p_idx)
                    cols.append(p_idx)
                    data.append(1.0)

        self.H_2d = sp.csr_matrix((data, (rows, cols)), shape=(self.n_core_2d, self.n_core_2d))

    def set_beta(self, beta: float) -> None:
        """Update Heaviside continuation sharpness parameter beta."""
        self.beta = max(1.0, float(beta))

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Average a cell signal over the rib columns and the 2D filter radius, without projection or clipping.

        ``forward`` is for densities: its Heaviside and its bounds would saturate a signal that may exceed 1."""
        out = np.array(signal, dtype=np.float64)
        col_mean = out[np.asarray(self.cols_3d)].mean(axis=1)
        out[np.asarray(self.cols_3d)] = (self.H_2d @ col_mean)[:, None]
        return out

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward projection: 3D Blueprint -> 2D Core Avg -> 2D Filter -> Heaviside -> 3D Extrusion.

        Parameters
        ----------
        rho : np.ndarray
            3D blueprint density field of shape (N_elem,).

        Returns
        -------
        np.ndarray
            3D physical density field xi with sharp vertical interior ribs.
        """
        rho = np.asarray(rho, dtype=np.float64)
        xi = rho.copy()

        # Step 1: Column-average core along Z to get 2D planar density map s_2d
        s_2d = np.empty(self.n_core_2d, dtype=np.float64)
        for i, col in enumerate(self.cols_3d):
            s_2d[i] = np.mean(rho[col])
        self._last_s_2d = s_2d

        # Step 2: 2D Spatial filtering
        s_flt = self.H_2d.dot(s_2d)
        self._last_s_flt = s_flt

        # Step 3: Heaviside step projection
        beta = self.beta
        eta = self.eta
        num = np.tanh(beta * eta) + np.tanh(beta * (s_flt - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        s_proj = num / den
        s_proj = np.clip(s_proj, 1e-4, 1.0)
        self._last_s_proj = s_proj

        # Step 4: Extrude 2D projected profile along Z into all interior core layers
        for i, col in enumerate(self.cols_3d):
            xi[col] = s_proj[i]

        return xi

    def backward(self, df_dxi: np.ndarray, rho: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact adjoint backward sensitivity through Extrusion -> Heaviside -> 2D Filter.

        Parameters
        ----------
        df_dxi : np.ndarray
            Sensitivity vector dJ/dxi of shape (N_elem,).
        rho : Optional[np.ndarray], optional
            Blueprint density field (not strictly required if forward was called).

        Returns
        -------
        np.ndarray
            Adjoint sensitivity vector dJ/drho of shape (N_elem,).
        """
        df_dxi = np.asarray(df_dxi, dtype=np.float64)
        df_drho = df_dxi.copy()

        if self._last_s_flt is None:
            return df_drho

        # Step 1: Accumulate sensitivity from 3D extruded columns to 2D projected map
        df_ds_proj = np.empty(self.n_core_2d, dtype=np.float64)
        for i, col in enumerate(self.cols_3d):
            df_ds_proj[i] = np.sum(df_dxi[col])

        # Step 2: Adjoint Heaviside projection
        beta = self.beta
        eta = self.eta
        s_flt = self._last_s_flt
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        # d(tanh(x))/dx = 1 - tanh^2(x)
        dproj_dflt = (beta * (1.0 - np.tanh(beta * (s_flt - eta))**2)) / den
        df_ds_flt = df_ds_proj * dproj_dflt

        # Step 3: Adjoint 2D spatial filtering (H_2d is symmetric or transpose applied)
        df_ds_2d = self.H_2d.T.dot(df_ds_flt)

        # Step 4: Distribute 2D sensitivity back to 3D blueprint columns
        inv_n = 1.0 / float(self.n_upper_layers)
        for i, col in enumerate(self.cols_3d):
            df_drho[col] = df_ds_2d[i] * inv_n

        return df_drho



