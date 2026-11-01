"""Regularization and topology control filters for density-based topology optimization.

Implements FilterOperator abstract base class, spatial DensityFilter (with volume weighting
and KD-Tree accelerated sparse stencil), Sigmund's heuristic SensitivityFilter,
and the Lazarov-Sigmund HelmholtzPDEFilter with exact adjoint sensitivity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


class FilterOperator(ABC):
    """Abstract Base Class for topology optimization filter and projection operators."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Map design variables to filtered field: x -> x_tilde."""
        pass

    @abstractmethod
    def backward(self, df_dx_tilde: np.ndarray) -> np.ndarray:
        """Adjoint sensitivity vector-Jacobian product: df/dx_tilde -> df/dx."""
        pass

    def jacobian_matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute directional derivative (Jacobian-vector product) J @ v."""
        return self.forward(v)


class DensityFilter(FilterOperator):
    """Spatial cone/hat filter for mesh-independent density filtering.

    Weight kernel:
        h_ei = max(0, r_min - d_ei)

    Volume-weighted normalized filter matrix:
        W_ei = (h_ei * v_i) / sum_j (h_ej * v_j)
        x_tilde = W @ x
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        r_min: Optional[float] = None,
        volume_weighted: bool = True,
        *,
        rmin: Optional[float] = None,
    ) -> None:
        """Initialize DensityFilter with KD-Tree neighbor search.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh.
        r_min : float, optional
            Filter radius (in physical length units; or pass ``rmin``).
        volume_weighted : bool
            Whether to weight neighbor contributions by element volumes (default True).
        """
        r_val = rmin if r_min is None else r_min
        if r_val is None or r_val <= 0.0:
            raise ValueError(f"Filter radius r_min must be positive, got {r_val}")

        self.mesh = mesh
        self.r_min: float = float(r_val)
        self.volume_weighted: bool = volume_weighted
        self.n_elem: int = mesh.n_elem

        # Build sparse filter matrix W
        self.W: sp.csr_matrix = self._build_filter_matrix()

    def _build_filter_matrix(self) -> sp.csr_matrix:
        """Build row-normalized sparse filter matrix W via cKDTree."""
        tree = cKDTree(self.mesh.elem_centers)
        v = self.mesh.elem_volumes

        # Query all pairs within r_min
        dist_mat = tree.sparse_distance_matrix(tree, self.r_min, output_type="coo_matrix")

        rows = dist_mat.row
        cols = dist_mat.col
        dists = dist_mat.data

        # Kernel h = r_min - dist
        h = np.maximum(0.0, self.r_min - dists)

        if self.volume_weighted:
            # Multiply weight by source element volume v_j
            h *= v[cols]

        H_coo = sp.coo_matrix((h, (rows, cols)), shape=(self.n_elem, self.n_elem))
        H_csr = H_coo.tocsr()

        # Row normalization: W_ei = H_ei / sum_j H_ej
        row_sums = np.asarray(H_csr.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        inv_sums = sp.diags(1.0 / row_sums)

        W = inv_sums @ H_csr
        return W.tocsr()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Filter density field: x_tilde = W @ x."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return self.W @ x
        else:
            return self.W @ x

    def backward(self, df_dx_tilde: np.ndarray) -> np.ndarray:
        """Adjoint reverse-mode sensitivity: df/dx = W^T @ (df/dx_tilde)."""
        df_dx_tilde = np.asarray(df_dx_tilde, dtype=np.float64)
        return self.W.T @ df_dx_tilde

    def jacobian_matvec(self, v: np.ndarray) -> np.ndarray:
        """Jacobian-vector product J @ v = W @ v."""
        return self.forward(v)


class SensitivityFilter:
    """Sigmund's heuristic sensitivity filter for compliance optimization."""

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        r_min: float,
        gamma: float = 1e-3,
    ) -> None:
        """Initialize SensitivityFilter.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh.
        r_min : float
            Filter radius in physical length units.
        gamma : float
            Division-by-zero protection parameter for low densities (default 1e-3).
        """
        if r_min <= 0.0:
            raise ValueError(f"Filter radius r_min must be positive, got {r_min}")
        self.mesh = mesh
        self.r_min: float = float(r_min)
        self.gamma: float = float(gamma)
        self.n_elem: int = mesh.n_elem

        # Build raw unnormalized distance weight matrix H
        tree = cKDTree(self.mesh.elem_centers)
        dist_mat = tree.sparse_distance_matrix(tree, self.r_min, output_type="coo_matrix")
        h = np.maximum(0.0, self.r_min - dist_mat.data)
        H_coo = sp.coo_matrix((h, (dist_mat.row, dist_mat.col)), shape=(self.n_elem, self.n_elem))
        self.H: sp.csr_matrix = H_coo.tocsr()
        self.row_sums: np.ndarray = np.asarray(self.H.sum(axis=1)).ravel()

    def filter_sensitivity(self, rho: np.ndarray, dc: np.ndarray) -> np.ndarray:
        """Apply Sigmund sensitivity filter: dc_tilde = H @ (rho * dc) / (max(gamma, rho) * sum_H)."""
        rho = np.asarray(rho, dtype=np.float64)
        dc = np.asarray(dc, dtype=np.float64)

        denom = np.maximum(self.gamma, rho) * self.row_sums
        num = self.H @ (rho * dc)
        return num / denom


class HelmholtzPDEFilter(FilterOperator):
    """Lazarov-Sigmund Helmholtz Differential Equation Filter for topology optimization.

    Solves the continuous boundary value problem:
        -l^2 * div(grad(rho_tilde)) + rho_tilde = rho   in Omega
        d(rho_tilde)/dn = 0                             on dOmega (Neumann)

    where characteristic length scale is l = r_min / (2 * sqrt(3)).

    FE Discretization:
        K_F * rho_N = T * rho
        rho_tilde = T_avg * rho_N
    """

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        r_min: float,
        bc_type: str = "neumann",
        robin_alpha: float = 1.0,
    ) -> None:
        """Initialize and factorize Helmholtz PDE filter.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh.
        r_min : float
            Filter radius (physical length).
        bc_type : str
            'neumann' (default) or 'robin'.
        robin_alpha : float
            Robin parameter (used if bc_type == 'robin').
        """
        if r_min <= 0.0:
            raise ValueError(f"r_min must be positive, got {r_min}")

        self.mesh = mesh
        self.r_min: float = float(r_min)
        self.ell: float = self.r_min / (2.0 * np.sqrt(3.0))
        self.bc_type: str = bc_type.lower()
        self.robin_alpha: float = float(robin_alpha)

        self.n_nodes: int = mesh.n_nodes
        self.n_elem: int = mesh.n_elem

        # Assemble scalar FE matrices (K_diff, M_mass, T)
        self._assemble_pde_system()

        # Pre-factorize symmetric positive definite system K_F = ell^2 * K_diff + M_mass
        K_F_csc = self.K_F.tocsc()
        self._solver = spla.splu(K_F_csc)

    def _assemble_pde_system(self) -> None:
        """Assemble scalar diffusion, mass, and element-node transfer matrices."""
        n_nodes = self.n_nodes
        n_elem = self.n_elem

        # Determine element topology
        if isinstance(self.mesh, StructuredMesh2D):
            etype = "Q4"
            ndim = 2
        elif isinstance(self.mesh, StructuredMesh3D):
            etype = "HEX8"
            ndim = 3
        else:
            etype = self.mesh.element_type
            ndim = self.mesh.ndim

        nodes = self.mesh.node_coords
        elements = self.mesh.elements
        n_elem_nodes = elements.shape[1]

        # COO assembly buffers
        rows_K = np.repeat(elements, n_elem_nodes, axis=1).ravel()
        cols_K = np.tile(elements, (1, n_elem_nodes)).ravel()

        rows_T = elements.ravel()
        cols_T = np.repeat(np.arange(n_elem), n_elem_nodes)

        vals_diff = np.zeros(n_elem * n_elem_nodes * n_elem_nodes, dtype=np.float64)
        vals_mass = np.zeros(n_elem * n_elem_nodes * n_elem_nodes, dtype=np.float64)
        vals_T = np.zeros(n_elem * n_elem_nodes, dtype=np.float64)

        # Standard element scalar matrices for uniform structured mesh fast-path
        if etype == "Q4" and isinstance(self.mesh, StructuredMesh2D):
            dx, dy = self.mesh.dx, self.mesh.dy
            # Analytical scalar diffusion and mass for Q4
            # k_diff = (dy/(6*dx))*[2, -2, -1, 1; -2, 2, 1, -1; ...] + (dx/(6*dy))*...
            kx = (dy / (6.0 * dx)) * np.array([
                [2, -2, -1, 1],
                [-2, 2, 1, -1],
                [-1, 1, 2, -2],
                [1, -1, -2, 2],
            ], dtype=np.float64)
            ky = (dx / (6.0 * dy)) * np.array([
                [2, 1, -1, -2],
                [1, 2, -2, -1],
                [-1, -2, 2, 1],
                [-2, -1, 1, 2],
            ], dtype=np.float64)
            ke_diff = kx + ky
            ke_mass = ((dx * dy) / 36.0) * np.array([
                [4, 2, 1, 2],
                [2, 4, 2, 1],
                [1, 2, 4, 2],
                [2, 1, 2, 4],
            ], dtype=np.float64)
            te = np.full(4, (dx * dy) / 4.0, dtype=np.float64)

            vals_diff[:] = np.tile(ke_diff.ravel(), n_elem)
            vals_mass[:] = np.tile(ke_mass.ravel(), n_elem)
            vals_T[:] = np.tile(te, n_elem)

        elif etype == "HEX8" and isinstance(self.mesh, StructuredMesh3D):
            dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
            vol = dx * dy * dz
            # 2x2x2 Gauss integration for scalar Laplacian and Mass
            gp = 1.0 / np.sqrt(3.0)
            from .fe_builders import _hex8_dN, _HEX8_SIGNS

            coords = np.array([
                [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
                [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz],
            ], dtype=np.float64)

            ke_diff = np.zeros((8, 8), dtype=np.float64)
            ke_mass = np.zeros((8, 8), dtype=np.float64)
            te = np.zeros(8, dtype=np.float64)

            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        xi, eta, zeta = sx * gp, sy * gp, sz * gp
                        dN = _hex8_dN(xi, eta, zeta)
                        J = dN.T @ coords
                        detJ = np.linalg.det(J)
                        dNdx = np.linalg.solve(J, dN.T).T  # (8, 3)
                        ke_diff += (dNdx @ dNdx.T) * detJ

                        s = _HEX8_SIGNS
                        N = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
                        ke_mass += np.outer(N, N) * detJ
                        te += N * detJ

            vals_diff[:] = np.tile(ke_diff.ravel(), n_elem)
            vals_mass[:] = np.tile(ke_mass.ravel(), n_elem)
            vals_T[:] = np.tile(te, n_elem)

        else:
            # General element-by-element integration
            stride = n_elem_nodes * n_elem_nodes
            for e in range(n_elem):
                c = nodes[elements[e]]
                if etype == "TRI3":
                    x = c[:, 0]
                    y = c[:, 1]
                    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
                    area = 0.5 * two_area
                    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / two_area
                    d = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / two_area
                    gradN = np.column_stack([b, d])
                    ke_diff = area * (gradN @ gradN.T)
                    ke_mass = (area / 12.0) * np.array([
                        [2, 1, 1],
                        [1, 2, 1],
                        [1, 1, 2],
                    ], dtype=np.float64)
                    te = np.full(3, area / 3.0, dtype=np.float64)
                elif etype == "TET4":
                    from .fe_builders import _TET4_DN
                    J = _TET4_DN.T @ c
                    detJ = np.linalg.det(J)
                    vol = detJ / 6.0
                    dNdx = np.linalg.solve(J, _TET4_DN.T).T
                    ke_diff = vol * (dNdx @ dNdx.T)
                    ke_mass = (vol / 20.0) * np.array([
                        [2, 1, 1, 1],
                        [1, 2, 1, 1],
                        [1, 1, 2, 1],
                        [1, 1, 1, 2],
                    ], dtype=np.float64)
                    te = np.full(4, vol / 4.0, dtype=np.float64)
                elif etype == "Q4":
                    from .fe_builders import _q4_dN
                    gp = 1.0 / np.sqrt(3.0)
                    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
                    ke_diff = np.zeros((4, 4), dtype=np.float64)
                    ke_mass = np.zeros((4, 4), dtype=np.float64)
                    te = np.zeros(4, dtype=np.float64)
                    for sx in (-1, 1):
                        for sy in (-1, 1):
                            xi, eta = sx * gp, sy * gp
                            dN = _q4_dN(xi, eta)
                            J = dN.T @ c[:, :2]
                            detJ = abs(np.linalg.det(J))
                            dNdx = np.linalg.solve(J, dN.T).T
                            ke_diff += (dNdx @ dNdx.T) * detJ
                            N = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
                            ke_mass += np.outer(N, N) * detJ
                            te += N * detJ
                elif etype == "HEX8":
                    from .fe_builders import _hex8_dN, _HEX8_SIGNS
                    gp = 1.0 / np.sqrt(3.0)
                    ke_diff = np.zeros((8, 8), dtype=np.float64)
                    ke_mass = np.zeros((8, 8), dtype=np.float64)
                    te = np.zeros(8, dtype=np.float64)
                    for sx in (-1, 1):
                        for sy in (-1, 1):
                            for sz in (-1, 1):
                                xi, eta, zeta = sx * gp, sy * gp, sz * gp
                                dN = _hex8_dN(xi, eta, zeta)
                                J = dN.T @ c
                                detJ = abs(np.linalg.det(J))
                                dNdx = np.linalg.solve(J, dN.T).T
                                ke_diff += (dNdx @ dNdx.T) * detJ
                                s = _HEX8_SIGNS
                                N = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
                                ke_mass += np.outer(N, N) * detJ
                                te += N * detJ
                else:
                    # High-order / shell element scalar diffusion & mass via nodal graph Laplacian
                    vol = self.mesh.elem_volumes[e]
                    h_char = vol ** (1.0 / max(1, ndim))
                    ones = np.ones((n_elem_nodes, 1), dtype=np.float64)
                    proj = np.eye(n_elem_nodes) - (ones @ ones.T) / n_elem_nodes
                    ke_diff = (vol / (h_char**2 + 1e-12)) * proj
                    ke_mass = (vol / (n_elem_nodes + 1.0)) * (
                        np.eye(n_elem_nodes) + (ones @ ones.T) / n_elem_nodes
                    )
                    te = np.full(n_elem_nodes, vol / n_elem_nodes, dtype=np.float64)

                vals_diff[e * stride : (e + 1) * stride] = ke_diff.ravel()
                vals_mass[e * stride : (e + 1) * stride] = ke_mass.ravel()
                vals_T[e * n_elem_nodes : (e + 1) * n_elem_nodes] = te

        self.K_diff = sp.coo_matrix((vals_diff, (rows_K, cols_K)), shape=(n_nodes, n_nodes)).tocsr()
        self.M_mass = sp.coo_matrix((vals_mass, (rows_K, cols_K)), shape=(n_nodes, n_nodes)).tocsr()
        self.T = sp.coo_matrix((vals_T, (rows_T, cols_T)), shape=(n_nodes, n_elem)).tocsr()

        # Averaging matrix T_avg: (n_elem, n_nodes) = diag(1/v_e) @ T.T
        inv_v = sp.diags(1.0 / self.mesh.elem_volumes)
        self.T_avg = (inv_v @ self.T.T).tocsr()

        # System matrix K_F = ell^2 * K_diff + M_mass
        self.K_F = (self.ell**2) * self.K_diff + self.M_mass

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """Forward Helmholtz filter: rho -> rho_tilde = T_avg @ K_F^{-1} @ T @ rho."""
        rho = np.asarray(rho, dtype=np.float64)
        b = self.T @ rho
        rho_N = self._solver.solve(b)
        rho_tilde = self.T_avg @ rho_N
        return rho_tilde

    def backward(self, df: np.ndarray) -> np.ndarray:
        """Adjoint backward sensitivity: df -> df/drho = T^T @ K_F^{-1} @ T_avg^T @ df."""
        df = np.asarray(df, dtype=np.float64)
        b_N = self.T_avg.T @ df
        # K_F is symmetric, so K_F^{-T} = K_F^{-1}
        w_N = self._solver.solve(b_N)
        df_drho = self.T.T @ w_N
        return df_drho

    def jacobian_matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute J @ v."""
        return self.forward(v)
