"""Finite element mesh abstractions for structured and unstructured topology optimization.

Provides StructuredMesh2D, StructuredMesh3D, and UnstructuredMesh containers with
isoparametric element geometric properties, counter-clockwise node ordering,
exact element volume calculation, and spatial search utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


class StructuredMesh2D:
    """Regular rectangular 2D quadrilateral mesh with counter-clockwise Q4 numbering.

    Local node ordering:
        3: (-1, +1) (top-left)      2: (+1, +1) (top-right)
                    *-----------------*
                    |                 |
                    |                 |
                    *-----------------*
        0: (-1, -1) (bottom-left)   1: (+1, -1) (bottom-right)

    Node indexing:
        node_id = jy * (nx + 1) + ix  (0 <= ix <= nx, 0 <= jy <= ny)

    Element indexing:
        elem_id = ey * nx + ex        (0 <= ex < nx, 0 <= ey < ny)
    """

    def __init__(
        self,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        lx: float = 1.0,
        ly: float = 1.0,
        *,
        nelx: Optional[int] = None,
        nely: Optional[int] = None,
    ) -> None:
        """Initialize regular 2D structured mesh.

        Parameters
        ----------
        nx : int, optional
            Number of elements along the x-axis (or pass ``nelx``).
        ny : int, optional
            Number of elements along the y-axis (or pass ``nely``).
        lx : float
            Total physical length along the x-axis.
        ly : float
            Total physical length along the y-axis.
        """
        nx_val = nelx if nx is None else nx
        ny_val = nely if ny is None else ny
        if nx_val is None or ny_val is None or nx_val <= 0 or ny_val <= 0:
            raise ValueError(f"Element counts nx and ny must be positive, got ({nx_val}, {ny_val})")
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError(f"Domain dimensions lx and ly must be positive, got ({lx}, {ly})")

        self.ndim: int = 2
        self.nelx: int = int(nx_val)
        self.nely: int = int(ny_val)
        self.lx: float = float(lx)
        self.ly: float = float(ly)

        self.dx: float = self.lx / self.nelx
        self.dy: float = self.ly / self.nely

        self.n_elem: int = self.nelx * self.nely
        self.n_nodes: int = (self.nelx + 1) * (self.nely + 1)
        self.ndof_per_node: int = 2
        self.ndof: int = self.ndof_per_node * self.n_nodes

        # Generate node coordinates: (n_nodes, 2)
        x_coords = np.linspace(0.0, self.lx, self.nelx + 1)
        y_coords = np.linspace(0.0, self.ly, self.nely + 1)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
        self.node_coords: np.ndarray = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)

        # Generate element connectivity: (n_elem, 4) in CCW order [0, 1, 2, 3]
        ex, ey = np.meshgrid(np.arange(self.nelx), np.arange(self.nely), indexing="xy")
        ex = ex.ravel()
        ey = ey.ravel()

        n0 = ey * (self.nelx + 1) + ex
        n1 = n0 + 1
        n2 = (ey + 1) * (self.nelx + 1) + ex + 1
        n3 = (ey + 1) * (self.nelx + 1) + ex

        self.elements: np.ndarray = np.column_stack([n0, n1, n2, n3]).astype(np.int64)

        # Element DOF matrix: (n_elem, 8)
        # Order: [u0_x, u0_y, u1_x, u1_y, u2_x, u2_y, u3_x, u3_y]
        edof = np.empty((self.n_elem, 8), dtype=np.int64)
        for i in range(4):
            node_idx = self.elements[:, i]
            edof[:, 2 * i] = 2 * node_idx
            edof[:, 2 * i + 1] = 2 * node_idx + 1
        self.edofMat: np.ndarray = edof

        # Element centers: (n_elem, 2)
        cx = (ex + 0.5) * self.dx
        cy = (ey + 0.5) * self.dy
        self.elem_centers: np.ndarray = np.column_stack([cx, cy]).astype(np.float64)

        # Element volumes (areas): (n_elem,)
        self.elem_volumes: np.ndarray = np.full(self.n_elem, self.dx * self.dy, dtype=np.float64)

    @property
    def nodes(self) -> np.ndarray:
        """Alias for ``node_coords``."""
        return self.node_coords

    @property
    def element_centroids(self) -> np.ndarray:
        """Alias for ``elem_centers``."""
        return self.elem_centers

    @property
    def n_elements(self) -> int:
        """Alias for ``n_elem``."""
        return self.n_elem

    @property
    def n_dofs(self) -> int:
        """Alias for ``ndof``."""
        return self.ndof

    def get_nodes_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find node indices within an axis-aligned 2D bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        mask = (
            (self.node_coords[:, 0] >= x_min - tol)
            & (self.node_coords[:, 0] <= x_max + tol)
            & (self.node_coords[:, 1] >= y_min - tol)
            & (self.node_coords[:, 1] <= y_max + tol)
        )
        return np.flatnonzero(mask)

    def get_nodes_at_x(self, x: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on the vertical line X = x."""
        return np.flatnonzero(np.abs(self.node_coords[:, 0] - x) <= tol)

    def get_nodes_at_y(self, y: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on the horizontal line Y = y."""
        return np.flatnonzero(np.abs(self.node_coords[:, 1] - y) <= tol)

    def get_elements_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        mask = (
            (self.elem_centers[:, 0] >= x_min - tol)
            & (self.elem_centers[:, 0] <= x_max + tol)
            & (self.elem_centers[:, 1] >= y_min - tol)
            & (self.elem_centers[:, 1] <= y_max + tol)
        )
        return np.flatnonzero(mask)

    def get_elements_in_circle(
        self,
        center: Tuple[float, float],
        radius: float,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a circular region."""
        dist_sq = (self.elem_centers[:, 0] - center[0]) ** 2 + (self.elem_centers[:, 1] - center[1]) ** 2
        return np.flatnonzero(dist_sq <= (radius + tol) ** 2)

    def to_unstructured(self, thickness: float = 1.0) -> UnstructuredMesh:
        """Convert structured mesh to a general UnstructuredMesh representation."""
        return UnstructuredMesh(
            nodes=self.node_coords.copy(),
            elements=self.elements.copy(),
            element_type="Q4",
            thickness=thickness,
        )


class StructuredMesh3D:
    """Regular rectangular 3D hexahedral mesh with consistent 8-node HEX8 numbering.

    Local node ordering:
        Bottom face (-z):
            3: (-1, +1, -1)             2: (+1, +1, -1)
                        *-----------------*
                        |                 |
                        |                 |
                        *-----------------*
            0: (-1, -1, -1)             1: (+1, -1, -1)
        Top face (+z):
            7: (-1, +1, +1)             6: (+1, +1, +1)
                        *-----------------*
                        |                 |
                        |                 |
                        *-----------------*
            4: (-1, -1, +1)             5: (+1, -1, +1)
    """

    def __init__(
        self,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        lx: float = 1.0,
        ly: float = 1.0,
        lz: float = 1.0,
        *,
        nelx: Optional[int] = None,
        nely: Optional[int] = None,
        nelz: Optional[int] = None,
    ) -> None:
        """Initialize regular 3D structured mesh.

        Parameters
        ----------
        nx, ny, nz : int, optional
            Number of elements along the x, y, and z axes (or pass ``nelx``, ``nely``, ``nelz``).
        lx, ly, lz : float
            Domain lengths along the x, y, and z axes.
        """
        nx_val = nelx if nx is None else nx
        ny_val = nely if ny is None else ny
        nz_val = nelz if nz is None else nz
        if nx_val is None or ny_val is None or nz_val is None or nx_val <= 0 or ny_val <= 0 or nz_val <= 0:
            raise ValueError(f"Element counts must be positive, got ({nx_val}, {ny_val}, {nz_val})")
        if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
            raise ValueError(f"Domain lengths must be positive, got ({lx}, {ly}, {lz})")

        self.ndim: int = 3
        self.nelx: int = int(nx_val)
        self.nely: int = int(ny_val)
        self.nelz: int = int(nz_val)
        self.lx: float = float(lx)
        self.ly: float = float(ly)
        self.lz: float = float(lz)

        self.dx: float = self.lx / self.nelx
        self.dy: float = self.ly / self.nely
        self.dz: float = self.lz / self.nelz

        self.n_elem: int = self.nelx * self.nely * self.nelz
        self.n_nodes: int = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        self.ndof_per_node: int = 3
        self.ndof: int = self.ndof_per_node * self.n_nodes

        # Generate node coordinates: (n_nodes, 3)
        # Node ordering: (kz, jy, ix) -> node_id = (kz * (ny + 1) + jy) * (nx + 1) + ix
        x_coords = np.linspace(0.0, self.lx, self.nelx + 1)
        y_coords = np.linspace(0.0, self.ly, self.nely + 1)
        z_coords = np.linspace(0.0, self.lz, self.nelz + 1)

        grid_z, grid_y, grid_x = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        self.node_coords: np.ndarray = np.column_stack(
            [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
        ).astype(np.float64)

        # Generate element connectivity: (n_elem, 8)
        ez, ey, ex = np.meshgrid(
            np.arange(self.nelz), np.arange(self.nely), np.arange(self.nelx), indexing="ij"
        )
        ez = ez.ravel()
        ey = ey.ravel()
        ex = ex.ravel()

        stride_y = self.nelx + 1
        stride_z = (self.nelx + 1) * (self.nely + 1)

        # Bottom face
        n0 = ez * stride_z + ey * stride_y + ex
        n1 = n0 + 1
        n2 = ez * stride_z + (ey + 1) * stride_y + ex + 1
        n3 = ez * stride_z + (ey + 1) * stride_y + ex

        # Top face
        n4 = (ez + 1) * stride_z + ey * stride_y + ex
        n5 = n4 + 1
        n6 = (ez + 1) * stride_z + (ey + 1) * stride_y + ex + 1
        n7 = (ez + 1) * stride_z + (ey + 1) * stride_y + ex

        self.elements: np.ndarray = np.column_stack([n0, n1, n2, n3, n4, n5, n6, n7]).astype(np.int64)

        # Element DOF matrix: (n_elem, 24)
        edof = np.empty((self.n_elem, 24), dtype=np.int64)
        for i in range(8):
            node_idx = self.elements[:, i]
            edof[:, 3 * i] = 3 * node_idx
            edof[:, 3 * i + 1] = 3 * node_idx + 1
            edof[:, 3 * i + 2] = 3 * node_idx + 2
        self.edofMat: np.ndarray = edof

        # Element centers: (n_elem, 3)
        cx = (ex + 0.5) * self.dx
        cy = (ey + 0.5) * self.dy
        cz = (ez + 0.5) * self.dz
        self.elem_centers: np.ndarray = np.column_stack([cx, cy, cz]).astype(np.float64)

        # Element volumes: (n_elem,)
        self.elem_volumes: np.ndarray = np.full(
            self.n_elem, self.dx * self.dy * self.dz, dtype=np.float64
        )

    @property
    def nodes(self) -> np.ndarray:
        """Alias for ``node_coords``."""
        return self.node_coords

    @property
    def element_centroids(self) -> np.ndarray:
        """Alias for ``elem_centers``."""
        return self.elem_centers

    @property
    def n_elements(self) -> int:
        """Alias for ``n_elem``."""
        return self.n_elem

    @property
    def n_dofs(self) -> int:
        """Alias for ``ndof``."""
        return self.ndof

    def get_nodes_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find node indices within an axis-aligned 3D bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        z_min, z_max = min(z_range), max(z_range)
        mask = (
            (self.node_coords[:, 0] >= x_min - tol)
            & (self.node_coords[:, 0] <= x_max + tol)
            & (self.node_coords[:, 1] >= y_min - tol)
            & (self.node_coords[:, 1] <= y_max + tol)
            & (self.node_coords[:, 2] >= z_min - tol)
            & (self.node_coords[:, 2] <= z_max + tol)
        )
        return np.flatnonzero(mask)

    def get_nodes_at_x(self, x: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on plane X = x."""
        return np.flatnonzero(np.abs(self.node_coords[:, 0] - x) <= tol)

    def get_nodes_at_y(self, y: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on plane Y = y."""
        return np.flatnonzero(np.abs(self.node_coords[:, 1] - y) <= tol)

    def get_nodes_at_z(self, z: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on plane Z = z."""
        return np.flatnonzero(np.abs(self.node_coords[:, 2] - z) <= tol)

    def get_elements_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        z_min, z_max = min(z_range), max(z_range)
        mask = (
            (self.elem_centers[:, 0] >= x_min - tol)
            & (self.elem_centers[:, 0] <= x_max + tol)
            & (self.elem_centers[:, 1] >= y_min - tol)
            & (self.elem_centers[:, 1] <= y_max + tol)
            & (self.elem_centers[:, 2] >= z_min - tol)
            & (self.elem_centers[:, 2] <= z_max + tol)
        )
        return np.flatnonzero(mask)

    def get_elements_in_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a spherical region."""
        dist_sq = (
            (self.elem_centers[:, 0] - center[0]) ** 2
            + (self.elem_centers[:, 1] - center[1]) ** 2
            + (self.elem_centers[:, 2] - center[2]) ** 2
        )
        return np.flatnonzero(dist_sq <= (radius + tol) ** 2)

    def to_unstructured(self) -> UnstructuredMesh:
        """Convert structured mesh to a general UnstructuredMesh representation."""
        return UnstructuredMesh(
            nodes=self.node_coords.copy(),
            elements=self.elements.copy(),
            element_type="HEX8",
        )


class UnstructuredMesh:
    """General unstructured finite element mesh supporting solid, shell, and 2D elements.

    Supported element types:
        - 2D Continuum (ndof_per_node = 2): 'TRI3', 'Q4'
        - 3D Solid Continuum (ndof_per_node = 3): 'TET4', 'TET10', 'HEX8', 'HEX20'
        - 3D Shell (ndof_per_node = 6): 'MITC3', 'MITC4'
    """

    SUPPORTED_ELEMENTS = {"TRI3", "Q4", "TET4", "TET10", "HEX8", "HEX20", "MITC3", "MITC4"}

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        element_type: str,
        thickness: float = 1.0,
        node_sets: Optional[Dict[str, np.ndarray]] = None,
        element_sets: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Initialize general unstructured mesh with exact volume and center calculation.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates array of shape (n_nodes, ndim).
        elements : np.ndarray
            Element connectivity array of shape (n_elem, n_node_per_elem).
        element_type : str
            Element family code (e.g., 'TRI3', 'Q4', 'TET4', 'TET10', 'HEX8', 'HEX20', 'MITC3', 'MITC4').
        thickness : float, optional
            Plate/shell or 2D plane thickness (default 1.0).
        node_sets : dict, optional
            Named node index groups.
        element_sets : dict, optional
            Named element index groups.
        """
        self.node_coords: np.ndarray = np.asarray(nodes, dtype=np.float64)
        self.elements: np.ndarray = np.asarray(elements, dtype=np.int64)
        self.element_type: str = element_type.upper()
        self.thickness: float = float(thickness)

        if self.element_type not in self.SUPPORTED_ELEMENTS:
            raise ValueError(
                f"Unsupported element_type '{self.element_type}'. Supported: {self.SUPPORTED_ELEMENTS}"
            )

        self.n_nodes: int = self.node_coords.shape[0]
        self.n_elem: int = self.elements.shape[0]
        self.ndim: int = self.node_coords.shape[1]
        self.n_node_per_elem: int = self.elements.shape[1]

        # Determine degrees of freedom per node
        if self.element_type in {"TRI3", "Q4"}:
            self.ndof_per_node = 2
        elif self.element_type in {"TET4", "TET10", "HEX8", "HEX20"}:
            self.ndof_per_node = 3
        elif self.element_type in {"MITC3", "MITC4"}:
            self.ndof_per_node = 6
        else:
            raise ValueError(f"Unknown element type: {self.element_type}")

        self.n_elem_dof: int = self.n_node_per_elem * self.ndof_per_node
        self.ndof: int = self.n_nodes * self.ndof_per_node

        # Build element DOF matrix: (n_elem, n_elem_dof)
        edof = np.empty((self.n_elem, self.n_elem_dof), dtype=np.int64)
        for i in range(self.n_node_per_elem):
            node_idx = self.elements[:, i]
            for d in range(self.ndof_per_node):
                edof[:, i * self.ndof_per_node + d] = node_idx * self.ndof_per_node + d
        self.edofMat: np.ndarray = edof

        # Named sets
        self.node_sets: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in (node_sets or {}).items()
        }
        self.element_sets: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in (element_sets or {}).items()
        }

        # Compute exact centroids and volumes
        self.elem_centers, self.elem_volumes = self._compute_centers_and_volumes()

    def _compute_centers_and_volumes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact element centroids and volumes/areas based on element geometry."""
        coords = self.node_coords[self.elements]  # shape (n_elem, n_nodes, ndim)
        n_elem = self.n_elem

        if self.element_type == "TRI3":
            # 2D Triangle: Shoelace / cross product area
            # Area = 0.5 * |(x1 - x0)(y2 - y0) - (x2 - x0)(y1 - y0)|
            v0 = coords[:, 0, :]
            v1 = coords[:, 1, :]
            v2 = coords[:, 2, :]
            cross2d = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (
                v1[:, 1] - v0[:, 1]
            )
            area = 0.5 * np.abs(cross2d)
            volumes = area * self.thickness
            centers = np.mean(coords, axis=1)
            return centers, volumes

        elif self.element_type == "Q4":
            # 2D Quadrilateral: 2x2 Gauss quadrature for exact area
            gp = 1.0 / np.sqrt(3.0)
            gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
            signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
            volumes = np.zeros(n_elem, dtype=np.float64)
            for xi, eta in gauss_pts:
                # Shape function derivatives
                dN_dxi = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * eta)
                dN_deta = 0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi)
                # J = [sum dN_dxi * x, sum dN_dxi * y; sum dN_deta * x, sum dN_deta * y]
                J11 = np.sum(dN_dxi[None, :] * coords[:, :, 0], axis=1)
                J12 = np.sum(dN_dxi[None, :] * coords[:, :, 1], axis=1)
                J21 = np.sum(dN_deta[None, :] * coords[:, :, 0], axis=1)
                J22 = np.sum(dN_deta[None, :] * coords[:, :, 1], axis=1)
                detJ = J11 * J22 - J12 * J21
                volumes += np.abs(detJ)
            volumes *= self.thickness
            centers = np.mean(coords, axis=1)
            return centers, volumes

        elif self.element_type == "TET4":
            # 3D 4-node Tetrahedron: V = 1/6 * |det([v1-v0, v2-v0, v3-v0])|
            v0 = coords[:, 0, :]
            v1 = coords[:, 1, :]
            v2 = coords[:, 2, :]
            v3 = coords[:, 3, :]
            e1 = v1 - v0
            e2 = v2 - v0
            e3 = v3 - v0
            cross = np.cross(e2, e3)
            det = np.sum(e1 * cross, axis=1)
            volumes = np.abs(det) / 6.0
            centers = np.mean(coords, axis=1)
            return centers, volumes

        elif self.element_type == "TET10":
            # 3D 10-node Quadratic Tetrahedron: 4-point Hammer-Marlowe Gauss quadrature
            a = (5.0 - np.sqrt(5.0)) / 20.0
            b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
            gps = [(a, a, a), (b, a, a), (a, b, a), (a, a, b)]
            w = 1.0 / 24.0  # Sum of weights = 1/6 (volume of ref tet)

            from .fe_builders import _tet10_shape_derivs

            volumes = np.zeros(n_elem, dtype=np.float64)
            for r, s, t in gps:
                dN = _tet10_shape_derivs(r, s, t)
                J = np.einsum("ka,nkb->nab", dN, coords)
                detJ = np.linalg.det(J)
                volumes += np.abs(detJ) * w

            # Centroid is at r=s=t=0.25 (L1=L2=L3=L4=0.25)
            # Evaluate shape functions at reference centroid:
            # Ni(0.25) = 0.25*(2*0.25 - 1) = -0.125 (for 4 corners)
            # Nj(0.25) = 4 * 0.25 * 0.25 = 0.25 (for 6 midsides)
            N_cent = np.array([-0.125] * 4 + [0.25] * 6, dtype=np.float64)
            centers = np.einsum("k,nkb->nb", N_cent, coords)
            return centers, volumes
            # Nj(0.25) = 4 * 0.25 * 0.25 = 0.25 (for 6 midsides)
            N_cent = np.array([-0.125] * 4 + [0.25] * 6, dtype=np.float64)
            centers = np.einsum("k,nkb->nb", N_cent, coords)
            return centers, volumes

        elif self.element_type == "HEX8":
            # 3D 8-node Hexahedron: 2x2x2 Gauss quadrature
            gp = 1.0 / np.sqrt(3.0)
            signs = np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=float,
            )
            volumes = np.zeros(n_elem, dtype=np.float64)
            for sx in (-gp, gp):
                for sy in (-gp, gp):
                    for sz in (-gp, gp):
                        dN = np.empty((8, 3), dtype=np.float64)
                        dN[:, 0] = 0.125 * signs[:, 0] * (1.0 + signs[:, 1] * sy) * (1.0 + signs[:, 2] * sz)
                        dN[:, 1] = 0.125 * signs[:, 1] * (1.0 + signs[:, 0] * sx) * (1.0 + signs[:, 2] * sz)
                        dN[:, 2] = 0.125 * signs[:, 2] * (1.0 + signs[:, 0] * sx) * (1.0 + signs[:, 1] * sy)

                        J = np.einsum("ka,nkb->nab", dN, coords)
                        detJ = np.linalg.det(J)
                        volumes += np.abs(detJ)
            # Reference centroid xi=eta=zeta=0 gives mean of 8 nodes
            centers = np.mean(coords, axis=1)
            return centers, volumes

        elif self.element_type == "HEX20":
            # 20-node Quadratic Hexahedron: 3x3x3 Gauss quadrature
            g1 = np.sqrt(3.0 / 5.0)
            gp_1d = [-g1, 0.0, g1]
            w_1d = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
            from .fe_builders import hex20_shape_derivs

            volumes = np.zeros(n_elem, dtype=np.float64)
            for i, xi in enumerate(gp_1d):
                for j, eta in enumerate(gp_1d):
                    for k, zeta in enumerate(gp_1d):
                        w = w_1d[i] * w_1d[j] * w_1d[k]
                        dN = hex20_shape_derivs(xi, eta, zeta)
                        J = np.einsum("ka,nkb->nab", dN, coords)
                        detJ = np.linalg.det(J)
                        volumes += np.abs(detJ) * w

            # Reference centroid xi=eta=zeta=0
            # For HEX20 serendipity: corner nodes have N = -0.25, midside nodes have N = 0.5?
            # Let's evaluate shape functions at (0, 0, 0)
            from .fe_builders import hex20_shape_functions

            N_cent = hex20_shape_functions(0.0, 0.0, 0.0)
            centers = np.einsum("k,nkb->nb", N_cent, coords)
            return centers, volumes

        elif self.element_type == "MITC3":
            # 3D 3-node Triangle Shell
            v0 = coords[:, 0, :]
            v1 = coords[:, 1, :]
            v2 = coords[:, 2, :]
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            volumes = area * self.thickness
            centers = np.mean(coords, axis=1)
            return centers, volumes

        elif self.element_type == "MITC4":
            # 3D 4-node Quad Shell: 2x2 Gauss quadrature on reference plane
            gp = 1.0 / np.sqrt(3.0)
            signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
            areas = np.zeros(n_elem, dtype=np.float64)
            for xi, eta in [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]:
                dN_dxi = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * eta)
                dN_deta = 0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi)
                # Tangent vectors in 3D: dx/dxi and dx/deta
                t_xi = np.einsum("k,nkb->nb", dN_dxi, coords)
                t_eta = np.einsum("k,nkb->nb", dN_deta, coords)
                normal = np.cross(t_xi, t_eta)
                dA = np.linalg.norm(normal, axis=1)
                areas += dA
            volumes = areas * self.thickness
            centers = np.mean(coords, axis=1)
            return centers, volumes

        else:
            centers = np.mean(coords, axis=1)
            volumes = np.ones(n_elem, dtype=np.float64)
            return centers, volumes

    def get_nodes_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Optional[Tuple[float, float]] = None,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find node indices within an axis-aligned bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        mask = (
            (self.node_coords[:, 0] >= x_min - tol)
            & (self.node_coords[:, 0] <= x_max + tol)
            & (self.node_coords[:, 1] >= y_min - tol)
            & (self.node_coords[:, 1] <= y_max + tol)
        )
        if self.ndim == 3 and z_range is not None:
            z_min, z_max = min(z_range), max(z_range)
            mask &= (self.node_coords[:, 2] >= z_min - tol) & (self.node_coords[:, 2] <= z_max + tol)
        return np.flatnonzero(mask)

    def get_nodes_at_x(self, x: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on coordinate X = x."""
        return np.flatnonzero(np.abs(self.node_coords[:, 0] - x) <= tol)

    def get_nodes_at_y(self, y: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on coordinate Y = y."""
        return np.flatnonzero(np.abs(self.node_coords[:, 1] - y) <= tol)

    def get_nodes_at_z(self, z: float, tol: float = 1e-7) -> np.ndarray:
        """Find node indices on coordinate Z = z."""
        if self.ndim < 3:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.abs(self.node_coords[:, 2] - z) <= tol)

    def get_elements_in_box(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Optional[Tuple[float, float]] = None,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a bounding box."""
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        mask = (
            (self.elem_centers[:, 0] >= x_min - tol)
            & (self.elem_centers[:, 0] <= x_max + tol)
            & (self.elem_centers[:, 1] >= y_min - tol)
            & (self.elem_centers[:, 1] <= y_max + tol)
        )
        if self.ndim == 3 and z_range is not None:
            z_min, z_max = min(z_range), max(z_range)
            mask &= (self.elem_centers[:, 2] >= z_min - tol) & (self.elem_centers[:, 2] <= z_max + tol)
        return np.flatnonzero(mask)

    def get_elements_in_sphere(
        self,
        center: Sequence[float],
        radius: float,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Find element indices whose centroids fall inside a sphere or circle."""
        c = np.asarray(center, dtype=np.float64)[: self.ndim]
        diff = self.elem_centers - c[None, :]
        dist_sq = np.sum(diff**2, axis=1)
        return np.flatnonzero(dist_sq <= (radius + tol) ** 2)
