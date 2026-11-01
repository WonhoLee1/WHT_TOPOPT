"""Isosurface and contour extraction for 2D/3D topology optimization results.

Provides IsosurfaceExtractor with:
- Volume-weighted element-to-nodal density interpolation with boundary padding.
- 3D Marching Cubes (Lorensen & Cline 1987) with bisection volume-matching level-set search.
- 2D Marching Squares contour extraction and extrusion to watertight 3D meshes.
- Taubin volume-preserving Laplacian smoothing (Taubin 1995) to eliminate staircase artifacts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
import skimage.measure

from ..core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized 2D point-in-polygon ray casting test.

    Parameters
    ----------
    points : np.ndarray of shape (M, 2)
        Query 2D points.
    polygon : np.ndarray of shape (N, 2)
        Vertices of the polygon in cyclic order.

    Returns
    -------
    np.ndarray of shape (M,) of bool
        True if query point is inside polygon.
    """
    x, y = points[:, 0], points[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        mask = (y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x))
        if np.any(mask):
            x_inters = (y[mask] - p1y) * (p2x - p1x) / (p2y - p1y + 1e-14) + p1x
            inside[mask] ^= (p1x == p2x) | (x[mask] <= x_inters)
        p1x, p1y = p2x, p2y
    return inside


def _signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute signed volume enclosed by a triangle surface mesh via divergence theorem.

    V = 1/6 * sum_f (v0 . (v1 x v2))
    """
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1, v2)
    vol = np.sum(v0 * cross) / 6.0
    return float(vol)


class IsosurfaceExtractor:
    """Extracts smooth geometric boundary representations from continuous density fields."""

    def __init__(self, mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]) -> None:
        """Initialize extractor on a mesh domain.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Underlying FE mesh instance.
        """
        self.mesh = mesh
        self.ndim: int = mesh.ndim

    def densities_to_nodal(
        self,
        rho: np.ndarray,
        pad_boundary: bool = False,
        pad_value: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[float, ...], Tuple[float, ...]]:
        """Interpolate element densities to regular nodal grid with optional zero-padding.

        Parameters
        ----------
        rho : np.ndarray of shape (n_elem,)
            Physical density field x_phys in [0, 1].
        pad_boundary : bool, default False
            If True, pads domain by 1 voxel layer so isosurface forms a closed watertight shell.
        pad_value : float, default 0.0
            Padding density value.

        Returns
        -------
        grid : np.ndarray
            Nodal scalar field of shape (ny+1, nx+1) for 2D or (nz+1, ny+1, nx+1) for 3D.
        spacing : tuple of float
            Grid spacing per axis: (dy, dx) for 2D, or (dz, dy, dx) for 3D.
        origin : tuple of float
            Coordinates of grid (0, 0) or (0, 0, 0) corner (shifted if padded).
        """
        rho_flat = np.asarray(rho, dtype=np.float64).ravel()
        if len(rho_flat) != self.mesh.n_elem:
            raise ValueError(
                f"Density array length {len(rho_flat)} does not match mesh element count {self.mesh.n_elem}"
            )

        # 1. Volume-weighted nodal averaging
        nodal_sum = np.zeros(self.mesh.n_nodes, dtype=np.float64)
        nodal_weight = np.zeros(self.mesh.n_nodes, dtype=np.float64)

        for el_idx in range(self.mesh.n_elem):
            nodes = self.mesh.elements[el_idx]
            val = rho_flat[el_idx]
            vol = float(self.mesh.elem_volumes[el_idx])
            nodal_sum[nodes] += val * vol
            nodal_weight[nodes] += vol

        nodal_vals = nodal_sum / np.maximum(nodal_weight, 1e-14)

        if isinstance(self.mesh, StructuredMesh2D):
            # Reshape: row=y (0..ny), col=x (0..nx)
            grid = nodal_vals.reshape((self.mesh.nely + 1, self.mesh.nelx + 1))
            spacing = (self.mesh.dy, self.mesh.dx)
            origin = (0.0, 0.0)

            if pad_boundary:
                grid = np.pad(grid, pad_width=1, mode="constant", constant_values=pad_value)
                origin = (-self.mesh.dx, -self.mesh.dy)

            return grid, spacing, origin

        elif isinstance(self.mesh, StructuredMesh3D):
            # Reshape: slice=z (0..nz), row=y (0..ny), col=x (0..nx)
            grid = nodal_vals.reshape((self.mesh.nelz + 1, self.mesh.nely + 1, self.mesh.nelx + 1))
            spacing = (self.mesh.dz, self.mesh.dy, self.mesh.dx)
            origin = (0.0, 0.0, 0.0)

            if pad_boundary:
                grid = np.pad(grid, pad_width=1, mode="constant", constant_values=pad_value)
                origin = (-self.mesh.dx, -self.mesh.dy, -self.mesh.dz)

            return grid, spacing, origin

        else:
            raise NotImplementedError("Arbitrary unstructured volumetric marching cubes requires structured resampler.")

    def extract_3d(
        self,
        rho: np.ndarray,
        tau: float = 0.5,
        pad_boundary: bool = True,
        pad_value: float = 0.0,
        target_volume: Optional[float] = None,
        bisection_tol: float = 1e-3,
        max_bisection_iters: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D triangle surface mesh using Marching Cubes.

        Parameters
        ----------
        rho : np.ndarray
            Element physical densities.
        tau : float, default 0.5
            Iso-level threshold. Ignored if target_volume is specified.
        pad_boundary : bool, default True
            Pads boundary so isosurface forms a sealed, watertight solid.
        pad_value : float, default 0.0
            Value for boundary padding voxels.
        target_volume : float, optional
            If provided, automatically determines volume-preserving threshold tau*
            via bisection such that enclosed volume matches target_volume.
        bisection_tol : float, default 1e-3
            Relative tolerance for volume matching.
        max_bisection_iters : int, default 15
            Maximum bisection iterations.

        Returns
        -------
        vertices : np.ndarray of shape (V, 3)
            Cartesian (x, y, z) coordinates.
        faces : np.ndarray of shape (F, 3)
            Triangle node indices in CCW winding.
        """
        if not isinstance(self.mesh, StructuredMesh3D):
            raise TypeError(f"extract_3d requires StructuredMesh3D, got {type(self.mesh)}")

        grid, spacing, origin = self.densities_to_nodal(
            rho, pad_boundary=pad_boundary, pad_value=pad_value
        )
        dz, dy, dx = spacing
        ox, oy, oz = origin

        def _run_mc(level_val: float) -> Tuple[np.ndarray, np.ndarray]:
            level_clamped = float(np.clip(level_val, 1e-4, 1.0 - 1e-4))
            try:
                verts_zyx, faces, _, _ = skimage.measure.marching_cubes(
                    grid, level=level_clamped, spacing=(dz, dy, dx)
                )
                if len(verts_zyx) == 0:
                    return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

                # Convert (z, y, x) -> (x, y, z)
                verts_xyz = np.column_stack([
                    verts_zyx[:, 2] + ox,
                    verts_zyx[:, 1] + oy,
                    verts_zyx[:, 0] + oz,
                ]).astype(np.float64)
                return verts_xyz, faces.astype(np.int64)
            except (ValueError, RuntimeError):
                return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

        if target_volume is not None and target_volume > 0.0:
            # Volume-preserving bisection search for tau*
            # As tau increases, the enclosed volume monotonically decreases
            tau_low = 0.02
            tau_high = 0.98
            best_verts, best_faces = _run_mc(tau)
            best_vol_err = float("inf")

            for _ in range(max_bisection_iters):
                tau_mid = 0.5 * (tau_low + tau_high)
                verts_m, faces_m = _run_mc(tau_mid)
                if len(faces_m) == 0:
                    tau_high = tau_mid
                    continue

                vol_m = abs(_signed_mesh_volume(verts_m, faces_m))
                rel_err = abs(vol_m - target_volume) / target_volume

                if rel_err < best_vol_err:
                    best_vol_err = rel_err
                    best_verts, best_faces = verts_m, faces_m

                if rel_err <= bisection_tol:
                    break

                if vol_m > target_volume:
                    # Enclosed volume too large -> increase threshold to shrink solid
                    tau_low = tau_mid
                else:
                    # Enclosed volume too small -> decrease threshold
                    tau_high = tau_mid

            return best_verts, best_faces

        return _run_mc(tau)

    def extract_2d(
        self,
        rho: np.ndarray,
        tau: float = 0.5,
        pad_boundary: bool = True,
        pad_value: float = 0.0,
    ) -> List[np.ndarray]:
        """Extract 2D contour loops using Marching Squares.

        Parameters
        ----------
        rho : np.ndarray
            Element physical densities.
        tau : float, default 0.5
            Iso-level threshold.
        pad_boundary : bool, default True
            Pads boundary so contours close against domain boundaries.
        pad_value : float, default 0.0
            Value for padding pixels.

        Returns
        -------
        contours : list of np.ndarray
            List of 2D physical curves, each array of shape (N_pts, 2) in (x, y).
        """
        if not isinstance(self.mesh, StructuredMesh2D):
            raise TypeError(f"extract_2d requires StructuredMesh2D, got {type(self.mesh)}")

        grid, spacing, origin = self.densities_to_nodal(
            rho, pad_boundary=pad_boundary, pad_value=pad_value
        )
        dy, dx = spacing
        ox, oy = origin

        raw_contours = skimage.measure.find_contours(grid, level=float(tau))
        physical_contours: List[np.ndarray] = []

        for c in raw_contours:
            if len(c) < 3:
                continue
            # c has indices [row, col] -> physical [x = col*dx + ox, y = row*dy + oy]
            x_pts = c[:, 1] * dx + ox
            y_pts = c[:, 0] * dy + oy
            poly_2d = np.column_stack([x_pts, y_pts]).astype(np.float64)

            # Ensure cyclic closure if start and end are within numerical tolerance
            if np.linalg.norm(poly_2d[0] - poly_2d[-1]) > 1e-5 and pad_boundary:
                poly_2d = np.vstack([poly_2d, poly_2d[0:1]])

            physical_contours.append(poly_2d)

        return physical_contours

    def extrude_2d_to_3d(
        self,
        contours: List[np.ndarray],
        thickness: float = 1.0,
        z_origin: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrude 2D contour loops into a guaranteed watertight 3D triangle mesh.

        Parameters
        ----------
        contours : list of np.ndarray of shape (N, 2)
            Closed 2D boundary polygons.
        thickness : float, default 1.0
            Physical extrusion thickness along +z axis.
        z_origin : float, default 0.0
            Starting z coordinate.

        Returns
        -------
        vertices : np.ndarray of shape (V, 3)
            3D mesh vertex coordinates.
        faces : np.ndarray of shape (F, 3)
            Triangle connectivity in CCW outward winding.
        """
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        vert_offset = 0

        for raw_poly in contours:
            # Drop redundant duplicate closing point if present
            poly = raw_poly.copy()
            if len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
                poly = poly[:-1]

            n_pts = len(poly)
            if n_pts < 3:
                continue

            # Triangulate 2D cap using Delaunay + point-in-polygon ray casting
            tri = Delaunay(poly)
            simplices = tri.simplices
            centroids = poly[simplices].mean(axis=1)
            inside_mask = _point_in_polygon(centroids, poly)
            cap_faces_2d = simplices[inside_mask]

            if len(cap_faces_2d) == 0:
                continue

            # Build 3D extruded prism vertices
            bot_verts = np.column_stack([poly, np.full(n_pts, z_origin, dtype=np.float64)])
            top_verts = np.column_stack([poly, np.full(n_pts, z_origin + thickness, dtype=np.float64)])
            prism_verts = np.vstack([bot_verts, top_verts])

            # Bottom cap faces (invert winding [0, 2, 1] so normal points -z)
            f_bot = cap_faces_2d[:, [0, 2, 1]] + vert_offset

            # Top cap faces (normal points +z)
            f_top = cap_faces_2d + (n_pts + vert_offset)

            # Side walls (two triangles per quad edge)
            side_faces = []
            for i in range(n_pts):
                nxt = (i + 1) % n_pts
                v_b0 = i + vert_offset
                v_b1 = nxt + vert_offset
                v_t0 = i + n_pts + vert_offset
                v_t1 = nxt + n_pts + vert_offset
                # Winding: outward facing
                side_faces.append([v_b0, v_b1, v_t1])
                side_faces.append([v_b0, v_t1, v_t0])

            all_verts.append(prism_verts)
            all_faces.append(f_bot)
            all_faces.append(f_top)
            all_faces.append(np.array(side_faces, dtype=np.int64))

            vert_offset += len(prism_verts)

        if not all_verts:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

        combined_vertices = np.vstack(all_verts)
        combined_faces = np.vstack(all_faces)
        return combined_vertices, combined_faces

    def extract(
        self,
        rho: np.ndarray,
        level: float = 0.5,
        thickness: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D triangle mesh from 2D (extruded) or 3D density field."""
        if isinstance(self.mesh, StructuredMesh3D):
            return self.extract_3d(rho, tau=level)
        elif isinstance(self.mesh, StructuredMesh2D):
            contours = self.extract_2d(rho, tau=level)
            return self.extrude_2d_to_3d(contours, thickness=thickness)
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    @staticmethod
    def taubin_smooth(
        vertices: np.ndarray,
        faces: np.ndarray,
        lambda_factor: float = 0.5,
        mu_factor: float = -0.53,
        iterations: int = 10,
        lam: Optional[float] = None,
        mu: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Taubin volume-preserving lambda-mu Laplacian smoothing.

        Alternates positive shrinking pass v <- v + lambda * L * v and negative
        expansion pass v <- v + mu * L * v using the vertex-umbrella graph Laplacian.
        Eliminates high-frequency Marching Cubes staircase voxelization artifacts
        while preserving low-frequency geometric volume and shape.

        Parameters
        ----------
        vertices : np.ndarray of shape (V, 3)
            Input mesh vertex coordinates.
        faces : np.ndarray of shape (F, 3)
            Mesh triangle connectivity.
        lambda_factor : float, default 0.5
            Positive shrinking factor (0 < lambda < -mu).
        mu_factor : float, default -0.53
            Negative expansion factor (mu < -lambda < 0).
        iterations : int, default 10
            Number of paired smoothing passes.

        Returns
        -------
        np.ndarray of shape (V, 3)
            Smoothed vertex coordinates.
        """
        if len(vertices) == 0 or len(faces) == 0:
            return vertices.copy()

        n_v = len(vertices)
        # Assemble undirected vertex graph edges
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ])
        # Symmetric edges
        edges_sym = np.vstack([edges, edges[:, [1, 0]]])
        edges_unique = np.unique(edges_sym, axis=0)

        i_idx = edges_unique[:, 0]
        j_idx = edges_unique[:, 1]
        data = np.ones(len(i_idx), dtype=np.float64)

        adj = sp.csr_matrix((data, (i_idx, j_idx)), shape=(n_v, n_v))
        degrees = np.array(adj.sum(axis=1)).flatten()
        deg_inv = sp.diags(1.0 / np.maximum(degrees, 1.0))

        # Umbrella Laplacian: L = D^-1 * A - I
        L = deg_inv @ adj - sp.eye(n_v, format="csr")

        v_curr = vertices.astype(np.float64).copy()
        l_val = float(lam) if lam is not None else float(lambda_factor)
        m_val = float(mu) if mu is not None else float(mu_factor)
        for _ in range(iterations):
            # 1. Shrinking pass (lambda > 0)
            v_curr += l_val * (L @ v_curr)
            # 2. Expansion pass (mu < 0)
            v_curr += m_val * (L @ v_curr)

        return v_curr
