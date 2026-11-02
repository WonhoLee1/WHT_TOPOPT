"""High-Performance Unstructured Tetrahedral (TET4 / TET10) FEA Assembler and Solver.

Fully self-contained within wht_topopt.
Features:
1. Analytical and vectorized element stiffness matrices (Ke0, B0, Volume) for TET4 and TET10.
2. Numba JIT accelerated batch assembly of element stiffness matrices.
3. Zero-copy COO/CSR global stiffness assembly with pattern caching.
4. Direct interface to Intel MKL PARDISO multi-RHS linear static solver.
5. Exact element centroid von Mises stress evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback dummy decorators if numba is not installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from wht_topopt.core.fe_builders import (
    isotropic_elasticity_3d,
    tet4_ke0_me0,
    tet10_ke0_me0,
    _b_matrix_3d,
    _TET4_DN,
    _tet10_shape_derivs,
)


# ==============================================================================
# 1. TetraMesh Data Structure
# ==============================================================================

@dataclass
class TetraMesh:
    """Unstructured 3D Tetrahedral Mesh (TET4 or TET10).

    Attributes:
        nodes: (n_nodes, 3) nodal coordinates.
        elems: (n_elems, 4) for TET4 or (n_elems, 10) for TET10 connectivity.
        element_type: 'TET4' or 'TET10'.
    """
    nodes: np.ndarray
    elems: np.ndarray
    element_type: str = "TET4"

    def __post_init__(self):
        self.nodes = np.ascontiguousarray(self.nodes, dtype=np.float64)
        self.elems = np.ascontiguousarray(self.elems, dtype=np.int64)
        self.element_type = self.element_type.upper()
        if self.element_type not in {"TET4", "TET10"}:
            raise ValueError(f"Unsupported tetrahedral element type: {self.element_type}")

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elems(self) -> int:
        return self.elems.shape[0]

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * 3

    @property
    def ndof_per_elem(self) -> int:
        return 12 if self.element_type == "TET4" else 30

    @property
    def nnode_per_elem(self) -> int:
        return 4 if self.element_type == "TET4" else 10

    def element_dofs(self) -> np.ndarray:
        """Return (n_elems, ndof_per_elem) global degree-of-freedom indices."""
        edofs = np.zeros((self.n_elems, self.ndof_per_elem), dtype=np.int64)
        for i in range(self.nnode_per_elem):
            node_idx = self.elems[:, i]
            edofs[:, 3 * i] = 3 * node_idx
            edofs[:, 3 * i + 1] = 3 * node_idx + 1
            edofs[:, 3 * i + 2] = 3 * node_idx + 2
        return edofs

    def centroids(self) -> np.ndarray:
        """Compute (n_elems, 3) element centroid coordinates."""
        # For TET4, mean of 4 corner nodes; for TET10, corner nodes [0..3] determine the geometry
        corner_count = 4
        return np.mean(self.nodes[self.elems[:, :corner_count]], axis=1)

    def compute_volumes(self) -> np.ndarray:
        """Compute exact volumes of all tetrahedral elements."""
        coords = self.nodes[self.elems[:, :4]]  # (n_elems, 4, 3)
        v1 = coords[:, 1] - coords[:, 0]
        v2 = coords[:, 2] - coords[:, 0]
        v3 = coords[:, 3] - coords[:, 0]
        # Scalar triple product V = |(v1 x v2) . v3| / 6
        cross = np.cross(v1, v2)
        detJ = np.einsum("nd,nd->n", cross, v3)
        return np.abs(detJ) / 6.0


# ==============================================================================
# 2. Vectorized and Numba-Accelerated Batch Element Assembly
# ==============================================================================

@njit(parallel=True, fastmath=True)
def _assemble_tet4_batch_numba(
    coords: np.ndarray,      # (n_elems, 4, 3)
    C0: np.ndarray,          # (6, 6)
    youngs: np.ndarray,      # (n_elems,)
    ke_all: np.ndarray,      # Output (n_elems, 12, 12)
    b_all: np.ndarray,       # Output (n_elems, 6, 12)
    volumes: np.ndarray,     # Output (n_elems,)
):
    """Numba-accelerated parallel computation of all TET4 element stiffness matrices."""
    n_elems = coords.shape[0]

    for e in prange(n_elems):
        x0, y0, z0 = coords[e, 0, 0], coords[e, 0, 1], coords[e, 0, 2]
        x1, y1, z1 = coords[e, 1, 0], coords[e, 1, 1], coords[e, 1, 2]
        x2, y2, z2 = coords[e, 2, 0], coords[e, 2, 1], coords[e, 2, 2]
        x3, y3, z3 = coords[e, 3, 0], coords[e, 3, 1], coords[e, 3, 2]

        # Edge vectors
        v1x, v1y, v1z = x1 - x0, y1 - y0, z1 - z0
        v2x, v2y, v2z = x2 - x0, y2 - y0, z2 - z0
        v3x, v3y, v3z = x3 - x0, y3 - y0, z3 - z0

        # Determinant of Jacobian: (v1 x v2) . v3
        cx = v1y * v2z - v1z * v2y
        cy = v1z * v2x - v1x * v2z
        cz = v1x * v2y - v1y * v2x
        detJ = cx * v3x + cy * v3y + cz * v3z

        vol = detJ / 6.0
        volumes[e] = vol

        # Invert 3x3 Jacobian: J = [v1, v2, v3]^T
        # Rows of J are v1, v2, v3
        inv_det = 1.0 / detJ
        invJ00 = (v2y * v3z - v2z * v3y) * inv_det
        invJ01 = (v1z * v3y - v1y * v3z) * inv_det
        invJ02 = (v1y * v2z - v1z * v2y) * inv_det

        invJ10 = (v2z * v3x - v2x * v3z) * inv_det
        invJ11 = (v1x * v3z - v1z * v3x) * inv_det
        invJ12 = (v1z * v2x - v1x * v2z) * inv_det

        invJ20 = (v2x * v3y - v2y * v3x) * inv_det
        invJ21 = (v1y * v3x - v1x * v3y) * inv_det
        invJ22 = (v1x * v2y - v1y * v2x) * inv_det

        # Shape function gradients in physical coordinates: dNdx = _TET4_DN @ (invJ).T
        # Note: rows of invJ.T are columns of invJ!
        gx1, gy1, gz1 = invJ00, invJ10, invJ20
        gx2, gy2, gz2 = invJ01, invJ11, invJ21
        gx3, gy3, gz3 = invJ02, invJ12, invJ22

        gx0 = -gx1 - gx2 - gx3
        gy0 = -gy1 - gy2 - gy3
        gz0 = -gz1 - gz2 - gz3

        # Populate B matrix (6, 12)
        # Node 0
        b_all[e, 0, 0] = gx0; b_all[e, 1, 1] = gy0; b_all[e, 2, 2] = gz0
        b_all[e, 3, 0] = gy0; b_all[e, 3, 1] = gx0
        b_all[e, 4, 1] = gz0; b_all[e, 4, 2] = gy0
        b_all[e, 5, 0] = gz0; b_all[e, 5, 2] = gx0

        # Node 1
        b_all[e, 0, 3] = gx1; b_all[e, 1, 4] = gy1; b_all[e, 2, 5] = gz1
        b_all[e, 3, 3] = gy1; b_all[e, 3, 4] = gx1
        b_all[e, 4, 4] = gz1; b_all[e, 4, 5] = gy1
        b_all[e, 5, 3] = gz1; b_all[e, 5, 5] = gx1

        # Node 2
        b_all[e, 0, 6] = gx2; b_all[e, 1, 7] = gy2; b_all[e, 2, 8] = gz2
        b_all[e, 3, 6] = gy2; b_all[e, 3, 7] = gx2
        b_all[e, 4, 7] = gz2; b_all[e, 4, 8] = gy2
        b_all[e, 5, 6] = gz2; b_all[e, 5, 8] = gx2

        # Node 3
        b_all[e, 0, 9] = gx3; b_all[e, 1, 10] = gy3; b_all[e, 2, 11] = gz3
        b_all[e, 3, 9] = gy3; b_all[e, 3, 10] = gx3
        b_all[e, 4, 10] = gz3; b_all[e, 4, 11] = gy3
        b_all[e, 5, 9] = gz3; b_all[e, 5, 11] = gx3

        # Ke = E * vol * B.T @ C0 @ B
        E_scaled = youngs[e] * vol
        # Temp CB = C0 @ B (6, 12)
        cb = np.zeros((6, 12), dtype=np.float64)
        for r in range(6):
            for c_idx in range(12):
                s = 0.0
                for k in range(6):
                    s += C0[r, k] * b_all[e, k, c_idx]
                cb[r, c_idx] = s

        # ke_all[e] = B.T @ cb * E_scaled
        for i in range(12):
            for j in range(12):
                s = 0.0
                for k in range(6):
                    s += b_all[e, k, i] * cb[k, j]
                ke_all[e, i, j] = s * E_scaled


def assemble_tetra_elements(
    mesh: TetraMesh,
    youngs: np.ndarray,
    nu: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble all element stiffness matrices for TET4 or TET10.

    Args:
        mesh: TetraMesh instance.
        youngs: (n_elems,) element Young's moduli.
        nu: Poisson's ratio.

    Returns:
        ke_all: (n_elems, ndof_per_elem, ndof_per_elem) stiffness matrices.
        b_all: (n_elems, 6, ndof_per_elem) strain-displacement matrices.
        volumes: (n_elems,) element volumes.
    """
    n_elems = mesh.n_elems
    C0 = isotropic_elasticity_3d(nu)

    if mesh.element_type == "TET4":
        coords = mesh.nodes[mesh.elems]  # (n_elems, 4, 3)
        ke_all = np.zeros((n_elems, 12, 12), dtype=np.float64)
        b_all = np.zeros((n_elems, 6, 12), dtype=np.float64)
        volumes = np.zeros(n_elems, dtype=np.float64)

        if HAS_NUMBA:
            _assemble_tet4_batch_numba(coords, C0, youngs, ke_all, b_all, volumes)
        else:
            for e in range(n_elems):
                ke0, _, b0 = tet4_ke0_me0(coords[e], nu)
                vol = np.linalg.det(_TET4_DN.T @ coords[e]) / 6.0
                volumes[e] = vol
                b_all[e] = b0
                ke_all[e] = youngs[e] * ke0

        return ke_all, b_all, volumes

    elif mesh.element_type == "TET10":
        # 10-node quadratic tetrahedron
        coords = mesh.nodes[mesh.elems]
        ke_all = np.zeros((n_elems, 30, 30), dtype=np.float64)
        b_all = np.zeros((n_elems, 6, 30), dtype=np.float64)
        volumes = np.zeros(n_elems, dtype=np.float64)

        for e in range(n_elems):
            ke0, _, b0 = tet10_ke0_me0(coords[e], nu)
            vol = np.abs(np.linalg.det(_TET4_DN.T @ coords[e, :4])) / 6.0
            volumes[e] = vol
            b_all[e] = b0
            ke_all[e] = youngs[e] * ke0

        return ke_all, b_all, volumes

    else:
        raise ValueError(f"Unknown element type: {mesh.element_type}")


# ==============================================================================
# 3. TetraFEASolver (PARDISO Multi-RHS Direct Solver)
# ==============================================================================

class TetraFEASolver:
    """High-performance FEA solver for unstructured tetrahedral meshes."""

    def __init__(self, mesh: TetraMesh, nu: float):
        self.mesh = mesh
        self.nu = nu
        self.edofs = mesh.element_dofs()
        d = self.mesh.ndof_per_elem
        n_elems = self.mesh.n_elems

        # Precompute sparsity pattern (COO rows and cols)
        self.rows = np.broadcast_to(self.edofs[:, :, None], (n_elems, d, d)).ravel()
        self.cols = np.broadcast_to(self.edofs[:, None, :], (n_elems, d, d)).ravel()

    def assemble_global_stiffness(self, youngs: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """Assemble global sparse stiffness matrix K and return (K, b_all, volumes)."""
        ke_all, b_all, volumes = assemble_tetra_elements(self.mesh, youngs, self.nu)
        data = ke_all.ravel()
        K = sparse.coo_matrix((data, (self.rows, self.cols)), shape=(self.mesh.n_dofs, self.mesh.n_dofs)).tocsr()
        return K, b_all, volumes

    def solve(
        self,
        youngs: np.ndarray,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,  # (n_cases, n_dofs) or (n_dofs,)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve linear static system K u = F for single or multiple load cases.

        Args:
            youngs: (n_elems,) element Young's moduli.
            fixed_dofs: 1D array of constrained global DOF indices.
            loads: (n_cases, n_dofs) or (n_dofs,) load vectors.

        Returns:
            u: (n_cases, n_dofs) nodal displacement solutions.
            b_all: (n_elems, 6, ndof_per_elem) strain matrices.
            volumes: (n_elems,) element volumes.
            info: Solver telemetry dict.
        """
        t0 = time.perf_counter()
        K, b_all, volumes = self.assemble_global_stiffness(youngs)
        t_assemble = time.perf_counter() - t0

        t0_solve = time.perf_counter()
        fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        F = np.asarray(loads, dtype=np.float64)
        if F.ndim == 1:
            F = F[None, :]

        n_cases = F.shape[0]
        free = np.setdiff1d(np.arange(self.mesh.n_dofs), fixed)

        Kff = K[free][:, free].tocsc()
        rhs = F[:, free].T  # (n_free, n_cases)

        solved_with_pardiso = False
        try:
            import pypardiso
            sol = pypardiso.spsolve(Kff, rhs)
            solved_with_pardiso = True
        except Exception:
            sol = spla.spsolve(Kff, rhs)

        if sol.ndim == 1:
            sol = sol[:, None]

        u = np.zeros((n_cases, self.mesh.n_dofs), dtype=np.float64)
        u[:, free] = sol.T
        t_solve = time.perf_counter() - t0_solve

        info = {
            "solver": "pardiso" if solved_with_pardiso else "superlu",
            "t_assemble_sec": t_assemble,
            "t_solve_sec": t_solve,
            "n_free_dofs": len(free),
            "n_cases": n_cases,
        }
        return u, b_all, volumes, info


# ==============================================================================
# 4. Centroid von Mises Stress Evaluation
# ==============================================================================

def calculate_tetra_von_mises(
    mesh: TetraMesh,
    E_field: np.ndarray,
    nu: float,
    u: np.ndarray,           # (n_cases, n_dofs)
    b_all: np.ndarray,       # (n_elems, 6, ndof_per_elem)
) -> np.ndarray:
    """Calculate (n_cases, n_elems) centroid von Mises stress for tetrahedral elements."""
    if u.ndim == 1:
        u = u[None, :]
    n_cases = u.shape[0]
    n_elems = mesh.n_elems
    edofs = mesh.element_dofs()
    C0 = isotropic_elasticity_3d(nu)

    # u_e: (n_cases, n_elems, ndof_per_elem)
    ue = u[:, edofs]

    # Strains: eps = B @ u_e -> (n_cases, n_elems, 6)
    eps = np.einsum("nsd,cnd->cns", b_all, ue)

    # Stresses: sig = E * C0 @ eps -> (n_cases, n_elems, 6)
    sig = np.einsum("st,cnt->cns", C0, eps) * E_field[None, :, None]

    sx, sy, sz = sig[..., 0], sig[..., 1], sig[..., 2]
    txy, tyz, tzx = sig[..., 3], sig[..., 4], sig[..., 5]

    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy**2 + tyz**2 + tzx**2))
    return vm
