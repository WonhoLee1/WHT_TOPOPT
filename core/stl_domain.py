"""STL Domain Voxelization, Rigid-Body MPC Coupling, and Hex8 TopOpt Core.

Completely self-contained implementation for wht_topopt, eliminating all external dependencies on aieng.
Provides:
1. STL ray-casting voxelization (inside_grid, read_stl, build_stl_problem)
2. Hex8 reference element stiffness & elasticity (isotropic_D, _b_matrix, _hex8_dN, reference_stiffness)
3. Grid and Mesh definitions (Mesh, structured_hex)
4. Rigid-Body MPC coupling transformations (RigidBody, rigid_body_transform)
5. Direct / PARDISO solver adapter (BuiltinAdapter, TopOptEvaluator)
6. Problem definitions & topology optimization runner (LoadCase, TopOptProblem, StlCase, run_topology_optimization)
7. Progress GIF generator (ProgressGif)
"""
from __future__ import annotations

from types import SimpleNamespace
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

# ==============================================================================
# 1. STL Reading & Ray-Casting Grid Voxelization
# ==============================================================================

def read_stl(path: str | Path) -> np.ndarray:
    """Read a binary/ASCII STL file and return (m, 3, 3) triangle vertex coordinates."""
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.triangles, dtype=float)


def inside_grid(
    tris: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    chunk: int = 20000,
) -> np.ndarray:
    """Boolean mask (nz, ny, nx) indicating whether grid centroids (xs, ys, zs) lie inside the closed STL.

    Each grid column (x_i, y_j) is cast as a ray along +z; an interior point has an odd number of surface crossings below it.
    Irrational coordinate shifts prevent rays from grazing edges or vertices.
    """
    span = max(np.ptp(xs) if len(xs) > 1 else 1.0, np.ptp(ys) if len(ys) > 1 else 1.0, 1.0)
    xs_s = xs + span * 1.2345678e-7
    ys_s = ys + span * 2.3456789e-7
    nx, ny = len(xs), len(ys)
    col_ids, col_z = [], []
    for s in range(0, len(tris), chunk):
        t = tris[s : s + chunk]
        lo, hi = t[:, :, :2].min(axis=1), t[:, :, :2].max(axis=1)
        i0, i1 = np.searchsorted(xs_s, lo[:, 0]), np.searchsorted(xs_s, hi[:, 0], side="right")
        j0, j1 = np.searchsorted(ys_s, lo[:, 1]), np.searchsorted(ys_s, hi[:, 1], side="right")
        ni, nj = np.maximum(i1 - i0, 0), np.maximum(j1 - j0, 0)
        cnt = ni * nj
        if cnt.sum() == 0:
            continue
        tri = np.repeat(np.arange(len(t)), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ii = i0[tri] + offs % np.maximum(ni[tri], 1)
        jj = j0[tri] + offs // np.maximum(ni[tri], 1)
        px, py = xs_s[ii], ys_s[jj]
        a, b, c = t[tri, 0], t[tri, 1], t[tri, 2]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        ok = np.abs(det) > 0
        l1 = ((px - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (py - a[:, 1])) / np.where(ok, det, 1.0)
        l2 = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (px - a[:, 0]) * (b[:, 1] - a[:, 1])) / np.where(ok, det, 1.0)
        hit = ok & (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        z = a[:, 2] + l1 * (b[:, 2] - a[:, 2]) + l2 * (c[:, 2] - a[:, 2])
        col_ids.append((jj * nx + ii)[hit])
        col_z.append(z[hit])

    out = np.zeros((len(zs), ny, nx), dtype=bool)
    if not col_ids:
        return out
    cid, cz = np.concatenate(col_ids), np.concatenate(col_z)
    order = np.lexsort((cz, cid))
    cid, cz = cid[order], cz[order]
    starts = np.searchsorted(cid, np.arange(nx * ny))
    ends = np.searchsorted(cid, np.arange(nx * ny), side="right")
    for col in np.flatnonzero(ends > starts):
        below = np.searchsorted(cz[starts[col] : ends[col]], zs)
        out[:, col // nx, col % nx] = (below % 2) == 1
    return out


# ==============================================================================
# 2. Hex8 Solid Mesh & Structured Hex Grid
# ==============================================================================

NODES_PER_ELEMENT = {"hex8": 8, "tet4": 4}
DOF_PER_NODE = 3


@dataclass
class Mesh:
    """Solid mesh: nodes (nn, 3) and elems (ne, 8) node indices."""
    nodes: np.ndarray
    elems: np.ndarray
    etype: str = "hex8"
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    grid: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=float)
        self.elems = np.ascontiguousarray(self.elems, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elems(self) -> int:
        return len(self.elems)

    @property
    def n_dofs(self) -> int:
        return DOF_PER_NODE * self.n_nodes

    def element_dofs(self) -> np.ndarray:
        """(ne, 24) global dof indices (ux, uy, uz per node)."""
        d = DOF_PER_NODE * self.elems[:, :, None] + np.arange(DOF_PER_NODE)[None, None, :]
        return d.reshape(self.n_elems, -1)

    def centroids(self) -> np.ndarray:
        return self.nodes[self.elems].mean(axis=1)

    def node_dofs(self, node_ids: Any, components: Tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64).ravel()
        comps = np.asarray(components, dtype=np.int64).ravel()
        return (DOF_PER_NODE * node_ids[:, None] + comps[None, :]).ravel()


def structured_hex(nx: int, ny: int, nz: int, size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Mesh:
    """Build a regular Hex8 box mesh."""
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    hx, hy, hz = (float(s) for s in size)
    xs, ys, zs = np.arange(nx + 1) * hx, np.arange(ny + 1) * hy, np.arange(nz + 1) * hz
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    nodes = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    def nid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    kz, jy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    i, j, k = ix.ravel(), jy.ravel(), kz.ravel()
    elems = np.stack([
        nid(i, j, k),
        nid(i + 1, j, k),
        nid(i + 1, j + 1, k),
        nid(i, j + 1, k),
        nid(i, j, k + 1),
        nid(i + 1, j, k + 1),
        nid(i + 1, j + 1, k + 1),
        nid(i, j + 1, k + 1),
    ], axis=1)
    return Mesh(nodes, elems, "hex8", grid=(nx, ny, nz))


# ==============================================================================
# 3. Hex8 Finite Element Matrices & Reference Stiffness
# ==============================================================================

_HEX_SIGNS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)
_GP = 1.0 / np.sqrt(3.0)
_HEX_GAUSS = [(sx * _GP, sy * _GP, sz * _GP) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]


def isotropic_D(E: float, nu: float) -> np.ndarray:
    """6x6 elasticity matrix in Voigt engineering notation."""
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=float)
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _b_matrix(dNdx: np.ndarray) -> np.ndarray:
    """(n, 3, k) shape function gradients -> (n, 6, 3k) strain-displacement matrix B."""
    n, _, k = dNdx.shape
    B = np.zeros((n, 6, 3 * k), dtype=float)
    for a in range(k):
        gx, gy, gz = dNdx[:, 0, a], dNdx[:, 1, a], dNdx[:, 2, a]
        c = 3 * a
        B[:, 0, c] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c], B[:, 3, c + 1] = gy, gx
        B[:, 4, c + 1], B[:, 4, c + 2] = gz, gy
        B[:, 5, c], B[:, 5, c + 2] = gz, gx
    return B


def _hex8_dN(xi: float, eta: float, zeta: float) -> np.ndarray:
    s = _HEX_SIGNS
    dN = np.empty((8, 3), dtype=float)
    dN[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    dN[:, 1] = 0.125 * s[:, 1] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 2] * zeta)
    dN[:, 2] = 0.125 * s[:, 2] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta)
    return dN


@dataclass
class RefStiffness:
    ke: np.ndarray        # (m, 24, 24) reference element stiffnesses
    volume: np.ndarray    # (ne,) element volumes
    index: np.ndarray     # (ne,) mapping from element to ke matrix


def reference_stiffness(mesh: Mesh, nu: float) -> RefStiffness:
    """Compute reference unit-modulus (E=1.0) element stiffness for mesh elements."""
    coords = mesh.nodes[mesh.elems]
    ne = len(coords)
    D = isotropic_D(1.0, nu)
    if mesh.grid is not None:
        # Uniform grid: all elements share one single Ke
        single_coords = coords[0:1]
        Ke = np.zeros((1, 24, 24), dtype=float)
        vol = 0.0
        for gp in _HEX_GAUSS:
            dN = _hex8_dN(*gp)
            J = np.einsum("ka,nkb->nab", dN, single_coords)
            detJ = np.linalg.det(J)[0]
            dNdx = np.linalg.solve(J, np.broadcast_to(dN.T, (1, 3, 8)))
            B = _b_matrix(dNdx)
            Ke += np.einsum("nsi,st,ntj->nij", B, D, B) * detJ
            vol += detJ
        return RefStiffness(ke=Ke, volume=np.full(ne, vol, dtype=float), index=np.zeros(ne, dtype=np.int64))
    else:
        Ke = np.zeros((ne, 24, 24), dtype=float)
        vol = np.zeros(ne, dtype=float)
        for gp in _HEX_GAUSS:
            dN = _hex8_dN(*gp)
            J = np.einsum("ka,nkb->nab", dN, coords)
            detJ = np.linalg.det(J)
            dNdx = np.linalg.solve(J, np.broadcast_to(dN.T, (ne, 3, 8)))
            B = _b_matrix(dNdx)
            Ke += np.einsum("nsi,st,ntj->nij", B, D, B) * detJ[:, None, None]
            vol += detJ
        return RefStiffness(ke=Ke, volume=vol, index=np.arange(ne, dtype=np.int64))


# ==============================================================================
# 4. Filters & Projection
# ==============================================================================

def build_weights(centroids: np.ndarray, rmin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse spatial filter weights within radius rmin."""
    if rmin <= 0.0:
        n = len(centroids)
        return np.arange(n), np.arange(n), np.ones(n, dtype=float)
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=rmin, output_type="ndarray")
    if len(pairs) == 0:
        n = len(centroids)
        return np.arange(n), np.arange(n), np.ones(n, dtype=float)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(centroids[i] - centroids[j], axis=1)
    w = rmin - d
    n = len(centroids)
    rows = np.concatenate([i, j, np.arange(n)])
    cols = np.concatenate([j, i, np.arange(n)])
    vals = np.concatenate([w, w, np.full(n, rmin, dtype=float)])
    return rows, cols, vals


def density_filter_matrix(weights: Tuple[np.ndarray, np.ndarray, np.ndarray], volume: Optional[np.ndarray] = None) -> sparse.csr_matrix:
    rows, cols, vals = weights
    n = int(max(rows.max(), cols.max()) + 1)
    if volume is not None:
        vals = vals * volume[cols]
    H_raw = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    row_sums = np.maximum(np.asarray(H_raw.sum(axis=1)).ravel(), 1e-12)
    inv_diag = sparse.diags(1.0 / row_sums)
    return (inv_diag @ H_raw).tocsr()


def heaviside(xt: np.ndarray, beta: float, eta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Regularized Heaviside projection and its derivative."""
    if beta <= 0.0:
        return xt.copy(), np.ones_like(xt)
    a = np.tanh(beta * eta)
    b = np.tanh(beta * (1.0 - eta))
    th = np.tanh(beta * (xt - eta))
    xp = (a + th) / max(a + b, 1e-12)
    dproj = beta * (1.0 - th**2) / max(a + b, 1e-12)
    return np.clip(xp, 0.0, 1.0), dproj


# ==============================================================================
# 5. Rigid-Body MPC Kinematic Transformation
# ==============================================================================

@dataclass(frozen=True)
class RigidBody:
    """Multipoint constraint tying slave node translations to a rigid master node (translations + rotations)."""
    nodes: np.ndarray  # slave node IDs
    ref_node: int      # master translational reference node ID
    rot_node: int      # master rotational reference node ID


def rigid_body_transform(mesh: Mesh, rigid_bodies: List[RigidBody]) -> sparse.csr_matrix:
    """Build sparse kinematic constraint matrix T such that full_u = T @ reduced_q."""
    n = mesh.n_dofs
    slave = np.zeros(n, dtype=bool)
    for rb in rigid_bodies:
        nodes = np.asarray(rb.nodes, dtype=np.int64)
        slave[(3 * nodes[:, None] + np.arange(3)).ravel()] = True
    keep = np.flatnonzero(~slave)
    col = -np.ones(n, dtype=np.int64)
    col[keep] = np.arange(len(keep))

    rows = list(keep)
    cols = list(col[keep])
    vals = [1.0] * len(keep)

    for rb in rigid_bodies:
        nodes = np.asarray(rb.nodes, dtype=np.int64)
        diff = mesh.nodes[nodes] - mesh.nodes[rb.ref_node]  # (k, 3)
        ur = col[3 * rb.ref_node + np.arange(3)]
        th = col[3 * rb.rot_node + np.arange(3)]
        for k_idx, node in enumerate(nodes):
            dx, dy, dz = diff[k_idx]
            for a in range(3):
                rows.append(3 * node + a)
                cols.append(ur[a])
                vals.append(1.0)
            # u_s = u_r + theta x r:
            # ux += th_y*dz - th_z*dy
            # uy += th_z*dx - th_x*dz
            # uz += th_x*dy - th_y*dx
            cross_entries = [
                (0, th[1], dz), (0, th[2], -dy),
                (1, th[2], dx), (1, th[0], -dz),
                (2, th[0], dy), (2, th[1], -dx),
            ]
            for a_sub, c_sub, v_sub in cross_entries:
                if abs(v_sub) > 1e-15:
                    rows.append(3 * node + a_sub)
                    cols.append(c_sub)
                    vals.append(v_sub)

    n_reduced = len(keep)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n_reduced)).tocsr()


# ==============================================================================
# 6. Direct / PARDISO FEA Linear Static Solver Adapter
# ==============================================================================

class BuiltinAdapter:
    """High-performance FEA solver adapter utilizing Intel MKL PARDISO or SciPy SuperLU."""

    def __init__(self, method: str = "direct"):
        self.method = method
        self._cache: Dict[str, Any] = {}

    def solve(
        self,
        mesh: Mesh,
        youngs: np.ndarray,
        nu: float,
        fixed_dofs: np.ndarray,
        loads: np.ndarray,
        rigid_bodies: List[RigidBody] = (),
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        t0 = time.perf_counter()
        edofs = mesh.element_dofs()
        d = edofs.shape[1]

        # Pattern cache
        key = (id(mesh), nu)
        if self._cache.get("key") != key:
            rows = np.broadcast_to(edofs[:, :, None], (mesh.n_elems, d, d)).ravel()
            cols = np.broadcast_to(edofs[:, None, :], (mesh.n_elems, d, d)).ravel()
            ref = reference_stiffness(mesh, nu)
            self._cache = {"key": key, "rows": rows, "cols": cols, "ref": ref}
        else:
            rows, cols, ref = self._cache["rows"], self._cache["cols"], self._cache["ref"]

        m_ke, d_ke, _ = ref.ke.shape
        data = (ref.ke.reshape(m_ke, d_ke * d_ke)[ref.index] * youngs[:, None]).ravel()
        K = sparse.coo_matrix((data, (rows, cols)), shape=(mesh.n_dofs, mesh.n_dofs)).tocsr()

        fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        F = np.asarray(loads, dtype=float)

        T = None
        if rigid_bodies:
            T = rigid_body_transform(mesh, rigid_bodies)
            K = (T.T @ K @ T).tocsr()
            F = (T.T @ F.T).T
            n_red = T.shape[1]
            fixed = np.flatnonzero(np.asarray(T[fixed].sum(axis=0)).ravel() != 0)
        else:
            n_red = mesh.n_dofs

        free = np.setdiff1d(np.arange(n_red), fixed)
        Kff = K[free][:, free].tocsc()
        rhs = F[:, free].T  # (n_free, n_cases)

        # Solve with PyPardiso or Scipy SuperLU
        solved_with_pardiso = False
        try:
            import pypardiso
            sol = pypardiso.spsolve(Kff, rhs)
            solved_with_pardiso = True
        except Exception:
            sol = spla.spsolve(Kff, rhs)

        if sol.ndim == 1:
            sol = sol[:, None]
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("the linear solve returned non-finite displacements: the system is singular "
                               "(on a coarse grid the supports or the load interface may have no elements)")

        q = np.zeros((F.shape[0], n_red), dtype=float)
        q[:, free] = sol.T
        u = q if T is None else (T @ q.T).T
        wall_s = time.perf_counter() - t0
        return u, wall_s, {"solver": "pardiso" if solved_with_pardiso else "superlu"}


# ==============================================================================
# 7. Problem, Evaluation & TopOpt Evaluator
# ==============================================================================

@dataclass
class LoadCase:
    name: str
    force: np.ndarray
    weight: float = 1.0


@dataclass
class TopOptProblem:
    mesh: Mesh
    fixed_dofs: np.ndarray
    load_cases: List[LoadCase]
    E0: float = 1.0
    emin_ratio: float = 1e-9
    nu: float = 0.3
    volfrac: float = 0.4
    penalty: float = 3.0
    interpolation: str = "simp"
    filter_type: str = "density"
    rmin: float = 0.0
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    passive_solid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    passive_void: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    max_iters: int = 100
    tol: float = 0.01
    optimizer: str = "mma"
    mma_backend: str = "mmapy"
    variable_groups: Optional[np.ndarray] = None
    rigid_bodies: List[RigidBody] = field(default_factory=list)
    am_build_direction: Optional[str] = None
    local_volume: Optional[Any] = None
    robust: Optional[Any] = None
    displacement_limits: List[Any] = field(default_factory=list)

    @property
    def Emin(self) -> float:
        return self.E0 * self.emin_ratio


@dataclass
class Evaluation:
    compliance: float
    case_compliance: np.ndarray
    xphys: np.ndarray
    u: np.ndarray
    volume_fraction: float
    dc: np.ndarray
    dv: np.ndarray
    solve_wall_s: float = 0.0


class TopOptEvaluator:
    """Evaluates FE displacements, compliance, and sensitivities for structured Hex8 meshes."""

    def __init__(self, problem: TopOptProblem, adapter: BuiltinAdapter):
        self.problem = problem
        self.adapter = adapter
        mesh = problem.mesh
        self.ref = reference_stiffness(mesh, problem.nu)
        self.edofs = mesh.element_dofs()
        self.loads = np.stack([lc.force for lc in problem.load_cases])
        self.cw = np.array([lc.weight for lc in problem.load_cases], dtype=float)
        self.vol = self.ref.volume / self.ref.volume.sum()
        self.active = np.ones(mesh.n_elems, dtype=bool)
        self.active[problem.passive_solid] = False
        self.active[problem.passive_void] = False
        self.beta = problem.projection_beta

        weights = build_weights(mesh.centroids(), problem.rmin)
        self.H = density_filter_matrix(weights, self.ref.volume) if problem.filter_type == "density" else None

    def physical(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.problem
        xt = self.H @ x if self.H is not None else x
        xp, dproj = heaviside(xt, self.beta, p.projection_eta)
        xp = xp.copy()
        xp[p.passive_solid] = 1.0
        xp[p.passive_void] = 0.0
        return xp, dproj * self.active

    def evaluate(self, x: np.ndarray, gradients: bool = True) -> Evaluation:
        p = self.problem
        xp, dproj = self.physical(x)
        penalty = p.penalty
        E = p.Emin + (p.E0 - p.Emin) * (xp ** penalty)

        u, solve_wall_s, _ = self.adapter.solve(
            p.mesh,
            youngs=E,
            nu=p.nu,
            fixed_dofs=p.fixed_dofs,
            loads=self.loads,
            rigid_bodies=p.rigid_bodies,
        )

        case_c = np.einsum("cd,cd->c", self.loads, u)
        c = float(self.cw @ case_c)
        vf = float(self.vol @ xp)

        if not gradients:
            return Evaluation(
                compliance=c,
                case_compliance=case_c,
                xphys=xp,
                u=u,
                volume_fraction=vf,
                dc=np.zeros(len(x)),
                dv=np.zeros(len(x)),
            )

        # Element compliance & sensitivity
        dE = penalty * (p.E0 - p.Emin) * (xp ** (penalty - 1.0))
        ue = u[:, self.edofs]
        ke_batch = self.ref.ke[self.ref.index]
        k_ue = np.einsum("nab,cnb->cna", ke_batch, ue)
        ce_all = np.sum(ue * k_ue, axis=2)  # (n_cases, n_elem); c = f.u = u^T K u, so dc/dE_e = -u_e^T Ke0 u_e
        dc_xp = -(self.cw @ ce_all) * dE
        dc = (self.H.T @ (dc_xp * dproj)) if self.H is not None else (dc_xp * dproj)
        dv = (self.H.T @ (self.vol * dproj)) if self.H is not None else (self.vol * dproj)

        return Evaluation(
            compliance=c,
            case_compliance=case_c,
            xphys=xp,
            u=u,
            volume_fraction=vf,
            dc=dc,
            dv=dv,
            solve_wall_s=solve_wall_s,
        )


# ==============================================================================
# 8. Constraints & Optimization Loop
# ==============================================================================

@dataclass
class TopOptResult:
    x: np.ndarray
    xphys: np.ndarray
    compliance: float
    volume_fraction: float
    iterations: int
    converged: bool
    history: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)  # method-specific results (SEMDOT: the optimizer, for its boundary)


def _constraints(problem: TopOptProblem, e: Evaluation) -> Tuple[np.ndarray, np.ndarray]:
    """Volume fraction constraint g <= 0 and derivative dg."""
    g = np.array([e.volume_fraction / problem.volfrac - 1.0])
    dg = np.vstack([e.dv / problem.volfrac])
    return g, dg


def measured_phase_times(result: "TopOptResult") -> Tuple[float, float, float]:
    """(FE solve, sensitivity and the rest of the evaluation, optimizer update) wall time in seconds, summed over the iterations."""
    h = result.history
    fem = float(sum(r.get("solve_wall_s", 0.0) for r in h))
    return fem, float(sum(r.get("t_eval_s", 0.0) for r in h)) - fem, float(sum(r.get("t_opt_s", 0.0) for r in h))


def _run_semdot(problem: TopOptProblem, adapter: BuiltinAdapter, x0: Optional[np.ndarray], callback: Optional[Callable], n_sub: int = 4) -> TopOptResult:
    """SEMDOT loop (Fu et al. 2020): the design variable x is updated by OC, filtered, interpolated to the nodes and cut on a
    sub-element grid; the analysis uses the resulting element volume fraction Psi with a linear modulus (no SIMP penalty) and
    the weighted sensitivity -((1 - Psi) xmin + Psi) u_e^T K0 u_e, averaged over the iterations."""
    from wht_topopt.algorithms.oc import OCOptimizer
    from wht_topopt.algorithms.semdot import SEMDOTOptimizer

    xmin = 1e-6
    design = TopOptEvaluator(problem, adapter)  # for the filter and the volumes only
    analysis = TopOptEvaluator(
        replace(problem, penalty=1.0, emin_ratio=xmin, filter_type="none", rmin=0.0, projection_beta=0.0), adapter
    )
    H = design.H
    n_elems = problem.mesh.n_elems
    x = np.full(n_elems, problem.volfrac, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n_elems,):
        raise ValueError(f"x0 must have shape ({n_elems},), got {x.shape}")
    oc = OCOptimizer(volfrac=problem.volfrac, move=0.20, rho_min=1e-3)
    semdot = SEMDOTOptimizer(mesh=problem.mesh, volfrac=problem.volfrac, n_sub=n_sub, beta_init=0.5, beta_max=8.0, rho_min=xmin, move=0.20)
    project = lambda xt: semdot.project_filtered(xt, target_volfrac=problem.volfrac, passive_solid=problem.passive_solid, passive_void=problem.passive_void)
    psi = project(H @ x if H is not None else x)
    de = analysis.problem.E0 - analysis.problem.Emin
    dv = H.T @ design.vol if H is not None else design.vol
    dc_prev = None
    history: List[Dict[str, Any]] = []
    converged = False
    it = 0
    for it in range(1, problem.max_iters + 1):
        t0 = time.perf_counter()
        e = analysis.evaluate(psi, gradients=True)
        unit_energy = -e.dc / de  # u_e^T K0_e u_e (zero on passive elements)
        dc = -((1.0 - e.xphys) * xmin + e.xphys) * de * unit_energy
        if dc_prev is not None:
            dc = 0.35 * dc + 0.65 * dc_prev
        dc_prev = dc
        dc_x = H.T @ dc if H is not None else dc
        history.append({"iter": it, "compliance": e.compliance, "volume_fraction": e.volume_fraction,
                        "t_eval_s": time.perf_counter() - t0, "solve_wall_s": e.solve_wall_s, "beta": semdot.beta, "psi": semdot.psi})
        if callback is not None:
            callback(it, e, None)
        t1 = time.perf_counter()
        x_new = oc.update(x, dc_x, None, dv, elem_volumes=design.ref.volume, passive_solid=problem.passive_solid, passive_void=problem.passive_void)
        psi = project(H @ x_new if H is not None else x_new)
        history[-1]["t_opt_s"] = time.perf_counter() - t1
        change = float(np.mean(np.abs(x_new - x)))
        x = x_new
        if change < problem.tol and it >= 25:
            converged = True
            break
    final = analysis.evaluate(psi, gradients=False)
    return TopOptResult(x=x, xphys=final.xphys, compliance=final.compliance, volume_fraction=final.volume_fraction, iterations=it,
                        converged=converged, history=history, extras={"semdot": semdot})


def _beso_load_path_guard(problem: TopOptProblem) -> Dict[str, Any]:
    """Keyword arguments arming BESO's connectivity check: a 0/1 design can cut the load path, and then its compliance explodes."""
    mesh = problem.mesh
    loaded = np.any(np.stack([lc.force for lc in problem.load_cases]) != 0.0, axis=0)
    load_nodes = np.unique(np.flatnonzero(loaded) // 3)
    for rb in problem.rigid_bodies:  # a load on the master nodes enters the elements through the slave nodes
        if np.isin([rb.ref_node, rb.rot_node], load_nodes).any():
            load_nodes = np.union1d(np.setdiff1d(load_nodes, [rb.ref_node, rb.rot_node]), rb.nodes)
    # at least two held dofs: a symmetry plane or roller holds one and is not an anchor by itself
    held = np.bincount(np.asarray(problem.fixed_dofs, dtype=np.int64) // 3, minlength=mesh.nodes.shape[0])
    return {"mesh": SimpleNamespace(elements=mesh.elems, n_nodes=mesh.nodes.shape[0]), "load_nodes": load_nodes,
            "fixed_nodes": np.flatnonzero(held >= 2)}


def run_topology_optimization(
    problem: TopOptProblem,
    adapter: BuiltinAdapter,
    *,
    x0: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, Evaluation, Any], None]] = None,
) -> TopOptResult:
    """Standalone MMA / GCMMA / OC / BESO / SEMDOT optimization driver.

    ``problem.optimizer`` selects the update; an unknown value is refused, never replaced by another method.
    With ``variable_groups`` the design vector ``x`` holds one value per group (MMA and GCMMA only); ``x0`` may be given per
    group or per element (then averaged over each group).
    """
    from wht_topopt.algorithms.adapter import AlgorithmRegistry, OptimizationStepInput
    from wht_topopt.algorithms.mma_wrapper import MMAOptimizer

    name = problem.optimizer.strip().lower()
    if name not in ("mma", "gcmma", "oc", "beso", "semdot"):
        raise ValueError(f"unknown optimizer '{problem.optimizer}': use 'mma', 'gcmma', 'oc', 'beso' or 'semdot'")

    groups = problem.variable_groups
    if groups is not None and name not in ("mma", "gcmma"):
        raise ValueError(f"variable_groups needs optimizer 'mma' or 'gcmma', got '{name}'")
    if name == "semdot":
        return _run_semdot(problem, adapter, x0, callback)

    evaluator = TopOptEvaluator(problem, adapter)
    n_elems = problem.mesh.n_elems
    n_groups = int(groups.max()) + 1 if groups is not None else None
    n_var = n_elems if groups is None else n_groups

    if x0 is None:
        # BESO evolves from the full design down to the target; the others start from the uniform feasible design
        x = np.full(n_var, 1.0 if name == "beso" else problem.volfrac, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if groups is not None and x.shape == (n_elems,) and n_groups != n_elems:
            x = np.bincount(groups, weights=x, minlength=n_groups) / np.bincount(groups, minlength=n_groups)
        if x.shape != (n_var,):
            raise ValueError(f"x0 must have shape ({n_var},), got {x.shape}")

    opt_kw = {"rho_min": 1e-3, "move": 0.20}
    if name == "mma":
        opt = MMAOptimizer(rho_max=1.0, **opt_kw)
    elif name == "gcmma":
        opt = AlgorithmRegistry.get("gcmma", rho_max=1.0, **opt_kw)
    elif name == "beso":
        opt = AlgorithmRegistry.get("beso", target_volfrac=problem.volfrac, **_beso_load_path_guard(problem), **opt_kw)
    else:  # oc
        opt = AlgorithmRegistry.get(name, volfrac=problem.volfrac, **opt_kw)
    solid_mask = np.zeros(n_elems, dtype=bool)
    solid_mask[problem.passive_solid] = True
    void_mask = np.zeros(n_elems, dtype=bool)
    void_mask[problem.passive_void] = True
    if name == "beso":  # a 0/1 update must not remove the elements that carry a load or a support
        guard = _beso_load_path_guard(problem)
        touches = np.zeros(problem.mesh.nodes.shape[0], dtype=bool)
        touches[np.concatenate([guard["load_nodes"], guard["fixed_nodes"]])] = True
        solid_mask |= touches[problem.mesh.elems].any(axis=1) & ~void_mask

    history = []
    converged = False

    c_ref: Optional[float] = None
    last_e: Optional[Evaluation] = None

    for it in range(1, problem.max_iters + 1):
        x_full = x[groups] if groups is not None else x  # map group variables to the element array

        t_eval0 = time.perf_counter()
        e = evaluator.evaluate(x_full, gradients=True)
        t_eval = time.perf_counter() - t_eval0
        last_e = e

        if c_ref is None:
            c_ref = max(abs(e.compliance), 1e-12)

        g, dg = _constraints(problem, e)
        f0 = e.compliance / c_ref
        df0 = e.dc / c_ref

        if groups is not None:  # gradients with respect to the group variables
            df_curr = np.bincount(groups, weights=df0, minlength=n_groups)
            dg_curr = np.vstack([np.bincount(groups, weights=dg[0], minlength=n_groups)])
        else:
            df_curr, dg_curr = df0, dg
        x_cur = x

        history.append({
            "iter": it,
            "compliance": e.compliance,
            "volume_fraction": e.volume_fraction,
            "t_eval_s": t_eval,
            "solve_wall_s": e.solve_wall_s,
        })

        if callback is not None:
            callback(it, e, None)

        t_opt0 = time.perf_counter()
        if name == "mma":
            x_next = opt.update(rho=x_cur, df_drho=df_curr, g_vals=g, dg_drho=dg_curr)
        else:
            def _true_values(xt: np.ndarray, _ref: float = c_ref) -> Tuple[float, np.ndarray]:
                et = evaluator.evaluate(xt[groups] if groups is not None else xt, gradients=False)
                return et.compliance / _ref, _constraints(problem, et)[0]

            x_next = opt.step(OptimizationStepInput(
                iteration=it,
                rho=x_cur,
                df_drho=df_curr,
                g_vals=g,
                dg_drho=dg_curr,
                elem_volumes=None if groups is not None else evaluator.ref.volume,
                passive_solid=None if groups is not None else solid_mask,
                passive_void=None if groups is not None else void_mask,
                f0_val=f0,
                eval_fn=_true_values,
            )).rho_next

        history[-1]["t_opt_s"] = time.perf_counter() - t_opt0

        chg = float(np.max(np.abs(x_next - x_cur)))
        x = x_next
        if chg < problem.tol and it >= 15:
            converged = True
            break

    x_final_full = x[groups] if groups is not None else x
    e_final = evaluator.evaluate(x_final_full, gradients=False)
    return TopOptResult(
        x=x,
        xphys=e_final.xphys,
        compliance=e_final.compliance,
        volume_fraction=e_final.volume_fraction,
        iterations=it,
        converged=converged,
        history=history,
    )


# ==============================================================================
# 9. StlCase & Complete STL Problem Construction
# ==============================================================================

@dataclass
class StlCase:
    domain: str
    forces: List[Tuple[str, Tuple[float, float, float]]] = field(default_factory=list)
    fixed: List[str] = field(default_factory=list)
    xfixed: List[str] = field(default_factory=list)
    yfixed: List[str] = field(default_factory=list)
    zfixed: List[str] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)
    keep_bc: bool = True
    symmetric_in: Optional[str] = None
    pin_region: Optional[str] = None
    pin_point: Optional[Tuple[float, float, float]] = None
    pin_loads: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=list)


def build_stl_problem(
    case: StlCase,
    mesh_control: int,
    volfrac: float,
    base: str | Path = ".",
    **kw: Any,
) -> Tuple[TopOptProblem, np.ndarray]:
    """Construct complete TopOptProblem with inside-domain element mask directly from closed STL files."""
    base = Path(base)
    dom = read_stl(base / case.domain)
    lo = dom.reshape(-1, 3).min(axis=0)
    hi = dom.reshape(-1, 3).max(axis=0)
    h = float((hi - lo).max()) / mesh_control
    n = np.maximum(np.ceil((hi - lo) / h - 1e-9).astype(int), 1)

    mesh = structured_hex(int(n[0]), int(n[1]), int(n[2]), size=(h, h, h))
    mesh.nodes += lo - 0.5 * (n * h - (hi - lo))  # Centering
    g = [mesh.nodes[0, a] + h * (np.arange(n[a]) + 0.5) for a in range(3)]
    inside = inside_grid(dom, *g).ravel()
    gn = [mesh.nodes[0, a] + h * np.arange(n[a] + 1) for a in range(3)]

    def nodes_in(stl_name: str) -> np.ndarray:
        tris = read_stl(base / stl_name)
        mask = inside_grid(tris, *gn).ravel()
        mask = np.concatenate([mask, np.zeros(mesh.n_nodes - len(mask), dtype=bool)])
        if not mask.any():
            from scipy.spatial import cKDTree
            pts = tris.reshape(-1, 3)
            d, idx = cKDTree(mesh.nodes).query(pts)
            mask = np.zeros(mesh.n_nodes, dtype=bool)
            mask[np.unique(idx[d <= h])] = True
        return mask

    in_design_node = np.zeros(mesh.n_nodes, dtype=bool)
    in_design_node[mesh.elems[inside].ravel()] = True

    fixed, bc_nodes = [], np.zeros(mesh.n_nodes, dtype=bool)
    for regions, comps in ((case.fixed, (0, 1, 2)), (case.xfixed, (0,)), (case.yfixed, (1,)), (case.zfixed, (2,))):
        for stl in regions:
            nodes = np.flatnonzero(nodes_in(stl) & in_design_node)
            if nodes.size == 0:
                raise ValueError(f"Support region {stl} contains no nodes of the design domain")
            fixed.append(mesh.node_dofs(nodes, comps))
            if comps == (0, 1, 2):
                bc_nodes[nodes] = True

    loads: List[LoadCase] = []
    for k, (stl, vec) in enumerate(case.forces):
        nodes = np.flatnonzero(nodes_in(stl) & in_design_node)
        if nodes.size == 0:
            raise ValueError(f"Force region {stl} contains no nodes of the design domain")
        f = np.zeros(mesh.n_dofs, dtype=float)
        for a in range(3):
            f[3 * nodes + a] = vec[a] / len(nodes)
        loads.append(LoadCase(f"F{k + 1}", f))
        bc_nodes[nodes] = True

    bodies: List[RigidBody] = []
    if case.pin_region is not None:
        slaves = np.flatnonzero(nodes_in(case.pin_region) & in_design_node)
        if slaves.size == 0:
            raise ValueError(f"Pin region {case.pin_region} contains no nodes of the design domain")
        point = np.asarray(case.pin_point if case.pin_point is not None else mesh.nodes[slaves].mean(axis=0), dtype=float)

        nn = mesh.n_nodes
        # Extra reference and rotation nodes
        mesh = Mesh(np.vstack([mesh.nodes, point, point + [0.0, 0.0, h]]), mesh.elems, "hex8", grid=mesh.grid)
        ref, rot = nn, nn + 1
        bodies.append(RigidBody(slaves, ref, rot))
        bc_nodes = np.concatenate([bc_nodes, [False, False]])
        bc_nodes[slaves] = True

        for k_idx, lc in enumerate(loads):
            loads[k_idx] = LoadCase(lc.name, np.concatenate([lc.force, np.zeros(6)]), lc.weight)
        for k_idx, (force, moment) in enumerate(case.pin_loads):
            f = np.zeros(mesh.n_dofs, dtype=float)
            f[3 * ref : 3 * ref + 3] = force
            f[3 * rot : 3 * rot + 3] = moment
            loads.append(LoadCase(f"pin{k_idx + 1}", f))

    if not loads:
        raise ValueError("No load case: provide force regions or pin loads")

    solid = np.zeros(mesh.n_elems, dtype=bool)
    if case.keep_bc:
        solid |= bc_nodes[mesh.elems].any(axis=1) & inside
    for stl in case.keep:
        solid |= nodes_in(stl)[mesh.elems].all(axis=1) & inside

    frac = (volfrac * inside.sum()) / mesh.n_elems

    groups = None
    if case.symmetric_in is not None:
        nx_, ny_, nz_ = mesh.grid
        k, j, i = np.meshgrid(np.arange(nz_), np.arange(ny_), np.arange(nx_), indexing="ij")
        a = "xyz".index(case.symmetric_in)
        ii, jj, kk = [i, j, k][0].copy(), j.copy(), k.copy()
        if a == 0:
            ii = nx_ - 1 - i
        elif a == 1:
            jj = ny_ - 1 - j
        else:
            kk = nz_ - 1 - k
        e_arr = (i + nx_ * (j + ny_ * k)).ravel()
        mirror = (ii + nx_ * (jj + ny_ * kk)).ravel()
        _, groups = np.unique(np.minimum(e_arr, mirror), return_inverse=True)
        groups = groups.ravel()
        kw = dict(kw, variable_groups=groups)

    p = TopOptProblem(
        mesh=mesh,
        fixed_dofs=np.unique(np.concatenate(fixed)),
        load_cases=loads,
        volfrac=frac,
        passive_void=np.flatnonzero(~inside),
        passive_solid=np.flatnonzero(solid),
        rigid_bodies=bodies,
        **kw,
    )
    return p, inside


# ==============================================================================
# 10. Progress GIF Utility
# ==============================================================================

class ProgressGif:
    """Lightweight 2D projection convergence GIF generator."""

    def __init__(
        self,
        grid: Tuple[int, int, int],
        path: str | Path,
        views: Tuple[str, ...] = ("z", "y"),
        every: int = 1,
        max_frames: int = 120,
        frame_ms: int = 100,
    ):
        self.grid = grid
        self.path = Path(path)
        self.views = views
        self.every = max(1, int(every))
        self.max_frames = max_frames
        self.frame_ms = frame_ms
        self.frames: List[np.ndarray] = []
        self.n_calls = 0

    def append(self, x: np.ndarray) -> None:
        self.n_calls += 1
        if (self.n_calls - 1) % self.every != 0:
            return
        import imageio.v2 as imageio
        nx, ny, nz = self.grid
        x3 = x.reshape(nz, ny, nx)
        imgs = []
        for v in self.views:
            axis = {"x": 2, "y": 1, "z": 0}[v]
            proj = x3.mean(axis=axis)
            img = np.clip(255.0 * (1.0 - proj), 0, 255).astype(np.uint8)
            imgs.append(img)
        row = np.hstack(imgs)
        self.frames.append(row)

    def save(self) -> None:
        if not self.frames:
            return
        import imageio.v2 as imageio
        self.path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(self.path), self.frames, duration=self.frame_ms / 1000.0, loop=0)
