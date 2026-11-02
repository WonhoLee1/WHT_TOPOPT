"""FE model shared by the examples (Q4 plane stress or Hex8): structured grids, passive regions, rigid-body interfaces, stiffness, mass, modes, transient.

Every number of the pillars comes from this model: there are no amplification factors, modal shortcuts or fallback values. A modal
solve that fails raises, it does not return a default frequency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wht_topopt.core.fe_builders import hex8_ke0_me0, isotropic_elasticity_2d, isotropic_elasticity_3d, q4_ke0_me0
from wht_topopt.core.stl_domain import rigid_body_transform
from wht_topopt.core.time_integration import (
    du_olhoff_mass_density,
    generalized_alpha_lhs,
    generalized_alpha_parameters,
    generalized_alpha_rhs,
    newmark_update,
)


class LinearSystem:
    """A factorised sparse matrix: PARDISO when installed, SuperLU otherwise."""

    def __init__(self, A: sp.spmatrix) -> None:
        try:
            import pypardiso

            self._solver = pypardiso.PyPardisoSolver(mtype=11)
            self._A = A.tocsr()
            self._solver.factorize(self._A)
            self._lu = None
        except ImportError:
            self._solver = None
            self._lu = spla.splu(A.tocsc())

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        x = self._solver.solve(self._A, np.asfortranarray(b)) if self._solver is not None else self._lu.solve(b)
        if not np.all(np.isfinite(x)):
            raise RuntimeError("the linear solve returned non-finite values: the system is singular")
        return x


@dataclass
class TransientResult:
    """Transient response on the free dofs. ``u`` holds the states at ``stored`` steps; energy covers every step."""

    times: np.ndarray
    stored: np.ndarray
    u: np.ndarray  # (n_stored, n_free)
    energy: np.ndarray  # 0.5 u^T K u per step
    omega_dt: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class Structure:
    """Uniform Q4 or Hex8 grid. Densities live on all grid elements; passive-void elements and their nodes leave the system.

    ``yield_stress`` None: non-dimensional problem, no safety factor. ``density`` None: no mass, modes or transient. Both are required arguments."""

    def __init__(
        self,
        mesh,
        fixed_dofs: np.ndarray,
        e0: float,
        nu: float,
        density: Optional[float],
        yield_stress: Optional[float],
        passive_solid: Sequence[int] = (),
        passive_void: Sequence[int] = (),
        rigid_bodies: Sequence = (),
        lumped_mass: Optional[np.ndarray] = None,
        emin_ratio: float = 1e-6,
        penal: float = 3.0,
        rayleigh: Tuple[float, float] = (5.0, 1e-5),
    ) -> None:
        self.mesh = mesh
        self.nodes = np.asarray(getattr(mesh, "node_coords", getattr(mesh, "nodes")), dtype=np.float64)
        self.conn = np.asarray(mesh.elements if hasattr(mesh, "elements") else mesh.elems, dtype=np.int64)
        self.dim = self.nodes.shape[1]
        if (self.dim, self.conn.shape[1]) not in ((2, 4), (3, 8)):
            raise ValueError("Structure needs Q4 or Hex8 elements")
        if self.dim == 2 and rigid_bodies:
            raise ValueError("rigid bodies are implemented for Hex8 only")
        self.n_elem = self.conn.shape[0]
        self.ndof = self.dim * self.nodes.shape[0]
        self.grid = self._grid_shape(mesh)
        if not e0 > 0.0 or not -1.0 < nu < 0.5:
            raise ValueError(f"invalid material E0={e0}, nu={nu}")
        self.e0, self.nu = float(e0), float(nu)
        self.density = None if density is None else float(density)
        self.yield_stress = None if yield_stress is None else float(yield_stress)
        self.emin = self.e0 * float(emin_ratio)
        self.penal = float(penal)
        self.rayleigh = rayleigh
        self.rigid_bodies = list(rigid_bodies)
        self.solid_mask = np.zeros(self.n_elem, dtype=bool)
        self.solid_mask[np.asarray(passive_solid, dtype=np.int64)] = True
        self.void_mask = np.zeros(self.n_elem, dtype=bool)
        self.void_mask[np.asarray(passive_void, dtype=np.int64)] = True
        if np.any(self.solid_mask & self.void_mask):
            raise ValueError("an element cannot be both passive solid and passive void")
        self.design_mask = ~(self.solid_mask | self.void_mask)

        first = self.nodes[self.conn[0]]
        self.size = first.max(axis=0) - first.min(axis=0)
        self.volumes = np.full(self.n_elem, float(np.prod(self.size)))
        if self.dim == 2:
            self.Ke0, self.Me0, B0 = q4_ke0_me0(first, self.nu)
            self.S0 = isotropic_elasticity_2d(self.nu) @ B0  # unit-modulus centroid stress operator
        else:
            self.Ke0, self.Me0, B0 = hex8_ke0_me0(first, self.nu)
            self.S0 = isotropic_elasticity_3d(self.nu) @ B0
        d = self.conn.shape[1] * self.dim
        self.edofs = (self.dim * self.conn[:, :, None] + np.arange(self.dim)).reshape(self.n_elem, d)
        self.centroids = self.nodes[self.conn].mean(axis=1)

        keep = ~self.void_mask
        self._keep = keep
        ek = self.edofs[keep]
        self._rows = np.repeat(ek, d, axis=1).ravel()
        self._cols = np.tile(ek, (1, d)).ravel()

        T = rigid_body_transform(SimpleNamespace(n_dofs=self.ndof, nodes=self.nodes), self.rigid_bodies) if self.rigid_bodies else sp.identity(self.ndof, format="csr")
        T = T.tocsr()
        fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self.fixed_dofs = fixed
        fixed_cols = np.flatnonzero(np.asarray(np.abs(T[fixed]).sum(axis=0)).ravel() != 0.0)
        used_cols = np.flatnonzero(np.asarray(np.abs(T[np.unique(ek)]).sum(axis=0)).ravel() != 0.0)
        self.free = np.setdiff1d(used_cols, fixed_cols)
        self.Tf = T[:, self.free].tocsr()
        self.Tft = self.Tf.T.tocsr()
        self.n_free = self.free.size
        self.lumped = None if lumped_mass is None else np.asarray(lumped_mass, dtype=np.float64)

    @staticmethod
    def _grid_shape(mesh) -> Tuple[int, int, int]:
        if hasattr(mesh, "nelz"):
            return int(mesh.nelx), int(mesh.nely), int(mesh.nelz)
        if hasattr(mesh, "nelx"):
            return int(mesh.nelx), int(mesh.nely)
        return tuple(int(v) for v in mesh.grid)

    @property
    def grid_mesh(self):
        """The grid without rigid-body reference nodes, for code that reads the lattice from the coordinates."""
        if self.dim == 2:
            return self.mesh
        return SimpleNamespace(elements=self.conn, node_coords=self.nodes[: int(self.conn.max()) + 1], n_nodes=int(self.conn.max()) + 1)

    @property
    def total_active_volume(self) -> float:
        return float(self.volumes[self._keep].sum())

    def mass_g(self, rho: np.ndarray) -> float:
        """Mass in grams (density in tonne/mm^3)."""
        if self.density is None:
            raise ValueError("the problem has no density")
        return float(np.sum(rho[self._keep] * self.volumes[self._keep]) * self.density * 1e6)

    def volume_fraction(self, rho: np.ndarray) -> float:
        return float(np.sum(rho[self._keep] * self.volumes[self._keep])) / self.total_active_volume

    def grayness(self, rho: np.ndarray) -> float:
        x = np.clip(rho[self._keep], 0.0, 1.0)
        return float(np.mean(x * (1.0 - x)))

    def young(self, rho: np.ndarray) -> np.ndarray:
        return self.emin + (self.e0 - self.emin) * np.maximum(rho, 0.0) ** self.penal

    def dyoung(self, rho: np.ndarray) -> np.ndarray:
        return self.penal * (self.e0 - self.emin) * np.maximum(rho, 1e-12) ** (self.penal - 1.0)

    def _assemble(self, values: np.ndarray, Ae0: np.ndarray) -> sp.csc_matrix:
        data = np.outer(values[self._keep], Ae0.ravel()).ravel()
        A = sp.coo_matrix((data, (self._rows, self._cols)), shape=(self.ndof, self.ndof)).tocsr()
        return A

    def stiffness(self, rho: np.ndarray) -> sp.csc_matrix:
        return (self.Tft @ self._assemble(self.young(rho), self.Ke0) @ self.Tf).tocsc()

    def mass(self, rho: np.ndarray) -> sp.csc_matrix:
        if self.density is None:
            raise ValueError("the problem has no density")
        M = self._assemble(du_olhoff_mass_density(rho) * self.density, self.Me0)
        if self.lumped is not None:
            M = M + sp.diags(self.lumped, format="csr")
        return (self.Tft @ M @ self.Tf).tocsc()

    def reduce(self, f: np.ndarray) -> np.ndarray:
        return self.Tft @ f

    def expand(self, q: np.ndarray) -> np.ndarray:
        return self.Tf @ q

    def solve_static(self, K: sp.spmatrix, loads: np.ndarray) -> np.ndarray:
        """Full-space displacements (ncase, ndof) of full-space loads (ncase, ndof)."""
        loads = np.atleast_2d(loads)
        q = LinearSystem(K).solve((self.Tft @ loads.T))
        return np.atleast_2d((self.Tf @ q.reshape(self.n_free, -1)).T)

    def unit_energy(self, u_full: np.ndarray) -> np.ndarray:
        """u_e^T Ke0 u_e per element."""
        ue = u_full[self.edofs]
        return np.einsum("ni,ij,nj->n", ue, self.Ke0, ue, optimize=True)

    def von_mises(self, u_full: np.ndarray) -> np.ndarray:
        """Element-centre von Mises stress with the solid modulus (no density factor)."""
        s = self.e0 * (u_full[self.edofs] @ self.S0.T)
        if self.dim == 2:
            sx, sy, t = s.T
            return np.sqrt(np.maximum(sx**2 - sx * sy + sy**2 + 3.0 * t**2, 0.0))
        sx, sy, sz, a, b, c = s.T
        return np.sqrt(np.maximum(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (a**2 + b**2 + c**2), 0.0))

    def natural_frequencies(self, K: sp.spmatrix, M: sp.spmatrix, k: int = 3) -> List[float]:
        """Lowest natural frequencies in Hz by shift-invert Lanczos about zero. Failure raises."""
        system = LinearSystem(K)
        op = spla.LinearOperator(K.shape, matvec=lambda v: system.solve(M @ np.asarray(v).ravel()), dtype=np.float64)
        vals = spla.eigs(op, k=k, which="LM", tol=1e-6, return_eigenvectors=False).real  # 1 / omega^2 of K^-1 M (not symmetric as an operator)
        vals = vals[vals > 0.0]
        return sorted(float(np.sqrt(1.0 / v) / (2.0 * np.pi)) for v in vals)

    def transient(self, K: sp.spmatrix, M: sp.spmatrix, force_at: Callable[[float], np.ndarray], times: np.ndarray, rho_inf: float = 0.85, store_every: int = 1) -> TransientResult:
        """Generalized-alpha march from rest; ``force_at(t)`` is the full-space load, zero at t = 0."""
        alpha_r, beta_r = self.rayleigh
        C = (alpha_r * M + beta_r * K).tocsc()
        am, af, beta, gamma = generalized_alpha_parameters(rho_inf)
        dt = float(times[1] - times[0])
        if not np.allclose(np.diff(times), dt):
            raise ValueError("the time steps must be uniform")
        f_prev = self.Tft @ force_at(float(times[0]))
        if np.any(f_prev != 0.0):
            raise ValueError("the transient starts from rest: the load must vanish at t = 0")
        system = LinearSystem(generalized_alpha_lhs(M, C, K, dt, am, af, beta, gamma))
        u = np.zeros(self.n_free)
        v = np.zeros(self.n_free)
        a = np.zeros(self.n_free)
        stored = np.arange(0, len(times), max(1, int(store_every)))
        if stored[-1] != len(times) - 1:
            stored = np.append(stored, len(times) - 1)
        keep = np.zeros((stored.size, self.n_free), dtype=np.float64)
        energy = np.zeros(len(times))
        slot = 1
        for s in range(1, len(times)):
            f_now = self.Tft @ force_at(float(times[s]))
            f_mid = (1.0 - af) * f_now + af * f_prev
            un = system.solve(generalized_alpha_rhs(M, C, K, u, v, a, f_mid, dt, am, af, beta, gamma))
            v, a = newmark_update(u, v, a, un, dt, beta, gamma)
            u, f_prev = un, f_now
            energy[s] = 0.5 * float(u @ (K @ u))
            if slot < stored.size and stored[slot] == s:
                keep[slot] = u
                slot += 1
        return TransientResult(times=np.asarray(times), stored=stored, u=keep, energy=energy)


def steps_for_first_mode(f1_hz: float, duration: float, limit: float = 0.5, max_steps: Optional[int] = None) -> Tuple[int, List[str]]:
    """Time steps over ``duration`` for omega_1 dt <= limit; a cap below that is reported, not hidden."""
    needed = int(np.ceil(duration * 2.0 * np.pi * f1_hz / limit)) + 1
    if max_steps is not None and needed > max_steps:
        return max_steps, [f"time step above the limit: {max_steps} steps used, omega_1 dt <= {limit:g} needs {needed} (f1 = {f1_hz:.1f} Hz)"]
    return needed, []
