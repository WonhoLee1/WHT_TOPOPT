"""Base interfaces and data structures for FEA solvers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


@dataclass
class LoadCase:
    """Represents a structural load case with force vector and optimization weight.
    
    Attributes:
        name: Identifier for the load case.
        force_vector: Global nodal force vector (N_dofs,).
        weight: Objective weighting factor for multi-load compliance.
    """
    name: str
    force_vector: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.force_vector = np.asarray(self.force_vector, dtype=float)
        if self.weight <= 0.0:
            raise ValueError(f"LoadCase weight must be positive, got {self.weight}")


@dataclass
class InertiaReliefConfig:
    """Configuration for inertia relief analysis of unconstrained structures.
    
    Attributes:
        enabled: Whether inertia relief rigid-body mode projection is active.
        reference_node: Optional reference node ID used as origin for rotation modes.
    """
    enabled: bool = False
    reference_node: Optional[int] = None


@dataclass
class SolverResult:
    """Standardized result returned by all FEASolver implementations.
    
    Attributes:
        u: Nodal displacement vector or matrix of shape (N_dofs,) or (N_cases, N_dofs).
        residual_norm: Relative residual norm ||K u - F|| / max(1.0, ||F||).
        iterations: Number of linear solver iterations (1 for direct solvers).
        solve_time_sec: Wall-clock solution time in seconds.
        solver_name: Identifier of the solver that produced the result.
        extra: Additional solver diagnostics (e.g. condition number, AMG levels).
    """
    u: np.ndarray
    residual_norm: float
    iterations: int
    solve_time_sec: float
    solver_name: str
    extra: Dict[str, Any] = field(default_factory=dict)


def build_rigid_body_modes(
    node_coords: np.ndarray,
    dim: int = 3,
    reference_point: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct exact rigid body motion candidate matrix Phi.
    
    Args:
        node_coords: Array of shape (N_nodes, dim) or (N_nodes, 3).
        dim: Spatial dimension (2 or 3).
        reference_point: Optional (dim,) origin coordinate for rotation modes.
        
    Returns:
        Phi: Matrix of shape (N_dofs, n_modes) where n_modes = 3 in 2D, 6 in 3D.
    """
    coords = np.asarray(node_coords, dtype=float)
    n_nodes = coords.shape[0]
    n_dofs = n_nodes * dim

    if reference_point is None:
        ref = np.mean(coords[:, :dim], axis=0)
    else:
        ref = np.asarray(reference_point, dtype=float)[:dim]

    rel_coords = coords[:, :dim] - ref

    if dim == 2:
        # 3 rigid-body modes: Tx, Ty, Rz
        phi = np.zeros((n_dofs, 3), dtype=float)
        # Tx
        phi[0::2, 0] = 1.0
        # Ty
        phi[1::2, 1] = 1.0
        # Rz: u = -y, v = x
        phi[0::2, 2] = -rel_coords[:, 1]
        phi[1::2, 2] = rel_coords[:, 0]
        return phi

    elif dim == 3:
        # 6 rigid-body modes: Tx, Ty, Tz, Rx, Ry, Rz
        phi = np.zeros((n_dofs, 6), dtype=float)
        # Translations
        phi[0::3, 0] = 1.0
        phi[1::3, 1] = 1.0
        phi[2::3, 2] = 1.0
        # Rotations: theta x r
        # Rx: (0, -z, y)
        phi[1::3, 3] = -rel_coords[:, 2]
        phi[2::3, 3] = rel_coords[:, 1]
        # Ry: (z, 0, -x)
        phi[0::3, 4] = rel_coords[:, 2]
        phi[2::3, 4] = -rel_coords[:, 0]
        # Rz: (-y, x, 0)
        phi[0::3, 5] = -rel_coords[:, 1]
        phi[1::3, 5] = rel_coords[:, 0]
        return phi
    else:
        raise ValueError(f"Unsupported spatial dimension dim={dim}; must be 2 or 3.")


class FEASolver(ABC):
    """Abstract base class for all finite element linear static solvers."""

    solver_name: str = "fea_solver"

    @abstractmethod
    def solve(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        free_dofs: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
        M: Optional[sp.spmatrix] = None,
        node_coords: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve structural linear equilibrium equation K u = F subject to boundary conditions.
        
        Args:
            K: Global stiffness matrix of size (N_dof, N_dof) in sparse format.
            F: Global load vector (N_dof,) or matrix (N_cases, N_dof) / (N_dof, N_cases).
            free_dofs: 1D array of unconstrained degree-of-freedom indices.
            fixed_dofs: Optional 1D array of constrained degree-of-freedom indices.
            u_fixed: Optional prescribed values for fixed_dofs (defaults to 0).
            M: Optional mass matrix of size (N_dof, N_dof) for inertia relief or nullspace.
            node_coords: Optional nodal coordinates (N_nodes, dim) for rigid-body projection.
            
        Returns:
            SolverResult containing displacement vector u, residual norm, and metadata.
        """
        pass

    def solve_cached(self, rhs: np.ndarray) -> np.ndarray:
        """Solve K_ff u_f = rhs_f reusing pre-factorized hierarchy/preconditioner for adjoint solves.
        
        Args:
            rhs: Adjoint right-hand-side vector (N_dof,) or (N_free,).
            
        Returns:
            Full adjoint displacement vector u_adj of shape (N_dof,).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement solve_cached")

    @staticmethod
    def partition_system(
        K: sp.spmatrix,
        F: np.ndarray,
        free_dofs: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
    ) -> Tuple[sp.spmatrix, np.ndarray, np.ndarray, np.ndarray]:
        """Extract free-free block K_ff and effective right-hand-side F_ff.
        
        K_ff * u_f = F_f - K_fp * u_p
        
        Returns:
            (K_ff, F_ff, free_dofs, fixed_dofs)
        """
        K_csr = K.tocsr()
        n_dof = K_csr.shape[0]

        free_dofs = np.asarray(free_dofs, dtype=np.int64)
        if fixed_dofs is None:
            mask = np.ones(n_dof, dtype=bool)
            mask[free_dofs] = False
            fixed_dofs = np.flatnonzero(mask)
        else:
            fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)

        K_ff = K_csr[free_dofs, :][:, free_dofs]

        # Handle 1D vs 2D F
        F_arr = np.asarray(F, dtype=float)
        is_1d = (F_arr.ndim == 1)
        if is_1d:
            F_f = F_arr[free_dofs].copy()
            if len(fixed_dofs) > 0 and u_fixed is not None:
                u_p = np.asarray(u_fixed, dtype=float)
                if np.any(u_p != 0.0):
                    K_fp = K_csr[free_dofs, :][:, fixed_dofs]
                    F_f -= K_fp @ u_p
        else:
            # multiple load cases: determine whether shape is (n_dof, n_cases) or (n_cases, n_dof)
            if F_arr.shape[0] == n_dof:
                F_f = F_arr[free_dofs, :].copy()
                if len(fixed_dofs) > 0 and u_fixed is not None:
                    u_p = np.asarray(u_fixed, dtype=float)
                    if np.any(u_p != 0.0):
                        K_fp = K_csr[free_dofs, :][:, fixed_dofs]
                        F_f -= K_fp @ u_p
            elif F_arr.shape[1] == n_dof:
                F_f = F_arr[:, free_dofs].T.copy()  # (n_free, n_cases)
                if len(fixed_dofs) > 0 and u_fixed is not None:
                    u_p = np.asarray(u_fixed, dtype=float).T
                    if np.any(u_p != 0.0):
                        K_fp = K_csr[free_dofs, :][:, fixed_dofs]
                        F_f -= K_fp @ u_p
            else:
                raise ValueError(f"F shape {F_arr.shape} is incompatible with K shape ({n_dof}, {n_dof})")

        return K_ff, F_f, free_dofs, fixed_dofs

    @staticmethod
    def reconstruct_displacement(
        u_free: np.ndarray,
        n_dof: int,
        free_dofs: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reassemble global displacement vector/matrix u of shape (n_dof,) or (n_cases, n_dof)."""
        u_free = np.asarray(u_free, dtype=float)
        free_dofs = np.asarray(free_dofs, dtype=np.int64)

        if u_free.ndim == 1:
            u_full = np.zeros(n_dof, dtype=float)
            u_full[free_dofs] = u_free
            if fixed_dofs is not None and u_fixed is not None and len(fixed_dofs) > 0:
                u_full[fixed_dofs] = u_fixed
            return u_full
        else:
            # u_free could be (n_free, n_cases) or (n_cases, n_free)
            if u_free.shape[0] == len(free_dofs):
                n_cases = u_free.shape[1]
                u_full = np.zeros((n_cases, n_dof), dtype=float)
                u_full[:, free_dofs] = u_free.T
                if fixed_dofs is not None and u_fixed is not None and len(fixed_dofs) > 0:
                    u_full[:, fixed_dofs] = np.asarray(u_fixed, dtype=float).T
                return u_full
            else:
                n_cases = u_free.shape[0]
                u_full = np.zeros((n_cases, n_dof), dtype=float)
                u_full[:, free_dofs] = u_free
                if fixed_dofs is not None and u_fixed is not None and len(fixed_dofs) > 0:
                    u_full[:, fixed_dofs] = np.asarray(u_fixed, dtype=float)
                return u_full

    @staticmethod
    def apply_inertia_relief(
        F: np.ndarray,
        M: Optional[sp.spmatrix],
        node_coords: np.ndarray,
        free_dofs: Optional[np.ndarray] = None,
        dim: int = 3,
    ) -> np.ndarray:
        """Apply exact rigid-body projection P_IR to balance forces on unconstrained structures.
        
        P_IR = I - M * Phi * (Phi^T * M * Phi)^(-1) * Phi^T
        F_relieved = P_IR * F
        
        Such that Phi^T * F_relieved = 0.
        
        Args:
            F: Force vector (N_dof,) or (N_dof, N_cases).
            M: Mass matrix (N_dof, N_dof). If None, identity matrix is used.
            node_coords: Coordinates (N_nodes, dim) of the FE mesh nodes.
            free_dofs: Optional active degree of freedom indices.
            dim: Spatial dimension (2 or 3).
            
        Returns:
            F_relieved: Balanced force vector matching F shape.
        """
        F_arr = np.asarray(F, dtype=float)
        n_dof = len(node_coords) * dim

        phi = build_rigid_body_modes(node_coords, dim=dim)
        if phi.shape[0] != n_dof:
            raise ValueError(f"Phi shape {phi.shape} does not match expected N_dofs={n_dof}")

        # Compute M * Phi
        if M is None:
            M_phi = phi.copy()
        else:
            M_csr = M.tocsr()
            M_phi = M_csr @ phi

        # Generalized mass matrix for rigid body modes: Phi^T @ M @ Phi
        m_rbm = phi.T @ M_phi  # (n_modes, n_modes)
        cond = np.linalg.cond(m_rbm)
        if cond > 1e14 or np.isnan(cond):
            raise ValueError(f"Rigid-body mass matrix is degenerate (condition number={cond})")

        m_rbm_inv = np.linalg.inv(m_rbm)

        if F_arr.ndim == 1:
            # F is (n_dof,)
            f_rbm = phi.T @ F_arr  # (n_modes,)
            c = m_rbm_inv @ f_rbm   # (n_modes,)
            f_bal = M_phi @ c       # (n_dof,)
            return F_arr - f_bal
        else:
            if F_arr.shape[0] == n_dof:
                f_rbm = phi.T @ F_arr       # (n_modes, n_cases)
                c = m_rbm_inv @ f_rbm        # (n_modes, n_cases)
                f_bal = M_phi @ c            # (n_dof, n_cases)
                return F_arr - f_bal
            else:
                f_rbm = phi.T @ F_arr.T     # (n_modes, n_cases)
                c = m_rbm_inv @ f_rbm        # (n_modes, n_cases)
                f_bal = (M_phi @ c).T        # (n_cases, n_dof)
                return F_arr - f_bal
