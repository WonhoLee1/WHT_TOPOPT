"""Hilbertian projection Level-Set topology optimization algorithm.

References:
    - Wegert, Roberts, Challis (2023), "A Hilbertian projection method for constrained
      level set-based topology optimisation", arXiv:2305.03913.
    - Jezek et al. (2026), "Sequential topology optimization: SIMP initialization
      for level-set boundary refinement", arXiv:2605.04735.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import GradientSource, Optimizer, TrustRegionController
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from wht_topopt.core.numba_kernels import HAS_NUMBA, numba_reinit_distance_2d


class LevelSetOptimizer(Optimizer):
    """Hilbertian projection Level-Set optimizer.

    Features:
        - Smoothed Heaviside mapping rho(phi) = H_eta(phi) in [rho_min, 1.0].
        - Wegert (2023) H^1 Hilbertian velocity extension-regularization:
            (M_Omega + ell^2 K_Omega) v = -M_Omega nabla f
        - Gram-Schmidt orthogonal projection onto constraint tangent space:
            P_{C_perp} v_f = v_f - (<mu, v_f>_H / ||mu||_H^2) * mu
        - Dynamic constraint correction alpha to enforce volume constraint satisfaction.
        - First-order Godunov Upwind Hamilton-Jacobi advection update.
        - Support for passive solid, passive void, and trust-region step limits.
    """

    def __init__(
        self,
        mesh: Optional[Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]] = None,
        volfrac: float = 0.5,
        ell: float = 0.1,
        gamma: float = 0.1,
        reinit_interval: int = 50,
        alpha_min: float = 0.1,
        rho_min: float = 1e-3,
        move: float = 0.2,
        trust_region: Optional[TrustRegionController] = None,
        gradient_source: GradientSource = GradientSource.EXACT_ADJOINT,
    ) -> None:
        """Initialize LevelSetOptimizer.

        Parameters
        ----------
        mesh : Optional computational mesh
            Used for spatial grid dimensions and Laplacian assembly.
        volfrac : float
            Prescribed target volume fraction V* / V0.
        ell : float
            Hilbertian regularization length scale beta (default 0.1).
        gamma : float
            CFL time step coefficient in (0, 1) (default 0.1).
        reinit_interval : int
            Iteration interval for signed-distance field reinitialization.
        alpha_min : float
            Constraint correction lower limit (default 0.1).
        rho_min : float
            Void density bound (default 1e-3).
        move : float
            Move limit for maximum level set or density shift.
        trust_region : Optional[TrustRegionController]
            Adaptive trust-region controller.
        gradient_source : GradientSource
            Sensitivity information source.
        """
        self.mesh = mesh
        self.volfrac = float(volfrac)
        self.ell = float(ell)
        self.gamma = float(gamma)
        self.reinit_interval = int(reinit_interval)
        self.alpha_min = float(alpha_min)
        self.rho_min = float(rho_min)
        self.move = float(move)
        self.trust_region = trust_region
        self.gradient_source = gradient_source

        # Internal level set field phi (stored across iterations)
        self.phi: Optional[np.ndarray] = None
        self.iter: int = 0

        # Cached H^1 operator
        self._A_H: Optional[sp.csc_matrix] = None
        self._lu_H: Optional[spla.SuperLU] = None
        self._h_mesh: float = 1.0
        self._eta: float = 1.5

    def reset(self) -> None:
        """Reset internal optimizer state."""
        self.phi = None
        self.iter = 0
        self._A_H = None
        self._lu_H = None

    def _build_hilbertian_system(self, n_elem: int) -> None:
        """Construct discrete H^1 Hilbertian operator A_H = M + ell^2 K_diff."""
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            dx, dy = self.mesh.dx, self.mesh.dy
            self._h_mesh = float(min(dx, dy))
            self._eta = 1.5 * self._h_mesh

            # 5-point discrete 2D Laplacian on element centers with Neumann BCs
            # Element index e = i + j*nx, with i in [0..nx-1], j in [0..ny-1]
            main_diag = np.ones(n_elem, dtype=np.float64)
            off_x = np.full(n_elem - 1, - (self.ell / dx) ** 2, dtype=np.float64)
            # Remove connections across rows
            off_x[np.arange(1, ny) * nx - 1] = 0.0

            off_y = np.full(n_elem - nx, - (self.ell / dy) ** 2, dtype=np.float64)

            # Diagonal accumulation
            d_diag = np.zeros(n_elem, dtype=np.float64)
            for j in range(ny):
                for i in range(nx):
                    idx = i + j * nx
                    deg_x = (1 if i > 0 else 0) + (1 if i < nx - 1 else 0)
                    deg_y = (1 if j > 0 else 0) + (1 if j < ny - 1 else 0)
                    d_diag[idx] = 1.0 + deg_x * ((self.ell / dx) ** 2) + deg_y * ((self.ell / dy) ** 2)

            diags = [off_y, off_x, d_diag, off_x, off_y]
            offsets = [-nx, -1, 0, 1, nx]
            A_H = sp.diags(diags, offsets, shape=(n_elem, n_elem), format="csc")
            self._A_H = A_H
            self._lu_H = spla.splu(A_H)
        else:
            # Fallback 1D/generic Laplacian for unstructured meshes
            self._h_mesh = 1.0
            self._eta = 1.5 * self._h_mesh
            main_diag = np.full(n_elem, 1.0 + 2.0 * (self.ell ** 2), dtype=np.float64)
            main_diag[0] = 1.0 + self.ell ** 2
            main_diag[-1] = 1.0 + self.ell ** 2
            off_diag = np.full(n_elem - 1, -(self.ell ** 2), dtype=np.float64)
            A_H = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_elem, n_elem), format="csc")
            self._A_H = A_H
            self._lu_H = spla.splu(A_H)

    def smoothed_heaviside(self, phi: np.ndarray) -> np.ndarray:
        r"""Evaluate smoothed Heaviside function H_eta(phi) in [0, 1].
        
        Wegert (2023) / Jezek (2026) convention:
            phi < 0 : solid (Omega) -> H_eta = 1.0
            phi > 0 : void (D \ Omega) -> H_eta = 0.0
        """
        eta = self._eta
        res = np.zeros_like(phi)
        in_band = np.abs(phi) <= eta
        res[phi < -eta] = 1.0
        res[in_band] = 0.5 - phi[in_band] / (2.0 * eta) - (1.0 / (2.0 * np.pi)) * np.sin(np.pi * phi[in_band] / eta)
        return res

    def heaviside_derivative(self, phi: np.ndarray) -> np.ndarray:
        """Evaluate smoothed shape delta function |H'_eta(phi)|."""
        eta = self._eta
        d_res = np.zeros_like(phi)
        in_band = np.abs(phi) <= eta
        d_res[in_band] = (1.0 / (2.0 * eta)) * (1.0 + np.cos(np.pi * phi[in_band] / eta))
        return d_res

    def reinitialize_distance(self, phi: np.ndarray, n_steps: int = 2, use_numba: bool = True) -> np.ndarray:
        """Reinitialize level-set field towards signed distance |nabla phi| = 1 (Wegert Eq. 5)."""
        if self.mesh is None or not isinstance(self.mesh, StructuredMesh2D):
            return phi.copy()

        nx, ny = self.mesh.nelx, self.mesh.nely
        dx, dy = self.mesh.dx, self.mesh.dy
        if use_numba and HAS_NUMBA:
            return numba_reinit_distance_2d(
                np.ascontiguousarray(phi, dtype=np.float64),
                int(nx),
                int(ny),
                float(dx),
                float(dy),
                int(n_steps),
            )
        phi_2d = phi.reshape((ny, nx)).copy()
        h = min(dx, dy)
        dt_tau = 0.2 * h

        # S(phi0) smoothed sign function
        grad_x = np.gradient(phi_2d, dx, axis=1)
        grad_y = np.gradient(phi_2d, dy, axis=0)
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)
        s_phi0 = phi_2d / np.sqrt(phi_2d ** 2 + (grad_norm * dx) ** 2 + 1e-12)

        for _ in range(n_steps):
            dx_m = np.zeros_like(phi_2d)
            dx_p = np.zeros_like(phi_2d)
            dy_m = np.zeros_like(phi_2d)
            dy_p = np.zeros_like(phi_2d)

            dx_m[:, 1:] = (phi_2d[:, 1:] - phi_2d[:, :-1]) / dx
            dx_p[:, :-1] = (phi_2d[:, 1:] - phi_2d[:, :-1]) / dx
            dy_m[1:, :] = (phi_2d[1:, :] - phi_2d[:-1, :]) / dy
            dy_p[:-1, :] = (phi_2d[1:, :] - phi_2d[:-1, :]) / dy

            pos = s_phi0 > 0.0
            neg = ~pos
            G_sq = np.zeros_like(phi_2d)
            G_sq[pos] = (
                np.maximum(dx_m[pos], 0.0) ** 2 + np.minimum(dx_p[pos], 0.0) ** 2 +
                np.maximum(dy_m[pos], 0.0) ** 2 + np.minimum(dy_p[pos], 0.0) ** 2
            )
            G_sq[neg] = (
                np.minimum(dx_m[neg], 0.0) ** 2 + np.maximum(dx_p[neg], 0.0) ** 2 +
                np.minimum(dy_m[neg], 0.0) ** 2 + np.maximum(dy_p[neg], 0.0) ** 2
            )
            phi_2d -= dt_tau * s_phi0 * (np.sqrt(np.maximum(G_sq, 1e-12)) - 1.0)

        return phi_2d.ravel()

    def update(
        self,
        rho: np.ndarray,
        df_drho: np.ndarray,
        g_vals: np.ndarray,
        dg_drho: np.ndarray,
        elem_volumes: Optional[np.ndarray] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute one Hilbertian projection Level-Set step.

        Parameters
        ----------
        rho : np.ndarray
            Current density field (N_elem,).
        df_drho : np.ndarray
            Objective sensitivity array (N_elem,).
        g_vals : np.ndarray
            Constraint function values.
        dg_drho : np.ndarray
            Constraint sensitivity array.
        elem_volumes : Optional[np.ndarray]
            Element volume array (N_elem,).
        passive_solid : Optional[np.ndarray]
            Passive solid elements.
        passive_void : Optional[np.ndarray]
            Passive void elements.

        Returns
        -------
        rho_new : np.ndarray
            Updated physical density field of shape (N_elem,).
        """
        rho_curr = np.asarray(rho, dtype=np.float64).copy()
        n_elem = len(rho_curr)

        if self._lu_H is None:
            self._build_hilbertian_system(n_elem)

        if elem_volumes is None:
            volumes = np.ones(n_elem, dtype=np.float64)
        else:
            volumes = np.asarray(elem_volumes, dtype=np.float64)

        total_vol = float(np.sum(volumes))
        target_vol = self.volfrac * total_vol
        cur_vol = float(np.sum(rho_curr * volumes))
        vol_violation = (cur_vol - target_vol) / total_vol

        # -------------------------------------------------------------------
        # 1. Level-Set Field Initialization (if first step)
        # -------------------------------------------------------------------
        self.iter += 1
        if self.phi is None:
            eta = self._eta
            norm_r = 1.0 - 2.0 * (rho_curr - self.rho_min) / (1.0 - self.rho_min)
            norm_r = np.clip(norm_r, -0.999, 0.999)
            self.phi = eta * norm_r

        phi = self.phi.copy()

        # -------------------------------------------------------------------
        # 2. Shape Derivatives and Hilbertian Regularization
        # -------------------------------------------------------------------
        # Shape sensitivity: g_f = -df/drho * H'(phi)
        H_prime = self.heaviside_derivative(phi)
        # Avoid zero shape gradient
        H_prime = np.maximum(H_prime, 0.05 / self._eta)

        # Raw shape sensitivities
        raw_g_f = -np.asarray(df_drho, dtype=np.float64).ravel() * H_prime
        if dg_drho is not None and dg_drho.size == n_elem:
            raw_g_c = -np.asarray(dg_drho, dtype=np.float64).ravel() * H_prime
        else:
            raw_g_c = -volumes * H_prime

        # Solve Hilbertian extension: A_H @ v = g
        v_f = self._lu_H.solve(raw_g_f)
        v_c = self._lu_H.solve(raw_g_c)

        # -------------------------------------------------------------------
        # 3. Gram-Schmidt Orthogonal Projection onto Constraint Tangent Space
        # -------------------------------------------------------------------
        # Inner product: <u, w>_H = u^T @ A_H @ w = u^T @ raw_g
        norm_c_sq = float(np.dot(v_c, self._A_H @ v_c))
        norm_c = np.sqrt(max(norm_c_sq, 1e-15))

        dot_fc = float(np.dot(v_f, self._A_H @ v_c))
        # P_{C_perp} v_f = v_f - (dot_fc / norm_c^2) * v_c
        v_proj = v_f - (dot_fc / max(norm_c_sq, 1e-15)) * v_c
        norm_proj_sq = float(np.dot(v_proj, self._A_H @ v_proj))
        norm_proj = np.sqrt(max(norm_proj_sq, 1e-15))

        # Constraint restoration coefficient alpha
        # Correction proportional to volume constraint violation
        alpha_cand = float(vol_violation) / max(self.volfrac * 0.05, 0.01)
        alpha = float(np.clip(alpha_cand, -0.85, 0.85))

        # Combined normal velocity field
        coeff_opt = np.sqrt(max(0.0, 1.0 - alpha ** 2))
        v_n = coeff_opt * (v_proj / norm_proj) + alpha * (v_c / norm_c)

        # -------------------------------------------------------------------
        # 4. Godunov Upwind Hamilton-Jacobi Advection
        # -------------------------------------------------------------------
        # CFL time step
        max_vel = float(np.max(np.abs(v_n)))
        dt = float(self.gamma * self._h_mesh / max(max_vel, 1e-8))
        if self.trust_region is not None:
            dt = min(dt, self.trust_region.move * self._h_mesh / max(max_vel, 1e-8))

        # 1st-order Godunov upwind spatial gradient
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            dx, dy = self.mesh.dx, self.mesh.dy
            phi_2d = phi.reshape((ny, nx))
            v_2d = v_n.reshape((ny, nx))

            # Backward and forward differences
            dx_m = np.zeros_like(phi_2d)
            dx_p = np.zeros_like(phi_2d)
            dy_m = np.zeros_like(phi_2d)
            dy_p = np.zeros_like(phi_2d)

            dx_m[:, 1:] = (phi_2d[:, 1:] - phi_2d[:, :-1]) / dx
            dx_p[:, :-1] = (phi_2d[:, 1:] - phi_2d[:, :-1]) / dx
            dy_m[1:, :] = (phi_2d[1:, :] - phi_2d[:-1, :]) / dy
            dy_p[:-1, :] = (phi_2d[1:, :] - phi_2d[:-1, :]) / dy

            # Godunov numerical Hamiltonian
            pos_v = v_2d > 0.0
            neg_v = ~pos_v

            grad_sq = np.zeros_like(phi_2d)
            # v > 0: use max(dx_m, 0)^2 + min(dx_p, 0)^2 + ...
            grad_sq[pos_v] = (
                np.maximum(dx_m[pos_v], 0.0) ** 2
                + np.minimum(dx_p[pos_v], 0.0) ** 2
                + np.maximum(dy_m[pos_v], 0.0) ** 2
                + np.minimum(dy_p[pos_v], 0.0) ** 2
            )
            # v < 0: use min(dx_m, 0)^2 + max(dx_p, 0)^2 + ...
            grad_sq[neg_v] = (
                np.minimum(dx_m[neg_v], 0.0) ** 2
                + np.maximum(dx_p[neg_v], 0.0) ** 2
                + np.minimum(dy_m[neg_v], 0.0) ** 2
                + np.maximum(dy_p[neg_v], 0.0) ** 2
            )

            grad_norm = np.maximum(np.sqrt(np.maximum(grad_sq, 1e-12)).ravel(), 0.2)
        else:
            grad_norm = np.ones(n_elem, dtype=np.float64)

        # Update phi
        phi_new = phi - dt * (v_n * grad_norm)

        # Periodic signed-distance reinitialization
        if self.reinit_interval > 0 and self.iter % self.reinit_interval == 0:
            phi_new = self.reinitialize_distance(phi_new)

        self.phi = phi_new

        # -------------------------------------------------------------------
        # 5. Physical Density Projection
        # -------------------------------------------------------------------
        H_val = self.smoothed_heaviside(self.phi)
        rho_cand = self.rho_min + (1.0 - self.rho_min) * H_val
        move_eff = self.move if self.trust_region is None else self.trust_region.move
        rho_new = np.clip(
            rho_cand,
            np.maximum(self.rho_min, rho_curr - move_eff),
            np.minimum(1.0, rho_curr + move_eff),
        )

        # Passive domains
        if passive_solid is not None:
            if np.asarray(passive_solid).dtype == bool:
                s_mask = np.asarray(passive_solid, dtype=bool)
            else:
                s_mask = np.zeros(n_elem, dtype=bool)
                s_mask[np.asarray(passive_solid, dtype=np.int64)] = True
            rho_new[s_mask] = 1.0

        if passive_void is not None:
            if np.asarray(passive_void).dtype == bool:
                v_mask = np.asarray(passive_void, dtype=bool)
            else:
                v_mask = np.zeros(n_elem, dtype=bool)
                v_mask[np.asarray(passive_void, dtype=np.int64)] = True
            rho_new[v_mask] = self.rho_min

        return rho_new
