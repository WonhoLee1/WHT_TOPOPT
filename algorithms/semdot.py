"""Smooth-Edged Material Distribution for Optimizing Topology (SEMDOT) algorithm.

References:
    - Fu, J., Shu, L., & Gao, L. (2020), "SEMDOT: Smooth-Edged Material Distribution
      for Optimizing Topology", arXiv:2005.09233.
    - Fu, J., et al. (2023), "Smooth-Edged Material Distribution for Optimizing Topology:
      Algorithm and Applications", Computer Methods in Applied Mechanics and Engineering.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from .base import GradientSource, Optimizer, TrustRegionController
from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh


class SEMDOTOptimizer(Optimizer):
    """Smooth-Edged Material Distribution for Optimizing Topology (SEMDOT) solver.

    Features:
        - Sub-grid sampling points (s_x, s_y) per element with bilinear interpolation from nodes.
        - Smooth boundary Heaviside threshold function H_beta(rho(x_g) - Psi).
        - Volume-matching bisection to find boundary cut level Psi.
        - Preserves clean 0/1 elements while providing smooth boundary representation on intermediate elements.
        - Support for passive solid, passive void, and trust-region controllers.
    """

    def __init__(
        self,
        mesh: Optional[Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh]] = None,
        volfrac: float = 0.5,
        n_sub: int = 4,
        beta_init: float = 0.5,
        beta_max: float = 8.0,
        beta_factor: float = 1.05,
        rho_min: float = 1e-3,
        move: float = 0.2,
        eta: float = 0.5,
        trust_region: Optional[TrustRegionController] = None,
        gradient_source: GradientSource = GradientSource.EXACT_ADJOINT,
    ) -> None:
        """Initialize SEMDOTOptimizer.

        Parameters
        ----------
        mesh : Optional computational mesh
            Grid structure for element-to-node topology.
        volfrac : float
            Target volume fraction V* / V0.
        n_sub : int
            Sub-grid discretization points per spatial dimension (default 4 -> 16 points in 2D).
        beta_init : float
            Initial Heaviside steepness parameter (default 0.5).
        beta_max : float
            Maximum Heaviside steepness parameter (default 8.0).
        beta_factor : float
            Continuation multiplier per iteration (default 1.05).
        rho_min : float
            Lower bound on density (default 1e-3).
        move : float
            Design change move limit per step (default 0.2).
        eta : float
            OC damping parameter (default 0.5).
        trust_region : Optional[TrustRegionController]
            Adaptive trust-region controller.
        gradient_source : GradientSource
            Sensitivity information source.
        """
        self.mesh = mesh
        self.volfrac = float(volfrac)
        self.n_sub = int(n_sub)
        self.beta = float(beta_init)
        self.beta_max = float(beta_max)
        self.beta_factor = float(beta_factor)
        self.rho_min = float(rho_min)
        self.move = float(move)
        self.eta = float(eta)
        self.trust_region = trust_region
        self.gradient_source = gradient_source

        # Stored state for smooth boundary extraction
        self.psi: float = 0.5
        self.iter: int = 0
        self.subgrid_level_set: Optional[np.ndarray] = None
        self.subgrid_points: Optional[np.ndarray] = None
        self.extracted_grid: Optional[np.ndarray] = None
        self._prev_sens: Optional[np.ndarray] = None
        self.sens_history: float = 0.5

    def reset(self) -> None:
        """Reset internal optimization state."""
        self.iter = 0
        self.psi = 0.5
        self.subgrid_level_set = None
        self.subgrid_points = None
        self.extracted_grid = None
        self._prev_sens = None

    def smooth_heaviside_cut(self, rho_samples: np.ndarray, psi: float, beta: float) -> np.ndarray:
        """Evaluate SEMDOT smooth boundary Heaviside function (Fu Eq. 16).

        H_beta(rho - psi) = (tanh(beta * psi) + tanh(beta * (rho - psi))) / (tanh(beta * psi) + tanh(beta * (1 - psi)))
        """
        t_bp = np.tanh(beta * psi)
        t_bm = np.tanh(beta * (1.0 - psi))
        den = t_bp + t_bm
        if abs(den) < 1e-15:
            den = 1e-15

        t_diff = np.tanh(beta * (rho_samples - psi))
        res = (t_bp + t_diff) / den
        return self.rho_min + (1.0 - self.rho_min) * np.clip(res, 0.0, 1.0)

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
        """Execute one SEMDOT iteration.

        Parameters
        ----------
        rho : np.ndarray
            Current density field (N_elem,).
        df_drho : np.ndarray
            Objective sensitivity array (N_elem,).
        g_vals : np.ndarray
            Constraint function values.
        dg_drho : np.ndarray
            Constraint sensitivities.
        elem_volumes : Optional[np.ndarray]
            Element volume array (N_elem,).
        passive_solid : Optional[np.ndarray]
            Passive solid elements.
        passive_void : Optional[np.ndarray]
            Passive void elements.

        Returns
        -------
        rho_new : np.ndarray
            Updated smooth-cut elemental density array of shape (N_elem,).
        """
        rho_curr = np.asarray(rho, dtype=np.float64).copy()
        n_elem = len(rho_curr)

        if elem_volumes is None:
            volumes = np.ones(n_elem, dtype=np.float64)
        else:
            volumes = np.asarray(elem_volumes, dtype=np.float64)

        total_volume = float(np.sum(volumes))
        target_volume = self.volfrac * total_volume

        current_move = self.move
        if self.trust_region is not None:
            current_move = self.trust_region.move

        # Passive masks
        solid_mask = np.zeros(n_elem, dtype=bool)
        if passive_solid is not None:
            if np.asarray(passive_solid).dtype == bool:
                solid_mask = np.asarray(passive_solid, dtype=bool)
            else:
                solid_mask[np.asarray(passive_solid, dtype=np.int64)] = True

        void_mask = np.zeros(n_elem, dtype=bool)
        if passive_void is not None:
            if np.asarray(passive_void).dtype == bool:
                void_mask = np.asarray(passive_void, dtype=bool)
            else:
                void_mask[np.asarray(passive_void, dtype=np.int64)] = True

        # -------------------------------------------------------------------
        # 1. Base Density Target Update via Optimality Criteria
        # -------------------------------------------------------------------
        self.iter += 1
        raw_num = np.maximum(-np.asarray(df_drho, dtype=np.float64).ravel(), 1e-30)
        if self._prev_sens is not None and self.sens_history > 0.0:
            num = (1.0 - self.sens_history) * raw_num + self.sens_history * self._prev_sens
        else:
            num = raw_num
        self._prev_sens = num.copy()
        den = volumes.copy()
        if dg_drho is not None and dg_drho.size == n_elem:
            d_arr = np.asarray(dg_drho, dtype=np.float64).ravel()
            den = np.where(d_arr > 0.0, d_arr, volumes)

        lb = np.maximum(self.rho_min, rho_curr - current_move)
        ub = np.minimum(1.0, rho_curr + current_move)

        # Logarithmic bisection to find multiplier lambda for target volume
        l1, l2 = 1e-15, 1e15
        lmid = 1.0
        for _ in range(60):
            lmid = np.sqrt(l1 * l2)
            be = (num / np.maximum(lmid * den, 1e-30)) ** self.eta
            cand = np.clip(rho_curr * be, lb, ub)
            if np.sum(cand * volumes) > target_volume:
                l1 = lmid
            else:
                l2 = lmid

        be_opt = (num / np.maximum(lmid * den, 1e-30)) ** self.eta
        rho_target = np.clip(rho_curr * be_opt, lb, ub)

        # -------------------------------------------------------------------
        # 2. Sub-grid Sampling Points Evaluation
        # -------------------------------------------------------------------
        ns = self.n_sub
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            # Bilinear interpolation inside each rectangular Q4 element
            # 1D sub-grid coordinates xi, eta in [0, 1]
            xi = np.linspace(0.1, 0.9, ns)
            XI, ETA = np.meshgrid(xi, xi)
            # Shape functions: N1 = (1-x)*(1-y), N2 = x*(1-y), N3 = x*y, N4 = (1-x)*y
            N1 = ((1.0 - XI) * (1.0 - ETA)).ravel() # (ns*ns,)
            N2 = (XI * (1.0 - ETA)).ravel()
            N3 = (XI * ETA).ravel()
            N4 = ((1.0 - XI) * ETA).ravel()
            N_mat = np.vstack([N1, N2, N3, N4])  # (4, ns*ns)

            # Node values per element
            elems = self.mesh.elements  # (n_elem, 4)
            n_nodes = self.mesh.n_nodes
            # Map element target rho to nodes via averaging
            node_sum = np.zeros(n_nodes, dtype=np.float64)
            node_cnt = np.zeros(n_nodes, dtype=np.float64)
            for k in range(4):
                np.add.at(node_sum, elems[:, k], rho_target)
                np.add.at(node_cnt, elems[:, k], 1.0)
            rho_nodes = node_sum / np.maximum(node_cnt, 1.0)

            # Element corner nodal values: shape (n_elem, 4)
            elem_node_rho = rho_nodes[elems]
            # Sub-grid point values: (n_elem, ns*ns) = elem_node_rho @ N_mat
            sub_samples = elem_node_rho @ N_mat
        else:
            # General mesh fallback: sample around element value with small variation
            offsets = np.linspace(-0.1, 0.1, ns * ns)
            sub_samples = np.clip(rho_target[:, None] + offsets[None, :], self.rho_min, 1.0)

        # Store sub-grid representation for boundary extraction
        self.subgrid_points = sub_samples

        # -------------------------------------------------------------------
        # 3. Volume-Matching Threshold Bisection for Cut Level Psi
        # -------------------------------------------------------------------
        psi_l, psi_u = 0.0, 1.0
        best_X = rho_curr.copy()
        beta_cur = self.beta

        for _ in range(80):
            psi_mid = 0.5 * (psi_l + psi_u)
            # Evaluate smooth cut fractions
            cut_samples = self.smooth_heaviside_cut(sub_samples, psi_mid, beta_cur)
            # Element volume fraction: average over sub-grid points
            X_cand = np.mean(cut_samples, axis=1)

            # Passive elements
            if np.any(solid_mask):
                X_cand[solid_mask] = 1.0
            if np.any(void_mask):
                X_cand[void_mask] = self.rho_min

            vol_cand = float(np.sum(X_cand * volumes))
            best_X = X_cand

            vol_err = abs(vol_cand - target_volume) / total_volume
            if vol_err < 1e-4 or (psi_u - psi_l) < 1e-8:
                break

            if vol_cand > target_volume:
                psi_l = psi_mid  # Increase threshold to reduce volume
            else:
                psi_u = psi_mid  # Decrease threshold to increase volume

        self.psi = 0.5 * (psi_l + psi_u)
        self.subgrid_level_set = sub_samples - self.psi

        # Construct 2D extracted binary sub-grid (ny*ns + 1, nx*ns + 1) when StructuredMesh2D is used
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            # Interpolate nodal rho_nodes (ny+1, nx+1) onto (ny*ns + 1, nx*ns + 1) grid
            rho_n2d = rho_nodes.reshape(ny + 1, nx + 1)
            ys_f = np.linspace(0, ny, ny * ns + 1)
            xs_f = np.linspace(0, nx, nx * ns + 1)
            j0 = np.clip(np.floor(ys_f).astype(int), 0, ny - 1)
            i0 = np.clip(np.floor(xs_f).astype(int), 0, nx - 1)
            ty = (ys_f - j0)[:, None]
            tx = (xs_f - i0)[None, :]
            v00 = rho_n2d[j0[:, None], i0[None, :]]
            v10 = rho_n2d[j0[:, None], i0[None, :] + 1]
            v01 = rho_n2d[j0[:, None] + 1, i0[None, :]]
            v11 = rho_n2d[j0[:, None] + 1, i0[None, :] + 1]
            sub_2d = (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11)
            self.extracted_grid = (sub_2d >= self.psi).astype(np.float64)

        # Update beta continuation
        self.beta = min(self.beta_max, self.beta * self.beta_factor)

        return best_X

    def project_filtered(
        self,
        x_filt: np.ndarray,
        target_volfrac: Optional[float] = None,
        passive_solid: Optional[np.ndarray] = None,
        passive_void: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project an already element-filtered field ``x_filt`` onto sub-grid points and return ``Psi_e``."""
        rho_target = np.asarray(x_filt, dtype=np.float64).copy()
        n_elem = len(rho_target)
        vf = self.volfrac if target_volfrac is None else float(target_volfrac)
        target_volume = vf * float(n_elem)

        solid_mask = np.zeros(n_elem, dtype=bool)
        if passive_solid is not None:
            solid_mask[np.asarray(passive_solid, dtype=np.int64)] = True
        void_mask = np.zeros(n_elem, dtype=bool)
        if passive_void is not None:
            void_mask[np.asarray(passive_void, dtype=np.int64)] = True

        ns = self.n_sub
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            xi = np.linspace(0.0, 1.0, ns + 1)
            XI, ETA = np.meshgrid(xi, xi)
            N1 = ((1.0 - XI) * (1.0 - ETA)).ravel()
            N2 = (XI * (1.0 - ETA)).ravel()
            N3 = (XI * ETA).ravel()
            N4 = ((1.0 - XI) * ETA).ravel()
            N_mat = np.vstack([N1, N2, N3, N4])

            elems = self.mesh.elements
            n_nodes = self.mesh.n_nodes
            node_sum = np.zeros(n_nodes, dtype=np.float64)
            node_cnt = np.zeros(n_nodes, dtype=np.float64)
            for k in range(4):
                np.add.at(node_sum, elems[:, k], rho_target)
                np.add.at(node_cnt, elems[:, k], 1.0)
            rho_nodes = node_sum / np.maximum(node_cnt, 1.0)
            elem_node_rho = rho_nodes[elems]
            sub_samples = elem_node_rho @ N_mat
        else:
            offsets = np.linspace(-0.05, 0.05, (ns + 1) ** 2)
            sub_samples = np.clip(rho_target[:, None] + offsets[None, :], self.rho_min, 1.0)

        psi_l, psi_u = 0.0, 1.0
        best_X = rho_target.copy()
        beta_cur = self.beta
        for _ in range(60):
            psi_mid = 0.5 * (psi_l + psi_u)
            cut_samples = self.smooth_heaviside_cut(sub_samples, psi_mid, beta_cur)
            X_cand = np.mean(cut_samples, axis=1)
            if np.any(solid_mask):
                X_cand[solid_mask] = 1.0
            if np.any(void_mask):
                X_cand[void_mask] = self.rho_min
            vol_cand = float(np.sum(X_cand))
            best_X = X_cand
            if abs(vol_cand - target_volume) / float(n_elem) < 1e-5 or (psi_u - psi_l) < 1e-9:
                break
            if vol_cand > target_volume:
                psi_l = psi_mid
            else:
                psi_u = psi_mid

        self.psi = 0.5 * (psi_l + psi_u)
        if self.mesh is not None and isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            rho_n2d = rho_nodes.reshape(ny + 1, nx + 1)
            ys_f = np.linspace(0, ny, ny * ns + 1)
            xs_f = np.linspace(0, nx, nx * ns + 1)
            j0 = np.clip(np.floor(ys_f).astype(int), 0, ny - 1)
            i0 = np.clip(np.floor(xs_f).astype(int), 0, nx - 1)
            ty = (ys_f - j0)[:, None]
            tx = (xs_f - i0)[None, :]
            v00 = rho_n2d[j0[:, None], i0[None, :]]
            v10 = rho_n2d[j0[:, None], i0[None, :] + 1]
            v01 = rho_n2d[j0[:, None] + 1, i0[None, :]]
            v11 = rho_n2d[j0[:, None] + 1, i0[None, :] + 1]
            sub_2d = (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11)
            # Threshold binary sub-grid to match target_volfrac
            lo_g, hi_g = float(sub_2d.min()), float(sub_2d.max())
            for _ in range(60):
                mid_g = 0.5 * (lo_g + hi_g)
                if float(np.mean(sub_2d >= mid_g)) > vf:
                    lo_g = mid_g
                else:
                    hi_g = mid_g
            self.extracted_grid = (sub_2d >= 0.5 * (lo_g + hi_g)).astype(np.float64)

        self.beta = min(self.beta_max, self.beta + 0.35)
        return best_X
