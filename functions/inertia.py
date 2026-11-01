r"""Analytical Center of Mass (CoM) and Moment of Inertia (MoI) response & sensitivities.

Rigorous mathematical derivation:
---------------------------------
For a 3D discretized domain with elements $e = 1, \dots, N_e$, element centroid $\mathbf{r}_e = [x_e, y_e, z_e]^T$,
element volume $V_e$, and base material density $\rho_0$:
Element mass: $m_e = \rho_e \rho_0 V_e$
Total mass: $M = \sum_{e=1}^{N_e} m_e = \rho_0 \sum_{e=1}^{N_e} \rho_e V_e$

1. Center of Mass (CoM, $\mathbf{r}_c$):
   $\mathbf{r}_c = \frac{1}{M} \sum_{e=1}^{N_e} m_e \mathbf{r}_e = \frac{\sum_{e=1}^{N_e} \rho_e V_e \mathbf{r}_e}{\sum_{e=1}^{N_e} \rho_e V_e}$
   Sensitivity w.r.t element density $\rho_k$:
   $\frac{\partial \mathbf{r}_c}{\partial \rho_k} = \frac{\rho_0 V_k}{M} (\mathbf{r}_k - \mathbf{r}_c)$
   Target deviation sensitivity for $g_{CoM} = \frac{1}{2} \|\mathbf{r}_c - \mathbf{r}_{target}\|^2$:
   $\frac{\partial g_{CoM}}{\partial \rho_k} = \frac{\rho_0 V_k}{M} (\mathbf{r}_c - \mathbf{r}_{target}) \cdot (\mathbf{r}_k - \mathbf{r}_c)$

2. Moment of Inertia Tensor (MoI, $\mathbf{I}^{(c)}$ about instantaneous CoM):
   For brick elements with dimensions $[dx_e, dy_e, dz_e]$:
   $I_{xx}^{(c)} = \sum_{e=1}^{N_e} \rho_e \rho_0 V_e \left[ \frac{dy_e^2 + dz_e^2}{12} + (y_e - y_c)^2 + (z_e - z_c)^2 \right]$
   $I_{xy}^{(c)} = - \sum_{e=1}^{N_e} \rho_e \rho_0 V_e (x_e - x_c)(y_e - y_c)$
   Sensitivity w.r.t element density $\rho_k$ (accounting for moving CoM via chain rule):
   $\frac{\partial I_{xx}^{(c)}}{\partial \rho_k} = \rho_0 V_k \left[ \frac{dy_k^2 + dz_k^2}{12} + (y_k - y_c)^2 + (z_k - z_c)^2 \right]$
   $\frac{\partial I_{xy}^{(c)}}{\partial \rho_k} = - \rho_0 V_k (x_k - x_c)(y_k - y_c)$
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class InertiaProperties:
    """Rigid body inertial properties for structural dynamics and topology optimization."""
    total_mass: float
    center_of_mass: np.ndarray  # shape (3,)
    inertia_tensor_origin: np.ndarray  # shape (3, 3)
    inertia_tensor_com: np.ndarray  # shape (3, 3)
    principal_moments: np.ndarray  # shape (3,)
    principal_axes: np.ndarray  # shape (3, 3)


class InertiaSensitivityCalculator:
    """Calculates exact analytical rigid-body inertial properties and sensitivities.

    Supports 3D hexahedral and tetrahedral element meshes.
    No FEA solver calls required; operates with O(N) vectorized computational complexity.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        base_density: float = 1.0,
        elem_sizes: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize calculator with mesh geometry.

        Parameters
        ----------
        nodes : np.ndarray
            Nodal coordinates array of shape (n_nodes, 3).
        elements : np.ndarray
            Element connectivity array of shape (n_elements, n_nodes_per_elem).
        base_density : float, default 1.0
            Base solid material density (e.g. 2700.0 kg/m^3 for aluminum or 1.0 normalized).
        elem_sizes : Optional[np.ndarray], default None
            Element bounding dimensions [dx, dy, dz] of shape (n_elements, 3) for local inertia.
            If None, inferred automatically from element node bounding boxes.
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.base_density = float(base_density)
        self.n_elements = len(self.elements)

        # Precompute element centroids
        elem_nodes = self.nodes[self.elements]  # shape (n_elements, n_nodes_per_elem, 3)
        self.elem_centroids = np.mean(elem_nodes, axis=1)  # shape (n_elements, 3)

        # Compute element bounding box sizes [dx, dy, dz] and approximate/exact volume
        min_coords = np.min(elem_nodes, axis=1)
        max_coords = np.max(elem_nodes, axis=1)
        inferred_sizes = np.maximum(max_coords - min_coords, 1e-12)
        if elem_sizes is not None:
            self.elem_sizes = np.asarray(elem_sizes, dtype=float)
        else:
            self.elem_sizes = inferred_sizes

        # Element volumes
        self.elem_volumes = self.elem_sizes[:, 0] * self.elem_sizes[:, 1] * self.elem_sizes[:, 2]

    def compute_properties(self, rho: np.ndarray) -> InertiaProperties:
        """Compute complete inertial properties for a given element density distribution.

        Parameters
        ----------
        rho : np.ndarray
            Element physical densities of shape (n_elements,).

        Returns
        -------
        InertiaProperties
            Dataclass containing total mass, CoM, inertia tensors, and principal axes.
        """
        rho_arr = np.asarray(rho, dtype=float).ravel()
        if len(rho_arr) != self.n_elements:
            raise ValueError(f"Density size {len(rho_arr)} != n_elements {self.n_elements}")

        elem_mass = rho_arr * self.base_density * self.elem_volumes
        total_mass = float(np.sum(elem_mass))
        safe_mass = max(total_mass, 1e-12)

        # Center of mass: r_c = sum(m_e * r_e) / M
        com = np.sum(elem_mass[:, None] * self.elem_centroids, axis=0) / safe_mass

        # Element local centroidal inertia (solid brick approximation)
        dx2 = self.elem_sizes[:, 0] ** 2
        dy2 = self.elem_sizes[:, 1] ** 2
        dz2 = self.elem_sizes[:, 2] ** 2
        i_local_xx = elem_mass * (dy2 + dz2) / 12.0
        i_local_yy = elem_mass * (dx2 + dz2) / 12.0
        i_local_zz = elem_mass * (dx2 + dy2) / 12.0

        # Inertia tensor about coordinate origin (0, 0, 0)
        x = self.elem_centroids[:, 0]
        y = self.elem_centroids[:, 1]
        z = self.elem_centroids[:, 2]

        I_xx_0 = float(np.sum(i_local_xx + elem_mass * (y**2 + z**2)))
        I_yy_0 = float(np.sum(i_local_yy + elem_mass * (x**2 + z**2)))
        I_zz_0 = float(np.sum(i_local_zz + elem_mass * (x**2 + y**2)))
        I_xy_0 = float(-np.sum(elem_mass * x * y))
        I_xz_0 = float(-np.sum(elem_mass * x * z))
        I_yz_0 = float(-np.sum(elem_mass * y * z))

        I_origin = np.array([
            [I_xx_0, I_xy_0, I_xz_0],
            [I_xy_0, I_yy_0, I_yz_0],
            [I_xz_0, I_yz_0, I_zz_0],
        ], dtype=float)

        # Inertia tensor about instantaneous Center of Mass (CoM)
        xc, yc, zc = com
        dx_com = x - xc
        dy_com = y - yc
        dz_com = z - zc

        I_xx_c = float(np.sum(i_local_xx + elem_mass * (dy_com**2 + dz_com**2)))
        I_yy_c = float(np.sum(i_local_yy + elem_mass * (dx_com**2 + dz_com**2)))
        I_zz_c = float(np.sum(i_local_zz + elem_mass * (dx_com**2 + dy_com**2)))
        I_xy_c = float(-np.sum(elem_mass * dx_com * dy_com))
        I_xz_c = float(-np.sum(elem_mass * dx_com * dz_com))
        I_yz_c = float(-np.sum(elem_mass * dy_com * dz_com))

        I_com = np.array([
            [I_xx_c, I_xy_c, I_xz_c],
            [I_xy_c, I_yy_c, I_yz_c],
            [I_xz_c, I_yz_c, I_zz_c],
        ], dtype=float)

        # Principal moments and axes (eigenvalues and eigenvectors of symmetric matrix)
        eigvals, eigvecs = np.linalg.eigh(I_com)
        sort_idx = np.argsort(eigvals)
        principal_moments = eigvals[sort_idx]
        principal_axes = eigvecs[:, sort_idx]

        return InertiaProperties(
            total_mass=total_mass,
            center_of_mass=com,
            inertia_tensor_origin=I_origin,
            inertia_tensor_com=I_com,
            principal_moments=principal_moments,
            principal_axes=principal_axes,
        )

    def compute_com_sensitivities(
        self,
        rho: np.ndarray,
        target_com: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute exact analytical derivatives of Center of Mass w.r.t element densities.

        Parameters
        ----------
        rho : np.ndarray
            Element physical densities of shape (n_elements,).
        target_com : Optional[np.ndarray], default None
            Target center of mass [x_target, y_target, z_target].
            If provided, also returns the gradient of objective g = 0.5 * ||r_c - r_target||^2.

        Returns
        -------
        d_com_drho : np.ndarray
            Jacobian matrix of shape (3, n_elements) where row i is d(r_{c, i}) / d(rho_e).
        d_g_drho : np.ndarray (optional)
            Gradient vector of shape (n_elements,) if target_com is provided.
        """
        rho_arr = np.asarray(rho, dtype=float).ravel()
        elem_mass = rho_arr * self.base_density * self.elem_volumes
        total_mass = float(np.sum(elem_mass))
        safe_mass = max(total_mass, 1e-12)

        com = np.sum(elem_mass[:, None] * self.elem_centroids, axis=0) / safe_mass

        # d(r_c)/d(rho_k) = (rho0 * V_k / M) * (r_k - r_c)
        scale = (self.base_density * self.elem_volumes) / safe_mass
        diff = self.elem_centroids - com  # shape (n_elements, 3)
        d_com_drho = (scale[:, None] * diff).T  # shape (3, n_elements)

        if target_com is None:
            return d_com_drho

        target = np.asarray(target_com, dtype=float).reshape(3)
        # g = 0.5 * (r_c - target)^T * (r_c - target)
        # dg/drho_k = (r_c - target) . d(r_c)/drho_k
        err = com - target
        d_g_drho = np.dot(err, d_com_drho)
        return d_com_drho, d_g_drho

    def compute_moi_sensitivities(
        self,
        rho: np.ndarray,
        component: str = "Ixx",
        about_com: bool = True,
    ) -> np.ndarray:
        """Compute exact analytical sensitivity of a moment of inertia component w.r.t element densities.

        Parameters
        ----------
        rho : np.ndarray
            Element physical densities of shape (n_elements,).
        component : str, default 'Ixx'
            One of 'Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz'.
        about_com : bool, default True
            If True, evaluates w.r.t the moving instantaneous Center of Mass.
            If False, evaluates w.r.t the fixed coordinate origin (0, 0, 0).

        Returns
        -------
        np.ndarray
            Analytical sensitivity vector of shape (n_elements,).
        """
        rho_arr = np.asarray(rho, dtype=float).ravel()
        comp = component.lower()

        elem_dm = self.base_density * self.elem_volumes  # dm_k / drho_k

        dx2 = self.elem_sizes[:, 0] ** 2
        dy2 = self.elem_sizes[:, 1] ** 2
        dz2 = self.elem_sizes[:, 2] ** 2

        if about_com:
            props = self.compute_properties(rho_arr)
            com = props.center_of_mass
            dx = self.elem_centroids[:, 0] - com[0]
            dy = self.elem_centroids[:, 1] - com[1]
            dz = self.elem_centroids[:, 2] - com[2]
        else:
            dx = self.elem_centroids[:, 0]
            dy = self.elem_centroids[:, 1]
            dz = self.elem_centroids[:, 2]

        if comp == "ixx":
            return elem_dm * ((dy2 + dz2) / 12.0 + dy**2 + dz**2)
        elif comp == "iyy":
            return elem_dm * ((dx2 + dz2) / 12.0 + dx**2 + dz**2)
        elif comp == "izz":
            return elem_dm * ((dx2 + dy2) / 12.0 + dx**2 + dy**2)
        elif comp == "ixy":
            return -elem_dm * dx * dy
        elif comp == "ixz":
            return -elem_dm * dx * dz
        elif comp == "iyz":
            return -elem_dm * dy * dz
        else:
            raise ValueError(f"Unknown inertia component: '{component}'. Expected Ixx, Iyy, Izz, Ixy, Ixz, Iyz.")
