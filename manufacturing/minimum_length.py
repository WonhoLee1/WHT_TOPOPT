"""Minimum length scale constraints and robust three-field projection.

Implements Sigmund (2009) & Wang, Lazarov, Sigmund (2011) three-field projection
(eroded, intermediate, dilated) and Zhou et al. (2015) structural indicator-based
geometric length scale constraints for solid and void phases with exact analytical
reverse-mode gradients.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from wht_topopt.core.projection import HeavisideProjection
from wht_topopt.functions.base import ConstraintFunction, GradientSource


def compute_eta_from_length_scale(r_min: float, l_min: float) -> Tuple[float, float]:
    """Compute eroded and dilated thresholds (eta_e, eta_d) from target length scale.

    For filter radius r_min and intermediate threshold eta_i = 0.5,
    the threshold variation delta_eta scales with target feature size l_min / r_min:
        delta_eta = 0.25 * (l_min / r_min)
        eta_e = 0.5 + delta_eta
        eta_d = 0.5 - delta_eta
    such that when l_min == r_min, eta_e = 0.75 and eta_d = 0.25 (Sigmund 2009).

    Parameters
    ----------
    r_min : float
        Filter radius in physical length units (r_min > 0).
    l_min : float
        Target minimum feature size (0 < l_min < 2 * r_min).

    Returns
    -------
    Tuple[float, float]
        Eroded threshold eta_e and dilated threshold eta_d.
    """
    if r_min <= 0.0:
        raise ValueError(f"r_min must be positive, got {r_min}")
    if not (0.0 < l_min < 2.0 * r_min):
        raise ValueError(f"l_min must be in (0, 2 * r_min), got l_min={l_min}, r_min={r_min}")

    delta_eta = 0.25 * (l_min / r_min)
    delta_eta = float(np.clip(delta_eta, 0.01, 0.49))
    eta_e = float(0.5 + delta_eta)
    eta_d = float(0.5 - delta_eta)
    return eta_e, eta_d


class ThreeFieldProjection:
    """Three-field robust Heaviside projection (Wang, Lazarov & Sigmund 2011).

    Computes eroded (eta_e), intermediate (eta_i = 0.5), and dilated (eta_d)
    crisp physical density fields from a filtered field rho_tilde, together
    with exact analytical derivatives and adjoint vector-Jacobian products.
    """

    def __init__(
        self,
        beta: float = 8.0,
        eta_i: float = 0.5,
        delta_eta: float = 0.25,
        eta_e: Optional[float] = None,
        eta_d: Optional[float] = None,
    ) -> None:
        """Initialize ThreeFieldProjection.

        Parameters
        ----------
        beta : float
            Sharpness parameter beta >= 0.0.
        eta_i : float
            Intermediate threshold (default 0.5).
        delta_eta : float
            Threshold offset: eta_e = eta_i + delta_eta, eta_d = eta_i - delta_eta.
        eta_e : Optional[float]
            Explicit eroded threshold (overrides delta_eta if provided with eta_d).
        eta_d : Optional[float]
            Explicit dilated threshold (overrides delta_eta if provided with eta_e).
        """
        self.beta: float = float(beta)
        self.eta_i: float = float(eta_i)

        if eta_e is not None and eta_d is not None:
            self.eta_e: float = float(eta_e)
            self.eta_d: float = float(eta_d)
            self.delta_eta: float = (self.eta_e - self.eta_d) / 2.0
        else:
            self.delta_eta: float = float(delta_eta)
            self.eta_e = self.eta_i + self.delta_eta
            self.eta_d = self.eta_i - self.delta_eta

        if not (0.0 < self.eta_d < self.eta_i < self.eta_e < 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < eta_d < eta_i < eta_e < 1, got "
                f"({self.eta_d}, {self.eta_i}, {self.eta_e})"
            )

        self.proj_e = HeavisideProjection(beta=self.beta, eta=self.eta_e)
        self.proj_i = HeavisideProjection(beta=self.beta, eta=self.eta_i)
        self.proj_d = HeavisideProjection(beta=self.beta, eta=self.eta_d)

    def project(
        self, rho_tilde: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project filtered field to eroded, intermediate, and dilated fields.

        Parameters
        ----------
        rho_tilde : np.ndarray
            Filtered density array of shape (N_elem,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (rho_hat_e, rho_hat_i, rho_hat_d).
        """
        rho = np.asarray(rho_tilde, dtype=np.float64)
        rho_e = self.proj_e.forward(rho)
        rho_i = self.proj_i.forward(rho)
        rho_d = self.proj_d.forward(rho)
        return rho_e, rho_i, rho_d

    def derivatives(
        self, rho_tilde: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute exact derivatives d(rho_hat)/d(rho_tilde) for all three fields.

        Parameters
        ----------
        rho_tilde : np.ndarray
            Filtered density array of shape (N_elem,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (d_rho_e, d_rho_i, d_rho_d).
        """
        rho = np.asarray(rho_tilde, dtype=np.float64)
        d_e = self.proj_e.derivative(rho)
        d_i = self.proj_i.derivative(rho)
        d_d = self.proj_d.derivative(rho)
        return d_e, d_i, d_d

    def backward(
        self,
        df_drho_e: Optional[np.ndarray],
        df_drho_i: Optional[np.ndarray],
        df_drho_d: Optional[np.ndarray],
        rho_tilde: np.ndarray,
    ) -> np.ndarray:
        """Adjoint vector-Jacobian product sum over all active projection fields.

        Parameters
        ----------
        df_drho_e : Optional[np.ndarray]
            Sensitivity with respect to eroded field.
        df_drho_i : Optional[np.ndarray]
            Sensitivity with respect to intermediate field.
        df_drho_d : Optional[np.ndarray]
            Sensitivity with respect to dilated field.
        rho_tilde : np.ndarray
            Filtered density field.

        Returns
        -------
        np.ndarray
            Sensitivity with respect to filtered field df/drho_tilde.
        """
        rho = np.asarray(rho_tilde, dtype=np.float64)
        grad = np.zeros_like(rho)
        d_e, d_i, d_d = self.derivatives(rho)

        if df_drho_e is not None:
            grad += np.asarray(df_drho_e, dtype=np.float64) * d_e
        if df_drho_i is not None:
            grad += np.asarray(df_drho_i, dtype=np.float64) * d_i
        if df_drho_d is not None:
            grad += np.asarray(df_drho_d, dtype=np.float64) * d_d

        return grad


def _build_1d_gradient_matrix(n: int, h: float) -> sp.csr_matrix:
    """Build 1D central difference matrix with one-sided boundary differences."""
    if n <= 1:
        return sp.csr_matrix((1, 1), dtype=np.float64)

    inv_h = 1.0 / h
    inv_2h = 0.5 / h

    row = []
    col = []
    data = []

    # Left boundary: forward difference (x[1] - x[0]) / h
    row.extend([0, 0])
    col.extend([0, 1])
    data.extend([-inv_h, inv_h])

    # Interior: central difference (x[i+1] - x[i-1]) / (2h)
    for i in range(1, n - 1):
        row.extend([i, i])
        col.extend([i - 1, i + 1])
        data.extend([-inv_2h, inv_2h])

    # Right boundary: backward difference (x[n-1] - x[n-2]) / h
    row.extend([n - 1, n - 1])
    col.extend([n - 2, n - 1])
    data.extend([-inv_h, inv_h])

    mat = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)
    return mat.tocsr()


class GeometricLengthScaleConstraint(ConstraintFunction):
    """Zhou et al. (2015) geometric minimum length scale constraint for solid or void.

    Evaluates structural indicator functions based on gradient norm ||grad rho_tilde||
    and penalizes under-threshold members in the structural skeleton:
        Inflection indicator:
            I(rho_tilde_e) = exp(-c_ls * ||grad rho_tilde_e||^2)
        Solid length scale indicator:
            g_s(rho_tilde) = (1/N) * sum_e [ rho_hat_i * I_e * (min(rho_tilde_e - eta_e, 0))^2 ] - epsilon_ls <= 0
        Void length scale indicator:
            g_v(rho_tilde) = (1/N) * sum_e [ (1 - rho_hat_i) * I_e * (min(eta_d - rho_tilde_e, 0))^2 ] - epsilon_ls <= 0

    Includes exact reverse-mode analytical sensitivities through both projection and
    discrete spatial gradient operators.
    """

    name: str = "geometric_length_scale"
    gradient_source: GradientSource = GradientSource.EXACT_ADJOINT

    def __init__(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        target: str = "solid",
        eta_e: float = 0.75,
        eta_d: float = 0.25,
        eta_i: float = 0.5,
        beta: float = 8.0,
        c_ls: float = 1.0,
        epsilon_ls: float = 1e-4,
    ) -> None:
        """Initialize GeometricLengthScaleConstraint.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational mesh.
        target : str
            Constraint target: 'solid' for minimum solid thickness g_s <= 0,
            or 'void' for minimum void cavity size g_v <= 0.
        eta_e : float
            Eroded threshold (default 0.75).
        eta_d : float
            Dilated threshold (default 0.25).
        eta_i : float
            Intermediate threshold (default 0.5).
        beta : float
            Sharpness parameter for intermediate projection (default 8.0).
        c_ls : float
            Inflection scaling parameter (default 1.0).
        epsilon_ls : float
            Constraint allowable tolerance (default 1e-4).
        """
        target = target.lower()
        if target not in {"solid", "void"}:
            raise ValueError(f"target must be 'solid' or 'void', got '{target}'")

        self.mesh = mesh
        self.target: str = target
        self.eta_e: float = float(eta_e)
        self.eta_d: float = float(eta_d)
        self.eta_i: float = float(eta_i)
        self.beta: float = float(beta)
        self.c_ls: float = float(c_ls)
        self.epsilon_ls: float = float(epsilon_ls)

        self.n_elem: int = mesh.n_elem
        self.proj_i = HeavisideProjection(beta=self.beta, eta=self.eta_i)

        # Build discrete gradient operators D_x, D_y, [D_z]
        self._build_gradient_operators()

    def _build_gradient_operators(self) -> None:
        """Assemble sparse gradient operators Dx, Dy, [Dz] mapping elem vectors to grad."""
        if isinstance(self.mesh, StructuredMesh2D):
            nx, ny = self.mesh.nelx, self.mesh.nely
            dx, dy = self.mesh.dx, self.mesh.dy
            d1x = _build_1d_gradient_matrix(nx, dx)
            d1y = _build_1d_gradient_matrix(ny, dy)
            # Element order: elem_id = ey * nx + ex
            self.Dx = sp.kron(sp.eye(ny, format="csr"), d1x, format="csr")
            self.Dy = sp.kron(d1y, sp.eye(nx, format="csr"), format="csr")
            self.Dz = None
        elif isinstance(self.mesh, StructuredMesh3D):
            nx, ny, nz = self.mesh.nelx, self.mesh.nely, self.mesh.nelz
            dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
            d1x = _build_1d_gradient_matrix(nx, dx)
            d1y = _build_1d_gradient_matrix(ny, dy)
            d1z = _build_1d_gradient_matrix(nz, dz)
            # Element order: elem_id = (ez * ny + ey) * nx + ex
            iy_ix = sp.eye(ny * nx, format="csr")
            iz_iy = sp.eye(nz * ny, format="csr")
            self.Dx = sp.kron(iz_iy, d1x, format="csr")
            self.Dy = sp.kron(sp.kron(sp.eye(nz, format="csr"), d1y), sp.eye(nx, format="csr"), format="csr")
            self.Dz = sp.kron(d1z, iy_ix, format="csr")
        else:
            # Unstructured mesh: least-squares gradient via k-nearest neighbors
            from scipy.spatial import cKDTree

            tree = cKDTree(self.mesh.elem_centers)
            k_neighbors = min(self.n_elem, 2 * self.mesh.ndim + 4)
            _, idxs = tree.query(self.mesh.elem_centers, k=k_neighbors)

            row_x, row_y, row_z = [], [], []
            col_x, col_y, col_z = [], [], []
            val_x, val_y, val_z = [], [], []

            ndim = self.mesh.ndim
            for i in range(self.n_elem):
                nbrs = idxs[i]
                diff = self.mesh.elem_centers[nbrs] - self.mesh.elem_centers[i]
                # Solve least squares (diff^T diff) grad = diff^T (rho_nbrs - rho_i)
                # grad = (diff^T diff)^{-1} diff^T (rho_nbrs - rho_i) = W @ (rho_nbrs - rho_i)
                ata = diff.T @ diff + 1e-10 * np.eye(ndim)
                w = np.linalg.solve(ata, diff.T)  # (ndim, k_neighbors)

                # Contribution of rho_i: -sum_j w[:, j]
                w_self = -np.sum(w, axis=1)

                for j_idx, nbr in enumerate(nbrs):
                    row_x.append(i)
                    col_x.append(nbr)
                    val_x.append(w[0, j_idx])
                    row_y.append(i)
                    col_y.append(nbr)
                    val_y.append(w[1, j_idx])
                    if ndim == 3:
                        row_z.append(i)
                        col_z.append(nbr)
                        val_z.append(w[2, j_idx])

                row_x.append(i)
                col_x.append(i)
                val_x.append(w_self[0])
                row_y.append(i)
                col_y.append(i)
                val_y.append(w_self[1])
                if ndim == 3:
                    row_z.append(i)
                    col_z.append(i)
                    val_z.append(w_self[2])

            self.Dx = sp.coo_matrix((val_x, (row_x, col_x)), shape=(self.n_elem, self.n_elem)).tocsr()
            self.Dy = sp.coo_matrix((val_y, (row_y, col_y)), shape=(self.n_elem, self.n_elem)).tocsr()
            if ndim == 3:
                self.Dz = sp.coo_matrix((val_z, (row_z, col_z)), shape=(self.n_elem, self.n_elem)).tocsr()
            else:
                self.Dz = None

    def evaluate(self, rho: np.ndarray, **kwargs: object) -> float:
        """Evaluate scalar constraint violation g(rho) <= 0.

        Parameters
        ----------
        rho : np.ndarray
            Filtered density field array of shape (N_elem,).

        Returns
        -------
        float
            Constraint value. Feasible when <= 0.
        """
        rho = np.asarray(rho, dtype=np.float64)
        rho_i = self.proj_i.forward(rho)

        # Gradient norm
        gx = self.Dx @ rho
        gy = self.Dy @ rho
        grad_norm_sq = gx**2 + gy**2
        if self.Dz is not None:
            gz = self.Dz @ rho
            grad_norm_sq += gz**2

        # Inflection indicator I = exp(-c_ls * ||grad||^2)
        I = np.exp(-self.c_ls * grad_norm_sq)

        if self.target == "solid":
            # Solid violation: rho_hat_i * I * (min(rho - eta_e, 0))^2
            diff = np.minimum(rho - self.eta_e, 0.0)
            viol = rho_i * I * (diff**2)
        else:
            # Void violation: (1 - rho_hat_i) * I * (min(eta_d - rho, 0))^2
            diff = np.minimum(self.eta_d - rho, 0.0)
            viol = (1.0 - rho_i) * I * (diff**2)

        val = float(np.mean(viol) - self.epsilon_ls)
        return val

    def gradient(self, rho: np.ndarray, **kwargs: object) -> np.ndarray:
        """Evaluate exact analytical constraint gradient dg/drho.

        Parameters
        ----------
        rho : np.ndarray
            Filtered density field array of shape (N_elem,).

        Returns
        -------
        np.ndarray
            Sensitivity array of shape (N_elem,).
        """
        rho = np.asarray(rho, dtype=np.float64)
        N = self.n_elem
        inv_N = 1.0 / N

        rho_i = self.proj_i.forward(rho)
        d_rho_i = self.proj_i.derivative(rho)

        gx = self.Dx @ rho
        gy = self.Dy @ rho
        grad_norm_sq = gx**2 + gy**2
        if self.Dz is not None:
            gz = self.Dz @ rho
            grad_norm_sq += gz**2
        else:
            gz = None

        I = np.exp(-self.c_ls * grad_norm_sq)

        if self.target == "solid":
            diff = np.minimum(rho - self.eta_e, 0.0)
            diff_sq = diff**2

            # Direct term w.r.t rho_e:
            # d/drho [ rho_i * I * diff^2 ] = (d_rho_i * I * diff^2) + (rho_i * I * 2 * diff)
            d_direct = inv_N * (d_rho_i * I * diff_sq + 2.0 * rho_i * I * diff)

            # Sensitivity w.r.t I: w_I = (1/N) * rho_i * diff^2
            w_I = inv_N * rho_i * diff_sq
        else:
            diff = np.minimum(self.eta_d - rho, 0.0)
            diff_sq = diff**2

            # Direct term w.r.t rho_e:
            # d/drho [ (1 - rho_i) * I * diff^2 ] = (-d_rho_i * I * diff^2) - (1 - rho_i) * I * 2 * diff
            d_direct = inv_N * (-d_rho_i * I * diff_sq - 2.0 * (1.0 - rho_i) * I * diff)

            # Sensitivity w.r.t I: w_I = (1/N) * (1 - rho_i) * diff^2
            w_I = inv_N * (1.0 - rho_i) * diff_sq

        # dI/dg_x = -2 * c_ls * gx * I
        # Adjoint sensitivities through Dx, Dy, Dz
        scale = -2.0 * self.c_ls * (w_I * I)
        vx = scale * gx
        vy = scale * gy

        grad = d_direct + self.Dx.T @ vx + self.Dy.T @ vy
        if self.Dz is not None and gz is not None:
            vz = scale * gz
            grad += self.Dz.T @ vz

        return grad
