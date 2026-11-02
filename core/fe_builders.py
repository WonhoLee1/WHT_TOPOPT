"""Exact isoparametric finite element builders for 2D, 3D solid, and shell structures.

Implements reference (E=1) stiffness matrices (Ke0), consistent mass matrices (Me0),
strain-displacement matrices (B0), and constitutive elasticity tensors (C0).
Includes Hughes B-bar volumetric locking-free formulation for Q4 and HEX8,
Hammer-Marlowe 4-point quadrature for quadratic TET10, 27-point quadrature for HEX20,
and Bathe-Dvorkin MITC formulation for 3D shells.
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import numpy as np


# ---------------------------------------------------------------------------
# 1. Constitutive Matrices (C0 for E = 1)
# ---------------------------------------------------------------------------

def isotropic_elasticity_2d(nu: float, plane: str = "plane_stress") -> np.ndarray:
    """2D isotropic elasticity matrix for E = 1.

    Voigt strain order: [eps_xx, eps_yy, gamma_xy].

    Parameters
    ----------
    nu : float
        Poisson's ratio (-1.0 < nu < 0.5).
    plane : str
        'plane_stress' or 'plane_strain'.

    Returns
    -------
    np.ndarray
        (3, 3) symmetric constitutive matrix C0.
    """
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")

    if plane.lower() == "plane_stress":
        factor = 1.0 / (1.0 - nu**2)
        C0 = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ], dtype=np.float64)
    elif plane.lower() == "plane_strain":
        factor = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C0 = factor * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown plane type '{plane}'. Must be 'plane_stress' or 'plane_strain'")
    return C0


def isotropic_elasticity_3d(nu: float) -> np.ndarray:
    """3D isotropic elasticity matrix for E = 1.

    Voigt strain order: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx].

    Parameters
    ----------
    nu : float
        Poisson's ratio (-1.0 < nu < 0.5).

    Returns
    -------
    np.ndarray
        (6, 6) symmetric constitutive matrix C0.
    """
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")

    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = 1.0 / (2.0 * (1.0 + nu))

    C0 = np.zeros((6, 6), dtype=np.float64)
    C0[:3, :3] = lam
    C0[0, 0] = lam + 2.0 * mu
    C0[1, 1] = lam + 2.0 * mu
    C0[2, 2] = lam + 2.0 * mu
    C0[3, 3] = mu
    C0[4, 4] = mu
    C0[5, 5] = mu
    return C0


# ---------------------------------------------------------------------------
# 2. 2D Elements: TRI3 and Q4
# ---------------------------------------------------------------------------

def tri3_ke0_me0(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    plane: str = "plane_stress",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact reference Ke0 (6x6), Me0 (6x6), and centroid B0 (3x6) for 2D TRI3.

    Parameters
    ----------
    coords : np.ndarray
        Shape (3, 2) corner node coordinates in CCW order.
    nu : float
        Poisson's ratio.
    thickness : float
        Element thickness.
    plane : str
        'plane_stress' or 'plane_strain'.

    Returns
    -------
    Ke0 : np.ndarray
        (6, 6) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (6, 6) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (3, 6) constant strain-displacement matrix.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # Determinant of Jacobian (twice the triangle area)
    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    if two_area <= 0.0:
        raise ValueError(f"Degenerate or clockwise TRI3 element, 2*Area = {two_area}")
    area = 0.5 * two_area

    # Shape function derivatives dN/dx and dN/dy
    # N1 = ((x2*y3 - x3*y2) + (y2 - y3)*x + (x3 - x2)*y) / (2*Area)
    b0 = y[1] - y[2]
    b1 = y[2] - y[0]
    b2 = y[0] - y[1]

    c0 = x[2] - x[1]
    c1 = x[0] - x[2]
    c2 = x[1] - x[0]

    B0 = (1.0 / two_area) * np.array([
        [b0, 0.0, b1, 0.0, b2, 0.0],
        [0.0, c0, 0.0, c1, 0.0, c2],
        [c0, b0, c1, b1, c2, b2],
    ], dtype=np.float64)

    C0 = isotropic_elasticity_2d(nu, plane=plane)
    vol = area * thickness
    Ke0 = vol * (B0.T @ C0 @ B0)

    # Consistent mass matrix (rho = 1)
    # Integral of Ni*Nj over triangle is Area/12 for i!=j, Area/6 for i==j
    m_block = (vol / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ], dtype=np.float64)
    Me0 = np.kron(m_block, np.eye(2, dtype=np.float64))

    return Ke0, Me0, B0


def tri3_kg0_tensor(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    plane: str = "plane_stress",
) -> np.ndarray:
    """Compute (6, 6, 6) geometric stiffness derivative tensor for 2D TRI3.

    KG_tensor[k, i, j] = d(KG_e^0) / d(u_{e,k}) such that for any displacement u_e:
        KG_e^0(u_e) = sum_k u_{e,k} * KG_tensor[k, :, :].
    """
    x = coords[:, 0]
    y = coords[:, 1]
    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    if two_area <= 0.0:
        raise ValueError(f"Degenerate or clockwise TRI3 element, 2*Area = {two_area}")
    area = 0.5 * two_area

    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=np.float64)
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=np.float64)

    B0 = (1.0 / two_area) * np.array([
        [b[0], 0.0, b[1], 0.0, b[2], 0.0],
        [0.0, c[0], 0.0, c[1], 0.0, c[2]],
        [c[0], b[0], c[1], b[1], c[2], b[2]],
    ], dtype=np.float64)

    # dNdx: row a is [b_a / 2A, c_a / 2A]
    dNdx = (1.0 / two_area) * np.column_stack([b, c])

    C0 = isotropic_elasticity_2d(nu, plane=plane)
    vol = area * thickness

    KG_tensor = np.zeros((6, 6, 6), dtype=np.float64)
    for k in range(6):
        ek = np.zeros(6, dtype=np.float64)
        ek[k] = 1.0
        sig = C0 @ (B0 @ ek)
        S = np.array([[sig[0], sig[2]], [sig[2], sig[1]]], dtype=np.float64)
        for a in range(3):
            for b_node in range(3):
                val = float(dNdx[a] @ S @ dNdx[b_node])
                KG_tensor[k, 2 * a, 2 * b_node] += vol * val
                KG_tensor[k, 2 * a + 1, 2 * b_node + 1] += vol * val

    return KG_tensor


def _q4_dN(xi: float, eta: float) -> np.ndarray:
    """Shape function derivatives (4, 2) for Q4 at (xi, eta)."""
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    dN = np.empty((4, 2), dtype=np.float64)
    dN[:, 0] = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * eta)
    dN[:, 1] = 0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi)
    return dN


def _q4_B_matrix(dNdx: np.ndarray) -> np.ndarray:
    """Build (3, 8) strain-displacement matrix from shape function derivatives (4, 2)."""
    B = np.zeros((3, 8), dtype=np.float64)
    for i in range(4):
        gx, gy = dNdx[i, 0], dNdx[i, 1]
        c = 2 * i
        B[0, c] = gx
        B[1, c + 1] = gy
        B[2, c] = gy
        B[2, c + 1] = gx
    return B


def q4_ke0_me0(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    plane: str = "plane_stress",
    formulation: str = "FI",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference Ke0 (8x8), Me0 (8x8), and centroid B0 (3x8) for 2D Q4.

    Supports:
        - formulation='FI': Full 2x2 Gauss quadrature integration.
        - formulation='BBAR': Hughes selective volumetric B-bar integration.

    Parameters
    ----------
    coords : np.ndarray
        Shape (4, 2) corner node coordinates in CCW order.
    nu : float
        Poisson's ratio.
    thickness : float
        Element thickness.
    plane : str
        'plane_stress' or 'plane_strain'.
    formulation : str
        'FI' (Full Integration) or 'BBAR' (Hughes B-bar volumetric locking-free).

    Returns
    -------
    Ke0 : np.ndarray
        (8, 8) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (8, 8) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (3, 8) strain-displacement matrix at centroid (0, 0).
    """
    C0 = isotropic_elasticity_2d(nu, plane=plane)

    # 1. Centroid B-matrix at (0, 0)
    dN0 = _q4_dN(0.0, 0.0)
    J0 = dN0.T @ coords
    detJ0 = np.linalg.det(J0)
    if detJ0 <= 0.0:
        raise ValueError(f"Degenerate or inverted Q4 element, detJ at centroid = {detJ0}")
    dNdx0 = np.linalg.solve(J0, dN0.T).T
    B0 = _q4_B_matrix(dNdx0)

    # Dilatational row at center for B-bar: eps_vol = 0.5 * (eps_xx + eps_yy)
    B0_vol = 0.5 * (B0[0, :] + B0[1, :])  # shape (8,)

    # 2. Gauss quadrature (2x2)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    Ke0 = np.zeros((8, 8), dtype=np.float64)
    Me0 = np.zeros((8, 8), dtype=np.float64)

    use_bbar = formulation.upper() == "BBAR"

    for xi, eta in gauss_pts:
        dN = _q4_dN(xi, eta)
        J = dN.T @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Inverted Q4 element at Gauss point ({xi}, {eta}), detJ = {detJ}")
        dNdx = np.linalg.solve(J, dN.T).T
        B = _q4_B_matrix(dNdx)

        if use_bbar:
            # Hughes B-bar: replace volumetric dilation with centroid dilation
            # B_bar[0] = B[0] + (B0_vol - 0.5*(B[0] + B[1]))
            # B_bar[1] = B[1] + (B0_vol - 0.5*(B[0] + B[1]))
            # B_bar[2] = B[2] (shear unchanged)
            diff_vol = B0_vol - 0.5 * (B[0, :] + B[1, :])
            B_eff = B.copy()
            B_eff[0, :] += diff_vol
            B_eff[1, :] += diff_vol
        else:
            B_eff = B

        w_det = detJ * thickness
        Ke0 += (B_eff.T @ C0 @ B_eff) * w_det

        # Consistent mass matrix (rho = 1)
        # N(xi, eta) = 0.25 * (1 + signs[:, 0]*xi) * (1 + signs[:, 1]*eta)
        signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        N = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
        N_mat = np.kron(N[:, None], np.eye(2, dtype=np.float64)).reshape(8, 2).T  # (2, 8)
        Me0 += (N_mat.T @ N_mat) * w_det

    return Ke0, Me0, B0


def q4_kg0_tensor(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    plane: str = "plane_stress",
) -> np.ndarray:
    """Compute (8, 8, 8) geometric stiffness derivative tensor for 2D Q4.

    Evaluated via full 2x2 Gauss quadrature.
    KG_tensor[k, i, j] = d(KG_e^0) / d(u_{e,k}) such that for any displacement u_e:
        KG_e^0(u_e) = sum_k u_{e,k} * KG_tensor[k, :, :].
    """
    C0 = isotropic_elasticity_2d(nu, plane=plane)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    KG_tensor = np.zeros((8, 8, 8), dtype=np.float64)

    for k in range(8):
        ek = np.zeros(8, dtype=np.float64)
        ek[k] = 1.0
        for xi, eta in gauss_pts:
            dN = _q4_dN(xi, eta)
            J = dN.T @ coords
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Inverted Q4 element at Gauss point ({xi}, {eta}), detJ = {detJ}")
            dNdx = np.linalg.solve(J, dN.T).T
            B = _q4_B_matrix(dNdx)
            sig = C0 @ (B @ ek)
            S = np.array([[sig[0], sig[2]], [sig[2], sig[1]]], dtype=np.float64)
            w = detJ * thickness
            for a in range(4):
                for b in range(4):
                    val = float(dNdx[a] @ S @ dNdx[b])
                    KG_tensor[k, 2 * a, 2 * b] += w * val
                    KG_tensor[k, 2 * a + 1, 2 * b + 1] += w * val

    return KG_tensor


# ---------------------------------------------------------------------------
# 3. 3D Solid Elements: TET4, TET10, HEX8, HEX20
# ---------------------------------------------------------------------------

_TET4_DN = np.array([
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def _b_matrix_3d(dNdx: np.ndarray) -> np.ndarray:
    """Build (6, 3*k) strain-displacement matrix from shape function gradients (k, 3)."""
    k = dNdx.shape[0]
    B = np.zeros((6, 3 * k), dtype=np.float64)
    for a in range(k):
        gx, gy, gz = dNdx[a, 0], dNdx[a, 1], dNdx[a, 2]
        c = 3 * a
        B[0, c] = gx
        B[1, c + 1] = gy
        B[2, c + 2] = gz
        B[3, c], B[3, c + 1] = gy, gx
        B[4, c + 1], B[4, c + 2] = gz, gy
        B[5, c], B[5, c + 2] = gz, gx
    return B


def tet4_ke0_me0(coords: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact reference Ke0 (12x12), Me0 (12x12), and constant B0 (6x12) for 3D TET4.

    Parameters
    ----------
    coords : np.ndarray
        Shape (4, 3) corner node coordinates.
    nu : float
        Poisson's ratio.

    Returns
    -------
    Ke0 : np.ndarray
        (12, 12) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (12, 12) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (6, 12) constant strain-displacement matrix.
    """
    C0 = isotropic_elasticity_3d(nu)
    J = _TET4_DN.T @ coords
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(f"Inverted or degenerate TET4 element, detJ = {detJ}")

    dNdx = np.linalg.solve(J, _TET4_DN.T).T
    B0 = _b_matrix_3d(dNdx)
    vol = detJ / 6.0
    Ke0 = vol * (B0.T @ C0 @ B0)

    # Consistent mass matrix (rho = 1)
    # Integral of Ni*Nj over tetrahedron is V/20 for i!=j, V/10 for i==j
    m_block = (vol / 20.0) * np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0],
    ], dtype=np.float64)
    Me0 = np.kron(m_block, np.eye(3, dtype=np.float64))

    return Ke0, Me0, B0


def tet4_kg0_tensor(coords: np.ndarray, nu: float) -> np.ndarray:
    """Compute (12, 12, 12) geometric stiffness derivative tensor for 3D TET4.

    Evaluated analytically via 1-point centroid integration (exact on linear tet).
    KG_tensor[k, i, j] = d(KG_e^0) / d(u_{e,k}) such that for any displacement u_e:
        KG_e^0(u_e) = sum_k u_{e,k} * KG_tensor[k, :, :].
    """
    C0 = isotropic_elasticity_3d(nu)
    J = _TET4_DN.T @ coords
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(f"Inverted or degenerate TET4 element, detJ = {detJ}")

    dNdx = np.linalg.solve(J, _TET4_DN.T).T  # (4, 3)
    B0 = _b_matrix_3d(dNdx)                 # (6, 12)
    vol = detJ / 6.0

    KG_tensor = np.zeros((12, 12, 12), dtype=np.float64)

    for k in range(12):
        ek = np.zeros(12, dtype=np.float64)
        ek[k] = 1.0
        sig = C0 @ (B0 @ ek)
        # Voigt: [xx, yy, zz, xy, yz, zx]
        S = np.array([
            [sig[0], sig[3], sig[5]],
            [sig[3], sig[1], sig[4]],
            [sig[5], sig[4], sig[2]],
        ], dtype=np.float64)
        for a in range(4):
            for b in range(4):
                val = float(dNdx[a] @ S @ dNdx[b])
                KG_tensor[k, 3 * a, 3 * b] += vol * val
                KG_tensor[k, 3 * a + 1, 3 * b + 1] += vol * val
                KG_tensor[k, 3 * a + 2, 3 * b + 2] += vol * val

    return KG_tensor


def tet4_nodal_smoothed_strains(
    node_coords: np.ndarray,
    elements: np.ndarray,
    elem_strains: np.ndarray,
    elem_volumes: np.ndarray,
) -> np.ndarray:
    """Nodal strain averaging helper for Edge-based/Node-based Smoothed FEM (ESNS).

    Averages constant element strains onto mesh nodes weighted by element volume.

    Parameters
    ----------
    node_coords : np.ndarray
        (n_nodes, 3) coordinates.
    elements : np.ndarray
        (n_elem, 4) TET4 connectivity.
    elem_strains : np.ndarray
        (n_elem, 6) element strains.
    elem_volumes : np.ndarray
        (n_elem,) element volumes.

    Returns
    -------
    nodal_strains : np.ndarray
        (n_nodes, 6) volume-weighted smoothed nodal strains.
    """
    n_nodes = node_coords.shape[0]
    nodal_strains = np.zeros((n_nodes, 6), dtype=np.float64)
    nodal_weights = np.zeros(n_nodes, dtype=np.float64)

    for i in range(4):
        n_idx = elements[:, i]
        v = elem_volumes * 0.25
        np.add.at(nodal_weights, n_idx, v)
        for s in range(6):
            np.add.at(nodal_strains[:, s], n_idx, elem_strains[:, s] * v)

    nonzero = nodal_weights > 0.0
    nodal_strains[nonzero] /= nodal_weights[nonzero, None]
    return nodal_strains


def _tet10_shape_derivs(r: float, s: float, t: float) -> np.ndarray:
    """Shape function derivatives (10, 3) [dN/dr, dN/ds, dN/dt] for 10-node TET10.

    Natural coordinates:
        Node 0: (0, 0, 0) -> L1 = 1 - r - s - t
        Node 1: (1, 0, 0) -> L2 = r
        Node 2: (0, 1, 0) -> L3 = s
        Node 3: (0, 0, 1) -> L4 = t
    """
    L1 = 1.0 - r - s - t
    L2 = r
    L3 = s
    L4 = t

    dN = np.zeros((10, 3), dtype=np.float64)

    # Corner nodes: N_i = L_i * (2*L_i - 1)
    # Node 0 (L1):
    dN[0, 0] = 1.0 - 4.0 * L1
    dN[0, 1] = 1.0 - 4.0 * L1
    dN[0, 2] = 1.0 - 4.0 * L1

    # Node 1 (L2):
    dN[1, 0] = 4.0 * L2 - 1.0
    dN[1, 1] = 0.0
    dN[1, 2] = 0.0

    # Node 2 (L3):
    dN[2, 0] = 0.0
    dN[2, 1] = 4.0 * L3 - 1.0
    dN[2, 2] = 0.0

    # Node 3 (L4):
    dN[3, 0] = 0.0
    dN[3, 1] = 0.0
    dN[3, 2] = 4.0 * L4 - 1.0

    # Midside nodes:
    # Node 4 (0-1): 4*L1*L2
    dN[4, 0] = 4.0 * (L1 - L2)
    dN[4, 1] = -4.0 * L2
    dN[4, 2] = -4.0 * L2

    # Node 5 (1-2): 4*L2*L3
    dN[5, 0] = 4.0 * L3
    dN[5, 1] = 4.0 * L2
    dN[5, 2] = 0.0

    # Node 6 (2-0): 4*L1*L3
    dN[6, 0] = -4.0 * L3
    dN[6, 1] = 4.0 * (L1 - L3)
    dN[6, 2] = -4.0 * L3

    # Node 7 (0-3): 4*L1*L4
    dN[7, 0] = -4.0 * L4
    dN[7, 1] = -4.0 * L4
    dN[7, 2] = 4.0 * (L1 - L4)

    # Node 8 (1-3): 4*L2*L4
    dN[8, 0] = 4.0 * L4
    dN[8, 1] = 0.0
    dN[8, 2] = 4.0 * L2

    # Node 9 (2-3): 4*L3*L4
    dN[9, 0] = 0.0
    dN[9, 1] = 4.0 * L4
    dN[9, 2] = 4.0 * L3

    return dN


_TET_MASS_RULE = None


def _tet_mass_quadrature():
    """Conical (collapsed Gauss-Legendre) rule on the reference tetrahedron, positive weights, exact to degree 4 (36 points)."""
    global _TET_MASS_RULE
    if _TET_MASS_RULE is None:
        def line(n):
            x, w = np.polynomial.legendre.leggauss(n)
            return 0.5 * (x + 1.0), 0.5 * w

        pts, wts = [], []
        for u, wu in zip(*line(4)):  # (1-u)^2 raises the degree in u by two, (1-v) the degree in v by one
            for v, wv in zip(*line(3)):
                for q, wq in zip(*line(3)):
                    pts.append((u, v * (1.0 - u), q * (1.0 - u) * (1.0 - v)))
                    wts.append(wu * wv * wq * (1.0 - u) ** 2 * (1.0 - v))
        _TET_MASS_RULE = (np.array(pts), np.array(wts))
    return _TET_MASS_RULE


def tet10_ke0_me0(
    coords: np.ndarray,
    nu: float,
    node_order: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference Ke0 (30x30), Me0 (30x30), and centroid B0 (6x30) for 10-node TET10.

    Stiffness: 4-point Hammer-Marlowe quadrature (2nd order exact). Mass: 36-point conical rule (4th order exact).

    Parameters
    ----------
    coords : np.ndarray
        Shape (10, 3) node coordinates.
    nu : float
        Poisson's ratio.
    node_order : str
        'standard' (VTK/Abaqus) or 'hecmw' (FrontISTR Table342).

    Returns
    -------
    Ke0 : np.ndarray
        (30, 30) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (30, 30) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (6, 30) strain-displacement matrix at reference centroid (0.25, 0.25, 0.25).
    """
    if node_order.lower() == "hecmw":
        # FrontISTR Table342 node permutation:
        # HEC-MW order maps to internal standard order: [0, 1, 2, 3, 6, 4, 5, 7, 8, 9]
        perm = [0, 1, 2, 3, 6, 4, 5, 7, 8, 9]
        c = coords[perm]
    else:
        c = coords

    C0 = isotropic_elasticity_3d(nu)

    # 4-point Hammer-Marlowe Gauss quadrature on tetrahedron
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    # 4 quadrature points in natural coordinates (r, s, t):
    gps = [(a, a, a), (b, a, a), (a, b, a), (a, a, b)]
    w = 1.0 / 24.0  # Sum of weights = 1/6 (volume of reference tet)

    Ke0 = np.zeros((30, 30), dtype=np.float64)
    Me0 = np.zeros((30, 30), dtype=np.float64)

    for r, s, t in gps:
        dN = _tet10_shape_derivs(r, s, t)
        J = dN.T @ c
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Inverted TET10 element at Gauss point ({r}, {s}, {t}), detJ = {detJ}")

        dNdx = np.linalg.solve(J, dN.T).T
        B = _b_matrix_3d(dNdx)
        w_det = detJ * w
        Ke0 += (B.T @ C0 @ B) * w_det

    # Consistent mass matrix (rho = 1): N^T N is of degree 4, so the 4-point rule above leaves it rank deficient
    for (r, s, t), w_m in zip(*_tet_mass_quadrature()):
        detJ = np.linalg.det(_tet10_shape_derivs(r, s, t).T @ c)
        L1, L2, L3, L4 = 1.0 - r - s - t, r, s, t
        N = np.array([
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            L4 * (2.0 * L4 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L1 * L3,
            4.0 * L1 * L4,
            4.0 * L2 * L4,
            4.0 * L3 * L4,
        ], dtype=np.float64)
        N_mat = np.kron(N[:, None], np.eye(3, dtype=np.float64)).reshape(30, 3).T
        Me0 += (N_mat.T @ N_mat) * (detJ * w_m)

    # Centroid B0 at r = s = t = 0.25 (L1 = L2 = L3 = L4 = 0.25)
    dN0 = _tet10_shape_derivs(0.25, 0.25, 0.25)
    J0 = dN0.T @ c
    dNdx0 = np.linalg.solve(J0, dN0.T).T
    B0 = _b_matrix_3d(dNdx0)

    if node_order.lower() == "hecmw":
        # Permute DOFs back to match original HEC-MW node ordering
        dof_perm = []
        for p in np.argsort(perm):
            dof_perm.extend([3 * p, 3 * p + 1, 3 * p + 2])
        Ke0 = Ke0[np.ix_(dof_perm, dof_perm)]
        Me0 = Me0[np.ix_(dof_perm, dof_perm)]
        B0 = B0[:, dof_perm]

    return Ke0, Me0, B0


_HEX8_SIGNS = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
], dtype=float)


def _hex8_dN(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape function derivatives (8, 3) for HEX8 at (xi, eta, zeta)."""
    s = _HEX8_SIGNS
    dN = np.empty((8, 3), dtype=np.float64)
    dN[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    dN[:, 1] = 0.125 * s[:, 1] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 2] * zeta)
    dN[:, 2] = 0.125 * s[:, 2] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta)
    return dN


def hex8_ke0_me0(
    coords: np.ndarray,
    nu: float,
    formulation: str = "FI",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference Ke0 (24x24), Me0 (24x24), and centroid B0 (6x24) for 3D HEX8.

    Supports:
        - formulation='FI': Full 2x2x2 Gauss quadrature integration.
        - formulation='BBAR': Hughes selective volumetric B-bar integration
          (as in FrontISTR Eq. 4.1 / static_LIB_C3D8.f90).

    Parameters
    ----------
    coords : np.ndarray
        Shape (8, 3) node coordinates.
    nu : float
        Poisson's ratio.
    formulation : str
        'FI' or 'BBAR'.

    Returns
    -------
    Ke0 : np.ndarray
        (24, 24) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (24, 24) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (6, 24) strain-displacement matrix at centroid (0, 0, 0).
    """
    C0 = isotropic_elasticity_3d(nu)

    # 1. Centroid B-matrix at (0, 0, 0)
    dN0 = _hex8_dN(0.0, 0.0, 0.0)
    J0 = dN0.T @ coords
    detJ0 = np.linalg.det(J0)
    if detJ0 <= 0.0:
        raise ValueError(f"Degenerate or inverted HEX8 element, detJ at centroid = {detJ0}")
    dNdx0 = np.linalg.solve(J0, dN0.T).T
    B0 = _b_matrix_3d(dNdx0)

    # Volumetric dilation vector at centroid: 1/3 * (B[0] + B[1] + B[2])
    B0_vol = (1.0 / 3.0) * (B0[0, :] + B0[1, :] + B0[2, :])  # shape (24,)

    # 2. 2x2x2 Gauss integration
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [
        (sx * gp, sy * gp, sz * gp)
        for sz in (-1, 1)
        for sy in (-1, 1)
        for sx in (-1, 1)
    ]

    Ke0 = np.zeros((24, 24), dtype=np.float64)
    Me0 = np.zeros((24, 24), dtype=np.float64)
    use_bbar = formulation.upper() == "BBAR"

    for xi, eta, zeta in gauss_pts:
        dN = _hex8_dN(xi, eta, zeta)
        J = dN.T @ coords
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Inverted HEX8 element at Gauss point ({xi}, {eta}, {zeta}), detJ = {detJ}")
        dNdx = np.linalg.solve(J, dN.T).T
        B = _b_matrix_3d(dNdx)

        if use_bbar:
            # Hughes B-bar: replace volumetric dilation in normal strain rows
            diff_vol = B0_vol - (1.0 / 3.0) * (B[0, :] + B[1, :] + B[2, :])
            B_eff = B.copy()
            B_eff[0, :] += diff_vol
            B_eff[1, :] += diff_vol
            B_eff[2, :] += diff_vol
        else:
            B_eff = B

        Ke0 += (B_eff.T @ C0 @ B_eff) * detJ

        # Consistent mass matrix (rho = 1)
        s = _HEX8_SIGNS
        N = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
        N_mat = np.kron(N[:, None], np.eye(3, dtype=np.float64)).reshape(24, 3).T
        Me0 += (N_mat.T @ N_mat) * detJ

    return Ke0, Me0, B0


def hex8_kg0_tensor(coords: np.ndarray, nu: float) -> np.ndarray:
    """Compute (24, 24, 24) geometric stiffness derivative tensor for 3D HEX8.

    Evaluated via 2x2x2 Gauss quadrature.
    KG_tensor[k, i, j] = d(KG_e^0) / d(u_{e,k}) such that for any displacement u_e:
        KG_e^0(u_e) = sum_k u_{e,k} * KG_tensor[k, :, :].
    """
    C0 = isotropic_elasticity_3d(nu)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [
        (sx * gp, sy * gp, sz * gp)
        for sz in (-1, 1)
        for sy in (-1, 1)
        for sx in (-1, 1)
    ]

    KG_tensor = np.zeros((24, 24, 24), dtype=np.float64)

    for k in range(24):
        ek = np.zeros(24, dtype=np.float64)
        ek[k] = 1.0
        for xi, eta, zeta in gauss_pts:
            dN = _hex8_dN(xi, eta, zeta)
            J = dN.T @ coords
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Inverted HEX8 element at Gauss point ({xi}, {eta}, {zeta}), detJ = {detJ}")
            dNdx = np.linalg.solve(J, dN.T).T
            B = _b_matrix_3d(dNdx)
            sig = C0 @ (B @ ek)
            S = np.array([
                [sig[0], sig[3], sig[5]],
                [sig[3], sig[1], sig[4]],
                [sig[5], sig[4], sig[2]],
            ], dtype=np.float64)
            for a in range(8):
                for b in range(8):
                    val = float(dNdx[a] @ S @ dNdx[b])
                    KG_tensor[k, 3 * a, 3 * b] += detJ * val
                    KG_tensor[k, 3 * a + 1, 3 * b + 1] += detJ * val
                    KG_tensor[k, 3 * a + 2, 3 * b + 2] += detJ * val

    return KG_tensor


# HEX20 Serendipity Shape Functions & Derivatives
_HEX20_CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)

_HEX20_MIDSIDE_NODES = [
    # Bottom face z = -1
    (8, 0, -1, -1), (9, 1, 0, -1), (10, 0, 1, -1), (11, -1, 0, -1),
    # Vertical edges z = 0
    (12, -1, -1, 0), (13, 1, -1, 0), (14, 1, 1, 0), (15, -1, 1, 0),
    # Top face z = +1
    (16, 0, -1, 1), (17, 1, 0, 1), (18, 0, 1, 1), (19, -1, 0, 1),
]


def hex20_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate 20 serendipity shape functions at (xi, eta, zeta)."""
    N = np.zeros(20, dtype=np.float64)

    # 8 Corner nodes
    for i in range(8):
        xi_i, eta_i, zeta_i = _HEX20_CORNER_SIGNS[i]
        xi_0 = xi * xi_i
        eta_0 = eta * eta_i
        zeta_0 = zeta * zeta_i
        N[i] = 0.125 * (1.0 + xi_0) * (1.0 + eta_0) * (1.0 + zeta_0) * (xi_0 + eta_0 + zeta_0 - 2.0)

    # 12 Midside nodes
    for idx, xi_i, eta_i, zeta_i in _HEX20_MIDSIDE_NODES:
        if xi_i == 0:
            N[idx] = 0.25 * (1.0 - xi**2) * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
        elif eta_i == 0:
            N[idx] = 0.25 * (1.0 + xi * xi_i) * (1.0 - eta**2) * (1.0 + zeta * zeta_i)
        elif zeta_i == 0:
            N[idx] = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * (1.0 - zeta**2)
    return N


def hex20_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate (20, 3) derivatives [dN/dxi, dN/deta, dN/dzeta] of HEX20 shape functions."""
    dN = np.zeros((20, 3), dtype=np.float64)

    # 8 Corner nodes
    for i in range(8):
        xi_i, eta_i, zeta_i = _HEX20_CORNER_SIGNS[i]
        xi_0 = xi * xi_i
        eta_0 = eta * eta_i
        zeta_0 = zeta * zeta_i
        term = xi_0 + eta_0 + zeta_0 - 2.0

        dN[i, 0] = 0.125 * xi_i * (1.0 + eta_0) * (1.0 + zeta_0) * term + 0.125 * (1.0 + xi_0) * (1.0 + eta_0) * (1.0 + zeta_0) * xi_i
        dN[i, 1] = 0.125 * eta_i * (1.0 + xi_0) * (1.0 + zeta_0) * term + 0.125 * (1.0 + xi_0) * (1.0 + eta_0) * (1.0 + zeta_0) * eta_i
        dN[i, 2] = 0.125 * zeta_i * (1.0 + xi_0) * (1.0 + eta_0) * term + 0.125 * (1.0 + xi_0) * (1.0 + eta_0) * (1.0 + zeta_0) * zeta_i

    # 12 Midside nodes
    for idx, xi_i, eta_i, zeta_i in _HEX20_MIDSIDE_NODES:
        if xi_i == 0:
            # N = 0.25 * (1 - xi^2) * (1 + eta*eta_i) * (1 + zeta*zeta_i)
            dN[idx, 0] = -0.5 * xi * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
            dN[idx, 1] = 0.25 * (1.0 - xi**2) * eta_i * (1.0 + zeta * zeta_i)
            dN[idx, 2] = 0.25 * (1.0 - xi**2) * (1.0 + eta * eta_i) * zeta_i
        elif eta_i == 0:
            # N = 0.25 * (1 + xi*xi_i) * (1 - eta^2) * (1 + zeta*zeta_i)
            dN[idx, 0] = 0.25 * xi_i * (1.0 - eta**2) * (1.0 + zeta * zeta_i)
            dN[idx, 1] = -0.5 * eta * (1.0 + xi * xi_i) * (1.0 + zeta * zeta_i)
            dN[idx, 2] = 0.25 * (1.0 + xi * xi_i) * (1.0 - eta**2) * zeta_i
        elif zeta_i == 0:
            # N = 0.25 * (1 + xi*xi_i) * (1 + eta*eta_i) * (1 - zeta^2)
            dN[idx, 0] = 0.25 * xi_i * (1.0 + eta * eta_i) * (1.0 - zeta**2)
            dN[idx, 1] = 0.25 * (1.0 + xi * xi_i) * eta_i * (1.0 - zeta**2)
            dN[idx, 2] = -0.5 * zeta * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    return dN


def hex20_ke0_me0(coords: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference Ke0 (60x60), Me0 (60x60), and centroid B0 (6x60) for 20-node HEX20.

    Uses full 3x3x3 = 27-point Gauss quadrature.

    Parameters
    ----------
    coords : np.ndarray
        Shape (20, 3) node coordinates.
    nu : float
        Poisson's ratio.

    Returns
    -------
    Ke0 : np.ndarray
        (60, 60) reference stiffness matrix for E = 1.
    Me0 : np.ndarray
        (60, 60) consistent mass matrix for unit density rho = 1.
    B0 : np.ndarray
        (6, 60) strain-displacement matrix at centroid (0, 0, 0).
    """
    C0 = isotropic_elasticity_3d(nu)

    # 3x3x3 Gauss quadrature
    g1 = np.sqrt(3.0 / 5.0)
    gp_1d = [-g1, 0.0, g1]
    w_1d = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

    Ke0 = np.zeros((60, 60), dtype=np.float64)
    Me0 = np.zeros((60, 60), dtype=np.float64)

    for i, xi in enumerate(gp_1d):
        for j, eta in enumerate(gp_1d):
            for k, zeta in enumerate(gp_1d):
                w = w_1d[i] * w_1d[j] * w_1d[k]
                dN = hex20_shape_derivs(xi, eta, zeta)
                J = dN.T @ coords
                detJ = np.linalg.det(J)
                if detJ <= 0.0:
                    raise ValueError(f"Inverted HEX20 element at Gauss point ({xi}, {eta}, {zeta}), detJ = {detJ}")

                dNdx = np.linalg.solve(J, dN.T).T
                B = _b_matrix_3d(dNdx)
                w_det = detJ * w
                Ke0 += (B.T @ C0 @ B) * w_det

                # Consistent mass matrix (rho = 1)
                N = hex20_shape_functions(xi, eta, zeta)
                N_mat = np.kron(N[:, None], np.eye(3, dtype=np.float64)).reshape(60, 3).T
                Me0 += (N_mat.T @ N_mat) * w_det

    # Centroid B0 at (0, 0, 0)
    dN0 = hex20_shape_derivs(0.0, 0.0, 0.0)
    J0 = dN0.T @ coords
    dNdx0 = np.linalg.solve(J0, dN0.T).T
    B0 = _b_matrix_3d(dNdx0)

    return Ke0, Me0, B0


def q4_kg0_tensor(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    plane: str = "plane_stress",
) -> np.ndarray:
    """Compute (8, 8, 8) geometric stiffness derivative tensor for 2D Q4.

    Evaluated via full 2x2 Gauss quadrature.
    KG_tensor[k, i, j] = d(KG_e^0) / d(u_{e,k}) such that for any displacement u_e:
        KG_e^0(u_e) = sum_k u_{e,k} * KG_tensor[k, :, :].
    """
    C0 = isotropic_elasticity_2d(nu, plane=plane)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    KG_tensor = np.zeros((8, 8, 8), dtype=np.float64)

    for k in range(8):
        ek = np.zeros(8, dtype=np.float64)
        ek[k] = 1.0
        for xi, eta in gauss_pts:
            dN = _q4_dN(xi, eta)
            J = dN.T @ coords
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Inverted Q4 element at Gauss point ({xi}, {eta}), detJ = {detJ}")
            dNdx = np.linalg.solve(J, dN.T).T
            B = _q4_B_matrix(dNdx)
            sig = C0 @ (B @ ek)
            S = np.array([[sig[0], sig[2]], [sig[2], sig[1]]], dtype=np.float64)
            w = detJ * thickness
            for a in range(4):
                for b in range(4):
                    val = float(dNdx[a] @ S @ dNdx[b])
                    KG_tensor[k, 2 * a, 2 * b] += w * val
                    KG_tensor[k, 2 * a + 1, 2 * b + 1] += w * val

    return KG_tensor


# ---------------------------------------------------------------------------
# 4. 3D Shell Elements: MITC3 and MITC4
# ---------------------------------------------------------------------------

def mitc3_ke0_me0(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    drill_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reference Ke0 (18x18) and Me0 (18x18) for 3-node MITC3 shell.

    6 DOFs per node: [u, v, w, rot_x, rot_y, rot_z].
    Combines CST membrane + Mindlin plate bending with mixed tensorial transverse shear
    and Hughes-Brezzi drilling DOF stabilization.

    Parameters
    ----------
    coords : np.ndarray
        Shape (3, 3) corner node coordinates.
    nu : float
        Poisson's ratio.
    thickness : float
        Shell thickness.
    drill_alpha : float
        Drilling DOF stabilization factor (default 1.0).

    Returns
    -------
    Ke0 : np.ndarray
        (18, 18) global coordinate shell stiffness matrix for E = 1.
    Me0 : np.ndarray
        (18, 18) consistent mass matrix for unit density rho = 1.
    """
    v0, v1, v2 = coords[0], coords[1], coords[2]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = np.cross(e1, e2)
    norm_len = np.linalg.norm(normal)
    if norm_len <= 1e-14:
        raise ValueError("Degenerate or collinear MITC3 shell element")
    area = 0.5 * norm_len
    ez = normal / norm_len
    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(ez, ex)

    # Local coordinates in element plane
    R = np.vstack([ex, ey, ez])  # (3, 3) local-to-global direction cosines
    loc_coords = (coords - v0) @ R.T  # (3, 3), z-component is 0

    # Local in-plane 2D coordinates
    loc2d = loc_coords[:, :2]
    x, y = loc2d[:, 0], loc2d[:, 1]
    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    b0 = y[1] - y[2]
    b1 = y[2] - y[0]
    b2 = y[0] - y[1]
    c0 = x[2] - x[1]
    c1 = x[0] - x[2]
    c2 = x[1] - x[0]

    # 1. Membrane part (CST): relates [u, v] to [eps_xx, eps_yy, gamma_xy]
    Bm = (1.0 / two_area) * np.array([
        [b0, 0.0, b1, 0.0, b2, 0.0],
        [0.0, c0, 0.0, c1, 0.0, c2],
        [c0, b0, c1, b1, c2, b2],
    ], dtype=np.float64)
    Cm = thickness * isotropic_elasticity_2d(nu, plane="plane_stress")
    Km_loc = area * (Bm.T @ Cm @ Bm)  # (6, 6)

    # 2. Mindlin bending part: relates rotations [rot_x, rot_y] to curvatures
    # Curvatures kappa = [d(rot_y)/dx, -d(rot_x)/dy, d(rot_y)/dy - d(rot_x)/dx]
    Bb = (1.0 / two_area) * np.array([
        [0.0, b0, 0.0, b1, 0.0, b2],
        [-c0, 0.0, -c1, 0.0, -c2, 0.0],
        [-b0, c0, -b1, c1, -b2, c2],
    ], dtype=np.float64)
    Cb = (thickness**3 / 12.0) * isotropic_elasticity_2d(nu, plane="plane_stress")
    Kb_loc = area * (Bb.T @ Cb @ Bb)  # (6, 6)

    # 3. Transverse shear part (Lee & Bathe 2004 MITC3 formulation)
    G = 1.0 / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    Cs = k_shear * G * thickness * np.eye(2, dtype=np.float64)

    dx01 = loc2d[1, 0] - loc2d[0, 0]
    dy01 = loc2d[1, 1] - loc2d[0, 1]
    dx12 = loc2d[2, 0] - loc2d[1, 0]
    dy12 = loc2d[2, 1] - loc2d[1, 1]
    dx02 = loc2d[2, 0] - loc2d[0, 0]
    dy02 = loc2d[2, 1] - loc2d[0, 1]

    # Row vectors for natural shear strains on DOFs [w0, rx0, ry0, w1, rx1, ry1, w2, rx2, ry2]
    g1 = np.zeros(9, dtype=np.float64)
    g1[0] = -1.0; g1[3] = 1.0
    g1[1] = 0.5 * dy01; g1[4] = 0.5 * dy01
    g1[2] = -0.5 * dx01; g1[5] = -0.5 * dx01

    g3 = np.zeros(9, dtype=np.float64)
    g3[0] = -1.0; g3[6] = 1.0
    g3[1] = 0.5 * dy02; g3[7] = 0.5 * dy02
    g3[2] = -0.5 * dx02; g3[8] = -0.5 * dx02

    g2 = np.zeros(9, dtype=np.float64)
    g2[3] = -1.0; g2[6] = 1.0
    g2[4] = 0.5 * dy12; g2[7] = 0.5 * dy12
    g2[5] = -0.5 * dx12; g2[8] = -0.5 * dx12

    c_val = g2 + g1 - g3

    J2 = np.array([[dx01, dy01], [dx02, dy02]], dtype=np.float64)
    invJ = np.linalg.inv(J2)

    Ks_loc = np.zeros((9, 9), dtype=np.float64)
    for r, s in [(1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0)]:
        Bs_nat = np.vstack([g1 + c_val * s, g3 + c_val * r])
        Bs_phys = invJ.T @ Bs_nat
        Ks_loc += (area / 3.0) * (Bs_phys.T @ Cs @ Bs_phys)

    # Assemble local (18, 18) element stiffness matrix
    # Local DOF layout per node: [u, v, w, rot_x, rot_y, rot_z]
    K_loc = np.zeros((18, 18), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            # Membrane: (u, v) -> DOFs (6*i + [0, 1])
            idx_mi = [6 * i, 6 * i + 1]
            idx_mj = [6 * j, 6 * j + 1]
            K_loc[np.ix_(idx_mi, idx_mj)] += Km_loc[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]

            # Bending: (rot_x, rot_y) -> DOFs (6*i + [3, 4])
            idx_bi = [6 * i + 3, 6 * i + 4]
            idx_bj = [6 * j + 3, 6 * j + 4]
            K_loc[np.ix_(idx_bi, idx_bj)] += Kb_loc[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]

            # Shear: (w, rot_x, rot_y) -> DOFs (6*i + [2, 3, 4])
            idx_si = [6 * i + 2, 6 * i + 3, 6 * i + 4]
            idx_sj = [6 * j + 2, 6 * j + 3, 6 * j + 4]
            K_loc[np.ix_(idx_si, idx_sj)] += Ks_loc[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]

    # Hughes-Brezzi drilling stabilization:
    # Penalizes difference between nodal rot_z and continuous in-plane rotation omega_z = 0.5*(dv/dx - du/dy)
    b = [b0, b1, b2]
    c_ = [c0, c1, c2]
    omega_z = np.zeros(18, dtype=np.float64)
    for i in range(3):
        omega_z[6 * i] = -0.5 * b[i] / two_area
        omega_z[6 * i + 1] = 0.5 * c_[i] / two_area

    k_drill = drill_alpha * 1e-3 * G * thickness * area
    for i in range(3):
        d_vec = np.zeros(18, dtype=np.float64)
        d_vec += -omega_z
        d_vec[6 * i + 5] = 1.0
        K_loc += k_drill * np.outer(d_vec, d_vec)

    # Transform from local to global coordinates:
    # T_node = block_diag(R, R) of size (6, 6)
    T_node = np.zeros((6, 6), dtype=np.float64)
    T_node[:3, :3] = R
    T_node[3:, 3:] = R
    T_elem = np.kron(np.eye(3, dtype=np.float64), T_node)  # (18, 18)

    Ke0 = T_elem.T @ K_loc @ T_elem

    # Consistent mass matrix (rho = 1)
    # Mass = area * thickness * rho; Rotational inertia = mass * thickness^2 / 12
    m_node = (area * thickness / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ], dtype=np.float64)
    I_rot = thickness**2 / 12.0
    diag_inertia = np.array([1.0, 1.0, 1.0, I_rot, I_rot, I_rot], dtype=np.float64)

    M_loc = np.zeros((18, 18), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M_loc[6 * i : 6 * i + 6, 6 * j : 6 * j + 6] = np.diag(m_node[i, j] * diag_inertia)
    Me0 = T_elem.T @ M_loc @ T_elem

    return Ke0, Me0


def mitc4_ke0_me0(
    coords: np.ndarray,
    nu: float,
    thickness: float = 1.0,
    drill_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reference Ke0 (24x24) and Me0 (24x24) for 4-node MITC4 shell.

    6 DOFs per node: [u, v, w, rot_x, rot_y, rot_z].
    Combines bilinear Q4 membrane + Mindlin bending with Bathe-Dvorkin MITC4
    transverse shear locking-free tying points and drilling stabilization.

    Parameters
    ----------
    coords : np.ndarray
        Shape (4, 3) corner node coordinates in CCW order.
    nu : float
        Poisson's ratio.
    thickness : float
        Shell thickness.
    drill_alpha : float
        Drilling DOF stabilization factor (default 1.0).

    Returns
    -------
    Ke0 : np.ndarray
        (24, 24) global coordinate shell stiffness matrix for E = 1.
    Me0 : np.ndarray
        (24, 24) consistent mass matrix for unit density rho = 1.
    """
    # 1. Element local coordinate system from diagonals
    d1 = coords[2] - coords[0]
    d2 = coords[3] - coords[1]
    normal = np.cross(d1, d2)
    norm_len = np.linalg.norm(normal)
    if norm_len <= 1e-14:
        raise ValueError("Degenerate or planar collapsed MITC4 shell element")
    ez = normal / norm_len
    ex = d1 / np.linalg.norm(d1)
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])  # (3, 3)

    # Project coordinates to local plane
    center = np.mean(coords, axis=0)
    loc_coords = (coords - center) @ R.T
    loc2d = loc_coords[:, :2]

    # Constitutive matrices
    Cm = thickness * isotropic_elasticity_2d(nu, plane="plane_stress")
    Cb = (thickness**3 / 12.0) * isotropic_elasticity_2d(nu, plane="plane_stress")
    G = 1.0 / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    Cs = k_shear * G * thickness * np.eye(2, dtype=np.float64)

    # 2x2 Gauss integration on local plane
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    K_loc = np.zeros((24, 24), dtype=np.float64)
    total_area = 0.0

    for xi, eta in gauss_pts:
        dN = _q4_dN(xi, eta)
        J2 = dN.T @ loc2d  # (2, 2)
        detJ = np.linalg.det(J2)
        if detJ <= 0.0:
            raise ValueError(f"Inverted local MITC4 geometry, detJ = {detJ}")
        total_area += detJ
        invJ = np.linalg.inv(J2)
        dNdx = (invJ @ dN.T).T  # (4, 2)

        # Membrane B-matrix (3, 8) on [u0, v0, u1, v1, ...]
        Bm = _q4_B_matrix(dNdx)
        Km_gp = (Bm.T @ Cm @ Bm) * detJ

        # Bending B-matrix (3, 8) on [rot_x0, rot_y0, rot_x1, rot_y1, ...]
        # kappa = [d(rot_y)/dx, -d(rot_x)/dy, d(rot_y)/dy - d(rot_x)/dx]
        Bb = np.zeros((3, 8), dtype=np.float64)
        for i in range(4):
            gx, gy = dNdx[i, 0], dNdx[i, 1]
            c = 2 * i
            Bb[0, c + 1] = gx
            Bb[1, c] = -gy
            Bb[2, c] = -gx
            Bb[2, c + 1] = gy
        Kb_gp = (Bb.T @ Cb @ Bb) * detJ

        # MITC4 Transverse Shear Tying Interpolation (Bathe & Dvorkin 1985)
        # Natural transverse shear strains at tying points A, B, C, D:
        # A: (0, -1), B: (1, 0), C: (0, 1), D: (-1, 0)
        # gamma_rz = 0.5 * (1 - eta) * gamma_rz_A + 0.5 * (1 + eta) * gamma_rz_C
        # gamma_sz = 0.5 * (1 - xi) * gamma_sz_D + 0.5 * (1 + xi) * gamma_sz_B
        # Map to physical: [gamma_xz, gamma_yz]^T = inv(J)^T * [gamma_rz, gamma_sz]^T
        Bs_nat = np.zeros((2, 12), dtype=np.float64)
        # w-components: index 3*i, rot_x: 3*i + 1, rot_y: 3*i + 2
        # Tying points edge differences:
        # A (edge 0-1, eta=-1): gamma_rz_A = 0.5*(w1 - w0) - 0.25*(x1 - x0)*(rot_y0 + rot_y1) + 0.25*(y1 - y0)*(rot_x0 + rot_x1)
        dx_01 = loc2d[1, 0] - loc2d[0, 0]
        dy_01 = loc2d[1, 1] - loc2d[0, 1]
        dx_32 = loc2d[2, 0] - loc2d[3, 0]
        dy_32 = loc2d[2, 1] - loc2d[3, 1]
        dx_03 = loc2d[3, 0] - loc2d[0, 0]
        dy_03 = loc2d[3, 1] - loc2d[0, 1]
        dx_12 = loc2d[2, 0] - loc2d[1, 0]
        dy_12 = loc2d[2, 1] - loc2d[1, 1]

        hA = 0.5 * (1.0 - eta)
        hC = 0.5 * (1.0 + eta)
        hD = 0.5 * (1.0 - xi)
        hB = 0.5 * (1.0 + xi)

        # Edge A (0-1):
        Bs_nat[0, 0] += -0.5 * hA
        Bs_nat[0, 3] += 0.5 * hA
        Bs_nat[0, 1] += 0.25 * dy_01 * hA
        Bs_nat[0, 4] += 0.25 * dy_01 * hA
        Bs_nat[0, 2] += -0.25 * dx_01 * hA
        Bs_nat[0, 5] += -0.25 * dx_01 * hA

        # Edge C (3-2):
        Bs_nat[0, 9] += -0.5 * hC
        Bs_nat[0, 6] += 0.5 * hC
        Bs_nat[0, 10] += 0.25 * dy_32 * hC
        Bs_nat[0, 7] += 0.25 * dy_32 * hC
        Bs_nat[0, 11] += -0.25 * dx_32 * hC
        Bs_nat[0, 8] += -0.25 * dx_32 * hC

        # Edge D (0-3):
        Bs_nat[1, 0] += -0.5 * hD
        Bs_nat[1, 9] += 0.5 * hD
        Bs_nat[1, 1] += 0.25 * dy_03 * hD
        Bs_nat[1, 10] += 0.25 * dy_03 * hD
        Bs_nat[1, 2] += -0.25 * dx_03 * hD
        Bs_nat[1, 11] += -0.25 * dx_03 * hD

        # Edge B (1-2):
        Bs_nat[1, 3] += -0.5 * hB
        Bs_nat[1, 6] += 0.5 * hB
        Bs_nat[1, 4] += 0.25 * dy_12 * hB
        Bs_nat[1, 7] += 0.25 * dy_12 * hB
        Bs_nat[1, 5] += -0.25 * dx_12 * hB
        Bs_nat[1, 8] += -0.25 * dx_12 * hB

        # Transform natural shear to Cartesian: Bs_phys = invJ.T @ Bs_nat
        Bs_phys = invJ.T @ Bs_nat  # (2, 12)
        Ks_gp = (Bs_phys.T @ Cs @ Bs_phys) * detJ

        # Assemble into K_loc
        for i in range(4):
            for j in range(4):
                # Membrane DOFs: [6*i, 6*i + 1]
                idx_mi = [6 * i, 6 * i + 1]
                idx_mj = [6 * j, 6 * j + 1]
                K_loc[np.ix_(idx_mi, idx_mj)] += Km_gp[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]

                # Bending DOFs: [6*i + 3, 6*i + 4]
                idx_bi = [6 * i + 3, 6 * i + 4]
                idx_bj = [6 * j + 3, 6 * j + 4]
                K_loc[np.ix_(idx_bi, idx_bj)] += Kb_gp[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]

                # Shear DOFs: [6*i + 2, 6*i + 3, 6*i + 4]
                idx_si = [6 * i + 2, 6 * i + 3, 6 * i + 4]
                idx_sj = [6 * j + 2, 6 * j + 3, 6 * j + 4]
                K_loc[np.ix_(idx_si, idx_sj)] += Ks_gp[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]

    # Hughes-Brezzi drilling DOF stabilization on (rot_z_i - omega_z(0)):
    # At centroid (xi=0, eta=0), continuum in-plane rotation is omega_z = 0.5 * (dv/dx - du/dy)
    dN0 = _q4_dN(0.0, 0.0)
    J2_0 = dN0.T @ loc2d
    dNdx0 = (np.linalg.inv(J2_0) @ dN0.T).T  # (4, 2)
    omega_z = np.zeros(24, dtype=np.float64)
    for a in range(4):
        omega_z[6 * a] = -0.5 * dNdx0[a, 1]      # -0.5 * dN_a/dy * u_a
        omega_z[6 * a + 1] = 0.5 * dNdx0[a, 0]   # +0.5 * dN_a/dx * v_a

    k_drill = drill_alpha * 1e-3 * G * thickness * total_area
    for i in range(4):
        d_vec = -omega_z.copy()
        d_vec[6 * i + 5] = 1.0
        K_loc += k_drill * np.outer(d_vec, d_vec)

    # Transform from local to global coordinates:
    T_node = np.zeros((6, 6), dtype=np.float64)
    T_node[:3, :3] = R
    T_node[3:, 3:] = R
    T_elem = np.kron(np.eye(4, dtype=np.float64), T_node)  # (24, 24)

    Ke0 = T_elem.T @ K_loc @ T_elem

    # Consistent mass matrix (rho = 1)
    # 2x2 Gauss integration of N(xi, eta)^T * N(xi, eta)
    I_rot = thickness**2 / 12.0
    diag_inertia = np.array([1.0, 1.0, 1.0, I_rot, I_rot, I_rot], dtype=np.float64)
    M_loc = np.zeros((24, 24), dtype=np.float64)
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    for xi, eta in gauss_pts:
        dN = _q4_dN(xi, eta)
        J2 = dN.T @ loc2d
        detJ = np.linalg.det(J2)
        N = 0.25 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
        m_block = np.outer(N, N) * (detJ * thickness)
        for i in range(4):
            for j in range(4):
                M_loc[6 * i : 6 * i + 6, 6 * j : 6 * j + 6] += np.diag(m_block[i, j] * diag_inertia)

    Me0 = T_elem.T @ M_loc @ T_elem
    return Ke0, Me0
