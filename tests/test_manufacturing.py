"""Comprehensive verification tests for Milestone 4 (Manufacturing Constraints Suite).

Tests exact analytical sensitivities against Central Finite Differences (h = 1e-6)
to relative error < 1e-6 for:
1. ThreeFieldProjection (eroded, intermediate, dilated)
2. GeometricLengthScaleConstraint (solid & void, 2D & 3D)
3. OverhangFilter (Langelaar AM 2D & 3D)
4. MillingFilter (Langelaar multi-axis CNC milling 2D & 3D)
5. CastingFilter (single-draw & parting-plane 2D & 3D)
6. ExtrusionFilter (volume-weighted 3D & 2D)

And tests geometric manufacturability requirements:
- Floating islands eliminated by OverhangFilter
- Trapped voids eliminated by CastingFilter
- Zero within-column variance for ExtrusionFilter (< 1e-14)
- Machine-precision adjoint dot-product identity for ExtrusionFilter (< 1e-14)
"""

from __future__ import annotations

import numpy as np
import pytest

from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D
from wht_topopt.manufacturing import (
    CastingFilter,
    ExtrusionFilter,
    GeometricLengthScaleConstraint,
    MillingFilter,
    OverhangFilter,
    ThreeFieldProjection,
    compute_eta_from_length_scale,
)


def _compute_central_fd_jacobian(func, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Compute numerical Jacobian J_ij = df_i/dx_j using central finite differences."""
    n = len(x)
    f0 = func(x)
    m = len(f0) if isinstance(f0, np.ndarray) else 1
    J_num = np.empty((m, n), dtype=np.float64)

    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        fp = func(xp)
        fm = func(xm)
        diff = (fp - fm) / (2.0 * h)
        if m == 1:
            J_num[0, j] = diff
        else:
            J_num[:, j] = diff

    return J_num if m > 1 else J_num.ravel()


def test_compute_eta_from_length_scale() -> None:
    """Verify Sigmund (2009) length scale threshold formula."""
    r_min = 4.0
    l_min = 4.0
    eta_e, eta_d = compute_eta_from_length_scale(r_min, l_min)
    assert abs(eta_e - 0.75) < 1e-12
    assert abs(eta_d - 0.25) < 1e-12
    assert abs(eta_e + eta_d - 1.0) < 1e-12

    # Smaller feature: eta_e closer to 0.5, eta_d closer to 0.5
    l_min2 = 2.0
    ee2, ed2 = compute_eta_from_length_scale(r_min, l_min2)
    assert 0.5 < ee2 < 0.75
    assert 0.25 < ed2 < 0.5
    assert abs(ee2 + ed2 - 1.0) < 1e-12

    with pytest.raises(ValueError):
        compute_eta_from_length_scale(-1.0, 1.0)
    with pytest.raises(ValueError):
        compute_eta_from_length_scale(2.0, 5.0)


def test_three_field_projection_derivatives() -> None:
    """Verify ThreeFieldProjection exact derivatives against Central Finite Differences."""
    proj = ThreeFieldProjection(beta=6.0, eta_i=0.5, delta_eta=0.25)
    x = np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95], dtype=np.float64)

    pe, pi, pd = proj.project(x)
    de, di, dd = proj.derivatives(x)

    h = 1e-6
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        pe_p, pi_p, pd_p = proj.project(xp)
        pe_m, pi_m, pd_m = proj.project(xm)

        num_e = (pe_p[i] - pe_m[i]) / (2.0 * h)
        num_i = (pi_p[i] - pi_m[i]) / (2.0 * h)
        num_d = (pd_p[i] - pd_m[i]) / (2.0 * h)

        assert abs(de[i] - num_e) < 1e-7
        assert abs(di[i] - num_i) < 1e-7
        assert abs(dd[i] - num_d) < 1e-7

    # Verify adjoint backward vector-Jacobian product
    np.random.seed(101)
    we = np.random.uniform(0.1, 1.0, size=len(x))
    wi = np.random.uniform(0.1, 1.0, size=len(x))
    wd = np.random.uniform(0.1, 1.0, size=len(x))

    grad_anal = proj.backward(we, wi, wd, x)
    grad_num = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        pe_p, pi_p, pd_p = proj.project(xp)
        pe_m, pi_m, pd_m = proj.project(xm)
        diff_J = (
            np.dot(we, pe_p - pe_m)
            + np.dot(wi, pi_p - pi_m)
            + np.dot(wd, pd_p - pd_m)
        ) / (2.0 * h)
        grad_num[i] = diff_J

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


@pytest.mark.parametrize("target", ["solid", "void"])
def test_geometric_length_scale_2d(target: str) -> None:
    """Verify Zhou et al. (2015) 2D geometric constraint sensitivities against FD."""
    mesh = StructuredMesh2D(8, 6, 8.0, 6.0)
    c = GeometricLengthScaleConstraint(mesh, target=target, beta=4.0, c_ls=0.5, epsilon_ls=1e-3)

    np.random.seed(123)
    rho = np.random.uniform(0.15, 0.85, size=mesh.n_elem)

    val = c.evaluate(rho)
    grad_anal = c.gradient(rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = (c.evaluate(rhop) - c.evaluate(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


@pytest.mark.parametrize("target", ["solid", "void"])
def test_geometric_length_scale_3d(target: str) -> None:
    """Verify Zhou et al. (2015) 3D geometric constraint sensitivities against FD."""
    mesh = StructuredMesh3D(5, 4, 3, 5.0, 4.0, 3.0)
    c = GeometricLengthScaleConstraint(mesh, target=target, beta=4.0, c_ls=0.5, epsilon_ls=1e-3)

    np.random.seed(456)
    rho = np.random.uniform(0.15, 0.85, size=mesh.n_elem)

    grad_anal = c.gradient(rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = (c.evaluate(rhop) - c.evaluate(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_overhang_filter_2d_sensitivity() -> None:
    """Verify Langelaar 2D AM overhang filter reverse-sweep sensitivities against FD."""
    mesh = StructuredMesh2D(8, 6, 8.0, 6.0)
    flt = OverhangFilter(mesh, build_axis="y", p=20.0)

    np.random.seed(789)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_overhang_filter_3d_sensitivity() -> None:
    """Verify Langelaar 3D AM overhang filter reverse-sweep sensitivities against FD."""
    mesh = StructuredMesh3D(5, 4, 3, 5.0, 4.0, 3.0)
    flt = OverhangFilter(mesh, build_axis="z", p=20.0)

    np.random.seed(321)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_overhang_filter_floating_island_removal() -> None:
    """Verify that OverhangFilter eliminates unsupported floating islands in mid-air."""
    mesh = StructuredMesh2D(10, 8, 10.0, 8.0)
    flt = OverhangFilter(mesh, build_axis="y", p=20.0)

    # Base layer 0 is entirely void, upper layer 4 has solid elements
    rho = np.zeros(mesh.n_elem)
    nx = mesh.nelx
    # Island elements at y=4
    rho[4 * nx + 4] = 1.0
    rho[4 * nx + 5] = 1.0

    xi = flt.forward(rho)

    # Floating island elements must be pruned (< 0.01)
    assert xi[4 * nx + 4] < 0.01
    assert xi[4 * nx + 5] < 0.01


def test_milling_filter_2d_sensitivity() -> None:
    """Verify Langelaar 2018 2D multi-axis milling filter sensitivities against FD."""
    mesh = StructuredMesh2D(8, 6, 8.0, 6.0)
    flt = MillingFilter(mesh, directions=["+y", "-y", "+x", "-x"], q=20.0)

    np.random.seed(654)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_milling_filter_3d_sensitivity() -> None:
    """Verify Langelaar 2018 3D multi-axis milling filter sensitivities against FD."""
    mesh = StructuredMesh3D(5, 4, 3, 5.0, 4.0, 3.0)
    flt = MillingFilter(mesh, directions=["+z", "-z", "+y", "-y", "+x", "-x"], q=20.0)

    np.random.seed(987)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


@pytest.mark.parametrize("mode", ["single_draw", "parting_plane"])
def test_casting_filter_2d_sensitivity(mode: str) -> None:
    """Verify CastingFilter 2D sensitivities for single-draw and parting-plane modes."""
    mesh = StructuredMesh2D(8, 6, 8.0, 6.0)
    flt = CastingFilter(mesh, draw_axis="y", mode=mode, parting_coord=3.0)

    np.random.seed(111)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_casting_filter_3d_sensitivity() -> None:
    """Verify CastingFilter 3D parting-plane sensitivities against FD."""
    mesh = StructuredMesh3D(5, 4, 4, 5.0, 4.0, 4.0)
    flt = CastingFilter(mesh, draw_axis="z", mode="parting_plane")

    np.random.seed(222)
    rho = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    w = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    grad_anal = flt.backward(w, rho=rho)

    h = 1e-6
    grad_num = np.zeros_like(rho)
    for i in range(mesh.n_elem):
        rhop = rho.copy()
        rhom = rho.copy()
        rhop[i] += h
        rhom[i] -= h
        grad_num[i] = np.dot(w, flt.forward(rhop) - flt.forward(rhom)) / (2.0 * h)

    rel_err = np.linalg.norm(grad_anal - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-7


def test_casting_filter_trapped_void_elimination() -> None:
    """Verify that CastingFilter eliminates trapped undercuts/voids along the draw direction."""
    mesh = StructuredMesh2D(10, 8, 10.0, 8.0)
    flt = CastingFilter(mesh, draw_axis="y", mode="single_draw")

    # Column 5: solid at base (y=0), void in middle (y=2), solid at top (y=7)
    rho = np.ones(mesh.n_elem)
    nx = mesh.nelx
    rho[2 * nx + 5] = 0.0

    xi = flt.forward(rho)

    # Any roof at y > 2 in column 5 must be eliminated (< 0.05) due to monotonicity
    assert xi[7 * nx + 5] < 0.05


def test_extrusion_filter_adjoint_and_fd() -> None:
    """Verify ExtrusionFilter adjoint dot-product identity and FD sensitivity."""
    mesh = StructuredMesh3D(5, 4, 6, 5.0, 4.0, 6.0)
    flt = ExtrusionFilter(mesh, extrusion_axis="z")

    np.random.seed(333)
    x = np.random.uniform(0.2, 0.8, size=mesh.n_elem)
    y = np.random.uniform(0.1, 1.0, size=mesh.n_elem)

    Px = flt.forward(x)
    PTy = flt.backward(y)

    # 1. Exact adjoint dot product identity: <P x, y> == <x, P^T y>
    dot1 = float(np.dot(Px, y))
    dot2 = float(np.dot(x, PTy))
    dot_rel_err = abs(dot1 - dot2) / abs(dot1)
    assert dot_rel_err < 1e-14

    # 2. FD sensitivity check (linear projection: step 1e-5 yields machine precision)
    h = 1e-5
    grad_num = np.zeros_like(x)
    for i in range(mesh.n_elem):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad_num[i] = np.dot(y, flt.forward(xp) - flt.forward(xm)) / (2.0 * h)

    rel_err = np.linalg.norm(PTy - grad_num) / np.linalg.norm(grad_num)
    assert rel_err < 1e-10

    # 3. Variance along extrusion axis must be zero (< 1e-14)
    var_max = flt.variance_along_axis(Px)
    assert var_max < 1e-14


def test_extrusion_filter_2d() -> None:
    """Verify ExtrusionFilter 2D along y-axis."""
    mesh = StructuredMesh2D(6, 8, 6.0, 8.0)
    flt = ExtrusionFilter(mesh, extrusion_axis="y")

    np.random.seed(444)
    x = np.random.uniform(0.1, 0.9, size=mesh.n_elem)
    y = np.random.uniform(0.1, 0.9, size=mesh.n_elem)

    Px = flt.forward(x)
    PTy = flt.backward(y)

    dot1 = float(np.dot(Px, y))
    dot2 = float(np.dot(x, PTy))
    assert abs(dot1 - dot2) / abs(dot1) < 1e-14

    var_max = flt.variance_along_axis(Px)
    assert var_max < 1e-14
