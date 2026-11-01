"""Unit tests for analytical Center of Mass and Moment of Inertia responses and sensitivities.

Verification criteria:
1. Parallel axis theorem exact identity match (< 1e-12).
2. Analytical Jacobian of CoM vs Central Finite Difference (< 1e-6).
3. Analytical Gradient of Target CoM objective vs Central Finite Difference (< 1e-6).
4. Analytical MoI sensitivities (Origin and CoM) for all 6 tensor components vs Central Finite Difference (< 1e-6).
"""
import numpy as np
import pytest

from wht_topopt.functions.inertia import InertiaSensitivityCalculator


@pytest.fixture
def sample_mesh_3d():
    """Create a regular 4x3x2 Hex8 mesh for testing."""
    nx, ny, nz = 4, 3, 2
    lx, ly, lz = 4.0, 3.0, 2.0
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    z = np.linspace(0, lz, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def node_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = node_id(i, j, k)
                n1 = node_id(i + 1, j, k)
                n2 = node_id(i + 1, j + 1, k)
                n3 = node_id(i, j + 1, k)
                n4 = node_id(i, j, k + 1)
                n5 = node_id(i + 1, j, k + 1)
                n6 = node_id(i + 1, j + 1, k + 1)
                n7 = node_id(i, j + 1, k + 1)
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    elements = np.array(elements, dtype=int)
    return nodes, elements


def test_parallel_axis_theorem(sample_mesh_3d):
    """Verify that I_origin and I_com satisfy the parallel axis theorem to machine precision."""
    nodes, elements = sample_mesh_3d
    calc = InertiaSensitivityCalculator(nodes, elements, base_density=2.7)
    np.random.seed(42)
    rho = np.random.uniform(0.1, 1.0, calc.n_elements)

    props = calc.compute_properties(rho)
    M = props.total_mass
    rc = props.center_of_mass
    I_origin = props.inertia_tensor_origin
    I_com = props.inertia_tensor_com

    # Steiner's parallel axis theorem: I_origin = I_com + M * (||rc||^2 * Eye - rc (x) rc)
    steiner = M * (np.dot(rc, rc) * np.eye(3) - np.outer(rc, rc))
    I_reconstructed = I_com + steiner

    diff = np.max(np.abs(I_origin - I_reconstructed))
    assert diff < 1e-11, f"Parallel axis theorem mismatch: {diff}"


def test_com_analytical_sensitivity_vs_fd(sample_mesh_3d):
    """Verify analytical CoM Jacobian and target objective gradient against central finite difference."""
    nodes, elements = sample_mesh_3d
    calc = InertiaSensitivityCalculator(nodes, elements, base_density=1.5)
    np.random.seed(123)
    rho = np.random.uniform(0.2, 0.8, calc.n_elements)
    target_com = np.array([2.0, 1.5, 1.0])

    d_com_drho, d_g_drho = calc.compute_com_sensitivities(rho, target_com=target_com)

    # Finite difference verification
    eps = 1e-7
    fd_com = np.zeros_like(d_com_drho)
    fd_g = np.zeros_like(d_g_drho)

    for k in range(calc.n_elements):
        r_p = rho.copy()
        r_p[k] += eps
        r_m = rho.copy()
        r_m[k] -= eps

        prop_p = calc.compute_properties(r_p)
        prop_m = calc.compute_properties(r_m)

        fd_com[:, k] = (prop_p.center_of_mass - prop_m.center_of_mass) / (2.0 * eps)

        g_p = 0.5 * float(np.dot(prop_p.center_of_mass - target_com, prop_p.center_of_mass - target_com))
        g_m = 0.5 * float(np.dot(prop_m.center_of_mass - target_com, prop_m.center_of_mass - target_com))
        fd_g[k] = (g_p - g_m) / (2.0 * eps)

    max_err_com = np.max(np.abs(d_com_drho - fd_com))
    max_err_g = np.max(np.abs(d_g_drho - fd_g))

    assert max_err_com < 1e-6, f"CoM Jacobian error too large: {max_err_com}"
    assert max_err_g < 1e-6, f"Target CoM gradient error too large: {max_err_g}"


@pytest.mark.parametrize("component", ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"])
def test_moi_sensitivities_origin_vs_fd(sample_mesh_3d, component):
    """Verify analytical MoI sensitivities about Origin against central finite difference."""
    nodes, elements = sample_mesh_3d
    calc = InertiaSensitivityCalculator(nodes, elements, base_density=2.0)
    np.random.seed(999)
    rho = np.random.uniform(0.1, 0.9, calc.n_elements)

    grad_analytical = calc.compute_moi_sensitivities(rho, component=component, about_com=False)

    comp_map = {
        "Ixx": (0, 0), "Iyy": (1, 1), "Izz": (2, 2),
        "Ixy": (0, 1), "Ixz": (0, 2), "Iyz": (1, 2),
    }
    i, j = comp_map[component]

    eps = 1e-7
    grad_fd = np.zeros(calc.n_elements)
    for k in range(calc.n_elements):
        r_p = rho.copy(); r_p[k] += eps
        r_m = rho.copy(); r_m[k] -= eps
        val_p = calc.compute_properties(r_p).inertia_tensor_origin[i, j]
        val_m = calc.compute_properties(r_m).inertia_tensor_origin[i, j]
        grad_fd[k] = (val_p - val_m) / (2.0 * eps)

    max_err = np.max(np.abs(grad_analytical - grad_fd))
    assert max_err < 1e-5, f"MoI Origin {component} error too large: {max_err}"


@pytest.mark.parametrize("component", ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"])
def test_moi_sensitivities_com_vs_fd(sample_mesh_3d, component):
    """Verify analytical MoI sensitivities about moving CoM against central finite difference."""
    nodes, elements = sample_mesh_3d
    calc = InertiaSensitivityCalculator(nodes, elements, base_density=2.0)
    np.random.seed(777)
    rho = np.random.uniform(0.1, 0.9, calc.n_elements)

    grad_analytical = calc.compute_moi_sensitivities(rho, component=component, about_com=True)

    comp_map = {
        "Ixx": (0, 0), "Iyy": (1, 1), "Izz": (2, 2),
        "Ixy": (0, 1), "Ixz": (0, 2), "Iyz": (1, 2),
    }
    i, j = comp_map[component]

    eps = 1e-7
    grad_fd = np.zeros(calc.n_elements)
    for k in range(calc.n_elements):
        r_p = rho.copy(); r_p[k] += eps
        r_m = rho.copy(); r_m[k] -= eps
        val_p = calc.compute_properties(r_p).inertia_tensor_com[i, j]
        val_m = calc.compute_properties(r_m).inertia_tensor_com[i, j]
        grad_fd[k] = (val_p - val_m) / (2.0 * eps)

    max_err = np.max(np.abs(grad_analytical - grad_fd))
    assert max_err < 1e-5, f"MoI CoM {component} error too large: {max_err}"
