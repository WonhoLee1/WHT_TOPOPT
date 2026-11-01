"""Numba JIT (@njit) accelerated computational kernels for wht_topopt.

Implements:
1. Pre-indexed CSC sparse matrix assembly (Ferrari & Sigmund 2020 `top99neo` pattern)
   eliminating per-iteration COO sorting and duplicate summation overhead.
2. Parallel element strain energy (`c_e0 = u_e^T Ke0 u_e`) and von Mises stress evaluation
   with zero intermediate `(n_elem, d)` heap allocations.
3. Parallel geometric stiffness `KG0` and buckling adjoint force evaluation.
4. Compiled layer-by-layer forward and backward sweeps for Langelaar (2016/2017) DfAM Overhang filter.
5. Compiled Godunov upwind signed-distance reinitialization for Hilbertian Level-Set optimization.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import scipy.sparse as sp

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range  # type: ignore[assignment]


@njit(cache=True)
def _build_gather_csr_from_sorted_order(
    sorted_keys: np.ndarray,
    order: np.ndarray,
    n_elem_dof_sq: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build unique CSC entry indices and CSR gather arrays from sorted 1D linear keys."""
    n_total = len(sorted_keys)
    # Count unique keys
    n_unique = 1
    for k in range(1, n_total):
        if sorted_keys[k] != sorted_keys[k - 1]:
            n_unique += 1

    unique_keys = np.empty(n_unique, dtype=np.int64)
    gather_ptr = np.empty(n_unique + 1, dtype=np.int32)
    gather_elem = np.empty(n_total, dtype=np.int32)
    gather_local = np.empty(n_total, dtype=np.int16)

    unique_keys[0] = sorted_keys[0]
    gather_ptr[0] = 0
    u_idx = 0

    for k in range(n_total):
        orig_idx = order[k]
        gather_elem[k] = orig_idx // n_elem_dof_sq
        gather_local[k] = orig_idx % n_elem_dof_sq
        if k > 0 and sorted_keys[k] != sorted_keys[k - 1]:
            u_idx += 1
            unique_keys[u_idx] = sorted_keys[k]
            gather_ptr[u_idx] = k

    gather_ptr[n_unique] = n_total
    return unique_keys, gather_ptr, gather_elem, gather_local


def build_csc_assembly_map(
    iK: np.ndarray,
    jK: np.ndarray,
    ndof: int,
    n_elem_dof_sq: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute CSC structure (indptr, indices) and gather map once per mesh.

    Because CSC stores columns contiguously (column j, then row i), linear key
    `key = j * ndof + i` sorts entries in exact CSC canonical order.
    """
    i_arr = np.asarray(iK, dtype=np.int64).ravel()
    j_arr = np.asarray(jK, dtype=np.int64).ravel()
    keys = j_arr * int(ndof) + i_arr
    order = np.argsort(keys, kind="quicksort").astype(np.int64)
    sorted_keys = keys[order]

    unique_keys, gather_ptr, gather_elem, gather_local = _build_gather_csr_from_sorted_order(
        sorted_keys, order, int(n_elem_dof_sq)
    )
    cols = unique_keys // int(ndof)
    rows = (unique_keys % int(ndof)).astype(np.int32)
    indptr = np.zeros(int(ndof) + 1, dtype=np.int32)
    np.add.at(indptr[1:], cols, 1)
    np.cumsum(indptr, out=indptr)

    return indptr, rows, gather_ptr, gather_elem, gather_local


@njit(parallel=True, fastmath=True, cache=True)
def numba_assemble_csc_uniform(
    E_e: np.ndarray,
    Ke0_flat: np.ndarray,
    gather_ptr: np.ndarray,
    gather_elem: np.ndarray,
    gather_local: np.ndarray,
) -> np.ndarray:
    """Assemble CSC data vector in parallel for uniform-element meshes (zero race conditions)."""
    n_nnz = len(gather_ptr) - 1
    data = np.empty(n_nnz, dtype=np.float64)
    for nz in prange(n_nnz):
        start = gather_ptr[nz]
        end = gather_ptr[nz + 1]
        acc = 0.0
        for p in range(start, end):
            acc += E_e[gather_elem[p]] * Ke0_flat[gather_local[p]]
        data[nz] = acc
    return data


@njit(parallel=True, fastmath=True, cache=True)
def numba_assemble_csc_batch(
    E_e: np.ndarray,
    Ke0_batch_flat: np.ndarray,
    gather_ptr: np.ndarray,
    gather_elem: np.ndarray,
    gather_local: np.ndarray,
) -> np.ndarray:
    """Assemble CSC data vector in parallel for non-uniform / unstructured meshes."""
    n_nnz = len(gather_ptr) - 1
    data = np.empty(n_nnz, dtype=np.float64)
    for nz in prange(n_nnz):
        start = gather_ptr[nz]
        end = gather_ptr[nz + 1]
        acc = 0.0
        for p in range(start, end):
            el = gather_elem[p]
            loc = gather_local[p]
            acc += E_e[el] * Ke0_batch_flat[el, loc]
        data[nz] = acc
    return data


@njit(parallel=True, fastmath=True, cache=True)
def numba_unit_compliance_uniform(
    u: np.ndarray,
    edofMat: np.ndarray,
    Ke0: np.ndarray,
) -> np.ndarray:
    """Compute c_e0 = u_e^T Ke0 u_e in parallel without temporary (n_elem, d) allocations."""
    n_elem = edofMat.shape[0]
    d = edofMat.shape[1]
    c_e0 = np.empty(n_elem, dtype=np.float64)

    for e in prange(n_elem):
        total = 0.0
        for i in range(d):
            ui = u[edofMat[e, i]]
            row_sum = 0.0
            for j in range(d):
                row_sum += Ke0[i, j] * u[edofMat[e, j]]
            total += ui * row_sum
        c_e0[e] = total
    return c_e0


@njit(parallel=True, fastmath=True, cache=True)
def numba_unit_compliance_batch(
    u: np.ndarray,
    edofMat: np.ndarray,
    Ke0_batch: np.ndarray,
) -> np.ndarray:
    """Compute c_e0 = u_e^T Ke0_e u_e in parallel for unstructured meshes."""
    n_elem = edofMat.shape[0]
    d = edofMat.shape[1]
    c_e0 = np.empty(n_elem, dtype=np.float64)

    for e in prange(n_elem):
        total = 0.0
        for i in range(d):
            ui = u[edofMat[e, i]]
            row_sum = 0.0
            for j in range(d):
                row_sum += Ke0_batch[e, i, j] * u[edofMat[e, j]]
            total += ui * row_sum
        c_e0[e] = total
    return c_e0


@njit(parallel=True, fastmath=True, cache=True)
def numba_element_stresses_uniform(
    u: np.ndarray,
    edofMat: np.ndarray,
    DB0: np.ndarray,
    V_vm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute microscopic Cauchy stress (E0 * C0 @ B0 @ u_e) and von Mises stress in parallel."""
    n_elem = edofMat.shape[0]
    d = edofMat.shape[1]
    n_comp = DB0.shape[0]

    cauchy = np.empty((n_elem, n_comp), dtype=np.float64)
    vm = np.empty(n_elem, dtype=np.float64)

    for e in prange(n_elem):
        for s in range(n_comp):
            acc = 0.0
            for j in range(d):
                acc += DB0[s, j] * u[edofMat[e, j]]
            cauchy[e, s] = acc

        vm_sq = 0.0
        for s1 in range(n_comp):
            sig1 = cauchy[e, s1]
            inner = 0.0
            for s2 in range(n_comp):
                inner += V_vm[s1, s2] * cauchy[e, s2]
            vm_sq += sig1 * inner
        vm[e] = np.sqrt(max(0.0, vm_sq))

    return cauchy, vm


@njit(parallel=True, fastmath=True, cache=True)
def numba_element_kg0_uniform(
    u: np.ndarray,
    edofMat: np.ndarray,
    KG_tensor: np.ndarray,
) -> np.ndarray:
    """Compute element geometric stiffness matrices KG0_e(u_e) in parallel."""
    n_elem = edofMat.shape[0]
    d = edofMat.shape[1]
    out = np.empty((n_elem, d, d), dtype=np.float64)

    for e in prange(n_elem):
        for i in range(d):
            for j in range(d):
                acc = 0.0
                for k in range(d):
                    acc += u[edofMat[e, k]] * KG_tensor[k, i, j]
                out[e, i, j] = acc
    return out


@njit(cache=True)
def numba_overhang_forward(
    rho: np.ndarray,
    layer_ptr: np.ndarray,
    layer_elems: np.ndarray,
    supp_ids_flat: np.ndarray,
    supp_mask_flat: np.ndarray,
    n_supp_cols: int,
    p_max: float,
    eps_0: float,
    eps_s: float,
) -> np.ndarray:
    """Compiled layer-by-layer forward sweep for Langelaar (2016/2017) DfAM overhang filter."""
    xi = np.empty_like(rho)
    n_layers = len(layer_ptr) - 1
    inv_p = 1.0 / p_max
    sqrt_eps_s = np.sqrt(eps_s)

    # Layer 0: supported by build plate
    for idx in range(layer_ptr[0], layer_ptr[1]):
        el = layer_elems[idx]
        xi[el] = rho[el]

    # Subsequent layers j = 1 .. n_layers - 1
    for j in range(1, n_layers):
        l_start = layer_ptr[j]
        l_end = layer_ptr[j + 1]
        for idx in range(l_start, l_end):
            el = layer_elems[idx]
            n_valid = 0
            U = 0.0
            for c in range(n_supp_cols):
                if supp_mask_flat[idx, c]:
                    n_valid += 1
                    s_val = xi[supp_ids_flat[idx, c]]
                    U += (s_val + eps_0) ** p_max

            U_pow = U ** inv_p
            S = U_pow - (float(n_valid) ** inv_p) * eps_0

            r_val = rho[el]
            diff = r_val - S
            rad = np.sqrt(diff * diff + eps_s)
            xi[el] = 0.5 * (r_val + S - rad + sqrt_eps_s)

    return xi


@njit(cache=True)
def numba_overhang_backward(
    df_dxi: np.ndarray,
    rho: np.ndarray,
    xi: np.ndarray,
    layer_ptr: np.ndarray,
    layer_elems: np.ndarray,
    supp_ids_flat: np.ndarray,
    supp_mask_flat: np.ndarray,
    n_supp_cols: int,
    p_max: float,
    eps_0: float,
    eps_s: float,
) -> np.ndarray:
    """Compiled reverse-layer adjoint sensitivity propagation for Langelaar DfAM overhang filter."""
    lam = df_dxi.copy()
    df_drho = np.empty_like(df_dxi)
    n_layers = len(layer_ptr) - 1
    inv_p = 1.0 / p_max

    for j in range(n_layers - 1, 0, -1):
        l_start = layer_ptr[j]
        l_end = layer_ptr[j + 1]
        for idx in range(l_start, l_end):
            el = layer_elems[idx]
            n_valid = 0
            U = 0.0
            for c in range(n_supp_cols):
                if supp_mask_flat[idx, c]:
                    n_valid += 1
                    s_val = xi[supp_ids_flat[idx, c]]
                    U += (s_val + eps_0) ** p_max

            U_pow = U ** inv_p
            S = U_pow - (float(n_valid) ** inv_p) * eps_0

            r_val = rho[el]
            diff = r_val - S
            rad = np.sqrt(diff * diff + eps_s)

            dxi_drho = 0.5 * (1.0 - diff / rad)
            dxi_dS = 0.5 * (1.0 + diff / rad)

            lam_cur = lam[el]
            df_drho[el] = lam_cur * dxi_drho
            mu = lam_cur * dxi_dS
            denom = U_pow + 1e-30

            for c in range(n_supp_cols):
                if supp_mask_flat[idx, c]:
                    s_id = supp_ids_flat[idx, c]
                    base = xi[s_id] + eps_0
                    ratio = base / denom
                    lam[s_id] += mu * (ratio ** (p_max - 1.0))

    # Base layer j = 0
    for idx in range(layer_ptr[0], layer_ptr[1]):
        el = layer_elems[idx]
        df_drho[el] = lam[el]

    return df_drho


@njit(cache=True)
def numba_reinit_distance_2d(
    phi_in: np.ndarray,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    n_steps: int,
) -> np.ndarray:
    """Compiled Godunov upwind signed-distance reinitialization on a 2D structured grid."""
    phi = phi_in.reshape((ny, nx)).copy()
    phi_next = np.empty_like(phi)
    s_phi0 = np.empty_like(phi)
    dt_tau = 0.2 * min(dx, dy)

    # Compute smoothed sign function S(phi0)
    for j in range(ny):
        jm = max(0, j - 1)
        jp = min(ny - 1, j + 1)
        scale_y = 1.0 / ((jp - jm) * dy) if jp > jm else 0.0
        for i in range(nx):
            im = max(0, i - 1)
            ip = min(nx - 1, i + 1)
            scale_x = 1.0 / ((ip - im) * dx) if ip > im else 0.0
            gx = (phi[j, ip] - phi[j, im]) * scale_x
            gy = (phi[jp, i] - phi[jm, i]) * scale_y
            gnorm = np.sqrt(gx * gx + gy * gy + 1e-12)
            p0 = phi[j, i]
            s_phi0[j, i] = p0 / np.sqrt(p0 * p0 + (gnorm * dx) ** 2 + 1e-12)

    for _ in range(n_steps):
        for j in range(ny):
            for i in range(nx):
                p_c = phi[j, i]
                dx_m = (p_c - phi[j, i - 1]) / dx if i > 0 else 0.0
                dx_p = (phi[j, i + 1] - p_c) / dx if i < nx - 1 else 0.0
                dy_m = (p_c - phi[j - 1, i]) / dy if j > 0 else 0.0
                dy_p = (phi[j + 1, i] - p_c) / dy if j < ny - 1 else 0.0

                s_val = s_phi0[j, i]
                if s_val > 0.0:
                    ax = max(dx_m, 0.0)
                    bx = min(dx_p, 0.0)
                    ay = max(dy_m, 0.0)
                    by = min(dy_p, 0.0)
                else:
                    ax = min(dx_m, 0.0)
                    bx = max(dx_p, 0.0)
                    ay = min(dy_m, 0.0)
                    by = max(dy_p, 0.0)

                g_sq = ax * ax + bx * bx + ay * ay + by * by
                phi_next[j, i] = p_c - dt_tau * s_val * (np.sqrt(max(g_sq, 1e-12)) - 1.0)

        for j in range(ny):
            for i in range(nx):
                phi[j, i] = phi_next[j, i]

    return phi.ravel()
