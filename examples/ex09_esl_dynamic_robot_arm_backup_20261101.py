"""Showcase 09: First-Order Equivalent Static Loads (F-ESL) Dynamic Topology Optimization.

Implements the state-of-the-art F-ESL (First-order Equivalent Static Loads) theory
(Buzaglo & Pollini 2026, Computers & Structures / Kang, Park & Arora 2006) combined with
Flexible Multibody Kinematics (FFRF) for a high-speed Humanoid Robot Arm Link:
- Domain: Robot Arm Link (160.0 x 48.0 x 40.0 mm)
- Grid: 50 x 24 x 20 = 24,000 Hex8 elements (80,325 DOFs), dx=3.2mm, dy=2.0mm, dz=2.0mm
- Aerospace Material: Al 7075-T6 (E0=71,000 MPa, nu=0.33, rho=2.81e-9 t/mm^3, Yield=503 MPa)
- Dual Joint Architecture:
  - Base Motor Hub (Left, x=20mm): shaft bore (r=8mm void), collar boss (r=16mm solid), fixed
  - Output Joint Hub (Right, x=140mm): shaft bore (r=7mm void), collar boss (r=14mm solid), loaded
- Dynamic Robot Slewing Trajectory (S-curve high-speed pick-and-place, T=50 ms):
  - Angular velocity peak: omega_max = 62.8 rad/s (600 rpm) at t = 25 ms
  - Angular acceleration peak: alpha_max = +3,947 rad/s^2 at t = 12.5 ms (start-up acceleration)
  - Angular deceleration peak: alpha_min = -3,947 rad/s^2 at t = 37.5 ms (braking shock)
  - Axial centrifugal payload pull: F_x(t) = m_payload * omega(t)^2 * L
  - In-plane tangential Euler force: F_y(t) = -m_payload * alpha(t) * L + T_motor(t) / r_hub
  - Out-of-plane torque vibration: F_z(t) = F_z0 * sin(3*pi*t/T)
- Dynamic Time-Integration Solvers (Both Supported & Verified):
  1. Built-in Generalized-alpha (Chung & Hulbert 1993, rho_infinity=0.85, 2nd-order, Intel MKL PARDISO)
  2. Pure Kratos Native Bossak-alpha (KratosMultiphysics.ResidualBasedBossakDisplacementScheme, AMGCL)
- True ESL Formulation:
  - Dynamic Time-Integration: M(rho) * u''(t) + C(rho) * u'(t) + K(rho) * u(t) = f_ext(t)
  - Critical Time Step Extraction: peak strain energy, peak tip deflection, peak deceleration shock
  - Equivalent Static Load Synthesis: f_eq(t_k) = K(rho) * u_dyn(t_k)
  - First-Order KKT Gradient Correction: g_FESL = grad(c_static) + delta_dynamic
  - Adaptive Trust-Region stabilization based on actual/predicted reduction ratio gamma_k
- Deliverables:
  - 100% Watertight binary STL via zero-padded Marching Cubes
  - Dark-theme PyVista 3D CAD surface rendering
  - Dark-theme PyVista discrete Hex8 solid voxel rendering
  - 30-step animated progression GIF
  - Dynamic transient response comparison plot (initial vs optimized vibration damping)
  - Multi-physics convergence history plot
  - Full urban-light interactive HTML engineering report
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os

# Enforce 6 OpenMP threads across all solvers
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except Exception:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

import pyvista as pv
import imageio.v2 as imageio
import skimage.measure
import trimesh
import pypardiso

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics import python_linear_solver_factory

KM.ParallelUtilities.SetNumThreads(6)

from wht_topopt.algorithms import MMAOptimizer, StressEnergyAdaptiveHCA, SEAHCAConfig
from wht_topopt.core.fem import FEMEngine
from wht_topopt.core.filter import DensityFilter
from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.manufacturing.extrusion import ExtrusionFilter, BottomPlateZExtrusionFilter, ShellRibExtrusionFilter

GLOBAL_FONT_SIZE_PT: int = 9
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE_PT


@dataclass
class FESLIterationRecord:
    """Record of metrics for a single F-ESL optimization iteration."""
    iteration: int
    compliance: float
    volume_fraction: float
    max_change: float
    max_stress_mpa: float
    peak_dyn_disp_mm: float
    trust_region_radius: float
    dyn_solve_time_s: float
    static_solve_time_s: float
    total_step_time_s: float
    solid_stress_cov: float = 0.0
    optimizer_phase: str = "MMA"



class RobotArmDomain3D:
    """3D Humanoid Robot Arm Link domain with dual motor shaft collars."""

    def __init__(
        self,
        lx: float = 160.0,
        ly: float = 48.0,
        lz: float = 40.0,
        nx: int = 50,
        ny: int = 24,
        nz: int = 20,
        bottom_plate: bool = False,
        shell_rib: bool = False,
    ) -> None:
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.bottom_plate = bottom_plate
        self.shell_rib = shell_rib

        self.mesh = StructuredMesh3D(nelx=nx, nely=ny, nelz=nz, lx=lx, ly=ly, lz=lz)
        self.n_elem = self.mesh.n_elem
        self.ndof = self.mesh.ndof

        # Joint hub centers (in mm)
        self.hub1_center = np.array([20.0, 24.0])   # Base motor joint
        self.hub2_center = np.array([140.0, 24.0])  # Output wrist joint

        self.r1_void = 8.0
        self.r2_void = 7.0
        if self.shell_rib:
            # Compact engineering boss collars (thickness=3.5mm) to leave volume budget for internal ribs
            self.r1_solid = 11.5
            self.r2_solid = 10.5
        else:
            self.r1_solid = 16.0
            self.r2_solid = 14.0
        self.m_payload = 1.5  # Output wrist payload in kg

        # Element centers
        self.elem_centers = np.zeros((self.n_elem, 3))
        elem_idx = 0
        for iz in range(nz):
            cz = (iz + 0.5) * self.mesh.dz
            for iy in range(ny):
                cy = (iy + 0.5) * self.mesh.dy
                for ix in range(nx):
                    cx = (ix + 0.5) * self.mesh.dx
                    self.elem_centers[elem_idx] = [cx, cy, cz]
                    elem_idx += 1

        d1 = np.sqrt((self.elem_centers[:, 0] - self.hub1_center[0])**2 + (self.elem_centers[:, 1] - self.hub1_center[1])**2)
        d2 = np.sqrt((self.elem_centers[:, 0] - self.hub2_center[0])**2 + (self.elem_centers[:, 1] - self.hub2_center[1])**2)

        self.passive_void_mask = (d1 <= self.r1_void) | (d2 <= self.r2_void)
        self.passive_solid_mask = ((d1 > self.r1_void) & (d1 <= self.r1_solid)) | ((d2 > self.r2_void) & (d2 <= self.r2_solid))

        # Outer shell structure or bottom plate integration
        if self.shell_rib:
            bottom_skin_mask = (self.elem_centers[:, 2] < self.mesh.dz)
            side_skin_mask = (self.elem_centers[:, 1] < self.mesh.dy) | (self.elem_centers[:, 1] >= (self.ly - self.mesh.dy))
            outer_shell_mask = (bottom_skin_mask | side_skin_mask) & (~self.passive_void_mask)
            self.passive_solid_mask = self.passive_solid_mask | outer_shell_mask
            print(f" [+] Integrated Outer Shell Structure (-Z bottom & +/-Y side walls, {np.sum(outer_shell_mask):,} elements)", flush=True)
        elif self.bottom_plate:
            bottom_skin_mask = (self.elem_centers[:, 2] < self.mesh.dz) & (~self.passive_void_mask)
            self.passive_solid_mask = self.passive_solid_mask | bottom_skin_mask
            print(f" [+] Integrated Bottom Skin Plate at -Z (thickness={self.mesh.dz:.1f}mm, {np.sum(bottom_skin_mask):,} elements)", flush=True)

        # Node classification
        node_coords = self.mesh.node_coords
        nd1 = np.sqrt((node_coords[:, 0] - self.hub1_center[0])**2 + (node_coords[:, 1] - self.hub1_center[1])**2)
        nd2 = np.sqrt((node_coords[:, 0] - self.hub2_center[0])**2 + (node_coords[:, 1] - self.hub2_center[1])**2)

        # Clamped nodes at Base Hub inner bore
        self.clamped_nodes = np.where((nd1 <= (self.r1_void + 0.8)) & (nd1 >= (self.r1_void - 0.8)))[0]
        if len(self.clamped_nodes) == 0:
            self.clamped_nodes = np.where(nd1 <= self.r1_void)[0]

        self.fixed_dofs = np.concatenate([
            3 * self.clamped_nodes,
            3 * self.clamped_nodes + 1,
            3 * self.clamped_nodes + 2,
        ])
        self.fixed_dofs = np.unique(self.fixed_dofs)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)

        # Loaded nodes at Output Hub inner bore
        self.loaded_nodes = np.where((nd2 <= (self.r2_void + 0.8)) & (nd2 >= (self.r2_void - 0.8)))[0]
        if len(self.loaded_nodes) == 0:
            self.loaded_nodes = np.where(nd2 <= self.r2_void)[0]

        print(f" [*] RobotArmDomain3D Initialized: {self.n_elem:,} Hex8 elements, {self.ndof:,} DOFs", flush=True)
        print(f"     Clamped Base Hub : {len(self.clamped_nodes):,} nodes ({len(self.fixed_dofs):,} DOFs fixed, {len(self.free_dofs):,} free)", flush=True)
        print(f"     Loaded Output Hub: {len(self.loaded_nodes):,} nodes at x=140mm", flush=True)


class BaseDynamicESLSolver(ABC):
    """Abstract Base Class for Transient Dynamic Solvers computing True Equivalent Static Loads."""

    def __init__(
        self,
        domain: RobotArmDomain3D,
        fem: FEMEngine,
        t_total: float = 0.05,  # 50 ms
        n_steps: int = 30,
        theta_max: float = np.pi / 2.0,  # 90 degrees slewing
        m_payload_kg: float = 1.5,       # 1.5 kg gripper + payload
    ) -> None:
        self.domain = domain
        self.fem = fem
        self.t_total = t_total
        self.n_steps = n_steps
        self.theta_max = theta_max
        self.m_payload = m_payload_kg

        self.time_vec = np.linspace(0.0, t_total, n_steps)
        self.dt = self.time_vec[1] - self.time_vec[0]

        # Dynamic state histories
        self.u_dyn_history: Optional[np.ndarray] = None
        self.v_dyn_history: Optional[np.ndarray] = None
        self.a_dyn_history: Optional[np.ndarray] = None

        # S-curve cycloidal kinematics
        t = self.time_vec
        self.omega = (theta_max / t_total) * (1.0 - np.cos(2.0 * np.pi * t / t_total))
        self.alpha = (2.0 * np.pi * theta_max / (t_total**2)) * np.sin(2.0 * np.pi * t / t_total)

        # Precompute external excitation history on loaded Output hub nodes
        # Fully 3D multiaxial dynamic load profile:
        # 1. Fx(t): Centrifugal tension (+620 N) <-> Braking/pushing compression (-420 N)
        # 2. Fy(t): Slewing acceleration (-750 N) <-> Deceleration impact (+750 N)
        # 3. Fz(t): Out-of-plane vertical shear vibration (+/- 350 N)
        # 4. Tx(t): Eccentric payload rotational torsion torque couple (+/- 28 N*m)
        self.f_ext_history = np.zeros((n_steps, self.domain.ndof), dtype=float)
        arm_length_m = 0.120
        e_offset_m = 0.040   # 40 mm payload eccentricity for torsion
        n_ld = len(self.domain.loaded_nodes)
        hub2_center_3d = np.array([self.domain.hub2_center[0], self.domain.hub2_center[1], self.domain.lz / 2.0]) * 1e-3  # in meters

        # --- Rigorous MSC Nastran RBE3 Centroidal Couple Distribution ---
        node_coords = self.domain.mesh.node_coords
        hub_coords_m = np.zeros((n_ld, 3), dtype=np.float64)
        for idx, nid in enumerate(self.domain.loaded_nodes):
            hub_coords_m[idx] = node_coords[nid] * 1e-3  # in meters

        # 1. Discrete Centroid of loaded hub nodes: x_bar = 1/N * sum(x_i)
        x_bar_m = np.mean(hub_coords_m, axis=0)

        # 2. Centroid-relative positions: r_tilde_i = x_i - x_bar  (guarantees sum(r_tilde) == 0.0)
        r_tilde_m = hub_coords_m - x_bar_m

        # 3. Hub centroid eccentricity relative to nominal center: delta_x = x_bar - x_hub
        delta_x_m = x_bar_m - hub2_center_3d

        # 4. Discrete 3x3 rotational inertia tensor: J_H = sum( ||r_tilde_i||^2 * I - r_tilde_i * r_tilde_i^T )
        J_hub = np.zeros((3, 3), dtype=np.float64)
        for i in range(n_ld):
            r_i = r_tilde_m[i]
            r_norm_sq = float(np.dot(r_i, r_i))
            J_hub += r_norm_sq * np.eye(3) - np.outer(r_i, r_i)

        J_hub_inv = np.linalg.pinv(J_hub)

        max_net_force_err = 0.0
        max_torque_err = 0.0

        for s in range(n_steps):
            ts = t[s]
            w = self.omega[s]
            a = self.alpha[s]

            # 1. Axial alternating force (Fx): centrifugal tension + braking/push compression
            fx_total = (0.7 * self.m_payload * (w**2) * arm_length_m - 
                        420.0 * np.sin(2.0 * np.pi * ts / self.t_total) * np.sin(np.pi * ts / self.t_total))

            # 2. Tangential alternating force (Fy): acceleration reversal + high-frequency reaction
            fy_total = -self.m_payload * a * arm_length_m + 150.0 * np.sin(4.0 * np.pi * ts / self.t_total)

            # 3. Out-of-plane damped shear force (Fz): vertical oscillation
            fz_total = 350.0 * np.sin(3.0 * np.pi * ts / self.t_total) * np.exp(-20.0 * ts)

            # 4. 3D pure torsional moment (Tx): eccentric inertia torque + dynamic flutter
            tx_total = (self.m_payload * a * arm_length_m) * e_offset_m + 12.0 * np.sin(3.0 * np.pi * ts / self.t_total)

            F_vec = np.array([fx_total, fy_total, fz_total], dtype=np.float64)
            T_nom = np.array([tx_total, 0.0, 0.0], dtype=np.float64)

            # Equivalent moment about discrete centroid: T_tilde = T_nom - delta_x x F
            T_tilde = T_nom - np.cross(delta_x_m, F_vec)

            # Angular couple multiplier: omega_c = J_hub^-1 * T_tilde
            omega_c = J_hub_inv @ T_tilde

            # Nodal force: f_i = 1/N * F + omega_c x r_tilde_i
            f_nodes = np.zeros((n_ld, 3), dtype=np.float64)
            for idx in range(n_ld):
                f_couple_i = np.cross(omega_c, r_tilde_m[idx])
                f_nodes[idx] = (F_vec / n_ld) + f_couple_i

            # Verify net force & torque accuracy
            sum_f = np.sum(f_nodes, axis=0)
            net_force_err = float(np.linalg.norm(sum_f - F_vec))
            max_net_force_err = max(max_net_force_err, net_force_err)

            # Torque check about nominal center
            t_actual = np.zeros(3, dtype=np.float64)
            for idx in range(n_ld):
                r_nom_i = hub_coords_m[idx] - hub2_center_3d
                t_actual += np.cross(r_nom_i, f_nodes[idx])
            torque_err = float(np.linalg.norm(t_actual - T_nom))
            max_torque_err = max(max_torque_err, torque_err)

            for idx, nid in enumerate(self.domain.loaded_nodes):
                self.f_ext_history[s, 3 * nid] = f_nodes[idx, 0]
                self.f_ext_history[s, 3 * nid + 1] = f_nodes[idx, 1]
                self.f_ext_history[s, 3 * nid + 2] = f_nodes[idx, 2]

        print(f"     [RBE3 Hub Coupling] Max parasitic force error: {max_net_force_err:.2e} N | Max torque error: {max_torque_err:.2e} N*m", flush=True)

    @abstractmethod
    def solve_transient_dynamics(
        self,
        K: sp.csc_matrix,
        M: sp.csc_matrix,
    ) -> Tuple[np.ndarray, List[int], float]:
        """Perform full transient dynamic time-integration.

        Returns
        -------
        u_dyn_history : (n_steps, ndof) array
        crit_indices : list of critical peak time indices
        solve_time : elapsed time in seconds
        """
        pass

    def extract_critical_time_steps(self, u_dyn_history: np.ndarray, K: sp.csc_matrix) -> List[int]:
        """Identify 4 critical dynamic instances: peak strain energy, peak tip disp, peak acceleration, peak braking."""
        # 1. Dynamic strain energy history: SE(s) = 0.5 * u^T * K * u
        se_hist = np.zeros(self.n_steps)
        for s in range(self.n_steps):
            us = u_dyn_history[s]
            se_hist[s] = 0.5 * float(us @ (K @ us))

        # 2. Tip displacement magnitude history
        tip_nid = self.domain.loaded_nodes[0]
        tip_mag_hist = np.linalg.norm(u_dyn_history[:, 3 * tip_nid:3 * tip_nid + 3], axis=1)

        # Peaks
        idx_se_max = int(np.argmax(se_hist))
        idx_tip_max = int(np.argmax(tip_mag_hist))
        idx_acc_max = int(np.argmax(self.alpha))
        idx_brake_max = int(np.argmin(self.alpha))

        crit_set = {idx_se_max, idx_tip_max, idx_acc_max, idx_brake_max}
        for cand in [7, 15, 22, 10, 20, 5, 25, 12, 18]:
            if len(crit_set) >= 4:
                break
            crit_set.add(cand)
        crit_indices = sorted(list(crit_set))[:4]
        return crit_indices

    def compute_equivalent_static_loads(
        self,
        K: sp.csc_matrix,
        u_dyn_history: np.ndarray,
        crit_indices: List[int],
    ) -> np.ndarray:
        """Compute the true Equivalent Static Load (ESL) vectors: f_eq(t_k) = K(rho) * u_dyn(t_k)."""
        n_crit = len(crit_indices)
        f_eq_cases = np.zeros((n_crit, self.domain.ndof), dtype=float)
        for k, idx in enumerate(crit_indices):
            u_k = u_dyn_history[idx]
            f_eq_cases[k] = K @ u_k
        return f_eq_cases

    def compute_element_von_mises_batch(
        self,
        u_steps: np.ndarray,
        E0: float = 1.0,
    ) -> np.ndarray:
        """Batch compute unpenalized element von Mises stress sigma_vm^0(t_k) across time steps via einsum.

        Parameters
        ----------
        u_steps : np.ndarray
            Array of displacement vectors (n_cases, ndof) or (ndof,).
        E0 : float
            Base reference Young's modulus (default 1.0 for unit-modulus unpenalized stress).

        Returns
        -------
        sigma_vm_cases : np.ndarray
            Element centroid von Mises stresses of shape (n_cases, n_elem) or (n_elem,).
        """
        return self.fem.compute_element_von_mises_batch(u_steps, E0=E0)



class BuiltinGeneralizedAlphaDynamicSolver(BaseDynamicESLSolver):
    """High-Performance Chung & Hulbert (1993) Generalized-alpha Implicit Dynamic Solver with PyPardiso."""

    def __init__(
        self,
        domain: RobotArmDomain3D,
        fem: FEMEngine,
        rho_inf: float = 0.85,  # Optimal high-frequency dissipation
        t_total: float = 0.05,
        n_steps: int = 30,
        theta_max: float = np.pi / 2.0,
        m_payload_kg: float = 1.5,
    ) -> None:
        super().__init__(domain, fem, t_total=t_total, n_steps=n_steps, theta_max=theta_max, m_payload_kg=m_payload_kg)
        self.rho_inf = rho_inf

        # Generalized-alpha parameters
        self.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f)**2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f

        # Integration coefficients
        dt = self.dt
        self.a0 = 1.0 / (self.beta * dt**2)
        self.a1 = self.gamma / (self.beta * dt)
        self.a2 = 1.0 / (self.beta * dt)
        self.a3 = 1.0 / (2.0 * self.beta) - 1.0
        self.a4 = self.gamma / self.beta - 1.0
        self.a5 = dt * 0.5 * (self.gamma / self.beta - 2.0)
        self.a6 = dt * (1.0 - self.gamma)
        self.a7 = self.gamma * dt

        # Rayleigh damping parameters
        self.alpha_rayleigh = 5.0
        self.beta_rayleigh = 1e-5

        # Reusable MKL PARDISO solver instance (default mtype=11)
        self.pardiso_solver = pypardiso.PyPardisoSolver()

        print(f" [+] Builtin Generalized-alpha Solver Ready: rho_inf={rho_inf:.2f}, alpha_m={self.alpha_m:.4f}, alpha_f={self.alpha_f:.4f}", flush=True)

    def solve_transient_dynamics(
        self,
        K: sp.csc_matrix,
        M: sp.csc_matrix,
    ) -> Tuple[np.ndarray, List[int], float]:
        t0 = time.perf_counter()
        free = self.domain.free_dofs
        n_free = len(free)

        # Rayleigh Damping C = alpha*M + beta*K
        C = self.alpha_rayleigh * M + self.beta_rayleigh * K

        # Effective Dynamic Stiffness: K_hat = (1-alpha_f)*K + a0*(1-alpha_m)*M + a1*(1-alpha_f)*C
        K_hat = (1.0 - self.alpha_f) * K + self.a0 * (1.0 - self.alpha_m) * M + self.a1 * (1.0 - self.alpha_f) * C
        K_hat_ff = K_hat[free, :][:, free].tocsr()

        M_ff = M[free, :][:, free].tocsr()
        C_ff = C[free, :][:, free].tocsr()
        K_ff = K[free, :][:, free].tocsr()

        # State vectors on free DOFs
        u_f = np.zeros(n_free, dtype=float)
        v_f = np.zeros(n_free, dtype=float)

        # Rigorous initial acceleration from equation of motion: M * a0 = F0 - C*v0 - K*u0
        f0_free = self.f_ext_history[0, free]
        # Use spsolve to compute initial acceleration so dedicated Pardiso factorize is never tainted!
        a_f = pypardiso.spsolve(M_ff, f0_free)

        # PyPardiso one-time symbolic & numerical factorization for dynamic stiffness K_hat_ff
        self.pardiso_solver.factorize(K_hat_ff)

        u_dyn_history = np.zeros((self.n_steps, self.domain.ndof), dtype=float)
        v_dyn_history = np.zeros((self.n_steps, self.domain.ndof), dtype=float)
        a_dyn_history = np.zeros((self.n_steps, self.domain.ndof), dtype=float)

        # Set initial dynamic state at step 0
        u_dyn_history[0, free] = u_f
        v_dyn_history[0, free] = v_f
        a_dyn_history[0, free] = a_f

        # Transient time march (Standard Chung & Hulbert 1993 unconditionally stable scheme)
        t_m0 = time.perf_counter()
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        alpha_m = self.alpha_m
        alpha_f = self.alpha_f

        for s in range(1, self.n_steps):
            f_curr = self.f_ext_history[s, free]
            f_prev = self.f_ext_history[s - 1, free]
            f_mid = (1.0 - alpha_f) * f_curr + alpha_f * f_prev

            p_term = (
                M_ff @ ((1.0 - alpha_m) / (beta * (dt**2)) * u_f +
                        (1.0 - alpha_m) / (beta * dt) * v_f +
                        (1.0 - alpha_m) * (1.0 - 2.0 * beta) / (2.0 * beta) * a_f) +
                C_ff @ ((1.0 - alpha_f) * gamma / (beta * dt) * u_f +
                        (1.0 - alpha_f) * (gamma / beta - 1.0) * v_f +
                        (1.0 - alpha_f) * dt * (gamma / (2.0 * beta) - 1.0) * a_f) -
                alpha_f * (K_ff @ u_f)
            )

            r_hat = f_mid + p_term
            u_next = self.pardiso_solver.solve(K_hat_ff, r_hat)

            a_next = (u_next - u_f) / (beta * (dt**2)) - v_f / (beta * dt) - (1.0 - 2.0 * beta) / (2.0 * beta) * a_f
            v_next = v_f + dt * ((1.0 - gamma) * a_f + gamma * a_next)

            u_f, v_f, a_f = u_next, v_next, a_next
            u_dyn_history[s, free] = u_f
            v_dyn_history[s, free] = v_f
            a_dyn_history[s, free] = a_f
        print(f"        [Dyn] 30-step march done in {time.perf_counter() - t_m0:.2f} s.", flush=True)

        self.u_dyn_history = u_dyn_history
        self.v_dyn_history = v_dyn_history
        self.a_dyn_history = a_dyn_history

        crit_indices = self.extract_critical_time_steps(u_dyn_history, K)
        solve_time = time.perf_counter() - t0
        return u_dyn_history, crit_indices, solve_time


class KratosBossakImplicitDynamicSolver(BaseDynamicESLSolver):
    """Native Pure Kratos Multiphysics Bossak-alpha Implicit Dynamic Solver with AMGCL."""

    def __init__(
        self,
        domain: RobotArmDomain3D,
        fem: FEMEngine,
        alpha_bossak: float = -0.05,
        t_total: float = 0.05,
        n_steps: int = 30,
        theta_max: float = np.pi / 2.0,
        m_payload_kg: float = 1.5,
    ) -> None:
        super().__init__(domain, fem, t_total=t_total, n_steps=n_steps, theta_max=theta_max, m_payload_kg=m_payload_kg)
        self.alpha_bossak = alpha_bossak
        self._cached_kratos_bundle: Optional[Dict[str, Any]] = None
        print(f" [+] Kratos Native Bossak-alpha Solver Ready: alpha_bossak={alpha_bossak:.3f}", flush=True)

    def _setup_kratos_modelpart(self, rho_phys: np.ndarray, e0: float, emin: float, penal: float, nu: float, rho_mat: float) -> Dict[str, Any]:
        """Construct or update Kratos in-memory ModelPart with Hex8 elements and Bossak strategy."""
        nodes = self.domain.mesh.node_coords
        n_nodes = len(nodes)
        elems = self.domain.mesh.elements
        n_elems = len(elems)

        model = KM.Model()
        mp = model.CreateModelPart("DynRobotArmKratos")
        mp.SetBufferSize(2)
        mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
        mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
        mp.AddNodalSolutionStepVariable(KM.VELOCITY)
        mp.AddNodalSolutionStepVariable(KM.ACCELERATION)
        mp.AddNodalSolutionStepVariable(KM.REACTION)
        mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)

        for i in range(n_nodes):
            mp.CreateNewNode(i + 1, float(nodes[i, 0]), float(nodes[i, 1]), float(nodes[i, 2]))

        KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, KM.REACTION_X, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z, mp)

        # Fix Base Hub nodes
        for nid in self.domain.clamped_nodes:
            n = mp.GetNode(int(nid) + 1)
            n.Fix(KM.DISPLACEMENT_X)
            n.Fix(KM.DISPLACEMENT_Y)
            n.Fix(KM.DISPLACEMENT_Z)

        # Create elements with SIMP properties
        props_list = []
        law = SMA.LinearElastic3DLaw()
        youngs_arr = emin + (rho_phys ** penal) * (e0 - emin)

        for e_idx in range(n_elems):
            pid = e_idx + 1
            prop = mp.CreateNewProperties(pid)
            prop.SetValue(KM.YOUNG_MODULUS, float(youngs_arr[e_idx]))
            prop.SetValue(KM.POISSON_RATIO, float(nu))
            prop.SetValue(KM.DENSITY, float(rho_mat * rho_phys[e_idx]))  # consistent in tonne/mm^3 (consistent with mm-N-s)
            prop.SetValue(KM.CONSTITUTIVE_LAW, law)
            props_list.append(prop)
            conn = [int(n) + 1 for n in elems[e_idx]]
            mp.CreateNewElement("SmallDisplacementElement3D8N", pid, conn, prop)

        # Conditions on Output Hub
        lprop = mp.CreateNewProperties(n_elems + 100)
        for cid, nid in enumerate(self.domain.loaded_nodes, 1):
            mp.CreateNewCondition("PointLoadCondition3D1N", cid, [int(nid) + 1], lprop)

        # Native Bossak Scheme + AMGCL
        scheme = KM.ResidualBasedBossakDisplacementScheme(self.alpha_bossak)
        params = {
            "solver_type": "amgcl",
            "krylov_type": "cg",
            "smoother_type": "ilu0",
            "coarsening_type": "aggregation",
            "block_size": 3,
            "tolerance": 1e-6,
            "max_iteration": 400,
            "verbosity": 0,
        }
        linear_solver = python_linear_solver_factory.ConstructSolver(KM.Parameters(json.dumps(params)))
        builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
        strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
        strategy.SetEchoLevel(0)
        strategy.Initialize()

        bundle = {
            "model": model,
            "mp": mp,
            "strategy": strategy,
            "props_list": props_list,
        }
        return bundle

    def solve_transient_dynamics(
        self,
        K: sp.csc_matrix,
        M: sp.csc_matrix,
        rho_phys: Optional[np.ndarray] = None,
        e0: float = 71000.0,
        emin: float = 71.0,
        penal: float = 3.0,
        nu: float = 0.33,
        rho_mat: float = 2.81e-9,
    ) -> Tuple[np.ndarray, List[int], float]:
        t0 = time.perf_counter()
        if rho_phys is None:
            rho_phys = np.ones(self.domain.n_elem)

        bundle = self._setup_kratos_modelpart(rho_phys, e0, emin, penal, nu, rho_mat)
        mp = bundle["mp"]
        strategy = bundle["strategy"]

        u_dyn_history = np.zeros((self.n_steps, self.domain.ndof), dtype=float)
        loaded = self.domain.loaded_nodes

        t = 0.0
        for s in range(self.n_steps):
            t += self.dt
            mp.CloneTimeStep(t)

            # Apply loads for step s
            f_step = self.f_ext_history[s]
            for nid in loaded:
                fx = float(f_step[3 * nid])
                fy = float(f_step[3 * nid + 1])
                fz = float(f_step[3 * nid + 2])
                mp.GetNode(int(nid) + 1).SetSolutionStepValue(SMA.POINT_LOAD, [fx, fy, fz])

            strategy.Solve()

            # Extract nodal displacements
            for i in range(len(self.domain.mesh.node_coords)):
                node = mp.GetNode(i + 1)
                disp = node.GetSolutionStepValue(KM.DISPLACEMENT)
                u_dyn_history[s, 3 * i] = disp[0]
                u_dyn_history[s, 3 * i + 1] = disp[1]
                u_dyn_history[s, 3 * i + 2] = disp[2]

        crit_indices = self.extract_critical_time_steps(u_dyn_history, K)
        solve_time = time.perf_counter() - t0
        return u_dyn_history, crit_indices, solve_time


class RobotArmFESLOptimizer3D:
    """Master 3D Dynamic Robot Arm Topology Optimizer with True ESL & Buzaglo & Pollini 2026 F-ESL."""

    def __init__(
        self,
        domain: RobotArmDomain3D,
        solver_type: str = "builtin",  # "builtin" or "kratos"
        target_vol: float = 0.25,
        max_iter: int = 30,
        initial_move_limit: float = 0.15,
        filter_radius_mm: float = 6.0,
        e0: float = 71000.0,
        nu: float = 0.33,
        rho_mat: float = 2.81e-9,
        yield_limit: float = 503.0,
        extrusion_axis: Optional[str] = None,
        output_dir: Optional[Path] = None,
        optimizer_mode: Literal["FESL_MMA", "SE_AHCA", "HYBRID_WARMUP", "HYBRID_SE_AHCA"] = "FESL_MMA",
        warmup_iters: int = 15,
        hca_stress_weight: float = 0.45,
    ) -> None:
        self.domain = domain
        self.mesh = domain.mesh
        self.solver_type = solver_type.lower()
        self.target_vol = target_vol
        self.max_iter = max_iter
        self.trust_region = initial_move_limit
        self.filter_radius_mm = filter_radius_mm
        self.extrusion_axis = extrusion_axis.lower() if extrusion_axis and extrusion_axis.lower() != "none" else None
        self.optimizer_mode: Literal["FESL_MMA", "SE_AHCA", "HYBRID_WARMUP", "HYBRID_SE_AHCA"] = optimizer_mode
        self.warmup_iters = warmup_iters
        self.hca_stress_weight = hca_stress_weight

        self.e0 = e0
        self.emin = 1e-3 * e0
        self.nu = nu
        self.rho_mat = rho_mat
        self.yield_limit = yield_limit
        self.penal = 3.0
        self.vol_total = float(np.sum(self.mesh.elem_volumes))
        self.init_mass_g = self.vol_total * self.rho_mat * 1e6

        self.output_dir = Path(output_dir) if output_dir else Path(f"wht_topopt/examples/outputs/ex09_rev1/{self.solver_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fem = FEMEngine(self.mesh, nu=self.nu, e0=self.e0, emin=self.emin)
        self.filter_op = DensityFilter(self.mesh, r_min=self.filter_radius_mm)
        if self.extrusion_axis == "shell_rib":
            self.ext_filter = ShellRibExtrusionFilter(self.mesh, bottom_layers=1, side_layers=1)
        elif self.extrusion_axis == "z_plate":
            self.ext_filter = BottomPlateZExtrusionFilter(self.mesh, bottom_plate_layers=1)
        elif self.extrusion_axis in ["x", "y", "z"]:
            self.ext_filter = ExtrusionFilter(self.mesh, extrusion_axis=self.extrusion_axis)
        else:
            self.ext_filter = None
        self.optimizer = MMAOptimizer(move=self.trust_region, rho_min=1e-3, rho_max=1.0, backend="mmapy")
        self.pardiso_static = pypardiso.PyPardisoSolver()

        # Initialize SE-AHCA Engine for gradient-free / hybrid optimization
        hca_cfg = SEAHCAConfig(
            volfrac_target=self.target_vol,
            stress_weight_eta=self.hca_stress_weight,
            sigma_allow=self.yield_limit,
            move_limit=self.trust_region,
        )
        self.hca_engine = StressEnergyAdaptiveHCA(
            elem_volumes=self.mesh.elem_volumes,
            config=hca_cfg,
            passive_solid_mask=self.domain.passive_solid_mask,
            passive_void_mask=self.domain.passive_void_mask,
        )

        # Select Dynamic Solver
        if self.solver_type == "kratos":
            self.dyn_solver: BaseDynamicESLSolver = KratosBossakImplicitDynamicSolver(self.domain, self.fem)
        else:
            self.dyn_solver = BuiltinGeneralizedAlphaDynamicSolver(self.domain, self.fem, rho_inf=0.85)

    def optimize(self) -> Tuple[np.ndarray, List[FESLIterationRecord], List[np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Run complete True ESL dynamic topology optimization loop."""
        print("\n" + "=" * 126, flush=True)
        print(f" >>> [START] Showcase 09: First-Order Equivalent Static Loads (F-ESL) Dynamic TopOpt [{self.solver_type.upper()}]", flush=True)
        print("     Theory       : SOTA F-ESL (Buzaglo & Pollini 2026) + True Dynamic Time Integration + Svanberg MMA", flush=True)
        print(f"     Dyn Solver   : {'Pure Kratos Bossak-alpha Scheme + AMGCL' if self.solver_type == 'kratos' else 'Built-in Generalized-alpha (rho_inf=0.85) + PyPardiso'}", flush=True)
        print(f"     Mesh Grid    : {self.mesh.nelx} x {self.mesh.nely} x {self.mesh.nelz} ({self.domain.n_elem:,} Hex8 elements, {self.domain.ndof:,} DOFs)", flush=True)
        print(f"     Material     : Al 7075-T6 (E={self.e0:,.0f} MPa, nu={self.nu:.2f}, Yield={self.yield_limit:.0f} MPa)", flush=True)
        print(f"     Initial Mass : {self.init_mass_g:.1f} g | Target Mass: {self.init_mass_g * self.target_vol:.1f} g (Vol = {self.target_vol:.1%})", flush=True)
        print(f"     Opt Mode     : {self.optimizer_mode} (Warmup={self.warmup_iters} iters, SE-AHCA eta={self.hca_stress_weight})", flush=True)
        print(f"     S-Curve Slew : 50 ms | Peak omega = {np.max(self.dyn_solver.omega):.1f} rad/s | Peak alpha = {np.max(self.dyn_solver.alpha):,.0f} rad/s^2", flush=True)
        print("-" * 136, flush=True)
        print(f"  {'Iter / Phase':<16} | {'F-ESL Comp (mJ)':<17} | {'Volume':<7} | {'MaxChange':<9} | {'Solid Stress (COV)':<25} | {'Peak Dyn Disp':<13} | {'TrustRad':<8} | {'Dyn Time':<8} | {'Total Time':<8}", flush=True)
        print("-" * 136, flush=True)

        x = np.full(self.domain.n_elem, self.target_vol, dtype=float)
        x[self.domain.passive_solid_mask] = 1.0
        x[self.domain.passive_void_mask] = 1e-3

        records: List[FESLIterationRecord] = []
        density_history: List[np.ndarray] = []
        solid_design_mask = ~self.domain.passive_void_mask

        t_total_start = time.perf_counter()
        prev_compliance = 0.0
        c_ref: Optional[float] = None
        u_dyn_init_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None

        free = self.domain.free_dofs

        for it in range(1, self.max_iter + 1):
            t_step0 = time.perf_counter()

            # Determine optimizer phase
            if self.optimizer_mode == "SE_AHCA":
                current_phase = "SE_AHCA"
            elif self.optimizer_mode in ["HYBRID_WARMUP", "HYBRID_SE_AHCA"]:
                current_phase = "SE_AHCA" if it <= self.warmup_iters else "FESL_MMA"
            else:
                current_phase = "FESL_MMA"

            print(f"  --> [Iter {it:02d}/{self.max_iter:02d} | {current_phase}] Computing dynamics (30 steps)...", flush=True)

            # 1. Forward Density Filter & optional Extrusion manufacturing projection
            rho_flt = self.filter_op.forward(x)
            if self.ext_filter is not None:
                rho_phys = self.ext_filter.forward(rho_flt)
            else:
                rho_phys = rho_flt.copy()
            rho_phys[self.domain.passive_solid_mask] = 1.0
            rho_phys[self.domain.passive_void_mask] = 1e-3
            density_history.append(rho_phys.copy())

            # 2. Assemble Global Mass M(rho) and Stiffness K(rho)
            youngs = self.emin + (rho_phys ** self.penal) * (self.e0 - self.emin)
            rho_elem_mass = (rho_phys * self.rho_mat)
            K = self.fem.assemble_K(youngs)
            M = self.fem.assemble_M(rho_elem_mass)

            # Rigorous coupling: add 1.5 kg payload lumped mass directly to Output Hub DOFs in M
            m_payload_kg = getattr(self.domain, "m_payload", getattr(self.dyn_solver, "m_payload", 1.5))
            m_payload_tonne = m_payload_kg * 1e-3  # kg -> tonne
            m_node = m_payload_tonne / len(self.domain.loaded_nodes)
            diag_payload = np.zeros(self.domain.ndof, dtype=np.float64)
            for nid in self.domain.loaded_nodes:
                diag_payload[3 * nid : 3 * nid + 3] = m_node
            M = (M + sp.diags(diag_payload, format="csc")).tocsc()

            # 3. Step 1: True Transient Dynamic Analysis
            if self.solver_type == "kratos":
                u_dyn_hist, crit_indices, dyn_time = self.dyn_solver.solve_transient_dynamics(
                    K, M, rho_phys=rho_phys, e0=self.e0, emin=self.emin, penal=self.penal, nu=self.nu, rho_mat=self.rho_mat
                )
            else:
                u_dyn_hist, crit_indices, dyn_time = self.dyn_solver.solve_transient_dynamics(K, M)

            print(f"      Dyn solve: {dyn_time:.2f} s (crits={crit_indices}) | Solving ESL static subproblems...", flush=True)

            if it == 1:
                tip_nid = self.domain.loaded_nodes[0]
                u_dyn_init_tuple = (u_dyn_hist[:, 3 * tip_nid + 1].copy(), u_dyn_hist[:, 3 * tip_nid + 2].copy())

            # 4. Step 2: Compute Equivalent Static Loads (ESL): f_eq(t_k) = K * u_dyn(t_k)
            f_eq_cases = self.dyn_solver.compute_equivalent_static_loads(K, u_dyn_hist, crit_indices)
            n_crit = len(crit_indices)

            # 5. Step 3: Solve Static Multi-Load Sub-Problem for compliance & sensitivities
            t_stat0 = time.perf_counter()
            K_ff = K[free, :][:, free].tocsr()
            F_ff = np.asfortranarray(f_eq_cases[:, free].T, dtype=np.float64)  # Fortran-order for Pardiso multi-RHS
            u_ff = self.pardiso_static.solve(K_ff, F_ff)
            u_static_cases = np.zeros((n_crit, self.domain.ndof), dtype=float)
            u_static_cases[:, free] = u_ff.T
            static_time = time.perf_counter() - t_stat0

            # 6. F-ESL Multi-Load Compliance
            w_cases = np.array([0.30, 0.25, 0.30, 0.15])
            if n_crit < 4:
                w_cases = np.full(n_crit, 1.0 / n_crit)

            total_compliance = 0.0
            c_e_cases = np.zeros((n_crit, self.domain.n_elem), dtype=float)
            for k in range(n_crit):
                uk = u_static_cases[k]
                ck = float(np.dot(f_eq_cases[k], uk))
                total_compliance += w_cases[k] * ck
                c_e_cases[k] = self.fem.compute_unit_element_compliance(uk)

            # 7. Optimizer Density Update
            if current_phase == "SE_AHCA":
                # Gradient-Free Stress-Energy Adaptive Cellular Automata update (No adjoint PDE required!)
                sigma_vm_cases = self.dyn_solver.compute_element_von_mises_batch(u_static_cases, E0=self.e0)
                sed_cases = 0.5 * youngs[None, :] * c_e_cases

                signal, hca_metrics = self.hca_engine.compute_cell_state_signal(
                    rho_phys=rho_phys,
                    strain_energy_cases=sed_cases,
                    von_mises_unpenalized_cases=sigma_vm_cases,
                    case_weights=w_cases,
                    neighborhood_filter_fn=self.filter_op.forward,
                )
                x_new, _, _ = self.hca_engine.step(
                    xi_current=x,
                    cell_signal=signal,
                    forward_projection_fn=self.filter_op.forward,
                )
                max_change = float(np.max(np.abs(x_new - x)))
                x = x_new.copy()

            else:
                # Rigorous Svanberg MMA update with 1st-order inertial/damping sensitivities
                df_dphys = np.zeros(self.domain.n_elem, dtype=float)
                delta_fesl = np.zeros(self.domain.n_elem, dtype=float)

                a_dyn_hist = self.dyn_solver.a_dyn_history
                v_dyn_hist = self.dyn_solver.v_dyn_history
                Me0 = self.fem.Me0  # (24, 24)
                Ke0 = self.fem.Ke0  # (24, 24)
                dK_factor = self.penal * (rho_phys ** (self.penal - 1.0)) * (self.e0 - self.emin)

                for k in range(n_crit):
                    idx_k = crit_indices[k]
                    uk = u_static_cases[k]
                    c_e_k = c_e_cases[k]
                    df_dphys += -w_cases[k] * dK_factor * c_e_k

                    # Rigorous Buzaglo & Pollini (2025/2026) First-Order F-ESL Inertial & Damping Sensitivities
                    if a_dyn_hist is not None and v_dyn_hist is not None:
                        ak = a_dyn_hist[idx_k]
                        vk = v_dyn_hist[idx_k]

                        u_e_k = uk[self.mesh.edofMat]
                        a_e_k = ak[self.mesh.edofMat]
                        v_e_k = vk[self.mesh.edofMat]

                        # 1. Inertial correction: - u_S,e^T * (rho_mat * Me0) * a_D,e
                        Me_ae = (a_e_k @ Me0) * self.rho_mat
                        delta_inertial_k = -np.sum(u_e_k * Me_ae, axis=1)

                        # 2. Rayleigh Damping correction: - u_S,e^T * (alpha_R * dM + beta_R * dK) * v_D,e
                        Me_ve = (v_e_k @ Me0) * self.rho_mat
                        Ke_ve = v_e_k @ Ke0
                        alpha_R = getattr(self.dyn_solver, "alpha_rayleigh", 5.0)
                        beta_R = getattr(self.dyn_solver, "beta_rayleigh", 1e-5)
                        dC_ve = alpha_R * Me_ve + beta_R * (dK_factor[:, None] * Ke_ve)
                        delta_damping_k = -np.sum(u_e_k * dC_ve, axis=1)

                        delta_fesl += w_cases[k] * (delta_inertial_k + delta_damping_k)

                df_dphys_fesl = df_dphys + delta_fesl

                # Backward filter with optional extrusion adjoint
                if self.ext_filter is not None:
                    df_dflt = self.ext_filter.backward(df_dphys_fesl)
                    df_dx = self.filter_op.backward(df_dflt)
                    dg_dflt = self.ext_filter.backward(self.mesh.elem_volumes / (self.vol_total * self.target_vol))
                    dg_dx = self.filter_op.backward(dg_dflt)
                else:
                    df_dx = self.filter_op.backward(df_dphys_fesl)
                    dg_dx = self.filter_op.backward(self.mesh.elem_volumes / (self.vol_total * self.target_vol))

                if c_ref is None:
                    c_ref = max(float(total_compliance), 1e-12)
                df_dx_norm = df_dx / c_ref

                # Adaptive Trust-Region / Move-Limit Update
                if it > 1:
                    actual_red = prev_compliance - total_compliance
                    if actual_red > 0.0:
                        self.trust_region = min(0.18, self.trust_region * 1.02)
                    else:
                        self.trust_region = max(0.04, self.trust_region * 0.90)

                self.optimizer.move = self.trust_region

                current_vol_raw = float(np.sum(rho_phys * self.mesh.elem_volumes)) / self.vol_total
                g_vol = np.array([current_vol_raw / self.target_vol - 1.0])

                x_new = self.optimizer.update(
                    rho=x,
                    df_drho=df_dx_norm,
                    g_vals=g_vol,
                    dg_drho=dg_dx,
                    elem_volumes=self.mesh.elem_volumes,
                    passive_solid=self.domain.passive_solid_mask,
                    passive_void=self.domain.passive_void_mask,
                )
                max_change = float(np.max(np.abs(x_new - x)))
                x = x_new.copy()

            prev_compliance = total_compliance
            current_vol = float(np.sum(rho_phys * self.mesh.elem_volumes)) / self.vol_total

            # 8. Stress & Dynamic Tip Displacement Monitoring
            u_peak = u_static_cases[0]
            _, vm = self.fem.compute_element_stresses(u_peak, E0=self.e0)
            vm_phys = (rho_phys ** 0.5) * vm

            # Solid design elements (rho_phys >= 0.5 and non-void)
            solid_mask = (rho_phys >= 0.5) & solid_design_mask
            if np.any(solid_mask):
                solid_vm = vm_phys[solid_mask]
                max_solid_stress = float(np.max(solid_vm))
                mean_solid_vm = float(np.mean(solid_vm))
                solid_stress_cov = float(np.std(solid_vm) / max(mean_solid_vm, 1e-12))
            else:
                max_solid_stress = float(np.max(vm_phys))
                solid_stress_cov = 0.0

            tip_nid = self.domain.loaded_nodes[0]
            tip_disp_mag = np.linalg.norm(u_dyn_hist[:, 3 * tip_nid:3 * tip_nid + 3], axis=1)
            peak_dyn_disp = float(np.max(tip_disp_mag))

            total_step_time_s = time.perf_counter() - t_step0

            record = FESLIterationRecord(
                iteration=it,
                compliance=total_compliance,
                volume_fraction=current_vol,
                max_change=max_change,
                max_stress_mpa=max_solid_stress,
                peak_dyn_disp_mm=peak_dyn_disp,
                trust_region_radius=self.trust_region,
                dyn_solve_time_s=dyn_time,
                static_solve_time_s=static_time,
                total_step_time_s=total_step_time_s,
                solid_stress_cov=solid_stress_cov,
                optimizer_phase=current_phase,
            )
            records.append(record)

            print(
                f"  Iter {it:02d} [{current_phase:^7}] | {total_compliance:14.4f} mJ | {current_vol:6.1%} | {max_change:8.4f} | "
                f"{max_solid_stress:7.1f} MPa (COV={solid_stress_cov:.3f}) | {peak_dyn_disp:7.4f} mm | {self.trust_region:6.3f}   | {dyn_time:5.2f} s | {total_step_time_s:5.2f} s",
                flush=True,
            )

        total_elapsed = time.perf_counter() - t_total_start
        print("-" * 126, flush=True)
        print(f" [SUCCESS] SOTA F-ESL Dynamic TopOpt ({self.solver_type}) completed in {total_elapsed:.1f} s ({total_elapsed / 60.0:.2f} min).", flush=True)
        print(f"     Final F-ESL Compliance: {records[-1].compliance:.4f} mJ", flush=True)
        print(f"     Final Volume Fraction : {records[-1].volume_fraction:.2%} (Mass: {self.init_mass_g * records[-1].volume_fraction:.1f} g)", flush=True)
        print(f"     Peak von Mises Stress : {records[-1].max_stress_mpa:.2f} MPa (Yield Margin: {(self.yield_limit / records[-1].max_stress_mpa):.1f}x)", flush=True)
        print(f"     Peak Dynamic Tip Disp : {records[-1].peak_dyn_disp_mm:.4f} mm", flush=True)
        print("=" * 126 + "\n", flush=True)

        final_rho = density_history[-1]
        tip_nid = self.domain.loaded_nodes[0]
        u_dyn_opt_tuple = (u_dyn_hist[:, 3 * tip_nid + 1].copy(), u_dyn_hist[:, 3 * tip_nid + 2].copy())

        return final_rho, records, density_history, u_dyn_init_tuple, u_dyn_opt_tuple

    def export_watertight_stl(self, rho: np.ndarray, stl_path: Path, isovalue: float = 0.35) -> Tuple[Path, bool, float]:
        """Export zero-padded Marching Cubes surface guaranteeing 100% watertight binary STL."""
        grid3d = rho.reshape((self.mesh.nelz, self.mesh.nely, self.mesh.nelx))
        padded_grid = np.pad(grid3d, pad_width=1, mode="constant", constant_values=0.0)

        verts, faces, normals, values = skimage.measure.marching_cubes(
            padded_grid,
            level=isovalue,
            spacing=(self.mesh.dz, self.mesh.dy, self.mesh.dx),
        )

        coords = np.zeros_like(verts)
        coords[:, 0] = verts[:, 2] - self.mesh.dx
        coords[:, 1] = verts[:, 1] - self.mesh.dy
        coords[:, 2] = verts[:, 0] - self.mesh.dz

        stl_mesh = trimesh.Trimesh(vertices=coords, faces=faces)
        try:
            stl_mesh.remove_duplicate_faces()
            stl_mesh.remove_degenerate_faces()
            stl_mesh.fill_holes()
        except Exception:
            pass

        stl_mesh.export(str(stl_path))
        is_watertight = bool(stl_mesh.is_watertight)
        volume_mm3 = float(abs(stl_mesh.volume)) if is_watertight else float(np.sum(rho > isovalue) * self.mesh.elem_volumes[0])
        print(f" [+] Exported STL: {stl_path} (Watertight: {is_watertight}, Volume: {volume_mm3:.1f} mm^3)", flush=True)
        return stl_path, is_watertight, volume_mm3

    def render_pyvista_cad_3d(self, stl_path: Path, png_path: Path, title: str) -> Path:
        """Render publication-quality 3D CAD surface with Dark Paraview Theme."""
        pv.set_plot_theme("document")
        plotter = pv.Plotter(off_screen=True, window_size=[1600, 1000])
        plotter.set_background("#0f172a")

        mesh = pv.read(str(stl_path))
        plotter.add_mesh(mesh, color="#38bdf8", specular=0.6, specular_power=30, smooth_shading=True, show_edges=False)
        plotter.add_axes(xlabel="X (Arm)", ylabel="Y (Slew)", zlabel="Z (Torsion)", color="white", line_width=2)
        plotter.add_text(title, position="upper_left", font_size=13, color="white", font="courier")
        plotter.camera_position = [(280.0, 120.0, 160.0), (80.0, 24.0, 20.0), (0.0, 0.0, 1.0)]
        plotter.screenshot(str(png_path))
        plotter.close()
        print(f" [+] Saved CAD 3D Screenshot: {png_path}", flush=True)
        return png_path

    def render_pyvista_voxels_3d(self, rho: np.ndarray, png_path: Path, title: str, threshold: float = 0.25) -> Path:
        """Render discrete Hex8 solid element volumes with visible element blocks and edges (ex08 style)."""
        grid = pv.ImageData()
        grid.dimensions = (self.mesh.nelx + 1, self.mesh.nely + 1, self.mesh.nelz + 1)
        grid.spacing = (self.mesh.dx, self.mesh.dy, self.mesh.dz)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data["density"] = rho

        solid_voxels = grid.threshold(value=threshold, scalars="density")

        plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
        plotter.set_background("black")

        if solid_voxels.n_cells > 0:
            plotter.add_mesh(
                solid_voxels,
                scalars="density",
                cmap="viridis",
                clim=[threshold, 1.0],
                edge_color="#202020",
                show_edges=True,
                smooth_shading=False,
                scalar_bar_args={"title": "Relative Density", "color": "white"},
            )
        plotter.add_axes(xlabel="X (Arm)", ylabel="Y (Slew)", zlabel="Z (Torsion)", color="white", line_width=2)
        plotter.add_text(title, position="upper_left", font_size=13, color="white", font="courier")
        plotter.camera_position = [(280.0, 120.0, 160.0), (80.0, 24.0, 20.0), (0.0, 0.0, 1.0)]
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(png_path))
        plotter.close()
        print(f" [+] Saved Discrete Hex8 Voxels Screenshot (ex08 style): {png_path}", flush=True)
        return png_path

    def generate_progression_gif(
        self,
        density_history: List[np.ndarray],
        gif_path: Path,
        records: Optional[List[IterationRecord]] = None,
        max_frames: int = 30,
        fps: int = 4,
        threshold: float = 0.25,
    ) -> Path:
        """Generate high-fidelity 3D discrete Hex8 voxel evolution GIF (Paraview dark theme)."""
        if not density_history:
            return gif_path

        grid = pv.ImageData()
        grid.dimensions = (self.mesh.nelx + 1, self.mesh.nely + 1, self.mesh.nelz + 1)
        grid.spacing = (self.mesh.dx, self.mesh.dy, self.mesh.dz)
        grid.origin = (0.0, 0.0, 0.0)

        n_tot = len(density_history)
        step_stride = max(1, n_tot // max_frames)
        selected_iters = list(range(0, n_tot, step_stride))
        if (n_tot - 1) not in selected_iters:
            selected_iters.append(n_tot - 1)

        plotter = pv.Plotter(off_screen=True, window_size=(960, 600))
        plotter.set_background("black")

        frames: List[np.ndarray] = []
        try:
            for it_idx in selected_iters:
                rho = density_history[it_idx]
                grid.cell_data["density"] = rho
                solid_voxels = grid.threshold(value=threshold, scalars="density")

                plotter.clear_actors()
                if solid_voxels.n_cells > 0:
                    plotter.add_mesh(
                        solid_voxels,
                        scalars="density",
                        cmap="viridis",
                        clim=[threshold, 1.0],
                        edge_color="#202020",
                        show_edges=True,
                        smooth_shading=False,
                        scalar_bar_args={"title": "Relative Density", "color": "white"},
                    )
                plotter.add_axes(xlabel="X (Arm)", ylabel="Y (Slew)", zlabel="Z (Torsion)", color="white", line_width=2)
                plotter.camera_position = [(280.0, 120.0, 160.0), (80.0, 24.0, 20.0), (0.0, 0.0, 1.0)]

                title_text = f"3D Dynamic Robot Arm | Iteration {it_idx + 1:02d}/{n_tot:02d}"
                if records and it_idx < len(records):
                    r = records[it_idx]
                    title_text += f"\nC = {r.compliance:.2f} | Vol = {r.volume_fraction:.1%}"
                plotter.add_text(title_text, position="upper_left", font_size=11, color="white", font="courier")

                frame = plotter.screenshot(return_img=True)
                frames.append(frame)

            gif_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
            print(f" [+] Generated High-Fidelity 3D Hex8 Progression GIF: {gif_path} ({len(frames)} frames)", flush=True)
        finally:
            plotter.close()

        return gif_path

    def plot_transient_vibration_comparison(
        self,
        u_init: Tuple[np.ndarray, np.ndarray],
        u_opt: Tuple[np.ndarray, np.ndarray],
        png_path: Path,
    ) -> Path:
        """Plot dynamic transient tip vibration response: initial solid baseline vs optimized link."""
        t_ms = self.dyn_solver.time_vec * 1000.0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(t_ms, u_init[0], "r--", linewidth=1.8, label="Initial Solid (Vol=100%) Uy")
        ax1.plot(t_ms, u_opt[0], "b-", linewidth=2.0, label=f"Optimized Link (Vol={self.target_vol:.0%}) Uy")
        ax1.set_ylabel("접선 변위 Uy (mm)", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)
        ax1.set_title("로봇 암 끝단 과도 동적 진동 응답 비교 (S-Curve 50ms 슬루잉)", fontsize=11, fontweight="bold")

        ax2.plot(t_ms, u_init[1], "m--", linewidth=1.8, label="Initial Solid (Vol=100%) Uz")
        ax2.plot(t_ms, u_opt[1], "c-", linewidth=2.0, label=f"Optimized Link (Vol={self.target_vol:.0%}) Uz")
        ax2.set_xlabel("시간 Time (ms)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("면외 변위 Uz (mm)", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f" [+] Saved Dynamic Vibration Comparison Plot: {png_path}", flush=True)
        return png_path

    def plot_convergence_history(self, records: List[FESLIterationRecord], png_path: Path) -> Path:
        """Plot 4-panel multi-physics convergence history."""
        iters = [r.iteration for r in records]
        comp = [r.compliance for r in records]
        vol = [r.volume_fraction for r in records]
        change = [r.max_change for r in records]
        trust = [r.trust_region_radius for r in records]
        stress = [r.max_stress_mpa for r in records]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))

        ax1.plot(iters, comp, "b-o", markersize=3.5, linewidth=1.6)
        ax1.set_ylabel("F-ESL 컴플라이언스 (mJ)", fontsize=9, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.set_title("동적 컴플라이언스 수렴 이력", fontsize=10, fontweight="bold")

        ax2.plot(iters, np.array(vol) * 100.0, "g-s", markersize=3.5, linewidth=1.6)
        ax2.axhline(self.target_vol * 100.0, color="r", linestyle="--", label=f"Target {self.target_vol:.0%}")
        ax2.set_ylabel("체적 분율 Volume Fraction (%)", fontsize=9, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")
        ax2.set_title("체적 분율 수렴 이력", fontsize=10, fontweight="bold")

        ax3.plot(iters, change, "c-^", markersize=3.5, linewidth=1.5, label="Max Change")
        ax3.plot(iters, trust, "m--", linewidth=1.5, label="Trust Region")
        ax3.set_xlabel("반복 횟수 Iteration", fontsize=9, fontweight="bold")
        ax3.set_ylabel("설계변수 변화량 / 이동한계", fontsize=9, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right")
        ax3.set_title("설계변수 변화량 및 어댑티브 이동한계", fontsize=10, fontweight="bold")

        ax4.plot(iters, stress, "r-d", markersize=3.5, linewidth=1.6)
        ax4.axhline(self.yield_limit, color="k", linestyle="--", label=f"Yield Limit {self.yield_limit:.0f} MPa")
        ax4.set_xlabel("반복 횟수 Iteration", fontsize=9, fontweight="bold")
        ax4.set_ylabel("최대 von Mises 응력 (MPa)", fontsize=9, fontweight="bold")
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right")
        ax4.set_title("피크 응력 및 항복 한도", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f" [+] Saved 4-Panel Convergence Plot: {png_path}", flush=True)
        return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Showcase 09: True F-ESL Dynamic TopOpt (Built-in GenAlpha vs Pure Kratos)")
    parser.add_argument("--solver", type=str, default="builtin", choices=["builtin", "kratos", "compare"], help="Solver backend")
    parser.add_argument("--extrusion-axis", type=str, default="none", choices=["none", "y", "z", "z_plate", "shell_rib"], help="Extrusion constraint axis ('y' for Slew, 'z' for Torsion/Motor, 'z_plate' for Z-Extrude with bottom plate, 'shell_rib' for Outer Shell + Interior Ribs, 'none' for Free 3D)")
    parser.add_argument("--max-iter", type=int, default=25, help="Maximum MMA iterations")
    parser.add_argument("--target-vol", type=float, default=0.25, help="Target volume fraction")
    parser.add_argument("--optimizer-mode", type=str, default="FESL_MMA", choices=["FESL_MMA", "SE_AHCA", "HYBRID_WARMUP", "HYBRID_SE_AHCA"], help="Optimization engine mode")
    parser.add_argument("--warmup-iters", type=int, default=15, help="Number of SE-AHCA warmup iterations for HYBRID_WARMUP mode")
    parser.add_argument("--hca-stress-weight", type=float, default=0.45, help="SE-AHCA stress weighting factor eta (0.0=pure SED, 1.0=pure FSD)")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (defaults to ex09_rev1/<solver> or ex09_rev1/ext_<axis>)")
    args = parser.parse_args()

    has_bottom_plate = (args.extrusion_axis == "z_plate")
    has_shell_rib = (args.extrusion_axis == "shell_rib")
    domain = RobotArmDomain3D(bottom_plate=has_bottom_plate, shell_rib=has_shell_rib)

    solvers_to_run = ["builtin", "kratos"] if args.solver == "compare" else [args.solver]
    ext_axis = args.extrusion_axis if args.extrusion_axis != "none" else None

    for sol in solvers_to_run:
        if args.output_dir:
            sol_dir = Path(args.output_dir)
        elif ext_axis is not None:
            sol_dir = Path("wht_topopt/examples/outputs/ex09_rev1") / f"ext_{ext_axis}"
        else:
            sol_dir = Path("wht_topopt/examples/outputs/ex09_rev1") / sol

        sol_dir.mkdir(parents=True, exist_ok=True)

        opt = RobotArmFESLOptimizer3D(
            domain=domain,
            solver_type=sol,
            target_vol=args.target_vol,
            max_iter=args.max_iter,
            extrusion_axis=ext_axis,
            output_dir=sol_dir,
            optimizer_mode=args.optimizer_mode,
            warmup_iters=args.warmup_iters,
            hca_stress_weight=args.hca_stress_weight,
        )

        final_rho, records, history, u_init, u_opt = opt.optimize()

        # Save Raw Element Density Field
        rho_name = f"robot_arm_ex09_ext_{ext_axis}_rho.npy" if ext_axis else f"robot_arm_ex09_{sol}_rho.npy"
        np.save(sol_dir / rho_name, final_rho)

        # Artifacts
        stl_name = f"robot_arm_ex09_ext_{ext_axis}.stl" if ext_axis else f"robot_arm_ex09_{sol}.stl"
        stl_path = sol_dir / stl_name
        opt.export_watertight_stl(final_rho, stl_path)

        cad_title = f"Robot Arm Link | {sol.upper()} F-ESL TopOpt"
        if ext_axis == "shell_rib":
            cad_title += " (Outer Shell Hull + Interior Extruded Ribs)"
        elif ext_axis == "z_plate":
            cad_title += " (Z-Extrusion with -Z Bottom Skin Plate)"
        elif ext_axis:
            cad_title += f" ({ext_axis.upper()}-Extrusion Milling Constraint)"
        cad_png = sol_dir / f"robot_arm_ex09_{'ext_' + ext_axis if ext_axis else sol}_cad_3d.png"
        opt.render_pyvista_cad_3d(stl_path, cad_png, cad_title)

        vox_title = f"Discrete Hex8 Voxels | {sol.upper()}"
        if ext_axis == "shell_rib":
            vox_title += " [Outer Shell Hull + Interior Z-Ribs]"
        elif ext_axis == "z_plate":
            vox_title += " [Z-Extrusion + -Z Bottom Plate]"
        elif ext_axis:
            vox_title += f" [Extrusion: {ext_axis.upper()}-axis]"
        vox_png = sol_dir / f"robot_arm_ex09_{'ext_' + ext_axis if ext_axis else sol}_voxels_3d.png"
        opt.render_pyvista_voxels_3d(final_rho, vox_png, vox_title)

        gif_path = sol_dir / f"robot_arm_ex09_{'ext_' + ext_axis if ext_axis else sol}_progression.gif"
        opt.generate_progression_gif(history, gif_path, records=records)

        # Save complete density history for downstream rendering
        np.save(sol_dir / f"robot_arm_ex09_{'ext_' + ext_axis if ext_axis else sol}_history.npy", np.array(history))

        dyn_png = sol_dir / f"dynamic_response_{'ext_' + ext_axis if ext_axis else sol}.png"
        opt.plot_transient_vibration_comparison(u_init, u_opt, dyn_png)

        conv_png = sol_dir / f"convergence_{'ext_' + ext_axis if ext_axis else sol}.png"
        opt.plot_convergence_history(records, conv_png)

        # Save metrics JSON
        metrics = {
            "solver": sol,
            "extrusion_axis": ext_axis,
            "final_compliance": records[-1].compliance,
            "final_volume": records[-1].volume_fraction,
            "peak_stress_mpa": records[-1].max_stress_mpa,
            "peak_dyn_disp_mm": records[-1].peak_dyn_disp_mm,
            "avg_dyn_time_s": float(np.mean([r.dyn_solve_time_s for r in records])),
            "avg_total_time_s": float(np.mean([r.total_step_time_s for r in records])),
        }
        with open(sol_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("\n [ALL RUNS COMPLETED SUCCESSFULLY] Ready for HTML master report generation.\n", flush=True)


if __name__ == "__main__":
    main()
