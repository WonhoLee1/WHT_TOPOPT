"""MuJoCo multibody dynamics trajectory load extraction and Inertia Relief projection.

Implements MuJoCoLoadProvider:
1. Runs forward multibody dynamics using the official `mujoco` API.
2. Extracts body wrenches (cfrc_int, cfrc_ext) and accelerations (cacc).
3. Identifies peak load cases and maps them to FE nodal forces via:
   - RBE3 surface coupling for joint reaction forces and moments.
   - Nodal D'Alembert inertial body forces:
     f_iner,i = -m_i * (a_0 + alpha x r_i + omega x (omega x r_i)).
4. Balances unconstrained dynamic components using FEASolver.apply_inertia_relief.
5. Computes optimized inertial properties (mass, COM, 3x3 inertia tensor) and updates
   the MJCF XML model for closed-loop dynamic re-simulation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import scipy.sparse as sp

from wht_topopt.core.mesh import StructuredMesh2D, StructuredMesh3D, UnstructuredMesh
from wht_topopt.solvers.base import FEASolver, LoadCase, build_rigid_body_modes


class MuJoCoLoadProvider:
    """Extracts dynamic load cases from MuJoCo multibody trajectories for topology optimization."""

    def __init__(
        self,
        xml_str: Optional[str] = None,
        xml_path: Optional[str] = None,
    ) -> None:
        """Initialize MuJoCoLoadProvider from XML string or path.

        Parameters
        ----------
        xml_str : Optional[str]
            MJCF model XML string.
        xml_path : Optional[str]
            Path to MJCF XML file.
        """
        if xml_str is not None:
            self.xml_str: str = xml_str
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_string(xml_str)
        elif xml_path is not None:
            with open(xml_path, "r", encoding="utf-8") as f:
                self.xml_str = f.read()
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            raise ValueError("Either xml_str or xml_path must be provided")

        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.trajectory: Optional[Dict[str, Any]] = None

    def simulate_trajectory(
        self,
        t_final: float = 1.0,
        dt: Optional[float] = None,
        qpos_init: Optional[np.ndarray] = None,
        qvel_init: Optional[np.ndarray] = None,
        control_callback: Optional[Callable[[mujoco.MjModel, mujoco.MjData, float], None]] = None,
    ) -> Dict[str, Any]:
        """Simulate multibody dynamics trajectory over [0, t_final].

        Parameters
        ----------
        t_final : float
            Total simulation time in seconds.
        dt : Optional[float]
            Optional simulation timestep (overrides model.opt.timestep if provided).
        qpos_init : Optional[np.ndarray]
            Initial generalized coordinate positions.
        qvel_init : Optional[np.ndarray]
            Initial generalized coordinate velocities.
        control_callback : Optional[Callable]
            Control callback called before each step: control_callback(model, data, time).

        Returns
        -------
        Dict[str, Any]
            Recorded trajectory history dictionary.
        """
        if t_final <= 0.0:
            raise ValueError(f"t_final must be positive, got {t_final}")

        if dt is not None:
            self.model.opt.timestep = float(dt)
        timestep = self.model.opt.timestep

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)
        if qpos_init is not None:
            self.data.qpos[:] = np.asarray(qpos_init, dtype=np.float64)
        if qvel_init is not None:
            self.data.qvel[:] = np.asarray(qvel_init, dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        n_steps = max(1, int(np.ceil(t_final / timestep)))
        n_bodies = self.model.nbody

        time_hist = np.empty(n_steps, dtype=np.float64)
        cfrc_int_hist = np.empty((n_steps, n_bodies, 6), dtype=np.float64)
        cfrc_ext_hist = np.empty((n_steps, n_bodies, 6), dtype=np.float64)
        cacc_hist = np.empty((n_steps, n_bodies, 6), dtype=np.float64)
        cvel_hist = np.empty((n_steps, n_bodies, 6), dtype=np.float64)
        xpos_hist = np.empty((n_steps, n_bodies, 3), dtype=np.float64)
        xmat_hist = np.empty((n_steps, n_bodies, 3, 3), dtype=np.float64)

        for step in range(n_steps):
            t_curr = float(self.data.time)
            if control_callback is not None:
                control_callback(self.model, self.data, t_curr)

            mujoco.mj_step(self.model, self.data)
            # Evaluate internal wrenches and accelerations after constraints
            mujoco.mj_rnePostConstraint(self.model, self.data)

            time_hist[step] = self.data.time
            cfrc_int_hist[step] = self.data.cfrc_int.copy()
            cfrc_ext_hist[step] = self.data.cfrc_ext.copy()
            cacc_hist[step] = self.data.cacc.copy()
            cvel_hist[step] = self.data.cvel.copy()
            xpos_hist[step] = self.data.xpos.copy()
            xmat_hist[step] = self.data.xmat.reshape(n_bodies, 3, 3).copy()

        self.trajectory = {
            "time": time_hist,
            "cfrc_int": cfrc_int_hist,
            "cfrc_ext": cfrc_ext_hist,
            "cacc": cacc_hist,
            "cvel": cvel_hist,
            "xpos": xpos_hist,
            "xmat": xmat_hist,
        }
        return self.trajectory

    def extract_peak_load_cases(
        self,
        body_name: str,
        top_k: int = 3,
        metric: str = "force_norm",
        time_snapshots: Optional[Sequence[float]] = None,
        min_step_distance: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify peak dynamic load cases for a target component body.

        Parameters
        ----------
        body_name : str
            Name of target body in MJCF model.
        top_k : int
            Number of peak load cases to extract (if time_snapshots is None).
        metric : str
            Metric for peak identification ('force_norm', 'moment_norm', 'total_norm', 'accel_norm').
        time_snapshots : Optional[Sequence[float]]
            Explicit time snapshots (in seconds) to extract.
        min_step_distance : int
            Minimum step separation between extracted peaks to avoid duplicate clusters.

        Returns
        -------
        List[Dict[str, Any]]
            List of peak load case records.
        """
        if self.trajectory is None:
            raise RuntimeError("Must run simulate_trajectory() before extracting peak load cases")

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MJCF model")

        times = self.trajectory["time"]
        cfrc_int = self.trajectory["cfrc_int"][:, body_id, :]  # (N_steps, 6)
        cfrc_ext = self.trajectory["cfrc_ext"][:, body_id, :]  # (N_steps, 6)
        cacc = self.trajectory["cacc"][:, body_id, :]          # (N_steps, 6)
        cvel = self.trajectory["cvel"][:, body_id, :]          # (N_steps, 6)
        xpos = self.trajectory["xpos"][:, body_id, :]          # (N_steps, 3)
        xmat = self.trajectory["xmat"][:, body_id, :, :]       # (N_steps, 3, 3)

        # In MuJoCo: 6D spatial vector convention is [moment_x, moment_y, moment_z, force_x, force_y, force_z]
        # and [alpha_x, alpha_y, alpha_z, a_x, a_y, a_z]
        total_wrench = cfrc_int + cfrc_ext
        moments = total_wrench[:, 0:3]
        forces = total_wrench[:, 3:6]
        ang_accels = cacc[:, 0:3]
        lin_accels = cacc[:, 3:6]
        ang_vels = cvel[:, 0:3]

        selected_indices: List[int] = []

        if time_snapshots is not None and len(time_snapshots) > 0:
            for t_target in time_snapshots:
                idx = int(np.argmin(np.abs(times - float(t_target))))
                selected_indices.append(idx)
        else:
            if metric == "force_norm":
                scores = np.linalg.norm(forces, axis=1)
            elif metric == "moment_norm":
                scores = np.linalg.norm(moments, axis=1)
            elif metric == "total_norm":
                scores = np.sqrt(np.sum(forces ** 2, axis=1) + np.sum(moments ** 2, axis=1))
            elif metric == "accel_norm":
                scores = np.linalg.norm(lin_accels, axis=1)
            else:
                raise ValueError(f"Unknown metric '{metric}'")

            # Peak finding with minimum step separation
            sorted_idx = np.argsort(scores)[::-1]
            for idx in sorted_idx:
                if len(selected_indices) >= top_k:
                    break
                if all(abs(idx - s) >= min_step_distance for s in selected_indices):
                    selected_indices.append(int(idx))

        load_cases: List[Dict[str, Any]] = []
        for i, idx in enumerate(selected_indices):
            load_cases.append({
                "case_id": i,
                "step_index": idx,
                "time": float(times[idx]),
                "force": forces[idx].copy(),
                "moment": moments[idx].copy(),
                "linear_accel": lin_accels[idx].copy(),
                "angular_accel": ang_accels[idx].copy(),
                "angular_vel": ang_vels[idx].copy(),
                "body_com": xpos[idx].copy(),
                "body_xmat": xmat[idx].copy(),
                "metric_value": float(np.linalg.norm(forces[idx])),
            })

        return load_cases

    def map_to_fe_loads(
        self,
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        peak_cases: List[Dict[str, Any]],
        attachment_node_indices: Optional[Sequence[int]] = None,
        density_mat: float = 1.0,
        apply_inertia_relief: bool = True,
        weights: Optional[Sequence[float]] = None,
    ) -> List[LoadCase]:
        """Map peak dynamic load cases onto finite element mesh nodes as balanced LoadCases.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Computational FE mesh.
        peak_cases : List[Dict[str, Any]]
            Extracted peak cases from extract_peak_load_cases.
        attachment_node_indices : Optional[Sequence[int]]
            Node indices where joint reaction force and moment are transferred via RBE3 coupling.
        density_mat : float
            Material density in kg/m^3.
        apply_inertia_relief : bool
            If True, applies exact rigid-body projection P_IR to eliminate residual imbalance.
        weights : Optional[Sequence[float]]
            Optimization multi-load case compliance weights.

        Returns
        -------
        List[LoadCase]
            List of LoadCase instances ready for topology optimization solver.
        """
        dim = mesh.ndim
        n_nodes = mesh.n_nodes
        ndof = mesh.ndof
        coords = mesh.node_coords

        # Compute lumped nodal masses: sum element volumes / nodes_per_elem
        nodal_mass = np.zeros(n_nodes, dtype=np.float64)
        elem_vols = mesh.elem_volumes * float(density_mat)
        elems = mesh.elements
        n_pe = elems.shape[1]
        for e in range(mesh.n_elem):
            m_e_node = elem_vols[e] / n_pe
            for n_idx in elems[e]:
                nodal_mass[n_idx] += m_e_node

        # Build diagonal mass matrix M for inertia relief
        diag_M = np.repeat(nodal_mass, dim)
        M_sp = sp.diags(diag_M, format="csr")

        # Resolve attachment nodes
        if attachment_node_indices is not None and len(attachment_node_indices) > 0:
            att_nodes = np.asarray(attachment_node_indices, dtype=np.int64)
        else:
            # Default: nodes on minimum z/y face (bottom interface)
            coord_min = np.min(coords[:, dim - 1])
            att_nodes = np.flatnonzero(coords[:, dim - 1] <= coord_min + 1e-4)

        n_att = len(att_nodes)
        att_coords = coords[att_nodes, :dim]
        att_center = np.mean(att_coords, axis=0)
        r_att = att_coords - att_center  # (n_att, dim)

        # RBE3 moment coupling tensor J = sum_i (||r_i||^2 * I - r_i * r_i^T)
        if dim == 3:
            J_att = np.zeros((3, 3), dtype=np.float64)
            for i in range(n_att):
                ri = r_att[i]
                J_att += np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)
            # Regularize if flat
            J_att += 1e-12 * np.eye(3)
            J_att_inv = np.linalg.inv(J_att)

        fe_cases: List[LoadCase] = []

        for k_idx, case in enumerate(peak_cases):
            F_global = np.zeros(ndof, dtype=np.float64)

            f_joint = np.asarray(case["force"], dtype=np.float64)[:dim]
            m_joint = np.asarray(case["moment"], dtype=np.float64)
            a_lin = np.asarray(case["linear_accel"], dtype=np.float64)[:dim]
            a_ang = np.asarray(case["angular_accel"], dtype=np.float64)
            omega = np.asarray(case["angular_vel"], dtype=np.float64)
            com = np.asarray(case["body_com"], dtype=np.float64)[:dim]

            # 1. Distribute joint reaction force & moment to attachment nodes
            for i, nid in enumerate(att_nodes):
                f_node = f_joint / n_att
                if dim == 3:
                    ri = r_att[i]
                    # Moment distribution: f_mom = (J^-1 * m) x r_i
                    f_mom = np.cross(J_att_inv @ m_joint, ri)
                    f_node = f_node + f_mom
                elif dim == 2:
                    # 2D torque about z: m_z * (r_y, -r_x) / sum(r^2)
                    r_sq = np.sum(r_att ** 2)
                    if r_sq > 1e-14:
                        m_z = m_joint[2] if len(m_joint) > 2 else 0.0
                        f_mom = (m_z / r_sq) * np.array([-r_att[i, 1], r_att[i, 0]])
                        f_node = f_node + f_mom

                F_global[nid * dim : (nid + 1) * dim] += f_node

            # 2. D'Alembert inertial forces on every node:
            # f_iner,i = -m_i * [a_lin + a_ang x r_i + omega x (omega x r_i)]
            r_all = coords[:, :dim] - com
            for nid in range(n_nodes):
                mi = nodal_mass[nid]
                if mi <= 0.0:
                    continue
                ri = r_all[nid]
                if dim == 3:
                    accel_i = (
                        a_lin
                        + np.cross(a_ang, ri)
                        + np.cross(omega, np.cross(omega, ri))
                    )
                else:
                    om_z = omega[2] if len(omega) > 2 else 0.0
                    al_z = a_ang[2] if len(a_ang) > 2 else 0.0
                    accel_i = (
                        a_lin
                        + np.array([-al_z * ri[1], al_z * ri[0]])
                        - (om_z ** 2) * ri
                    )
                F_global[nid * dim : (nid + 1) * dim] += -mi * accel_i

            # 3. Apply Inertia Relief projection P_IR to eliminate residual imbalance
            if apply_inertia_relief:
                F_balanced = FEASolver.apply_inertia_relief(
                    F=F_global,
                    M=M_sp,
                    node_coords=coords,
                    dim=dim,
                )
            else:
                F_balanced = F_global

            w = float(weights[k_idx]) if weights is not None else 1.0
            case_name = f"mujoco_peak_{case.get('case_id', k_idx)}_t{case.get('time', 0.0):.3f}s"
            fe_cases.append(LoadCase(name=case_name, force_vector=F_balanced, weight=w))

        return fe_cases

    @staticmethod
    def compute_optimized_inertial_properties(
        mesh: Union[StructuredMesh2D, StructuredMesh3D, UnstructuredMesh],
        rho: np.ndarray,
        density_mat: float = 1.0,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute mass, COM, and 3x3 inertia tensor from topology optimized density field.

        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or UnstructuredMesh
            Finite element mesh.
        rho : np.ndarray
            Element physical density array of shape (N_elem,).
        density_mat : float
            Material density in kg/m^3.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            (total_mass, center_of_mass, inertia_tensor_3x3).
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        vols = mesh.elem_volumes
        m_elem = rho_arr * vols * float(density_mat)
        total_mass = float(np.sum(m_elem))

        if total_mass < 1e-12:
            raise ValueError("Total mass is near zero; cannot compute inertial properties")

        centers = mesh.elem_centers
        dim = mesh.ndim

        # Center of mass
        com_nd = np.sum(m_elem[:, None] * centers, axis=0) / total_mass
        com_3d = np.zeros(3, dtype=np.float64)
        com_3d[:dim] = com_nd

        # 3x3 inertia tensor about COM
        r = np.zeros((mesh.n_elem, 3), dtype=np.float64)
        r[:, :dim] = centers - com_nd

        Ixx = np.sum(m_elem * (r[:, 1] ** 2 + r[:, 2] ** 2))
        Iyy = np.sum(m_elem * (r[:, 0] ** 2 + r[:, 2] ** 2))
        Izz = np.sum(m_elem * (r[:, 0] ** 2 + r[:, 1] ** 2))
        Ixy = -np.sum(m_elem * r[:, 0] * r[:, 1])
        Ixz = -np.sum(m_elem * r[:, 0] * r[:, 2])
        Iyz = -np.sum(m_elem * r[:, 1] * r[:, 2])

        # Add element local bounding-box moment of inertia
        if isinstance(mesh, StructuredMesh3D):
            dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
            Ixx += (1.0 / 12.0) * np.sum(m_elem) * (dy**2 + dz**2)
            Iyy += (1.0 / 12.0) * np.sum(m_elem) * (dx**2 + dz**2)
            Izz += (1.0 / 12.0) * np.sum(m_elem) * (dx**2 + dy**2)
        elif isinstance(mesh, StructuredMesh2D):
            dx, dy = mesh.dx, mesh.dy
            Izz += (1.0 / 12.0) * np.sum(m_elem) * (dx**2 + dy**2)

        inertia_3x3 = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ], dtype=np.float64)

        return total_mass, com_3d, inertia_3x3

    @staticmethod
    def update_mjcf_inertial(
        xml_str: str,
        body_name: str,
        mass: float,
        com: Sequence[float],
        inertia_tensor: np.ndarray,
    ) -> str:
        """Update <inertial> tag of body_name in MJCF XML string for closed-loop dynamic re-simulation.

        Parameters
        ----------
        xml_str : str
            MJCF model XML string.
        body_name : str
            Name of target body.
        mass : float
            Updated component mass in kg.
        com : Sequence[float]
            Updated component center of mass [x, y, z] in body frame.
        inertia_tensor : np.ndarray
            (3, 3) updated inertia tensor.

        Returns
        -------
        str
            Updated MJCF XML string.
        """
        root = ET.fromstring(xml_str)
        target_body = None
        for b in root.iter("body"):
            if b.attrib.get("name") == body_name:
                target_body = b
                break

        if target_body is None:
            raise ValueError(f"Body '{body_name}' not found in MJCF XML")

        inertial = target_body.find("inertial")
        if inertial is None:
            inertial = ET.Element("inertial")
            target_body.insert(0, inertial)

        I = np.asarray(inertia_tensor, dtype=np.float64)
        Ixx, Iyy, Izz = float(I[0, 0]), float(I[1, 1]), float(I[2, 2])
        Ixy, Ixz, Iyz = float(I[0, 1]), float(I[0, 2]), float(I[1, 2])

        com_arr = np.asarray(com, dtype=np.float64)
        pos_str = f"{com_arr[0]:.6g} {com_arr[1]:.6g} {com_arr[2]:.6g}"
        fullinertia_str = f"{Ixx:.6g} {Iyy:.6g} {Izz:.6g} {Ixy:.6g} {Ixz:.6g} {Iyz:.6g}"

        inertial.set("pos", pos_str)
        inertial.set("mass", f"{float(mass):.6g}")
        inertial.set("fullinertia", fullinertia_str)

        return ET.tostring(root, encoding="unicode")
