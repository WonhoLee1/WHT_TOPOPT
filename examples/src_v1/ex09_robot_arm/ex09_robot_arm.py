"""ex09: humanoid robot arm link, the full 6-pillar study.

    python wht_topopt/examples/src_v1/ex09_robot_arm/ex09_robot_arm.py [--smoke] [--algos mma semdot] [--iters 30]

160 x 48 x 40 mm link of Al 7075-T6 between two motor hubs, base hub bore clamped, 1.5 kg payload at the output hub, 50 ms S-curve slewing through 90 deg.
Pillars 2 and 3 use the skin-rib design map (bottom skin, side walls, vertical interior ribs). Replaces ex09_*, compare_dynamic_extrusion_all and
ex11_robot_arm_hca (see src_bak).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from typing import Optional

import numpy as np

from wht_topopt.core.mesh import StructuredMesh3D
from wht_topopt.examples.src_v1.common.pillar1_algorithms import LoadSet
from wht_topopt.examples.src_v1.common.pillar2_static_dynamic import LoadHistory
from wht_topopt.examples.src_v1.common.pillar4_manufacturing import ExtrusionMap, RibMap
from wht_topopt.examples.src_v1.common.pillar5_audit import StiffnessTest, TestRig
from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.common.structure import Structure

FAMILY = "ex09_robot_arm"
E_AL, NU_AL, YIELD_AL, DENSITY_AL = 71000.0, 0.33, 503.0, 2.81e-9  # MPa, -, MPa, tonne/mm^3
LX, LY, LZ = 160.0, 48.0, 40.0
HUB1, HUB2 = np.array([20.0, 24.0]), np.array([140.0, 24.0])
R1_VOID, R2_VOID, R1_SOLID, R2_SOLID = 8.0, 7.0, 12.0, 11.0
PAYLOAD_TONNE = 1.5e-3
T_TOTAL, THETA_MAX = 0.05, np.pi / 2.0
ARM_M, ECCENTRICITY_M = 0.120, 0.040


def build(nx: int = 50, ny: int = 24, nz: int = 20, volfrac: float = 0.28, delta_v: float = 0.07, rmin: float = 6.0, r_rib: float = 2.8, iters: Optional[int] = None, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    mesh = StructuredMesh3D(nelx=nx, nely=ny, nelz=nz, lx=LX, ly=LY, lz=LZ)
    c = mesh.elem_centers
    d1, d2 = np.hypot(*(c[:, :2] - HUB1).T), np.hypot(*(c[:, :2] - HUB2).T)
    void = (d1 <= R1_VOID) | (d2 <= R2_VOID)
    hubs = ((d1 > R1_VOID) & (d1 <= R1_SOLID)) | ((d2 > R2_VOID) & (d2 <= R2_SOLID))
    skin = ((c[:, 2] < mesh.dz) | (c[:, 1] < mesh.dy) | (c[:, 1] >= LY - mesh.dy)) & ~void
    n = mesh.node_coords
    nd1, nd2 = np.hypot(*(n[:, :2] - HUB1).T), np.hypot(*(n[:, :2] - HUB2).T)
    clamped = np.flatnonzero(np.abs(nd1 - R1_VOID) <= 0.8)
    loaded = np.flatnonzero(np.abs(nd2 - R2_VOID) <= 0.8)
    if clamped.size == 0 or loaded.size == 0:
        raise ValueError("no node on a hub bore: the mesh is too coarse")
    fixed = (3 * clamped[:, None] + np.arange(3)).ravel()
    lumped = np.zeros(mesh.ndof)
    lumped[(3 * loaded[:, None] + np.arange(3)).ravel()] = PAYLOAD_TONNE / loaded.size
    s = Structure(mesh, fixed, E_AL, NU_AL, DENSITY_AL, YIELD_AL, passive_solid=np.flatnonzero(hubs | skin), passive_void=np.flatnonzero(void), lumped_mass=lumped, rayleigh=(5.0, 1e-5))

    hub_center = np.array([HUB2[0], HUB2[1], LZ / 2.0])
    n_ld = loaded.size
    arm_y = (n[loaded, 1] - hub_center[1]) * 1e-3  # m
    arm_z = (n[loaded, 2] - hub_center[2]) * 1e-3

    def force_vector(fx: float, fy: float, fz: float, tx_nm: float) -> np.ndarray:
        """Hub load: resultants shared evenly, the torque as tangential forces (N: moment in N*m over an arm in m)."""
        f = np.zeros(mesh.ndof)
        r2 = arm_y**2 + arm_z**2 + 1e-12
        f[3 * loaded] = fx / n_ld
        f[3 * loaded + 1] = fy / n_ld - (tx_nm / n_ld) * arm_z / r2
        f[3 * loaded + 2] = fz / n_ld + (tx_nm / n_ld) * arm_y / r2
        return f

    loads = LoadSet(
        np.stack([force_vector(620.0, 0.0, 0.0, 0.0), force_vector(0.0, -750.0, 0.0, 0.0), force_vector(0.0, 0.0, 350.0, 0.0), force_vector(0.0, 0.0, 0.0, 28.0)]),
        np.array([0.25, 0.35, 0.15, 0.25]),
        ["axial_620N", "slew_750N", "shear_350N", "torsion_28Nm"],
    )
    mean = lambda u, a: float(np.mean(u[3 * loaded + a]))
    f_tors = force_vector(0.0, 0.0, 0.0, 50.0)
    tests = [
        StiffnessTest("K_x", force_vector(1000.0, 0.0, 0.0, 0.0), 1000.0, lambda u: mean(u, 0), "N/mm"),
        StiffnessTest("K_y", force_vector(0.0, 1000.0, 0.0, 0.0), 1000.0, lambda u: mean(u, 1), "N/mm"),
        StiffnessTest("K_z", force_vector(0.0, 0.0, 1000.0, 0.0), 1000.0, lambda u: mean(u, 2), "N/mm"),
        # the rotation conjugate to the torque: work (N*mm) / moment (N*mm)
        StiffnessTest("K_theta_x", f_tors, 50.0e3, lambda u: float(f_tors @ u) / 50.0e3, "N*mm/rad"),
    ]

    def slew(t: float) -> np.ndarray:
        omega = (THETA_MAX / T_TOTAL) * (1.0 - np.cos(2.0 * np.pi * t / T_TOTAL))
        alpha = (2.0 * np.pi * THETA_MAX / T_TOTAL**2) * np.sin(2.0 * np.pi * t / T_TOTAL)
        m = 1.5  # kg
        fx = 0.7 * m * omega**2 * ARM_M - 420.0 * np.sin(2.0 * np.pi * t / T_TOTAL) * np.sin(np.pi * t / T_TOTAL)
        fy = -m * alpha * ARM_M + 150.0 * np.sin(4.0 * np.pi * t / T_TOTAL)
        fz = 350.0 * np.sin(3.0 * np.pi * t / T_TOTAL) * np.exp(-20.0 * t)
        tx = m * alpha * ARM_M * ECCENTRICITY_M + 12.0 * np.sin(3.0 * np.pi * t / T_TOTAL)
        return force_vector(float(fx), float(fy), float(fz), float(tx))

    dynamic = LoadHistory(
        name="50 ms S-curve slewing",
        formula="omega(t) = (theta_max/T)(1 - cos(2 pi t/T)), theta_max = 90 deg, T = 50 ms; Fx = 0.7 m omega^2 L - 420 sin(2 pi t/T) sin(pi t/T), "
                "Fy = -m alpha L + 150 sin(4 pi t/T), Fz = 350 sin(3 pi t/T) exp(-20 t), Tx = m alpha L e + 12 sin(3 pi t/T); m = 1.5 kg, L = 120 mm, e = 40 mm",
        duration=T_TOTAL,
        force_at=slew,
        peak_force=750.0,
        tip_response=lambda u: np.array([mean(u, 1), mean(u, 2)]),
        load_end=0.8 * T_TOTAL,
    )
    return Study(
        name="ex09_robot_arm",
        family=FAMILY,
        structure=s,
        loads=loads,
        rig=TestRig(tests, loads, dynamic),
        volfrac=volfrac,
        delta_v=delta_v,
        rmin=rmin,
        make_map=lambda: RibMap(s, r_rib),
        # the skin-rib filter is already the design map of models 1 to 3
        manufacturing={
            "extrude_y": lambda: ExtrusionMap(s, rmin, "y"),
            "extrude_z": lambda: ExtrusionMap(s, rmin, "z"),
        },
        bending_key="K_y",
        torsion_key="Uz",
        smoke=smoke,
        iters=iters,
        max_steps=max_steps,
    )


FULL = dict(nx=50, ny=24, nz=20)
SMOKE = dict(nx=40, ny=24, nz=10, iters=4, max_steps=30, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE)
