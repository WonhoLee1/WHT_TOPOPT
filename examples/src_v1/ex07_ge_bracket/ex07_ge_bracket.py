"""ex07: GE jet engine bracket challenge (GrabCAD 2013), the full 6-pillar study.

    python wht_topopt/examples/src_v1/ex07_ge_bracket/ex07_ge_bracket.py [--smoke] [--algos mma semdot] [--iters 30]

Domain and interfaces from FreeTO's GE_domain.STL, GE_fixed.STL, GE_force.STL. Ti-6Al-4V, the 0.75 in pin is a rigid body on the lug bores, four bolts
held. Static cases: 8000 lbf up, 8500 lbf out, 9500 lbf at 42 deg, 5000 lb-in torque. Dynamic case: gust on the pin, F(t) = F_peak sin(pi t / T) exp(-1.5 t / T),
T = 50 ms, in the direction of 0.7 (case 1 + case 2). Replaces ex07_ge_bracket_*, ex07_se_ahca_signal_study, ex08_kratos_* and ex11_ge_bracket_hca (see src_bak).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from typing import Optional

import numpy as np

from wht_topopt.core.stl_domain import StlCase, build_stl_problem
from wht_topopt.examples.src_v1.common.pillar1_algorithms import LoadSet
from wht_topopt.examples.src_v1.common.pillar2_static_dynamic import LoadHistory
from wht_topopt.examples.src_v1.common.pillar4_manufacturing import DensityMap, ExtrusionMap, SkinMap, skin_elements
from wht_topopt.examples.src_v1.common.pillar5_audit import StiffnessTest, TestRig
from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.common.structure import Structure

ROOT = Path(__file__).resolve().parents[4]
FAMILY = "ex07_ge_bracket"
LBF, INCH = 4.4482216152605, 25.4
C42, S42 = float(np.cos(np.radians(42.0))), float(np.sin(np.radians(42.0)))
E_TI, NU_TI, YIELD_TI, DENSITY_TI = 113800.0, 0.342, 903.0, 4.43e-9  # MPa, -, MPa, tonne/mm^3
CASE_GE = StlCase(
    domain="GE_domain.STL",
    forces=[],
    fixed=["GE_fixed.STL"],
    pin_region="GE_force.STL",
    pin_loads=[
        ((0.0, 8000.0 * LBF, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 8500.0 * LBF), (0.0, 0.0, 0.0)),
        ((0.0, 9500.0 * LBF * C42, 9500.0 * LBF * S42), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 5000.0 * LBF * INCH, 0.0)),
    ],
)
CASE_NAMES = ["vertical_8000lbf", "horizontal_8500lbf", "angled_9500lbf", "torque_5000lbin"]
CASE_WEIGHTS = np.array([0.35, 0.35, 0.20, 0.10])
GUST_T = 0.050  # s
GUST_WINDOW = 0.070  # s
SKIN_THICKNESS = 2.0  # mm
SKIN_FACES = ("z-", "z+", "y+")


def build(mesh_control: int = 60, volfrac: float = 0.15, delta_v: float = 0.05, rmin_elements: float = 2.2, iters: Optional[int] = None, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    p, inside = build_stl_problem(CASE_GE, mesh_control, volfrac, base=ROOT / "external" / "freeto_stls", E0=E_TI, nu=NU_TI)
    void = np.flatnonzero(~inside)
    solid = np.setdiff1d(p.passive_solid, void)
    s = Structure(p.mesh, p.fixed_dofs, E_TI, NU_TI, DENSITY_TI, YIELD_TI, passive_solid=solid, passive_void=void, rigid_bodies=p.rigid_bodies)
    h = float(s.size[0])
    rmin = rmin_elements * h
    pin = p.rigid_bodies[0]
    forces = np.stack([lc.force for lc in p.load_cases])
    loads = LoadSet(forces, CASE_WEIGHTS, CASE_NAMES)

    def unit(node: int, axis: int, amount: float) -> np.ndarray:
        f = np.zeros(s.ndof)
        f[3 * node + axis] = amount
        return f

    tests = [
        StiffnessTest("K_x", unit(pin.ref_node, 0, 1000.0), 1000.0, lambda u: u[3 * pin.ref_node], "N/mm"),
        StiffnessTest("K_y", unit(pin.ref_node, 1, 1000.0), 1000.0, lambda u: u[3 * pin.ref_node + 1], "N/mm"),
        StiffnessTest("K_z", unit(pin.ref_node, 2, 1000.0), 1000.0, lambda u: u[3 * pin.ref_node + 2], "N/mm"),
        StiffnessTest("K_theta_x", unit(pin.rot_node, 0, 50.0e3), 50.0e3, lambda u: u[3 * pin.rot_node], "N*mm/rad"),
    ]
    direction = 0.7 * (forces[0] + forces[1])

    def gust(t: float) -> np.ndarray:
        if t <= 0.0 or t > GUST_T:
            return np.zeros(s.ndof)
        return direction * (np.sin(np.pi * t / GUST_T) * np.exp(-1.5 * t / GUST_T))

    resultant = float(np.linalg.norm(direction[3 * pin.ref_node : 3 * pin.ref_node + 3]))
    dynamic = LoadHistory(
        name="aircraft gust on the pin",
        formula=f"F(t) = 0.7 (F_case1 + F_case2) sin(pi t / T) exp(-1.5 t / T), T = {GUST_T * 1e3:.0f} ms, zero afterwards",
        duration=GUST_WINDOW,
        force_at=gust,
        peak_force=resultant,
        tip_response=lambda u: np.array([u[3 * pin.ref_node + 1], u[3 * pin.ref_node + 2]]),
        load_end=GUST_T,
    )
    rig = TestRig(tests, loads, dynamic)
    skin = skin_elements(inside, s.grid, s.size, SKIN_THICKNESS, SKIN_FACES)
    return Study(
        name="ex07_ge_bracket",
        family=FAMILY,
        structure=s,
        loads=loads,
        rig=rig,
        volfrac=volfrac,
        delta_v=delta_v,
        rmin=rmin,
        make_map=lambda: DensityMap(s, rmin, beta0=1.0, beta_max=8.0, beta_every=20),
        manufacturing={
            "extrude_y": lambda: ExtrusionMap(s, rmin, "y"),
            "extrude_z": lambda: ExtrusionMap(s, rmin, "z"),
            "skin_zm_zp_yp": lambda: SkinMap(s, rmin, skin),
        },
        bending_key="K_y",
        torsion_key="Uz",
        smoke=smoke,
        iters=iters,
        max_steps=max_steps,
    )


# no cap on the time steps in the full study: omega_1 dt <= 0.5 for every design, which makes the dynamic models the expensive part
FULL = dict(mesh_control=60)
SMOKE = dict(mesh_control=24, iters=4, max_steps=20, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE)
