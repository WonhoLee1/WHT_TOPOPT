"""Inspect the mathematical & geometric reasons why ex07 algorithms yielded identical compliances (within 0.03%).

Compares:
1. ex07 Synthetic Box Model (voxelize_ge_aircraft_bracket)
2. Actual GrabCAD Challenge STL Model (GE_domain.STL, GE_fixed.STL, GE_force.STL)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wht_topopt.core.voxelizer import voxelize_ge_aircraft_bracket
from wht_topopt.core.stl_domain import StlCase, build_stl_problem

def inspect():
    print("=" * 80)
    print(" [INSPECTION] ex07 Synthetic Box Model vs Real Challenge STL Model")
    print("=" * 80)

    # 1. ex07 Synthetic Box Model (from ex07_3d_bracket_benchmark.py: nx=40, ny=20, nz=16, or nx=20, ny=10, nz=8)
    # ex07 benchmark used resolution 1600 elements: 20 x 10 x 8
    dom_syn_1600 = voxelize_ge_aircraft_bracket(nx=20, ny=10, nz=8)
    n_elem_syn = dom_syn_1600.n_elements
    n_solid_syn = int(np.sum(dom_syn_1600.passive_solid_mask))
    n_void_syn = int(np.sum(dom_syn_1600.passive_void_mask))
    n_design_syn = int(np.sum(dom_syn_1600.design_mask))

    print(f"\n1. ex07 Synthetic Box Model (nx=20, ny=10, nz=8 = 1,600 elems):")
    print(f"   - Total elements: {n_elem_syn}")
    print(f"   - Passive Solid:  {n_solid_syn:4d} ({n_solid_syn / n_elem_syn * 100:.2f}%)")
    print(f"   - Passive Void:   {n_void_syn:4d} ({n_void_syn / n_elem_syn * 100:.2f}%)")
    print(f"   - Design Elements:{n_design_syn:4d} ({n_design_syn / n_elem_syn * 100:.2f}%)")
    print(f"   - Non-design Fixed Ratio: {(n_solid_syn + n_void_syn) / n_elem_syn * 100:.2f}%")
    print(f"   - Target VolFrac: 0.30")
    print(f"   - Target Solid Count: {int(round(0.30 * n_elem_syn))} elements")
    print(f"   - Free Elements to choose: {int(round(0.30 * n_elem_syn)) - n_solid_syn} elements only!")

    # Check for nx=40, ny=20, nz=16 (12,800 elems)
    dom_syn_12k = voxelize_ge_aircraft_bracket(nx=40, ny=20, nz=16)
    n_elem_12k = dom_syn_12k.n_elements
    n_solid_12k = int(np.sum(dom_syn_12k.passive_solid_mask))
    n_void_12k = int(np.sum(dom_syn_12k.passive_void_mask))
    n_design_12k = int(np.sum(dom_syn_12k.design_mask))
    print(f"\n   ex07 Synthetic Box Model (nx=40, ny=20, nz=16 = 12,800 elems):")
    print(f"   - Total elements: {n_elem_12k}")
    print(f"   - Passive Solid:  {n_solid_12k:4d} ({n_solid_12k / n_elem_12k * 100:.2f}%)")
    print(f"   - Passive Void:   {n_void_12k:4d} ({n_void_12k / n_elem_12k * 100:.2f}%)")
    print(f"   - Design Elements:{n_design_12k:4d} ({n_design_12k / n_elem_12k * 100:.2f}%)")
    print(f"   - Non-design Fixed Ratio: {(n_solid_12k + n_void_12k) / n_elem_12k * 100:.2f}%")

    # 2. Real Challenge STL Model
    stl_dir = ROOT / "external" / "freeto_stls"
    case = StlCase(
        domain="GE_domain.STL", forces=[], fixed=["GE_fixed.STL"], pin_region="GE_force.STL",
        pin_loads=[((0.0, 35585.8, 0.0), (0.0, 0.0, 0.0))],
    )
    p_real, inside_real = build_stl_problem(case, mesh_control=40, volfrac=0.15, base=stl_dir)
    n_elem_real = p_real.mesh.n_elems
    n_in_real = int(np.sum(inside_real))
    n_solid_real = len(p_real.passive_solid)
    n_void_real = len(p_real.passive_void)
    n_design_real = n_in_real - n_solid_real

    print(f"\n2. Real GrabCAD Challenge STL Model (MC=40, 15,000 elems):")
    print(f"   - Total Bounding Grid: {n_elem_real} elements (Grid {p_real.mesh.grid})")
    print(f"   - Actual Domain Envelope (Inside STL): {n_in_real:4d} elements ({n_in_real / n_elem_real * 100:.2f}%)")
    print(f"   - Passive Solid (Bolt & Pin surfaces): {n_solid_real:4d} ({n_solid_real / n_in_real * 100:.2f}% of in-domain)")
    print(f"   - Free Design Elements:                {n_design_real:4d} ({n_design_real / n_in_real * 100:.2f}% of in-domain)")
    print(f"   - Free Design Freedom in domain: {n_design_real / n_in_real * 100:.1f}% !!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect()
