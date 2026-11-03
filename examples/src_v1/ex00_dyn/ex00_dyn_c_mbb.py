"""ex00-dyn-c: the full-span MBB beam under the 20 s load history.

The half model of ex00c cannot carry this load: its load node lies on the symmetry plane, where ux is held, and an x load is not symmetric anyway.
So the whole span is modelled: load at the top centre, pin at the lower left corner, roller at the lower right corner.

    python wht_topopt/examples/src_v1/ex00_dyn/ex00_dyn_c_mbb.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from typing import Optional

import numpy as np

from wht_topopt.core.mesh import StructuredMesh2D
from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.ex00_dyn._common import DEFAULT_ALGOS, dynamic_study_2d


def build(nelx: int = 360, nely: int = 60, volfrac: float = 0.40, rmin: float = 2.4, iters: Optional[int] = 60, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    mesh = StructuredMesh2D(nelx=nelx, nely=nely, lx=float(nelx), ly=float(nely))
    n = mesh.nodes
    pin = np.flatnonzero((n[:, 0] < 1e-9) & (n[:, 1] < 1e-9))
    roller = np.flatnonzero((n[:, 0] > nelx - 1e-9) & (n[:, 1] < 1e-9))
    load = np.flatnonzero((np.abs(n[:, 0] - nelx / 2.0) < 1e-9) & (n[:, 1] > nely - 1e-9))
    if load.size != 1:
        raise ValueError(f"nelx must be even: the load needs a node at the top centre, got {load.size}")
    force = np.zeros(mesh.ndof)
    force[2 * load + 1] = -1.0
    return dynamic_study_2d("ex00-dyn-c-mbb", mesh, np.concatenate([2 * pin, 2 * pin + 1, 2 * roller + 1]), force, volfrac, rmin, iters=iters, max_steps=max_steps, smoke=smoke)


FULL = dict(nelx=360, nely=60)
SMOKE = dict(nelx=96, nely=16, iters=4, max_steps=140, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE, default_algos=DEFAULT_ALGOS)
