"""ex00b: 2D L-bracket. The upper right 60 % x 60 % is void, the top of the vertical arm is clamped, unit downward load at the end of
the horizontal arm, on a small passive solid interface.

    python wht_topopt/examples/src_v1/ex00_2d_basic/ex00b_lbracket.py [--smoke]
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
from wht_topopt.examples.src_v1.ex00_2d_basic._common import study_2d


def build(nel: int = 120, volfrac: float = 0.40, rmin: float = 2.4, cut: float = 0.4, iters: Optional[int] = None, smoke: bool = False) -> Study:
    mesh = StructuredMesh2D(nelx=nel, nely=nel, lx=float(nel), ly=float(nel))
    n, c = mesh.nodes, mesh.elem_centers
    corner = cut * nel
    void = np.flatnonzero((c[:, 0] > corner) & (c[:, 1] > corner))
    top = np.flatnonzero((n[:, 1] > nel - 1e-9) & (n[:, 0] <= corner + 1e-9))
    tip = np.flatnonzero((n[:, 0] > nel - 1e-9) & (n[:, 1] >= corner - 2.0 - 1e-9) & (n[:, 1] <= corner + 1e-9))
    force = np.zeros(mesh.ndof)
    force[2 * tip + 1] = -1.0 / len(tip)
    interface = np.flatnonzero((c[:, 0] > nel - 3.0) & (c[:, 1] > corner - 4.0) & (c[:, 1] < corner))
    return study_2d("ex00b_lbracket", mesh, np.concatenate([2 * top, 2 * top + 1]), force, volfrac, rmin, passive_solid=interface, passive_void=void, iters=iters, smoke=smoke)


FULL = dict(nel=120)
SMOKE = dict(nel=40, iters=12, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE)
