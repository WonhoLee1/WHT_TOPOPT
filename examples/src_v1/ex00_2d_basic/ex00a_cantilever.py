"""ex00a: 2D cantilever. Left edge clamped, unit downward load at the middle of the right edge.

    python wht_topopt/examples/src_v1/ex00_2d_basic/ex00a_cantilever.py [--smoke]
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


def build(nelx: int = 180, nely: int = 60, volfrac: float = 0.40, rmin: float = 2.4, iters: Optional[int] = None, smoke: bool = False) -> Study:
    mesh = StructuredMesh2D(nelx=nelx, nely=nely, lx=float(nelx), ly=float(nely))
    n = mesh.nodes
    left = np.flatnonzero(n[:, 0] < 1e-9)
    tip = np.flatnonzero((n[:, 0] > nelx - 1e-9) & (np.abs(n[:, 1] - nely / 2.0) <= 1.0 + 1e-9))
    force = np.zeros(mesh.ndof)
    force[2 * tip + 1] = -1.0 / len(tip)
    return study_2d("ex00a_cantilever", mesh, np.concatenate([2 * left, 2 * left + 1]), force, volfrac, rmin, iters=iters, smoke=smoke)


FULL = dict(nelx=180, nely=60)
SMOKE = dict(nelx=48, nely=16, iters=12, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE)
