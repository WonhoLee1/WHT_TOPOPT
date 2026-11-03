"""ex00-dyn-a: the ex00a cantilever under the 20 s load history.

    python wht_topopt/examples/src_v1/ex00_dyn/ex00_dyn_a_cantilever.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from typing import Optional

from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.ex00_2d_basic import ex00a_cantilever
from wht_topopt.examples.src_v1.ex00_dyn._common import DEFAULT_ALGOS, dynamic_study_2d


def build(nelx: int = 180, nely: int = 60, volfrac: float = 0.40, rmin: float = 2.4, iters: Optional[int] = 60, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    base = ex00a_cantilever.build(nelx=nelx, nely=nely, volfrac=volfrac, rmin=rmin)
    return dynamic_study_2d("ex00-dyn-a-cantilever", base.structure.mesh, base.structure.fixed_dofs, base.loads.forces[0], volfrac, rmin, iters=iters, max_steps=max_steps, smoke=smoke)


FULL = dict(nelx=180, nely=60)
SMOKE = dict(nelx=48, nely=16, iters=4, max_steps=140, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE, default_algos=DEFAULT_ALGOS)
