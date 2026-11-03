"""ex00-dyn-b: the ex00b L-bracket under the 20 s load history.

    python wht_topopt/examples/src_v1/ex00_dyn/ex00_dyn_b_lbracket.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from typing import Optional

import numpy as np

from wht_topopt.examples.src_v1.common.study import Study, main
from wht_topopt.examples.src_v1.ex00_2d_basic import ex00b_lbracket
from wht_topopt.examples.src_v1.ex00_dyn._common import DEFAULT_ALGOS, dynamic_study_2d


def build(nel: int = 120, volfrac: float = 0.40, rmin: float = 2.4, iters: Optional[int] = 60, max_steps: Optional[int] = None, smoke: bool = False) -> Study:
    base = ex00b_lbracket.build(nel=nel, volfrac=volfrac, rmin=rmin)
    s = base.structure
    return dynamic_study_2d("ex00-dyn-b-lbracket", s.mesh, s.fixed_dofs, base.loads.forces[0], volfrac, rmin, np.flatnonzero(s.solid_mask), np.flatnonzero(s.void_mask), iters=iters, max_steps=max_steps, smoke=smoke)


FULL = dict(nel=120)
SMOKE = dict(nel=40, iters=4, max_steps=140, smoke=True)

if __name__ == "__main__":
    main(build, FULL, SMOKE, default_algos=DEFAULT_ALGOS)
