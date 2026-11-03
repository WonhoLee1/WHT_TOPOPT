"""What the three 2D basics share: a non-dimensional material, one load case, and the pillars that do not apply to 2D statics."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from wht_topopt.examples.src_v1.common.pillar1_algorithms import LoadSet
from wht_topopt.examples.src_v1.common.pillar5_audit import StiffnessTest, TestRig
from wht_topopt.examples.src_v1.common.structure import Structure
from wht_topopt.examples.src_v1.common.study import Study

FAMILY = "ex00_2d_basic"
NOT_APPLICABLE = {
    2: "정적 2D 기본 예제: 과도 하중 없음",
    3: "정적 2D 기본 예제: 동적 성능 목표 없음",
    4: "2D 평면 문제: 압출, 스킨-리브 필터 대상 아님",
}


def study_2d(name: str, mesh, fixed: np.ndarray, force: np.ndarray, volfrac: float, rmin: float, passive_solid: Sequence[int] = (), passive_void: Sequence[int] = (), iters: Optional[int] = None, smoke: bool = False) -> Study:
    """Benchmark in E = 1, nu = 0.3, F = 1: no yield stress and no density."""
    s = Structure(mesh, fixed, e0=1.0, nu=0.3, density=None, yield_stress=None, passive_solid=passive_solid, passive_void=passive_void, emin_ratio=1e-9)
    total = float(np.sum(np.abs(force)))
    loads = LoadSet(force, np.ones(1), ["load"])
    test = StiffnessTest("K_load", force, total, lambda u: float(force @ u) / total, "F/u")  # displacement conjugate to the load
    return Study(name=name, family=FAMILY, structure=s, loads=loads, rig=TestRig([test], loads), volfrac=volfrac, rmin=rmin, not_applicable=NOT_APPLICABLE, smoke=smoke, iters=iters)
