"""Pillar 3: the dynamic optimisation with a relaxed volume, and whether it dominates both earlier models.

The spec's targets are checked against measured audits and reported as passed or missed; a missed target is a result, not an error.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .pillar1_algorithms import DesignRun
from .pillar2_static_dynamic import LoadHistory, optimize_dynamic
from .structure import Structure


def run_volume_relaxed(s: Structure, make_map: Callable[[], object], load: LoadHistory, volfrac: float, delta_v: float, iters: Optional[int] = None, max_steps: Optional[int] = None) -> DesignRun:
    """Model 3: the dynamic model at ``volfrac + delta_v``. The spec asks for delta_v of 0.05 to 0.07."""
    if not 0.0 < delta_v < 1.0 - volfrac:
        raise ValueError(f"delta_v {delta_v} does not fit volume {volfrac}")
    return optimize_dynamic(s, make_map(), load, volfrac + delta_v, iters or 40, max_steps, label="model3_relaxed")


def dominance_table(model1: Dict, model2: Dict, model3: Dict, bending_key: str, torsion_key: str) -> List[Dict]:
    """The spec's targets, Model 3 against Model 1, from the audit dictionaries (static and dynamic blocks)."""
    k1, k3 = float(model1["static"]["stiffness"][bending_key]), float(model3["static"]["stiffness"][bending_key])
    u1 = abs(model1["dynamic"]["peak"][torsion_key])
    u3 = abs(model3["dynamic"]["peak"][torsion_key])
    f1, f3 = model1["dynamic"]["frequencies_hz"][0], model3["dynamic"]["frequencies_hz"][0]
    rows = [
        {"target": f"정적 강성 {bending_key} +20% 이상", "model1": k1, "model3": k3, "change_percent": 100.0 * (k3 / k1 - 1.0), "passed": bool(k3 >= 1.2 * k1)},
        {"target": f"동적 피크 {torsion_key} -70% 이상", "model1": u1, "model3": u3, "change_percent": 100.0 * (u3 / u1 - 1.0), "passed": bool(u3 <= 0.3 * u1)},
        {"target": "f1 +20 Hz 이상", "model1": f1, "model3": f3, "change_percent": 100.0 * (f3 / f1 - 1.0), "passed": bool(f3 >= f1 + 20.0)},
    ]
    return rows
