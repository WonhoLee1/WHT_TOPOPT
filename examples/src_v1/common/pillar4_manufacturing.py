"""Pillar 4: manufacturing filters as design maps x -> rho, with their exact adjoints.

``DensityMap`` is the free design. ``ExtrusionMap`` averages along one axis. ``RibMap`` wraps ``PlanarRibExtrusionFilter`` (skin, side walls and
vertical interior ribs). ``skin_elements`` selects passive shell elements on chosen faces of the design domain.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from wht_topopt.core.stl_domain import build_weights, density_filter_matrix, heaviside
from wht_topopt.manufacturing.extrusion import PlanarRibExtrusionFilter

from .structure import Structure

RHO_MIN = 1e-3


class DensityMap:
    """Density filter, optional Heaviside with beta continuation, passive regions forced."""

    label = "free"

    def __init__(self, structure: Structure, rmin: float, beta0: float = 0.0, beta_max: float = 0.0, beta_every: int = 20, eta: float = 0.5) -> None:
        self.s = structure
        self.H = density_filter_matrix(build_weights(structure.centroids, rmin), structure.volumes)
        self.beta0, self.beta_max, self.beta_every, self.eta = float(beta0), float(beta_max), int(beta_every), float(eta)
        self.beta = self.beta0
        self._dproj: Optional[np.ndarray] = None

    def set_iteration(self, it: int) -> None:
        if self.beta0 > 0.0:
            self.beta = min(self.beta_max, self.beta0 * 2.0 ** ((it - 1) // self.beta_every))

    def fix_passive(self, field_: np.ndarray) -> np.ndarray:
        out = field_.copy()
        out[self.s.solid_mask] = 1.0
        out[self.s.void_mask] = RHO_MIN
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        rho, self._dproj = heaviside(self.H @ self._pre(x), self.beta, self.eta)
        return self.fix_passive(rho)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return self._pre_adjoint(self.H.T @ (g * self._dproj * self.s.design_mask))

    def smooth_signal(self, s: np.ndarray) -> np.ndarray:
        return self.H @ s

    def _pre(self, x: np.ndarray) -> np.ndarray:
        return x

    def _pre_adjoint(self, g: np.ndarray) -> np.ndarray:
        return g

    @property
    def is_design_element(self) -> np.ndarray:
        return self.s.design_mask


class SkinMap(DensityMap):
    """Free design inside a passive shell on some faces of the domain: extra passive solid for this design only."""

    def __init__(self, structure: Structure, rmin: float, skin: np.ndarray, **kw) -> None:
        super().__init__(structure, rmin, **kw)
        self.label = "skin"
        self.skin = np.zeros(structure.n_elem, dtype=bool)
        self.skin[np.asarray(skin, dtype=np.int64)] = True
        self.skin &= ~structure.void_mask

    def fix_passive(self, field_: np.ndarray) -> np.ndarray:
        out = super().fix_passive(field_)
        out[self.skin] = 1.0
        return out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return super().backward(g * ~self.skin)


class ExtrusionMap(DensityMap):
    """x is averaged along ``axis`` before the density filter, so every column of the grid has one density."""

    def __init__(self, structure: Structure, rmin: float, axis: str, **kw) -> None:
        super().__init__(structure, rmin, **kw)
        self.axis = axis.lower()
        self.label = f"extrude_{self.axis}"
        nx, ny, nz = structure.grid
        iz, iy, ix = np.unravel_index(np.arange(structure.n_elem), (nz, ny, nx))
        raw = {"x": iz * ny + iy, "y": iz * nx + ix, "z": iy * nx + ix}.get(self.axis)
        if raw is None:
            raise ValueError(f"unknown extrusion axis {axis!r}")
        _, self.groups = np.unique(raw, return_inverse=True)
        self.count = np.bincount(self.groups).astype(np.float64)

    def _pre(self, x: np.ndarray) -> np.ndarray:
        return (np.bincount(self.groups, weights=x) / self.count)[self.groups]

    def _pre_adjoint(self, g: np.ndarray) -> np.ndarray:
        return (np.bincount(self.groups, weights=g) / self.count)[self.groups]


class RibMap:
    """Bottom skin and side walls stay, the core is a vertical rib pattern (spec, Pillar 4). Structured grids only."""

    label = "skin_rib"

    def __init__(self, structure: Structure, r_min_xy: float, bottom_layers: int = 1, side_layers: int = 1, betas: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0), beta_every: int = 12) -> None:
        self.s = structure
        self.filter = PlanarRibExtrusionFilter(structure.mesh, r_min_xy=r_min_xy, bottom_layers=bottom_layers, side_layers=side_layers, beta=float(betas[0]))
        self.betas, self.beta_every = tuple(betas), int(beta_every)
        self._x: Optional[np.ndarray] = None

    def set_iteration(self, it: int) -> None:
        self.filter.set_beta(self.betas[min((it - 1) // self.beta_every, len(self.betas) - 1)])

    @property
    def beta(self) -> float:
        return self.filter.beta

    def fix_passive(self, field_: np.ndarray) -> np.ndarray:
        out = field_.copy()
        out[self.s.solid_mask] = 1.0
        out[self.s.void_mask] = RHO_MIN
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return self.fix_passive(self.filter.forward(x))

    def backward(self, g: np.ndarray) -> np.ndarray:
        return self.filter.backward(g * self.s.design_mask, self._x)

    def smooth_signal(self, s: np.ndarray) -> np.ndarray:
        return self.filter.smooth_signal(s)

    @property
    def is_design_element(self) -> np.ndarray:
        return self.s.design_mask


def skin_elements(inside: np.ndarray, grid: Sequence[int], size: Sequence[float], thickness: float, faces: Sequence[str]) -> np.ndarray:
    """Indices of the design elements within ``thickness`` of the exposed faces ('x-', 'x+', ... 'z+') of the domain ``inside``."""
    nx, ny, nz = grid
    ins = np.asarray(inside, dtype=bool).reshape(nz, ny, nx)
    axes = {"x": (2, size[0]), "y": (1, size[1]), "z": (0, size[2])}
    sel = np.zeros_like(ins)
    for f in faces:
        if f not in ("x-", "x+", "y-", "y+", "z-", "z+"):
            raise ValueError(f"unknown face {f!r}")
        ax, h = axes[f[0]]
        layers = max(1, int(round(thickness / h)))
        outward = 1 if f[1] == "+" else -1
        grown = ins & ~_shifted(ins, ax, -outward)  # the outward neighbour lies outside the domain
        layer = grown
        for _ in range(layers - 1):
            layer = _shifted(layer, ax, -outward) & ins  # one element further in
            grown = grown | layer
        sel |= grown
    return np.flatnonzero(sel.ravel())


def _shifted(a: np.ndarray, axis: int, step: int) -> np.ndarray:
    """``a`` moved by ``step`` cells along ``axis``, False where it enters."""
    out = np.zeros_like(a)
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    if step > 0:
        src[axis], dst[axis] = slice(0, -step), slice(step, None)
    else:
        src[axis], dst[axis] = slice(-step, None), slice(0, step)
    out[tuple(dst)] = a[tuple(src)]
    return out
