"""Neural operator and coordinate network topology optimization interfaces.

Exports:
- GINOTInterface, WarmStartResult: Geometry-Informed Neural Operator Transformer with Fourier features,
  physics encoder, and cross-attention decoding.
- TONICInitializer: Implicit coordinate-MLP density parameterization, analytical sensitivity
  backpropagation, Adam optimizer, and physics-informed warm-start synthesis.
"""

from __future__ import annotations

from .ginot_interface import GINOTInterface, WarmStartResult
from .tonic_interface import TONICInitializer

__all__ = [
    "GINOTInterface",
    "WarmStartResult",
    "TONICInitializer",
]
