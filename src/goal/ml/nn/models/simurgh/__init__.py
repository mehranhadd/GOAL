"""SIMURGH — native GOAL model.

**Symmetry-aware Interatomic Model with element-pair Routed Gated
artisans (an element-pair mixture of potential artisans).**

Variants exposed:

* :class:`backbone_arace.SimurghAraceBackbone` — **default**
    The ARACE architecture (ARtisan + Atomic Cluster Expansion):
    element-pair potential artisans are the primary computation at
    every layer, alternating with ACE message passing that aggregates
    the artisan edge features back onto nodes.  Implements the
    ``EquivariantBackbone`` protocol; combine with any standard
    ``TaskHead`` (e.g. ``energy_forces``).  Registered
    ``"simurgh_arace"``.

* :class:`backbone.SimurghBackbone` — legacy
    The original ACE-dressing-first pipeline (environment dressing →
    artisan bank as final readout).  Registered ``"simurgh"`` and
    ``"simurgh_ace_first"``.

* :class:`monolithic.SimurghMonolithic`
    Self-contained ``MonolithicModel`` counterpart of the legacy
    backbone (``"simurgh_monolithic"``).

* :class:`arace.MonolithicArace`
    Self-contained ``MonolithicModel`` counterpart of the ARACE
    backbone, for research/ablation (``"monolithic_arace"``).

The modular and monolithic variants of each architecture share the same
building blocks so the physics is identical; the difference is only in
how they integrate with the GOAL Lightning training loop.
"""

from __future__ import annotations

from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.nn.models.simurgh.backbone import SimurghBackbone
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
from goal.ml.nn.models.simurgh.monolithic import SimurghMonolithic

__all__ = [
    "MonolithicArace",
    "SimurghAraceBackbone",
    "SimurghBackbone",
    "SimurghMonolithic",
]
