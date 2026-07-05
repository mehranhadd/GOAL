"""SIMURGH — native GOAL model.

**Symmetry-aware Interatomic Model with element-pair Routed Gated
artisans (an element-pair mixture of potential artisans).**

Two variants exposed:

* :class:`backbone.SimurghBackbone`
    Implements the ``EquivariantBackbone`` protocol.  Combine with any
    standard ``TaskHead`` (e.g. ``energy_forces``, ``dual_forces``,
    ``multi``) for the **modular** training path.  This is the
    *default* variant for the project.

* :class:`monolithic.SimurghMonolithic`
    A self-contained ``MonolithicModel`` that returns the property
    dictionary directly — useful for inference deployments where a
    head is not needed, or as a single-file demo.

Both variants share the same building blocks
(:class:`EnvironmentDressing`, :class:`SimurghArtisanBank`,
:class:`DualForcesHead`) so the physics is identical.  The difference
is only in how they integrate with the GOAL Lightning training loop.
"""

from __future__ import annotations

from goal.ml.nn.models.simurgh.backbone import SimurghBackbone
from goal.ml.nn.models.simurgh.monolithic import SimurghMonolithic

__all__ = ["SimurghBackbone", "SimurghMonolithic"]
