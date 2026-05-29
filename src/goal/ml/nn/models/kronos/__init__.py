"""KRONOS — native GOAL model.

**K-order Routed Orthogonal Network of Symmetry with Element-Pair
Mixture-of-Experts.**

Two variants exposed:

* :class:`backbone.KronosBackbone`
    Implements the ``EquivariantBackbone`` protocol.  Combine with any
    standard ``TaskHead`` (e.g. ``energy_forces``, ``dual_forces``,
    ``multi``) for the **modular** training path.  This is the
    *default* variant for the project.

* :class:`monolithic.KronosMonolithic`
    A self-contained ``MonolithicModel`` that returns the property
    dictionary directly — useful for inference deployments where a
    head is not needed, or as a single-file demo.

Both variants share the same building blocks
(:class:`EnvironmentDressing`, :class:`KronosMoE`,
:class:`DualForcesHead`) so the physics is identical.  The difference
is only in how they integrate with the GOAL Lightning training loop.
"""

from __future__ import annotations

from goal.ml.nn.models.kronos.backbone import KronosBackbone
from goal.ml.nn.models.kronos.monolithic import KronosMonolithic

__all__ = ["KronosBackbone", "KronosMonolithic"]
