"""Trainable foundation-model backbones.

Wrap pre-trained interatomic potentials (MACE, UMA, …) as real
``nn.Module`` monolithic backbones so they can be fine-tuned inside the
standard :class:`goal.ml.training.module.GOALModule` training loop with
``head: null``.

Modules here are auto-discovered by ``goal.ml.nn.models`` — the
``@BACKBONE_REGISTRY.register(...)`` decorators fire on import.  Upstream
packages (mace-torch, fairchem-core, peft) are imported lazily, so these
modules import (and register) even when those packages are absent.
"""

from __future__ import annotations
