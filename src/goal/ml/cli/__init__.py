"""GOAL command-line interface — Hydra-based entry points."""

from __future__ import annotations

import os as _os

# PyTorch 2.6+ changed torch.load to weights_only=True by default.
# GOAL checkpoints include omegaconf DictConfig and other non-tensor
# objects in hyper_parameters, making weights_only=True incompatible.
# Patch Lightning's TorchCheckpointIO to always use weights_only=False
# for checkpoints that originate from this project.
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO as _TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import _load as _pl_load


def _patched_load_checkpoint(
    self: _TorchCheckpointIO,
    path: str,
    map_location: object = None,
    weights_only: bool = False,
) -> dict:
    return _pl_load(path, map_location=map_location, weights_only=False)


_TorchCheckpointIO.load_checkpoint = _patched_load_checkpoint  # type: ignore[method-assign]

# Absolute path to the project's ``configs/`` directory.  Used by every
# ``@hydra.main`` decorator in this package.
#
# Why absolute? Hydra's relative ``config_path`` is resolved differently
# depending on how the entrypoint is invoked:
#
# * ``python -m goal.ml.cli.train``  → ``__module__`` is ``"__main__"`` →
#   Hydra uses ``inspect.getfile(task_function)`` (filesystem resolution).
# * ``goal-train`` (console script) → ``__module__`` is ``"goal.ml.cli.train"`` →
#   Hydra uses *module-based* resolution and tries to import ``configs``
#   as a top-level Python package.  The project root isn't on ``sys.path``
#   (only ``src/`` is), so the import fails with a ``MissingConfigException``.
#
# Using an absolute filesystem path bypasses both modes and works
# identically for ``python -m`` and console-script invocations.
CONFIGS_DIR: str = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "configs")
)

# Absolute path to configs/ml/ — self-contained single-file experiment configs.
CONFIGS_ML_DIR: str = _os.path.join(CONFIGS_DIR, "ml")


import goal.ml.adapters  # noqa: F401,E402
import goal.ml.data.datasets  # noqa: F401,E402
import goal.ml.nn.heads  # noqa: F401,E402

# ---------------------------------------------------------------------------
# Populate the global registries used by every CLI entry point.
#
# Importing these subpackages runs their ``__init__.py`` side-effects, which
# call ``MODEL_REGISTRY.register_lazy(...)`` / ``HEAD_REGISTRY.register_lazy(...)``
# / ``DATASET_REGISTRY.register_lazy(...)``.  Without these imports the
# registries are empty and the CLIs cannot resolve names like
# ``cfg.model.backbone.name = "kronos"``.
# ---------------------------------------------------------------------------
import goal.ml.nn.models  # noqa: F401,E402
