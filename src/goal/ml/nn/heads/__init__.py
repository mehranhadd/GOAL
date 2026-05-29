"""Task-specific output heads.

Drop a new ``.py`` file into this folder with
``@HEAD_REGISTRY.register("name")`` on its class and it is discovered
automatically — no edit to this file required.
"""

from __future__ import annotations

from goal.ml.registry import auto_discover

auto_discover(__name__)
