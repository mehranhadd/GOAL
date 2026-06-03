"""Progress-bar callback that prepends the current training stage.

Lightning's built-in :class:`RichProgressBar` shows ``Epoch X/Y`` as
the train task description.  When a stage-based curriculum is active
the stage *name* and *epoch within stage* are what the user actually
wants to read at a glance — far more useful than the absolute epoch
counter alone.

Drop this callback into ``Trainer(callbacks=...)`` to replace the
default progress bar.  When :class:`GOALModule` exposes a
``stage_display(epoch)`` helper (the multi-stage curriculum schedule)
its return value becomes the progress-bar description.  Otherwise we
fall back to the stock ``Epoch X/Y`` string so the callback is safe
to enable globally.
"""

from __future__ import annotations

import typing

from lightning.pytorch.callbacks import RichProgressBar


class GOALRichProgressBar(RichProgressBar):
    """``RichProgressBar`` with stage-aware train description.

    Overrides :attr:`train_description` so the running progress line
    becomes e.g. ``"[energy_only 5/30] Epoch 5/199"`` while a stage
    schedule is active.
    """

    @property
    def train_description(self) -> str:
        if self.trainer is None:
            return super().train_description
        module: typing.Any = self.trainer.lightning_module
        if module is None or not hasattr(module, "stage_display"):
            return super().train_description
        try:
            return str(module.stage_display(int(self.trainer.current_epoch)))
        except Exception:
            # Never let a display helper crash the training loop —
            # fall back to the stock description if anything goes
            # wrong.
            return super().train_description
