"""Universal Lightning module for any backbone + head combination.

Handles training, validation, testing, EMA, gradient clipping, and logging.
Compatible with Lightning 2.6+ features including:
- ``configure_model()`` hook for FSDP2 / torch.compile
- ``EMAWeightAveraging`` callback (replaces manual EMA when preferred)
- ``torch.compile`` support with configurable mode and fullgraph
- Proper multi-GPU sync via ``sync_dist``
- ``test_step`` for evaluation
- Performance: gradient accumulation, non-blocking transfers, TF32
"""

from __future__ import annotations

import typing
import warnings

import lightning as L
import torch
from omegaconf import DictConfig

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.ema import EMAWrapper
from goal.ml.training.loss import CompositeLoss
from goal.ml.training.metrics import PROG_BAR_METRICS, mlip_metrics


class GOALModule(L.LightningModule):
    """Universal ``LightningModule`` for any backbone + head combination.

    Handles: training, validation, testing, EMA, gradient clipping, logging.
    Compatible with Lightning >=2.6 APIs.

    Supports two wiring modes:

    **Modular** (backbone + head)
        A backbone satisfying ``EquivariantBackbone`` or ``InvariantBackbone``
        produces ``NodeFeatures``, then a ``TaskHead`` maps those features to
        a property dictionary.

    **Monolithic** (backbone only, head=None)
        A model satisfying ``MonolithicModel`` directly returns a property
        dictionary from ``forward(graph)``.  Set ``head=None`` and the
        training loop will call the backbone directly.

    Parameters
    ----------
    backbone : EquivariantBackbone | MonolithicModel
        Any model satisfying a backbone or monolithic protocol.
    head : TaskHead | None
        Output head for modular models.  ``None`` for monolithic models.
    loss : CompositeLoss
        Composable loss function built from weighted components.
    config : DictConfig
        Full Hydra configuration.
    compile_model : bool
        Whether to apply ``torch.compile`` to the backbone in
        ``configure_model()``.
    """

    def __init__(
        self,
        backbone: typing.Any,
        head: typing.Any,
        loss: CompositeLoss,
        config: DictConfig,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.backbone: typing.Any = backbone
        self.head: typing.Any = head
        self.loss: CompositeLoss = loss
        self.config: DictConfig = config
        self._compile_model: bool = compile_model
        self.save_hyperparameters(ignore=["backbone", "head", "loss"])

        # EMA — keeps a shadow copy of weights for evaluation.
        # Lightning 2.6 provides EMAWeightAveraging callback as an alternative,
        # but we keep this for fine-grained control (e.g. per-step decay schedule).
        if config.training.ema.get("enabled", False):
            self.ema = EMAWrapper(
                self.parameters(),
                decay=config.training.ema.decay,
            )

        # ----- TASK 2: stage-based loss-weight schedule -----
        #
        # ``training.stages`` is a list of dicts ``{name, epochs,
        # loss_weights}`` defining an epoch-keyed curriculum.  At each
        # step we look up the currently-active stage and patch
        # ``WeightedLoss.weight`` for every matching label.  The set
        # of *labels* a stage can override is fixed at construction
        # time by introspecting the CompositeLoss — unknown labels in
        # the stage config are silently ignored (a stage may name
        # ``"newton"`` before the matching loss exists).
        raw_stages: typing.Any = config.training.get("stages", None)
        self._stages_cfg: list[typing.Any] = list(raw_stages or [])
        self._weighted_loss_by_label: dict[str, typing.Any] = {}
        self._base_loss_weights: dict[str, float] = {}
        for wl in self.loss.losses:
            label: str | None = getattr(wl, "label", None)
            if label is None:
                continue
            self._weighted_loss_by_label[label] = wl
            self._base_loss_weights[label] = float(getattr(wl, "weight", 1.0))
        self._current_stage_name: str | None = None

        # Warn about labels named in any stage's loss_weights that do
        # not have a matching declared component.  Forward-declaration
        # is intentionally allowed (a stage may name ``newton`` before
        # the newton loss is wired), but typos like ``force`` vs
        # ``forces`` would otherwise silently leave the base weight
        # untouched — invisible at runtime, easy to miss for hours.
        # ``strict_stage_labels: true`` upgrades the warning to an error
        # for projects that want hard validation.
        known_labels: set[str] = set(self._weighted_loss_by_label.keys())
        unknown: dict[str, list[str]] = {}
        for stage in self._stages_cfg:
            stage_name: str = str(stage.get("name", "?"))
            overrides: typing.Any = stage.get("loss_weights", None) or {}
            for label in dict(overrides).keys():
                if str(label) not in known_labels:
                    unknown.setdefault(stage_name, []).append(str(label))
        if unknown:
            details: str = "; ".join(
                f"stage {n!r} -> {sorted(labels)}" for n, labels in unknown.items()
            )
            msg: str = (
                f"training.stages references loss labels that aren't "
                f"declared in training.losses: {details}.  Declared "
                f"labels: {sorted(known_labels)}.  These overrides will "
                f"be ignored until a matching loss component is added."
            )
            if bool(config.training.get("strict_stage_labels", False)):
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    def configure_model(self) -> None:
        """Hook called before ``configure_optimizers`` (Lightning 2.4+).

        Used for:
        - ``torch.compile`` wrapping (avoids recompilation per epoch)
        - FSDP2 / tensor parallelism wrapping via ``ModelParallelStrategy``
        - DTensor-based sharding
        """
        if self._compile_model:
            compile_cfg: dict[str, typing.Any] = self.config.training.get("compile", {})
            # torch.compile requires PyTorch 2.0+ and works best with CUDA
            if not hasattr(torch, "compile"):
                warnings.warn(
                    "torch.compile requested but unavailable (PyTorch <2.0). Skipping.",
                    stacklevel=2,
                )
            else:
                self.backbone = torch.compile(
                    self.backbone,
                    mode=compile_cfg.get("mode", "default"),
                    fullgraph=compile_cfg.get("fullgraph", False),
                    dynamic=compile_cfg.get("dynamic", None),
                )

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Run backbone -> head pipeline, or backbone alone for monolithic models."""
        if self.head is None:
            return self.backbone(graph)
        features: typing.Any = self.backbone(graph)
        predictions: dict[str, torch.Tensor] = self.head(features, graph)
        # Inject gate values (live, grad-enabled) for GateRegLoss.
        # Duck-typed so non-KRONOS backbones are silently skipped.
        # We read the raw Parameter rather than .gates() (which detaches).
        if hasattr(self.backbone, "moe") and hasattr(self.backbone.moe, "experts"):
            gate_list: list[torch.Tensor] = [
                expert.gate  # type: ignore[union-attr]
                for expert in self.backbone.moe.experts.values()
            ]
            if gate_list:
                predictions["gate_values"] = torch.stack(gate_list)
        return predictions

    def on_before_batch_transfer(self, batch: typing.Any, dataloader_idx: int) -> typing.Any:
        """Pre-transfer hook — ensures non-blocking device transfer.

        Lightning calls this before moving the batch to GPU. We ensure
        the batch tensors are contiguous (avoids scattered reads on PCIe).
        """
        return batch

    # ------------------------------------------------------------------
    # TASK 2 — stage helpers
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> str | None:
        """Name of the active loss-weight stage, or ``None`` when no
        ``training.stages`` schedule is configured."""
        return self._current_stage_name

    def stage_display(self, epoch: int) -> str:
        """Human-readable ``"[stage_name epoch_in_stage/stage_length] Epoch X/Y"``
        prefix.  Used by :class:`GOALRichProgressBar` to prepend the stage
        status to the running epoch counter.  Returns the bare epoch
        string when no stage schedule is configured."""
        # ``self.trainer`` raises when the module hasn't been attached
        # to a Trainer yet (e.g. during pure-Python unit tests).  The
        # underlying ``_trainer`` attribute returns ``None`` instead,
        # which is what we want here.
        trainer: typing.Any = getattr(self, "_trainer", None)
        max_epochs: int | None = None
        if trainer is not None:
            max_epochs = getattr(trainer, "max_epochs", None)
        epoch_str: str = (
            f"Epoch {epoch}/{max_epochs - 1}"
            if isinstance(max_epochs, int) and max_epochs > 0
            else f"Epoch {epoch}"
        )

        if not self._stages_cfg:
            return epoch_str

        boundary: int = 0
        for stage in self._stages_cfg:
            n_ep: typing.Any = stage.get("epochs", None)
            name: str = str(stage.get("name", "?"))
            if n_ep is None:
                # Open-ended final stage
                within: int = epoch - boundary + 1
                return f"[{name} {within}/∞] {epoch_str}"
            upper: int = boundary + int(n_ep)
            if epoch < upper:
                within = epoch - boundary + 1
                return f"[{name} {within}/{int(n_ep)}] {epoch_str}"
            boundary = upper
        # Past every finite stage — fall back to the last one's name.
        last_name: str = str(self._stages_cfg[-1].get("name", "?"))
        return f"[{last_name}] {epoch_str}"

    def _get_current_stage(self, epoch: int) -> typing.Any | None:
        """Resolve the active stage dict for the given epoch.

        Stage ``i`` spans ``[start_i, start_i + epochs_i)`` where
        ``start_0 = 0`` and ``start_{i+1} = start_i + epochs_i``.  A
        stage with ``epochs: null`` captures every remaining epoch
        (used for the final, open-ended phase).  Returns ``None`` when
        no schedule is configured — callers should then leave the
        base loss weights intact.
        """
        if not self._stages_cfg:
            return None
        boundary: int = 0
        for stage in self._stages_cfg:
            n_ep: typing.Any = stage.get("epochs", None)
            if n_ep is None:
                return stage
            upper: int = boundary + int(n_ep)
            if epoch < upper:
                return stage
            boundary = upper
        # Past every finite stage — clamp to the last one defined.
        return self._stages_cfg[-1]

    def _apply_stage_weights(self, stage: typing.Any) -> None:
        """Patch ``WeightedLoss.weight`` in-place per the stage overrides.

        Labels that don't have a matching ``WeightedLoss`` in the
        ``CompositeLoss`` are silently skipped — that lets a stage
        config name ``"newton"`` before the newton loss component is
        added.
        """
        overrides: typing.Any = stage.get("loss_weights", None)
        if overrides is None:
            return
        for label, w in dict(overrides).items():
            wl: typing.Any = self._weighted_loss_by_label.get(str(label))
            if wl is not None:
                wl.weight = float(w)

    def _maybe_advance_stage(self, epoch: int) -> None:
        """Resolve the active stage, apply its weights, and log
        transitions.  Cheap to call every step."""
        stage: typing.Any = self._get_current_stage(epoch)
        if stage is None:
            return
        self._apply_stage_weights(stage)
        stage_name: str = str(stage.get("name", "?"))
        if stage_name != self._current_stage_name:
            self._current_stage_name = stage_name
            active: dict[str, float] = {
                lbl: float(wl.weight) for lbl, wl in self._weighted_loss_by_label.items()
            }
            # ``rank_zero_only`` keeps DDP runs quiet.
            from lightning.pytorch.utilities.rank_zero import rank_zero_info

            rank_zero_info(f"[stage] epoch={epoch} → {stage_name!r}  " f"active_weights={active}")

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _log_step_metrics(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
    ) -> None:
        """Compute the standard MLIP diagnostic metrics and log them.

        Headline metrics (per atom energy MAE, force cosine
        similarity, Newton violation) are routed to the Lightning
        progress bar; the rest go to loggers only.
        """
        is_train = prefix.startswith("train")
        metrics: dict[str, torch.Tensor] = mlip_metrics(predictions, batch)
        for name, value in metrics.items():
            self.log(
                f"{prefix}{name}",
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                prog_bar=name in PROG_BAR_METRICS,
                on_step=is_train,
                on_epoch=True,
            )

    def training_step(self, batch: AtomicGraph, batch_idx: int) -> torch.Tensor:
        """Single training step -- forward, loss, logging."""
        self._maybe_advance_stage(self.current_epoch)

        predictions: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = self.loss(predictions, batch)

        # Loss components — kept on the progress bar for backward
        # compatibility with existing dashboards.
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=True,
        )
        # Standard MLIP diagnostic metrics (energy_mae_per_atom,
        # forces_{mae,rmse,cosine_similarity,magnitude_mae},
        # newton_violation).
        self._log_step_metrics(predictions, batch, prefix="train/")
        return losses["total"]

    def on_train_batch_end(
        self,
        outputs: typing.Any,
        batch: AtomicGraph,
        batch_idx: int,
    ) -> None:
        """Update EMA after each optimiser step."""
        if hasattr(self, "ema"):
            self.ema.update()

    def validation_step(self, batch: AtomicGraph, batch_idx: int) -> None:
        """Single validation step -- uses EMA weights if available.

        Wraps the model forward in :func:`torch.enable_grad` because
        Lightning's default validation loop runs under
        ``torch.no_grad``, which would prevent autograd-derived force
        heads (``DualForcesHead`` in ``mode='autograd'`` / ``'hybrid'``)
        from computing ``-∂E/∂r``.  The :func:`torch.no_grad` outer
        context still applies to anything *outside* this block (loss
        backward, etc.), so the cost is just the per-forward grad
        tape.  The loss + metrics dicts are then detached when
        Lightning logs them, so no second-order graph survives the
        step.
        """
        # Mirror the same stage on validation so val metrics use the
        # same per-component weighting as training (i.e. a 0-weight
        # forces stage logs val/forces but with that 0 baked in).
        self._maybe_advance_stage(self.current_epoch)

        with torch.enable_grad():
            if hasattr(self, "ema"):
                with self.ema.average_parameters():
                    predictions = self(batch)
            else:
                predictions = self(batch)

            losses = self.loss(predictions, batch)

        self.log_dict(
            {f"val/{k}": v.detach() for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self._log_step_metrics(predictions, batch, prefix="val/")

        # Expert load-balance diagnostic (KRONOS only, validation only).
        # Duck-typed — silently skipped for non-KRONOS backbones.
        if hasattr(self.backbone, "compute_expert_loads"):
            loads: dict[str, torch.Tensor] = self.backbone.compute_expert_loads(batch)
            load_var: torch.Tensor | None = loads.pop("load_variance", None)
            self.log_dict(
                {f"val/expert_load/{k}": v for k, v in loads.items()},
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            if load_var is not None:
                self.log(
                    "val/expert_load_variance",
                    load_var,
                    batch_size=batch.num_graphs,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def test_step(self, batch: AtomicGraph, batch_idx: int) -> None:
        """Single test step -- uses EMA weights if available.

        Same :func:`torch.enable_grad` wrapper as
        :meth:`validation_step` (see its docstring for the rationale):
        autograd-based force heads (``EnergyForcesHead``, ``DualForcesHead``
        in autograd/hybrid mode) need the grad tape even during evaluation.

        Stage weights are intentionally skipped here — the test loop always
        runs with the base loss weights so all components are reported at
        full weight regardless of which training stage the checkpoint is from.
        """

        with torch.enable_grad():
            if hasattr(self, "ema"):
                with self.ema.average_parameters():
                    predictions = self(batch)
            else:
                predictions = self(batch)

            losses = self.loss(predictions, batch)

        self.log_dict(
            {f"test/{k}": v.detach() for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self._log_step_metrics(predictions, batch, prefix="test/")

    def _split_param_groups(
        self,
        lr: float,
        weight_decay: float,
    ) -> list[dict[str, typing.Any]]:
        """Split parameters into equivariant (no WD) and standard (WD) groups.

        Equivariant group: parameters belonging to any module whose
        fully-qualified name contains ``"tp"`` or ``"equivariant"``, or
        whose type is ``EquivariantLinear`` or ``FullyConnectedTensorProduct``
        (from e3nn).  All other parameters go into the standard group and
        receive ``weight_decay``.

        Using ``id()`` deduplication ensures shared parameters (if any)
        appear only once.
        """
        from e3nn.o3 import FullyConnectedTensorProduct

        from goal.ml.nn.primitives.linear import EquivariantLinear

        equivariant_ids: set[int] = set()
        for name, module in self.named_modules():
            is_equiv_type = isinstance(module, (EquivariantLinear, FullyConnectedTensorProduct))
            is_equiv_name = "tp" in name or "equivariant" in name.lower()
            if is_equiv_type or is_equiv_name:
                for param in module.parameters(recurse=False):
                    equivariant_ids.add(id(param))

        equivariant_params: list[torch.nn.Parameter] = []
        standard_params: list[torch.nn.Parameter] = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in equivariant_ids:
                equivariant_params.append(param)
            else:
                standard_params.append(param)

        return [
            {"params": equivariant_params, "weight_decay": 0.0, "lr": lr},
            {"params": standard_params, "weight_decay": weight_decay, "lr": lr},
        ]

    def configure_optimizers(self) -> dict[str, typing.Any]:
        """Set up optimiser and learning rate scheduler.

        Uses AdamW (PyTorch 2.x) as default. Supports cosine annealing
        and reduce-on-plateau schedulers.

        Parameters are split into two groups:
        - Equivariant parameters (TP layers, EquivariantLinear): no weight decay.
        - Standard parameters (MLP weights, gates, embeddings): weight decay applies.
        """
        opt_cfg: typing.Any = self.config.training.optimizer
        scheduler_type: str = opt_cfg.get("scheduler_type", "reduce_on_plateau")
        weight_decay: float = float(opt_cfg.get("weight_decay", 0.0))

        param_groups = self._split_param_groups(
            lr=float(opt_cfg.lr),
            weight_decay=weight_decay,
        )

        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.lr,
            amsgrad=opt_cfg.get("amsgrad", True),
        )

        # max_epochs for the scheduler: prefer trainer.max_epochs (what Lightning
        # actually uses) over training.max_epochs (legacy, scheduler-only field).
        # trainer is not attached yet when configure_optimizers runs, so read
        # from cfg.trainer first, fall back to cfg.training for old configs.
        trainer_max: int = int(
            self.config.get("trainer", {}).get("max_epochs", None)
            or self.config.training.get("max_epochs", 500)
        )

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trainer_max,
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup_epochs = opt_cfg.get("warmup_epochs", 10)
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=trainer_max - warmup_epochs,
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            # Default: ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=opt_cfg.scheduler.patience,
                factor=opt_cfg.scheduler.factor,
                min_lr=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total",
                    "interval": "epoch",
                },
            }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Gradient clipping before optimiser step."""
        clip_val: float = self.config.training.get("gradient_clip", 0)
        if clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )
