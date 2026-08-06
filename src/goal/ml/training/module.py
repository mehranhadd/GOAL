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
        # Duck-typed so non-SIMURGH backbones are silently skipped.
        # We read the raw Parameter rather than .gates() (which detaches).
        if hasattr(self.backbone, "artisan_bank") and hasattr(self.backbone.artisan_bank, "experts"):
            gate_list: list[torch.Tensor] = [
                expert.gate  # type: ignore[union-attr]
                for expert in self.backbone.artisan_bank.artisans.values()
            ]
            if gate_list:
                predictions["gate_values"] = torch.stack(gate_list)

        # Adaptive-depth-gate side channels (ARACE add-on).  Duck-typed,
        # so every other backbone is silently skipped; both attributes are
        # ``None`` unless ``model.backbone.adaptive_gate`` is configured.
        # ``last_aux_loss`` keeps its autograd graph — :meth:`_with_aux_loss`
        # adds it to the training total.
        aux_loss: torch.Tensor | None = getattr(self.backbone, "last_aux_loss", None)
        if aux_loss is not None:
            predictions["aux_loss"] = aux_loss
        round_gates: typing.Any = getattr(self.backbone, "last_gate_scores", None)
        if round_gates:
            populated: list[torch.Tensor] = [g for g in round_gates if g is not None]
            if populated:
                predictions["gate_scores"] = torch.stack(populated, dim=0)  # (L, N)
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
    # Logger sections
    # ------------------------------------------------------------------
    #
    # Everything a run logs lands in one of a few W&B / TensorBoard
    # sections, chosen by the text before the first ``/``:
    #
    #   train/      val/      — the metrics you actually watch: energy and
    #                           force MAE/RMSE, cosine similarity, Newton
    #                           violation, per-round energies, and the
    #                           loss total plus its physics components.
    #   train_gate/ val_gate/  — every gate diagnostic:
    #                             gate_mean_round_{L}   adaptive depth gate
    #                             aux_loss              its sparsity penalty
    #                             gate_reg              L2 on artisan gates
    #                             artisan/round{L}/{pair}
    #                                                   artisan gate values
    #                             artisan_load/{pair}   load balance (val
    #                             artisan_load_variance only — see below)
    #
    # Gate curves are numerous — one per round, plus one per element pair
    # per round — and they are for occasional inspection, not for judging
    # a run.  Keeping them in their own section is what stops them from
    # burying ``val/forces_mae`` in a wall of panels.

    #: Loss labels that belong in the gate section rather than next to the
    #: physics losses.  ``total`` deliberately stays in ``train/`` — it is
    #: the whole loss, including these components.
    GATE_LOSS_LABELS: typing.ClassVar[frozenset[str]] = frozenset({"aux_loss", "gate_reg"})

    @staticmethod
    def _gate_prefix(prefix: str) -> str:
        """``"train/"`` → ``"train_gate/"``, ``"val/"`` → ``"val_gate/"``."""
        return f"{prefix.rstrip('/')}_gate/"

    def _log_artisan_gates(self, batch: AtomicGraph, prefix: str) -> None:
        """Log every artisan gate as ``{split}_gate/artisan/round{L}/{pair}``.

        These are the learnable per-element-pair gates of the artisan bank
        — ``nn.Parameter`` scalars, so reading them costs nothing and they
        can be logged on both train and validation.  A gate drifting to
        near-zero means that element pair has been switched off, which is
        the artisan equivalent of expert collapse.

        Duck-typed via ``backbone.gates()``; a silent no-op for backbones
        without an artisan bank.  These are the most numerous curves a run
        produces — ``K*(K+1)/2 x num_rounds`` of them — which is exactly
        why they are filed under the gate section instead of next to the
        energy and force metrics.
        """
        gates_fn: typing.Any = getattr(self.backbone, "gates", None)
        if not callable(gates_fn):
            return
        gate_values: dict[str, torch.Tensor] = gates_fn()
        if not gate_values:
            return
        artisan_prefix: str = f"{self._gate_prefix(prefix)}artisan/"
        self.log_dict(
            {f"{artisan_prefix}{k}": v for k, v in gate_values.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            on_step=prefix.startswith("train"),
            on_epoch=True,
        )

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _log_losses(
        self,
        losses: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
        on_step: bool = True,
        prog_bar: bool = True,
    ) -> None:
        """Log the loss breakdown, routing gate terms to the gate section.

        Physics components (and ``total``) keep the ``{prefix}`` section
        and the progress bar; ``aux_loss`` / ``gate_reg`` move to
        ``{prefix}_gate/`` and never reach the progress bar.
        """
        physics: dict[str, torch.Tensor] = {}
        gate: dict[str, torch.Tensor] = {}
        for label, value in losses.items():
            target = gate if label in self.GATE_LOSS_LABELS else physics
            target[label] = value.detach()

        self.log_dict(
            {f"{prefix}{k}": v for k, v in physics.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=True,
        )
        if gate:
            self.log_dict(
                {f"{self._gate_prefix(prefix)}{k}": v for k, v in gate.items()},
                batch_size=batch.num_graphs,
                sync_dist=True,
                prog_bar=False,
                on_step=on_step,
                on_epoch=True,
            )

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

    def _log_round_energies(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
    ) -> None:
        """Log per-round ARACE energy contributions as
        ``{prefix}energy_round_{L}``.

        Duck-typed no-op for non-ARACE backbones: the per-round totals
        come either from the monolithic model's ``"layer_energies"``
        prediction key or from the modular backbone's
        ``last_round_energies`` attribute (a detached ``(L, B)``
        snapshot of the most recent forward).  Monitors whether every
        round keeps contributing — a round collapsing to zero is a
        warning sign.
        """
        layer_e: torch.Tensor | None = predictions.get("layer_energies", None)
        if layer_e is None:
            layer_e = getattr(self.backbone, "last_round_energies", None)
        if layer_e is None or layer_e.dim() != 2:
            return
        for round_idx in range(layer_e.shape[0]):
            self.log(
                f"{prefix}energy_round_{round_idx}",
                layer_e[round_idx].detach().mean(),
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=prefix.startswith("train"),
                on_epoch=True,
            )

    def _log_gate_scores(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
    ) -> None:
        """Log per-round adaptive-gate means to the **gate** section.

        Emits ``{train,val}_gate/gate_mean_round_{L}`` — deliberately not
        ``train/…``, so the gate curves stay out of the section holding
        the energy/force metrics (see "Logger sections" above).

        Duck-typed no-op unless the ARACE adaptive depth gate is enabled:
        the ``(L, N)`` gate tensor comes either from the monolithic
        model's ``"gate_scores"`` prediction key or from the modular
        backbone's ``last_gate_scores`` side channel (injected into
        ``predictions`` by :meth:`forward`).

        Reading the curve: ~1.0 means the gate is always open (not
        selective — raise ``aux_loss_weight``), ~0.5 means it is
        discriminating between atoms, and a collapse towards 0 means
        rounds are being skipped wholesale (lower ``aux_loss_weight``).
        """
        gates: torch.Tensor | None = predictions.get("gate_scores", None)
        if gates is None or gates.dim() != 2:
            return
        gate_prefix: str = self._gate_prefix(prefix)
        for round_idx in range(gates.shape[0]):
            self.log(
                f"{gate_prefix}gate_mean_round_{round_idx}",
                gates[round_idx].detach().mean(),
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=prefix.startswith("train"),
                on_epoch=True,
            )

    @staticmethod
    def _with_aux_loss(
        losses: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fold a model-supplied ``aux_loss`` into the loss breakdown.

        Some models emit their own regularisation term alongside the
        predictions — currently the ARACE adaptive depth gate's sparsity
        penalty ``λ · mean(g)``, already scaled by its configured weight.
        It is added to ``total`` and surfaced under the ``aux_loss`` label
        so it shows up in the logged breakdown like any other component.

        No config wiring is needed (and none should be: the weight lives
        with the module that defines the term).  Models that emit no
        ``aux_loss`` are untouched — the returned dict is the input dict.
        """
        aux: torch.Tensor | None = predictions.get("aux_loss", None)
        if aux is None or "aux_loss" in losses:
            return losses
        return {**losses, "aux_loss": aux, "total": losses["total"] + aux}

    def training_step(self, batch: AtomicGraph, batch_idx: int) -> torch.Tensor:
        """Single training step -- forward, loss, logging."""
        self._maybe_advance_stage(self.current_epoch)

        predictions: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = self._with_aux_loss(
            self.loss(predictions, batch), predictions
        )

        # Loss components — physics terms on the progress bar under
        # ``train/``, gate terms quietly under ``train_gate/``.
        self._log_losses(losses, batch, prefix="train/")
        # Standard MLIP diagnostic metrics (energy_mae_per_atom,
        # forces_{mae,rmse,cosine_similarity,magnitude_mae},
        # newton_violation).
        self._log_step_metrics(predictions, batch, prefix="train/")
        # ARACE per-round energy decomposition (no-op for other backbones).
        self._log_round_energies(predictions, batch, prefix="train/")
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix="train/")
        # Per-element-pair artisan gates (no-op without an artisan bank).
        self._log_artisan_gates(batch, prefix="train/")
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

            losses = self._with_aux_loss(self.loss(predictions, batch), predictions)

        self._log_losses(losses, batch, prefix="val/")
        self._log_step_metrics(predictions, batch, prefix="val/")
        # ARACE per-round energy decomposition (no-op for other backbones).
        self._log_round_energies(predictions, batch, prefix="val/")
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix="val/")

        # Per-element-pair artisan gates → val_gate/artisan/… (also logged
        # every train step; see :meth:`_log_artisan_gates`).
        self._log_artisan_gates(batch, prefix="val/")

        # Artisan load-balance diagnostic (legacy ACE-first SIMURGH only —
        # the ARACE backbones do not expose ``compute_artisan_loads``, so
        # this is a no-op for them).  Stays validation-only on purpose: it
        # runs a *full extra forward of every artisan* over the batch, far
        # too expensive per training step.  Filed under the gate section as
        # ``val_gate/artisan_load/…`` alongside the gate values it explains.
        if hasattr(self.backbone, "compute_artisan_loads"):
            loads: dict[str, torch.Tensor] = self.backbone.compute_artisan_loads(batch)
            load_var: torch.Tensor | None = loads.pop("load_variance", None)
            load_prefix: str = f"{self._gate_prefix('val/')}artisan_load/"
            self.log_dict(
                {f"{load_prefix}{k}": v for k, v in loads.items()},
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            if load_var is not None:
                self.log(
                    f"{self._gate_prefix('val/')}artisan_load_variance",
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

            losses = self._with_aux_loss(self.loss(predictions, batch), predictions)

        # Same split as train/val: gate terms under ``test_gate/``.
        # ``prog_bar=False`` keeps the test loop as quiet as it was.
        self._log_losses(losses, batch, prefix="test/", prog_bar=False)
        self._log_step_metrics(predictions, batch, prefix="test/")
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix="test/")

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
