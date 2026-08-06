"""Hydra-based training entry point with automatic SLURM resumption.

This is the main training command for GOAL. It:
1. Builds components via the registry system
2. Automatically resumes from ``last.ckpt`` if it exists
3. Writes a ``TRAINING_COMPLETE`` sentinel to prevent SLURM requeue loops
4. Supports DDP, FSDP, ModelParallel strategies via trainer configs
5. Compatible with Lightning 2.6+ features (EMAWeightAveraging, torch.compile)
6. Performance-tuned: TF32 matmul, cuDNN benchmark, gradient accumulation
"""

from __future__ import annotations

import typing
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.data.datamodule import GOALDataModule

try:
    import examples.datasets  # noqa: F401 — registers md17, qm9, ani1, spice, etc.
except ImportError:
    pass
from goal.ml.data.statistics import (
    ATOMIC_SYMBOLS,
    compute_atomic_references,
    compute_avg_num_neighbors,
    compute_energy_scale,
    compute_pair_counts,
    compute_unique_elements,
)
from goal.ml.nn.models.base import MonolithicModel
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager
from goal.ml.training.callbacks.logging import RichLoggingCallback
from goal.ml.training.loss import CompositeLoss, WeightedLoss
from goal.ml.training.module import GOALModule
from goal.ml.training.strategies.factory import build_strategy


def _instantiate_callbacks(cfg: DictConfig | None) -> list[Callback]:
    """Instantiate Lightning callbacks from Hydra config."""
    if not cfg:
        return []
    callbacks: list[Callback] = []
    for _, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def _instantiate_loggers(cfg: DictConfig | None) -> list[Logger]:
    """Instantiate Lightning loggers from Hydra config."""
    if not cfg:
        return []
    loggers: list[Logger] = []
    for _, lg_conf in cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def _gpu_supports_tf32() -> bool:
    """Check whether the current CUDA device supports TF32 (Ampere+, sm_80+)."""
    if not torch.cuda.is_available():
        return False
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    return capability[0] >= 8  # Ampere = 8.0, Hopper = 9.0


def _setup_performance(cfg: DictConfig) -> None:
    """Apply global PyTorch performance settings before training.

    Settings are applied conditionally based on GPU availability and
    hardware capability. TF32 and cuDNN options are silently skipped
    when running on CPU or pre-Ampere GPUs.

    - **TF32 matmul precision**: Uses TF32 tensor cores for float32 ops,
      giving ~3× speedup with negligible precision loss. Requires Ampere+
      (A100, H100, RTX 30xx/40xx).  Falls back to ``"highest"`` (full fp32)
      on older hardware.
    - **cuDNN benchmark**: Auto-tunes convolution algorithms for the
      given input sizes.  Adds startup overhead but faster thereafter.
      Only meaningful when CUDA is available.
    - **cuDNN deterministic**: Forces deterministic cuDNN algorithms.
      Only meaningful when CUDA is available.
    """
    perf: dict[str, typing.Any] = cfg.training.get("performance", {})
    has_cuda: bool = torch.cuda.is_available()

    # TF32 for matmul — only effective on Ampere+ GPUs (sm_80+)
    if _gpu_supports_tf32():
        matmul_precision: str = perf.get("float32_matmul_precision", "high")
    else:
        matmul_precision = perf.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)

    # cuDNN benchmark — only effective when CUDA is present
    if has_cuda:
        torch.backends.cudnn.benchmark = perf.get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = perf.get("cudnn_deterministic", False)


def _build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config.

    Each loss entry may specify ``fn`` as either:

    - A **string** (default ``"mse"``) — single loss function.
    - A **list** of ``{name, weight}`` dicts — multiple loss functions
      for the same property, each logged and weighted independently.

    Examples::

        # Single fn (backward compatible)
        losses:
          - name: energy
            weight: 4.0
            fn: mse

        # Composite fn per property
        losses:
          - name: forces
            fn:
              - name: mse
                weight: 4.0
              - name: rmse
                weight: 8.0
    """
    import inspect

    losses: list[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        fn_spec: typing.Any = loss_cfg.get("fn", "mse")

        # Forward only the kwargs that the loss class actually accepts.
        # This lets the config uniformly annotate entries (e.g. normalize_by_n_atoms)
        # without every loss class needing to declare it.
        sig = inspect.signature(loss_cls.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        accepted: set[str] = set(sig.parameters) - {"self"}
        loss_kwargs: dict[str, typing.Any] = {
            k: v
            for k, v in loss_cfg.items()
            if k not in ("name", "weight", "fn") and (has_var_keyword or k in accepted)
        }

        if isinstance(fn_spec, str):
            # Single loss function — backward compatible
            losses.append(
                WeightedLoss(
                    loss_cls(loss_fn=fn_spec, **loss_kwargs),
                    weight=loss_cfg.weight,
                    label=loss_cfg.name,
                )
            )
        else:
            # Composite: list of {name, weight} sub-fns
            for sub in fn_spec:
                sub_name: str = sub["name"] if isinstance(sub, dict) else sub.name
                sub_weight: float = float(sub["weight"] if isinstance(sub, dict) else sub.weight)
                fn_label: str = sub_name.rsplit(".", 1)[-1]
                losses.append(
                    WeightedLoss(
                        loss_cls(loss_fn=sub_name),
                        weight=sub_weight,
                        label=f"{loss_cfg.name}_{fn_label}",
                        group=loss_cfg.name,
                    )
                )
    return CompositeLoss(losses)


def _build_head(cfg: DictConfig, backbone: typing.Any) -> typing.Any:
    """Build the task head from config, or ``None`` for monolithic backbones.

    The pairing rule is enforced from the backbone's protocol:

    * A backbone satisfying the :class:`MonolithicModel` protocol
      *must* have ``head: null`` — it already returns the property
      dict directly, so a head would be silently ignored.
    * Any other backbone *must* declare a ``head:`` block — without one
      the backbone's ``NodeFeatures`` would have no consumer and the
      loss would see no predictions.

    Either mismatch raises ``ValueError`` here rather than failing
    deep inside the training loop.
    """
    head_cfg = cfg.model.get("head", None)
    is_monolithic: bool = isinstance(backbone, MonolithicModel)
    backbone_name: str = cfg.model.backbone.name

    if is_monolithic:
        if head_cfg is not None:
            raise ValueError(
                f"Backbone '{backbone_name}' is monolithic (returns a "
                f"property dict directly) and is incompatible with a "
                f"task head.  Set 'head: null' in the model config."
            )
        return None

    if head_cfg is None:
        raise ValueError(
            f"Backbone '{backbone_name}' is modular (produces "
            f"NodeFeatures) and requires a task head.  Declare a "
            f"'head:' block in the model config, or switch to a "
            f"monolithic backbone (e.g. 'simurgh_monolithic', "
            f"'monolithic_example')."
        )

    head_cls: typing.Any = HEAD_REGISTRY.get(head_cfg.name)
    head_kwargs: dict[str, typing.Any] = {k: v for k, v in head_cfg.items() if k != "name"}
    return head_cls(**head_kwargs)


def _run_training(cfg: DictConfig) -> None:
    """Core training logic — shared by train() and train_ml() entry points."""
    # Seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Apply global PyTorch performance settings
    _setup_performance(cfg)

    # Check for completion sentinel — stops SLURM requeue loop
    checkpoint_dir: Path = Path(cfg.training.checkpoint_dir)
    if (checkpoint_dir / "TRAINING_COMPLETE").exists():
        print("Training already complete. Exiting.")
        return

    # Build datamodule first and prime it so we can compute
    # dataset-derived buffers (per-element atomic energies, scale/shift,
    # avg_num_neighbors) before constructing the model.  Lightning will
    # call ``setup("fit")`` again later — the second call is a no-op
    # because the datasets are already populated.
    datamodule: GOALDataModule = GOALDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Build components via registry
    backbone_cls: typing.Any = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs: dict[str, typing.Any] = {
        k: v for k, v in cfg.model.backbone.items() if k != "name"
    }

    # Auto-extract elements from the training set for SIMURGH-family backbones.
    # This replaces any manually-specified elements list in the config —
    # the dataset is always authoritative. Other backbones (hyperspec,
    # invariant_gnn) use a fixed-size embedding table and ignore this key.
    backbone_name: str = cfg.model.backbone.name
    if backbone_name in (
        "simurgh",
        "simurgh_ace_first",
        "simurgh_arace",
        "simurgh_monolithic",
        "monolithic_arace",
    ):
        elements: list[int] = compute_unique_elements(datamodule.data_train)
        symbol_str: str = " ".join(f"{ATOMIC_SYMBOLS.get(z, '?')}({z})" for z in elements)
        print(f"[stats] elements discovered in training set: {symbol_str}")
        # Overwrite whatever the config said — dataset is authoritative.
        backbone_kwargs["elements"] = elements

    # Atomic-energy baseline + ScaleShift.  When the backbone exposes
    # an ``atomic_energies`` sub-config and its mode resolves to
    # ``"dataset"``, run the LSQ regression on the training set and
    # inject the values + scale.  ``"learned"`` and ``"provided"``
    # modes pass through unchanged.
    ae_cfg: dict[str, typing.Any] = dict(backbone_kwargs.get("atomic_energies") or {})
    mode: str = str(ae_cfg.get("mode", "learned"))
    if ae_cfg.get("compute_from_dataset", False) and mode == "learned":
        mode = "dataset"

    if mode == "dataset" and ae_cfg.get("values") is None:
        refs: dict[int, float] = compute_atomic_references(datamodule.data_train)
        ae_cfg["values"] = refs
        ae_cfg["mode"] = "dataset"
        backbone_kwargs["atomic_energies"] = ae_cfg
        print(
            "[stats] atomic_references "
            + ", ".join(f"Z={z}:{e:+.4f}" for z, e in sorted(refs.items()))
        )

    # ScaleShift gain — computed for both "dataset" and "provided" modes
    # so the interaction network starts at the right order of magnitude
    # regardless of where the baseline came from.  Skipped for "learned"
    # (no meaningful baseline yet) and when the user has already set a value.
    if mode in ("dataset", "provided") and backbone_kwargs.get("scale") is None:
        current_refs: dict[int, float] = {
            int(z): float(e) for z, e in (ae_cfg.get("values") or {}).items()
        }
        if current_refs:
            scale_val: float = compute_energy_scale(datamodule.data_train, current_refs)
            backbone_kwargs["scale"] = scale_val
            print(f"[stats] scale={scale_val:.6f}")

    # avg_num_neighbors — MACE-style sum-aggregation normaliser.
    dressing_cfg: typing.Any = backbone_kwargs.get("dressing_kwargs")
    if isinstance(dressing_cfg, DictConfig) or isinstance(dressing_cfg, dict):
        dressing_dict: dict[str, typing.Any] = dict(dressing_cfg)
        if dressing_dict.get("avg_num_neighbors") is None:
            avg_nn: float = compute_avg_num_neighbors(datamodule.data_train)
            dressing_dict["avg_num_neighbors"] = avg_nn
            backbone_kwargs["dressing_kwargs"] = dressing_dict
            print(f"[stats] avg_num_neighbors={avg_nn:.4f}")

    # avg_num_neighbors for the ARACE backbone — it has no dressing_kwargs
    # sub-config; the normaliser is a top-level backbone key instead.
    if backbone_name == "simurgh_arace" and backbone_kwargs.get("avg_num_neighbors") is None:
        avg_nn_arace: float = compute_avg_num_neighbors(datamodule.data_train)
        backbone_kwargs["avg_num_neighbors"] = avg_nn_arace
        print(f"[stats] avg_num_neighbors={avg_nn_arace:.4f}")

    # pair_counts — for data-driven expert routing.
    # Injected when the backbone has an artisan_config with rare_artisan.enabled=true.
    artisan_cfg_raw: typing.Any = backbone_kwargs.get("artisan_config")
    if isinstance(artisan_cfg_raw, (DictConfig, dict)):
        ecfg: dict[str, typing.Any] = dict(artisan_cfg_raw)
        ge: dict[str, typing.Any] = dict(ecfg.get("rare_artisan") or {})
        if bool(ge.get("enabled", False)) and ecfg.get("pair_counts") is None:
            pc: dict[tuple[int, int], int] = compute_pair_counts(datamodule.data_train)
            total: int = max(1, sum(pc.values()))
            print(
                f"[stats] pair_counts computed ({len(pc)} pairs, " f"{total} total directed edges)"
            )
            ecfg["pair_counts"] = {list(k): v for k, v in pc.items()}
            backbone_kwargs["artisan_config"] = ecfg

    backbone: typing.Any = backbone_cls(**backbone_kwargs)

    head: typing.Any = _build_head(cfg, backbone)
    loss: CompositeLoss = _build_loss(cfg)

    module: GOALModule = GOALModule(
        backbone=backbone,
        head=head,
        loss=loss,
        config=cfg,
        compile_model=cfg.training.get("compile_model", False),
    )

    # Automatic checkpoint resumption — last.ckpt if exists, else None
    last_ckpt: Path = checkpoint_dir / "last.ckpt"
    resume_path: str | None = str(last_ckpt) if last_ckpt.exists() else None

    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="epoch"),
        RichLoggingCallback(),
        # NOTE: the stage-aware ``GOALRichProgressBar`` is registered
        # via the Hydra callback config (``configs/callbacks/
        # rich_progress_bar.yaml``) so the user has a single knob to
        # disable / replace it.  Don't re-add it here — Lightning
        # rejects multiple progress-bar callbacks.
    ]

    # Optionally add SLURM plugin
    plugins: list[typing.Any] = []
    if cfg.training.get("slurm_mode", False):
        from lightning.pytorch.plugins.environments import SLURMEnvironment

        plugins.append(SLURMEnvironment(auto_requeue=True))

    # Instantiate callbacks and loggers from Hydra config (if present)
    hydra_callbacks: list[Callback] = _instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = _instantiate_loggers(cfg.get("logger"))

    # Instantiate GOALCheckpointManager from the top-level checkpoint_manager block.
    # It lives outside cfg.callbacks deliberately (it is a first-class concern, not
    # an optional callback), so we handle it here explicitly.
    ckpt_manager_cfg = cfg.get("checkpoint_manager")
    if ckpt_manager_cfg is not None:
        ckpt_manager: GOALCheckpointManager = hydra.utils.instantiate(ckpt_manager_cfg)
        # Restore pool state if resuming
        if resume_path is not None:
            GOALCheckpointManager.restore_pools_from_dir(ckpt_manager, str(checkpoint_dir))
            print(f"[ckpt/resume] Pool state restored from {checkpoint_dir}/checkpoint_state.json")
        hydra_callbacks.append(ckpt_manager)

    # Guard: ModelCheckpoint + GOALCheckpointManager active at the same time
    # leads to double saves and conflicting deletion logic.
    from lightning.pytorch.callbacks import ModelCheckpoint

    has_model_checkpoint = any(isinstance(cb, ModelCheckpoint) for cb in hydra_callbacks)
    has_goal_manager = any(isinstance(cb, GOALCheckpointManager) for cb in hydra_callbacks)
    if has_model_checkpoint and has_goal_manager:
        raise ValueError(
            "Both ModelCheckpoint and GOALCheckpointManager are active simultaneously. "
            "They conflict: use one or the other.  For SIMURGH experiments use the "
            "top-level checkpoint_manager block.  For other models use ModelCheckpoint "
            "inside callbacks:."
        )

    # Merge callbacks: GOAL-specific + Hydra-configured
    all_callbacks: list[Callback] = callbacks + (hydra_callbacks or [])

    # Strategy: if cfg.strategy exists, use the strategy factory;
    # otherwise fall through to hydra.utils.instantiate (backward compat).
    strategy_override: dict[str, typing.Any] = {}
    if cfg.get("strategy") is not None:
        strategy_override["strategy"] = build_strategy(cfg)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=all_callbacks,
        logger=loggers or True,
        plugins=plugins or None,
        **strategy_override,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)

    # Mark completion so SLURM doesn't requeue after success
    if trainer.is_global_zero:
        (checkpoint_dir / "TRAINING_COMPLETE").touch()


@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="simurgh_gmd26")
def train_ml(cfg: DictConfig) -> None:
    """GOAL training entry point for self-contained configs/ml/ experiment files.

    Usage:
        goal-train-ml                                  # loads simurgh_gmd26.yaml (default)
        goal-train-ml --config-name simurgh_md17        # loads simurgh_md17.yaml
        goal-train-ml --config-name hyperspec_md17     # loads hyperspec_md17.yaml

    All parameters live in a single file under configs/ml/.
    Override any parameter from the CLI:
        goal-train-ml trainer.max_epochs=100
        goal-train-ml training.optimizer.lr=0.001
    """
    _run_training(cfg)


if __name__ == "__main__":
    train_ml()
