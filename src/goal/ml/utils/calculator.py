"""ASE Calculator interface for trained GOAL models.

Wraps any trained ``GOALModule`` checkpoint into a standard ASE
``Calculator`` so it can be used for:

- Single-point energy / force / stress calculations
- Geometry optimisation (``ase.optimize``)
- Molecular dynamics (``ase.md``)
- Nudged elastic band (NEB) transition-state searches
- Phonon calculations (``ase.phonons``)

Three checkpoint formats are accepted:

A. ``.simurgh`` archive — a single self-contained zip (weights +
   resolved config + metadata + frozen model source).  Recommended for
   sharing and deployment; loads from the frozen source, so it works
   even after the codebase has changed.
B. Checkpoint *directory* written by ``GOALCheckpointManager`` — must
   contain ``config.yaml``, ``metadata.json`` and ``frozen_source/``.
   Select which weights to use via ``checkpoint=`` (``"best"``,
   ``"last"``, ``"epoch=N"``, or a ``.ckpt`` filename).
C. Bare ``.ckpt`` file (legacy) — NOT self-contained: loading rebuilds
   the model from the *live* source code and may fail if it changed
   since the checkpoint was saved.

Usage::

    from goal.ml.utils.calculator import GOALCalculator

    # From a self-contained archive (FORMAT A — recommended)
    calc = GOALCalculator(checkpoint_path="my_model.simurgh")

    # From a checkpoint directory (FORMAT B)
    calc = GOALCalculator(checkpoint_path="logs/.../checkpoints", checkpoint="best")

    # From a bare checkpoint file (FORMAT C — legacy)
    calc = GOALCalculator(checkpoint_path="logs/train/runs/.../last.ckpt")

    # From an already-loaded module
    calc = GOALCalculator(module=my_module, cutoff=5.0)

    # Attach to ASE Atoms and compute
    from ase.build import molecule
    atoms = molecule("H2O")
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())

    # Run MD
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from ase import units
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
    dyn.run(100)
"""

from __future__ import annotations

import logging
import tempfile
import typing
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from goal.ml.training.archive import (
    ARCHIVE_SUFFIX,
    FROZEN_SOURCE_DIRNAME,
    frozen_source_imports,
    is_managed_checkpoint_dir,
    resolve_checkpoint,
    unpack_simurgh_archive,
)

logger = logging.getLogger(__name__)


class GOALCalculator(Calculator):
    """ASE Calculator backed by a trained GOAL model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        A ``.simurgh`` archive, a checkpoint directory written by
        ``GOALCheckpointManager``, or a bare Lightning ``.ckpt`` file
        (legacy).  Mutually exclusive with ``module``.
    module : GOALModule, optional
        An already-instantiated ``GOALModule``.  Mutually exclusive with
        ``checkpoint_path``.
    cutoff : float, optional
        Neighbour-list cutoff in Ångström.  Required when ``module`` is
        provided directly.  When loading from checkpoint, read from the
        saved config automatically.
    device : str
        Torch device — ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    dtype : torch.dtype or None
        Precision for atomic positions and cell.  When ``None`` (the
        recommended default) the calculator sniffs the dtype of the
        loaded model's parameters and uses that, so the inputs always
        match the model.  Pass an explicit dtype only to override.
    head : str or None
        Multi-head tag for models trained with multiple heads.
    checkpoint : str
        Which weights to load when ``checkpoint_path`` is a checkpoint
        *directory*: ``"best"`` (default), ``"last"``, ``"epoch=N"``,
        or a ``.ckpt`` filename/path.  Ignored for the other formats.
    **kwargs
        Forwarded to ``ase.calculators.calculator.Calculator.__init__``.
    """

    implemented_properties: typing.ClassVar[list[str]] = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        module: typing.Any | None = None,
        cutoff: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        head: str | None = None,
        checkpoint: str = "best",
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)

        if checkpoint_path is not None and module is not None:
            raise ValueError("Provide either 'checkpoint_path' or 'module', not both.")
        if checkpoint_path is None and module is None:
            raise ValueError("Provide either 'checkpoint_path' or 'module'.")

        self.device: torch.device = torch.device(device)
        self.head: str | None = head
        self._checkpoint_selector: str = checkpoint
        # Keeps an unpacked .simurgh archive alive for the calculator's lifetime
        self._archive_tmpdir: tempfile.TemporaryDirectory | None = None

        if checkpoint_path is not None:
            self._module, self._cutoff = self._load_model(checkpoint_path)
        else:
            if cutoff is None:
                raise ValueError("'cutoff' is required when providing a module directly.")
            self._module = module
            self._cutoff = float(cutoff)

        self._module = self._module.to(self.device)
        self._module.eval()

        # Default the input dtype to the model's actual parameter dtype so
        # the AtomicGraph never disagrees with the network it feeds.  An
        # explicit ``dtype=...`` still wins.
        if dtype is None:
            try:
                dtype = next(self._module.parameters()).dtype
            except StopIteration:
                dtype = torch.float64
        self.dtype: torch.dtype = dtype

    # ------------------------------------------------------------------
    # Loading — format dispatch
    # ------------------------------------------------------------------

    def _load_model(self, path: str | Path) -> tuple[typing.Any, float]:
        """Dispatch on the three supported checkpoint formats."""
        path_str = str(path)

        if path_str.endswith(ARCHIVE_SUFFIX):
            # FORMAT A — self-contained .simurgh archive
            return self._load_from_archive(path)

        if Path(path).is_dir():
            # FORMAT B — checkpoint directory written by GOALCheckpointManager
            dirpath = Path(path)
            if not is_managed_checkpoint_dir(dirpath):
                raise FileNotFoundError(
                    f"Directory {dirpath} is missing config.yaml, metadata.json, "
                    f"or {FROZEN_SOURCE_DIRNAME}/. This is not a valid SIMURGH "
                    f"checkpoint directory. Pass a {ARCHIVE_SUFFIX} archive or a "
                    f"directory produced by GOALCheckpointManager."
                )
            ckpt_path = resolve_checkpoint(dirpath, self._checkpoint_selector)
            return self._load_from_directory(dirpath, ckpt_path)

        if path_str.endswith(".ckpt"):
            # FORMAT C — legacy bare checkpoint
            logger.warning(
                "Loading from a bare .ckpt file. This format is not "
                "self-contained — if source files have changed since this "
                "checkpoint was saved, loading may fail. Use a %s archive "
                "or the checkpoint directory for reliable loading.",
                ARCHIVE_SUFFIX,
            )
            return self._build_module_from_ckpt(path)

        raise ValueError(
            f"Unrecognised checkpoint path: {path}. Expected a {ARCHIVE_SUFFIX} "
            f"archive, a checkpoint directory, or a .ckpt file."
        )

    def _load_from_archive(self, path: str | Path) -> tuple[typing.Any, float]:
        """FORMAT A: unpack the archive and load from its frozen source."""
        archive = Path(path)
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {archive}")
        self._archive_tmpdir = tempfile.TemporaryDirectory(prefix="goal_simurgh_")
        dest = unpack_simurgh_archive(archive, self._archive_tmpdir.name)
        weights = dest / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Archive {archive} contains no weights.pt")
        logger.info("Loading %s from frozen source (self-contained archive).", archive.name)
        return self._load_from_directory(dest, weights)

    def _load_from_directory(
        self,
        dirpath: Path,
        ckpt_path: Path,
    ) -> tuple[typing.Any, float]:
        """FORMAT B: rebuild the model with imports served from frozen source."""
        frozen = dirpath / FROZEN_SOURCE_DIRNAME
        logger.info(
            "Loading %s using frozen source at %s — immune to later code changes.",
            ckpt_path.name,
            frozen,
        )
        with frozen_source_imports(frozen):
            return self._build_module_from_ckpt(ckpt_path)

    # ------------------------------------------------------------------
    # Loading — model reconstruction
    # ------------------------------------------------------------------

    def _build_module_from_ckpt(
        self,
        path: str | Path,
    ) -> tuple[typing.Any, float]:
        """Load a GOALModule from a Lightning checkpoint.

        Reconstructs backbone + head + loss from the saved Hydra config,
        loads the state dict, and extracts the neighbour-list cutoff.
        Lightning's ``load_from_checkpoint`` cannot be used directly because
        backbone/head/loss are excluded from ``save_hyperparameters``.

        All ``goal`` imports happen inside this function so that, when it
        runs under :func:`frozen_source_imports`, the model classes are
        executed from the frozen snapshot instead of the live codebase.
        """
        import torch

        # Populate the registries (auto-discovery walks the package path,
        # which under a frozen import context is the frozen snapshot).
        import goal.ml.nn.heads  # noqa: F401
        import goal.ml.nn.models  # noqa: F401
        from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
        from goal.ml.training.loss import CompositeLoss, WeightedLoss
        from goal.ml.training.module import GOALModule

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        raw = torch.load(str(path), map_location=self.device, weights_only=False)
        cfg = raw["hyper_parameters"]["config"]

        # Rebuild backbone from saved config
        bb_cls = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
        bb_kw = {k: v for k, v in cfg.model.backbone.items() if k != "name"}
        backbone = bb_cls(**bb_kw)

        # Rebuild head (None for monolithic models)
        head_cfg = cfg.model.get("head", None)
        if head_cfg is not None:
            head_cls = HEAD_REGISTRY.get(head_cfg.name)
            head_kw = {k: v for k, v in head_cfg.items() if k != "name"}
            head = head_cls(**head_kw)
        else:
            head = None

        # Rebuild loss components (not used for inference; skip gracefully on
        # API mismatch between frozen source and config, e.g. when the frozen
        # source pre-dates a new loss parameter).
        try:
            losses: list[WeightedLoss] = []
            for lc in cfg.training.losses:
                loss_cls = LOSS_REGISTRY.get(lc.name)
                fn_spec = lc.get("fn", "mse")
                lkw = {k: v for k, v in lc.items() if k not in ("name", "weight", "fn")}
                fn = fn_spec if isinstance(fn_spec, str) else "mse"
                losses.append(
                    WeightedLoss(loss_cls(loss_fn=fn, **lkw), weight=lc.weight, label=lc.name)
                )
            loss = CompositeLoss(losses)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not reconstruct loss components from frozen source (%s). "
                "Using an empty CompositeLoss — safe for inference only.",
                exc,
            )
            loss = CompositeLoss([])

        module = GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)
        module.load_state_dict(raw["state_dict"], strict=False)

        # Extract cutoff from saved config
        cutoff: float = float(
            cfg.data.get(
                "cutoff",
                cfg.model.backbone.get("cutoff", 5.0),
            )
        )
        return module, cutoff

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate energy, forces, and/or stress for the given Atoms.

        This method is called automatically by ASE when you access
        ``atoms.get_potential_energy()``, ``atoms.get_forces()``, etc.

        Runs under ``torch.enable_grad`` so that autograd-based force heads
        (``EnergyForcesHead``, ``DualForcesHead`` in autograd/hybrid mode)
        can compute ``-∂E/∂r``.  Gradients are not retained after this call.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            raise RuntimeError("No atoms object set on calculator.")

        from goal.ml.data.graph import AtomicGraph

        graph: AtomicGraph = AtomicGraph.from_ase(
            self.atoms,
            cutoff=self._cutoff,
            dtype=self.dtype,
            head=self.head,
        )
        graph = graph.to(self.device)

        with torch.enable_grad():
            predictions: dict[str, torch.Tensor] = self._module(graph)

        # Energy — scalar per structure
        if "energy" in predictions:
            energy: torch.Tensor = predictions["energy"]
            self.results["energy"] = energy.detach().cpu().item()

        # Forces — (N, 3) per atom
        if "forces" in predictions:
            forces: torch.Tensor = predictions["forces"]
            self.results["forces"] = forces.detach().cpu().numpy()

        # Stress — (3, 3) → Voigt (6,) in eV/Å³
        if "stress" in predictions:
            stress: torch.Tensor = predictions["stress"]
            stress_np: np.ndarray = stress.detach().cpu().numpy()
            # ASE expects Voigt notation with sign convention: positive = compressive
            if stress_np.shape == (3, 3):
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress_np)
            elif stress_np.shape == (6,):
                self.results["stress"] = stress_np
            else:
                self.results["stress"] = stress_np.flatten()[:6]

    def __repr__(self) -> str:
        return (
            f"GOALCalculator(cutoff={self._cutoff}, " f"device={self.device}, dtype={self.dtype})"
        )
