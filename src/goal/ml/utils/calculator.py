"""ASE Calculator interface for trained GOAL models.

Wraps any trained ``GOALModule`` checkpoint into a standard ASE
``Calculator`` so it can be used for:

- Single-point energy / force / stress calculations
- Geometry optimisation (``ase.optimize``)
- Molecular dynamics (``ase.md``)
- Nudged elastic band (NEB) transition-state searches
- Phonon calculations (``ase.phonons``)

Usage::

    from goal.ml.utils.calculator import GOALCalculator

    # From a checkpoint file
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

import typing
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


class GOALCalculator(Calculator):
    """ASE Calculator backed by a trained GOAL model.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a Lightning checkpoint (``.ckpt``).  Mutually exclusive
        with ``module``.
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
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)

        if checkpoint_path is not None and module is not None:
            raise ValueError("Provide either 'checkpoint_path' or 'module', not both.")
        if checkpoint_path is None and module is None:
            raise ValueError("Provide either 'checkpoint_path' or 'module'.")

        self.device: torch.device = torch.device(device)
        self.head: str | None = head

        if checkpoint_path is not None:
            self._module, self._cutoff = self._load_checkpoint(checkpoint_path)
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

    def _load_checkpoint(
        self,
        path: str | Path,
    ) -> tuple[typing.Any, float]:
        """Load a GOALModule from a Lightning checkpoint.

        Reconstructs backbone + head + loss from the saved Hydra config,
        loads the state dict, and extracts the neighbour-list cutoff.
        Lightning's ``load_from_checkpoint`` cannot be used directly because
        backbone/head/loss are excluded from ``save_hyperparameters``.
        """
        import torch

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

        # Rebuild loss components
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
