"""Model loading utilities for integrating goal.ml models in MD simulations."""

from __future__ import annotations

import typing
from pathlib import Path

from ase.calculators.calculator import Calculator

if typing.TYPE_CHECKING:
    from goal.ml.utils.calculator import GOALCalculator


def load_goal_calculator(
    checkpoint_path: str | Path,
    cutoff: float | None = None,
    device: str = "cpu",
    dtype: typing.Any = None,
    head: str | None = None,
) -> GOALCalculator:
    """Load a goal.ml trained model as an ASE calculator.

    This is the primary method for using trained models from goal.ml in
    molecular dynamics simulations.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to Lightning checkpoint (.ckpt) from goal.ml training
    cutoff : float, optional
        Neighbor list cutoff in Ångströms. If None, read from checkpoint config.
    device : str
        Torch device ("cpu", "cuda", "cuda:0", etc.). Default: "cpu"
    dtype : torch.dtype, optional
        Precision. Default: torch.float64 (from checkpoint if available)
    head : str, optional
        Multi-head identifier for multi-task models

    Returns
    -------
    goal.ml.utils.calculator.GOALCalculator
        ASE-compatible calculator wrapping the trained model

    Raises
    ------
    FileNotFoundError
        If checkpoint not found
    RuntimeError
        If model loading fails

    Examples
    --------
    Load a trained model and run MD:

    >>> from goal.md.adapters.model_loader import load_goal_calculator
    >>> from ase.md.langevin import Langevin
    >>> from ase import units
    >>>
    >>> calc = load_goal_calculator("outputs/train/run_001/last.ckpt")
    >>> atoms.calc = calc
    >>> dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
    >>> dyn.run(1000)
    """
    from goal.ml.utils.calculator import GOALCalculator

    if dtype is None:
        import torch

        dtype = torch.float64

    calc = GOALCalculator(
        checkpoint_path=str(checkpoint_path),
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        head=head,
    )

    return calc


def get_model_cutoff(checkpoint_path: str | Path) -> float:
    """Extract neighbor list cutoff from model checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to Lightning checkpoint

    Returns
    -------
    float
        Cutoff in Ångströms

    Raises
    ------
    FileNotFoundError
        If checkpoint not found
    KeyError
        If cutoff not found in config
    """
    from goal.ml.training.module import GOALModule

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    module = GOALModule.load_from_checkpoint(str(checkpoint_path))
    cutoff = float(
        module.config.data.get(
            "cutoff",
            module.config.model.backbone.get("cutoff", 5.0),
        )
    )
    return cutoff
