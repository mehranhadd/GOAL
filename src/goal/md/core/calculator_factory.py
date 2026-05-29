"""Calculator factory for creating ASE calculators from various backends.

Supports:
- ML models trained in goal.ml (via checkpoint loading)
- Quantum chemistry: ORCA, CP2K, PSI4
- Pretrained models: MACE, NequIP, FlashMD
- Semi-empirical: xTB
"""

from __future__ import annotations

import abc
import os
import pathlib
import shutil
import typing
from pathlib import Path

import ase.calculators.calculator
import torch

try:
    from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate

    HAS_ORCA = True
except ImportError:
    HAS_ORCA = False

try:
    from ase.calculators.cp2k import CP2K

    HAS_CP2K = True
except ImportError:
    HAS_CP2K = False

try:
    from flashmd import get_pretrained
    from flashmd.ase import EnergyCalculator

    HAS_FLASHMD = True
except ImportError:
    HAS_FLASHMD = False

try:
    from mace.calculators import MACECalculator, mace_mp

    HAS_MACE = True
except ImportError:
    HAS_MACE = False

try:
    from xtb.ase.calculator import XTB

    HAS_XTB = True
except ImportError:
    HAS_XTB = False

try:
    import nequip  # noqa: F401

    HAS_NEQUIP = True
except ImportError:
    HAS_NEQUIP = False


class CalculatorFactory:
    """Factory for creating ASE calculators.

    Supports registration of custom builders for any calculator backend.

    Examples
    --------
    >>> calc = CalculatorFactory.create("goal_model", checkpoint="model.ckpt")
    >>> calc = CalculatorFactory.create("flashmd")
    >>> calc = CalculatorFactory.create("orca", orca_path="/path/to/orca")
    """

    _builders: dict[str, CalculatorBuilder] = {}

    @classmethod
    def register(cls, key: str, builder: CalculatorBuilder) -> None:
        """Register a calculator builder."""
        cls._builders[key] = builder

    @classmethod
    def create(
        cls, key: str, *args: typing.Any, **kwargs: typing.Any
    ) -> ase.calculators.calculator.Calculator:
        """Create calculator using registered builder.

        Parameters
        ----------
        key : str
            Builder identifier
        *args, **kwargs
            Arguments passed to builder

        Returns
        -------
        ase.calculators.calculator.Calculator
            Calculator instance

        Raises
        ------
        ValueError
            If builder not found
        """
        builder = cls._builders.get(key)
        if builder is None:
            available = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Calculator builder '{key}' not registered. " f"Available: {available}"
            )
        return builder.build(*args, **kwargs)


class CalculatorBuilder(abc.ABC):
    """Abstract base class for calculator builders."""

    @abc.abstractmethod
    def build(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> ase.calculators.calculator.Calculator:
        """Build calculator from parameters."""
        raise NotImplementedError


def register_calculator(
    key: str,
) -> typing.Callable[[type[CalculatorBuilder]], type[CalculatorBuilder]]:
    """Decorator to register calculator builder."""

    def decorator(
        builder_cls: type[CalculatorBuilder],
    ) -> type[CalculatorBuilder]:
        instance = builder_cls()
        CalculatorFactory.register(key, instance)
        return builder_cls

    return decorator


# ============================================================================
# ML Models from goal.ml
# ============================================================================


@register_calculator("goal_model")
class GOALModelBuilder(CalculatorBuilder):
    """Load trained models from goal.ml checkpoints."""

    def build(
        self,
        checkpoint: str | Path,
        cutoff: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        head: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Load goal.ml trained model as ASE calculator.

        Parameters
        ----------
        checkpoint : str or Path
            Path to Lightning checkpoint (.ckpt)
        cutoff : float, optional
            Cutoff in Ångströms (read from config if None)
        device : str
            Torch device ("cpu" or "cuda")
        dtype : torch.dtype
            Precision (torch.float64 default)
        head : str, optional
            Multi-head identifier

        Returns
        -------
        GOALCalculator
            ASE-compatible calculator wrapping trained model
        """
        from goal.ml.utils.calculator import GOALCalculator

        calc = GOALCalculator(
            checkpoint_path=checkpoint,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            head=head,
        )
        return calc


# ============================================================================
# FlashMD
# ============================================================================


@register_calculator("flashmd")
class FlashMDBuilder(CalculatorBuilder):
    """Create FlashMD pretrained model calculator."""

    def build(
        self,
        model_name: str = "pet-omatpes-v2",
        device: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create FlashMD calculator.

        Parameters
        ----------
        model_name : str
            Model identifier (default: "pet-omatpes-v2")
        device : str
            Torch device. Auto-detects CUDA if None.

        Returns
        -------
        flashmd.ase.EnergyCalculator
            FlashMD calculator

        Raises
        ------
        ImportError
            If FlashMD not installed
        """
        if not HAS_FLASHMD:
            raise ImportError("FlashMD not installed. Install with: pip install flashmd")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        energy_model, _ = get_pretrained(model_name, 16)
        calc = EnergyCalculator(energy_model, device=device)
        return calc


# ============================================================================
# MACE
# ============================================================================


@register_calculator("mace")
class MACEBuilder(CalculatorBuilder):
    """Create MACE model calculator."""

    def build(
        self,
        model_name_or_path: str = "small",
        device: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create MACE calculator.

        Parameters
        ----------
        model_name_or_path : str
            "small"/"medium"/"large" (universal) or path to checkpoint
        device : str
            Torch device. Auto-detects CUDA if None.

        Returns
        -------
        mace.calculators.MACECalculator
            MACE calculator

        Raises
        ------
        ImportError
            If MACE not installed
        """
        if not HAS_MACE:
            raise ImportError("MACE not installed. Install with: pip install mace-torch")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name_or_path in ["small", "medium", "large"]:
            calc = mace_mp(model_name_or_path, device=device)
        else:
            calc = MACECalculator(model_name_or_path, device=device)

        return calc


# ============================================================================
# Quantum Chemistry
# ============================================================================


@register_calculator("orca")
class ORCABuilder(CalculatorBuilder):
    """Create ORCA quantum chemistry calculator."""

    def build(
        self,
        orca_path: str,
        keywords: str = "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        directory: str = "orca_files",
    ) -> ase.calculators.calculator.Calculator:
        """Create ORCA calculator.

        Parameters
        ----------
        orca_path : str
            Path to ORCA executable
        keywords : str
            ORCA input keywords
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity
        num_threads : int
            Number of threads
        directory : str
            Output directory

        Returns
        -------
        ase.calculators.orca.ORCA
            ORCA calculator

        Raises
        ------
        ImportError
            If ASE ORCA support not available
        """
        if not HAS_ORCA:
            raise ImportError("ORCA support not available. Check ASE installation.")

        profile = OrcaProfile(command=orca_path)
        calc = ORCA(
            template=OrcaTemplate(
                keywords=keywords,
                charge=charge,
                multiplicity=multiplicity,
            ),
            profile=profile,
            num_threads=num_threads,
            directory=directory,
        )
        return calc


@register_calculator("cp2k")
class CP2KBuilder(CalculatorBuilder):
    """Create CP2K quantum chemistry calculator."""

    def build(
        self,
        cp2k_path: str = "cp2k.ssmp",
        input_file: str | None = None,
        print_level: str = "LOW",
    ) -> ase.calculators.calculator.Calculator:
        """Create CP2K calculator.

        Parameters
        ----------
        cp2k_path : str
            Path to CP2K executable
        input_file : str, optional
            Path to CP2K input file content
        print_level : str
            CP2K print level

        Returns
        -------
        ase.calculators.cp2k.CP2K
            CP2K calculator

        Raises
        ------
        ImportError
            If ASE CP2K support not available
        FileNotFoundError
            If input file not found
        """
        if not HAS_CP2K:
            raise ImportError("CP2K support not available. Check ASE installation.")

        inp_content = None
        if input_file:
            inp_path = Path(input_file)
            if not inp_path.exists():
                raise FileNotFoundError(f"CP2K input file not found: {inp_path}")
            inp_content = inp_path.read_text()

        real_binary = shutil.which(cp2k_path)
        if not real_binary:
            raise FileNotFoundError(f"CP2K executable not found: {cp2k_path}")

        final_command = real_binary
        if "cp2k_shell" not in str(real_binary):
            symlink_name = "cp2k_shell.ssmp"
            symlink_path = Path.cwd() / symlink_name
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            os.symlink(real_binary, symlink_path)
            final_command = str(symlink_path.absolute())

        calc = CP2K(
            command=final_command,
            inp=inp_content,
            print_level=print_level,
        )
        return calc


@register_calculator("psi4")
class PSI4Builder(CalculatorBuilder):
    """Create PSI4 quantum chemistry calculator."""

    def build(self, **kwargs: typing.Any) -> ase.calculators.calculator.Calculator:
        """Create PSI4 calculator.

        Raises
        ------
        NotImplementedError
            PSI4 support not yet implemented
        """
        raise NotImplementedError("PSI4 support not yet implemented")


# ============================================================================
# Semi-empirical
# ============================================================================


@register_calculator("xtb")
class XTBBuilder(CalculatorBuilder):
    """Create xTB semi-empirical calculator (GFN2-xTB by default)."""

    def build(
        self,
        method: str = "GFN2-xTB",
        charge: int = 0,
        uhf: int = 0,
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        max_iterations: int = 250,
        solvent: str | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Create xTB semi-empirical calculator.

        Parameters
        ----------
        method : str
            xTB method: ``"GFN1-xTB"`` or ``"GFN2-xTB"`` (default).
        charge : int
            Total molecular charge.
        uhf : int
            Number of unpaired electrons.
        accuracy : float
            Numerical accuracy (1.0 = default).
        electronic_temperature : float
            Fermi smearing temperature in Kelvin.
        max_iterations : int
            Maximum SCF iterations.
        solvent : str, optional
            Implicit solvent name (e.g. ``"water"``).

        Returns
        -------
        xtb.ase.calculator.XTB
            xTB calculator instance.

        Raises
        ------
        ImportError
            If ``xtb-python`` is not installed.
        """
        if not HAS_XTB:
            raise ImportError(
                "xtb-python not installed. Install with: pip install xtb-python "
                "or: pip install 'goal[md]'"
            )

        kwargs: dict[str, typing.Any] = dict(
            method=method,
            charge=charge,
            uhf=uhf,
            accuracy=accuracy,
            electronic_temperature=electronic_temperature,
            max_iterations=max_iterations,
        )
        if solvent is not None:
            kwargs["solvent"] = solvent

        return XTB(**kwargs)


# ============================================================================
# NequIP
# ============================================================================


@register_calculator("nequip")
class NequIPBuilder(CalculatorBuilder):
    """Create a NequIP / Allegro trained-model calculator."""

    def build(
        self,
        model_path: str | Path,
        device: str | None = None,
        species_to_type_name: dict[str, str] | None = None,
    ) -> ase.calculators.calculator.Calculator:
        """Load a NequIP or Allegro model as an ASE calculator.

        Parameters
        ----------
        model_path : str or Path
            Path to a deployed NequIP model (``*.pth`` / ``*.pt``).
        device : str, optional
            Torch device. Auto-detected if ``None``.
        species_to_type_name : dict, optional
            Mapping of element symbol to NequIP type name when the model
            uses custom type names.

        Returns
        -------
        nequip.ase.NequIPCalculator
            Calculator wrapping the trained NequIP model.

        Raises
        ------
        ImportError
            If ``nequip`` is not installed.
        """
        if not HAS_NEQUIP:
            raise ImportError("NequIP not installed. Install with: pip install nequip")

        from nequip.ase import NequIPCalculator

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs: dict[str, typing.Any] = {"model_path": str(model_path), "device": device}
        if species_to_type_name is not None:
            kwargs["species_to_type_name"] = species_to_type_name

        return NequIPCalculator.from_deployed_model(**kwargs)
