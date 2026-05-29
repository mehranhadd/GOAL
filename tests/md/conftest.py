"""Shared fixtures for the goal.md test suite."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT


@pytest.fixture()
def water_atoms() -> Atoms:
    """Simple 3-atom water molecule with a dummy calculator."""
    atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    )
    atoms.calc = EMT()
    return atoms


@pytest.fixture()
def ethanol_atoms() -> Atoms:
    """9-atom ethanol molecule (from ASE build)."""
    from ase.build import molecule

    atoms = molecule("CH3CH2OH")
    atoms.calc = EMT()
    return atoms


@pytest.fixture()
def emt_calculator() -> Calculator:
    """Effective medium theory (EMT) calculator — no external dependencies."""
    return EMT()


@pytest.fixture()
def periodic_atoms() -> Atoms:
    """Periodic copper FCC slab for testing PBC paths."""
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    atoms.calc = EMT()
    return atoms
