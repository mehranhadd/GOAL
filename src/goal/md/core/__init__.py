"""Core molecular dynamics functionality.

Factory classes for creating molecules, calculators, and dynamics objects.
All factories support custom builder registration for extensibility.
"""

from goal.md.core.calculator_factory import CalculatorBuilder, CalculatorFactory
from goal.md.core.md_factory import DynamicsBuilder, DynamicsFactory
from goal.md.core.molecule_factory import MoleculeBuilder, MoleculeFactory
from goal.md.core.molecule_tools import (
    box_molecule,
    generate_3d_coordinates_from_smiles,
)

__all__ = [
    "CalculatorFactory",
    "CalculatorBuilder",
    "MoleculeFactory",
    "MoleculeBuilder",
    "DynamicsFactory",
    "DynamicsBuilder",
    "generate_3d_coordinates_from_smiles",
    "box_molecule",
]
