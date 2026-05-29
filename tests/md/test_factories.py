"""Tests for goal.md factory classes.

These tests cover:
- MoleculeFactory: from_smiles, from_file
- CalculatorFactory: registration, creation, error handling
- DynamicsFactory: registration, creation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

# ---------------------------------------------------------------------------
# MoleculeFactory
# ---------------------------------------------------------------------------


class TestMoleculeFactory:
    """Tests for MoleculeFactory."""

    def test_factory_is_importable(self) -> None:
        """Test MoleculeFactory is importable and has create method."""
        from goal.md.core.molecule_factory import MoleculeFactory

        assert hasattr(MoleculeFactory, "create")

    def test_from_file(self, tmp_path: Path) -> None:
        """Test MoleculeFactory.create creates Atoms from XYZ file."""
        from goal.md.core.molecule_factory import MoleculeFactory

        xyz = tmp_path / "mol.xyz"
        xyz.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")
        atoms = MoleculeFactory.create("from_file", path=str(xyz))
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("rdkit", reason="rdkit not installed"),
        reason="rdkit not installed",
    )
    def test_from_smiles_water(self) -> None:
        from goal.md.core.molecule_factory import MoleculeFactory

        atoms = MoleculeFactory.create("from_smiles", smiles="O")
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3  # O + 2H

    def test_from_file_missing_raises(self, tmp_path: Path) -> None:
        """Test MoleculeFactory raises when file does not exist."""
        from goal.md.core.molecule_factory import MoleculeFactory

        with pytest.raises((FileNotFoundError, Exception)):
            MoleculeFactory.create("from_file", path=str(tmp_path / "missing.xyz"))

    def test_unknown_method_raises(self) -> None:
        """Test MoleculeFactory raises for unknown creation method."""
        from goal.md.core.molecule_factory import MoleculeFactory

        with pytest.raises((ValueError, KeyError)):
            MoleculeFactory.create("unknown_method_xyz")


# ---------------------------------------------------------------------------
# CalculatorFactory
# ---------------------------------------------------------------------------


class TestCalculatorFactory:
    """Tests for CalculatorFactory."""

    def test_factory_is_importable(self) -> None:
        """Test CalculatorFactory is importable and has create method."""
        from goal.md.core.calculator_factory import CalculatorFactory

        assert hasattr(CalculatorFactory, "create")

    def test_registered_keys(self) -> None:
        """Test CalculatorFactory has all expected calculator types registered."""
        from goal.md.core.calculator_factory import CalculatorFactory

        keys = list(CalculatorFactory._builders.keys())
        assert "goal_model" in keys
        assert "flashmd" in keys
        assert "mace" in keys
        assert "orca" in keys
        assert "cp2k" in keys
        assert "psi4" in keys
        assert "xtb" in keys
        assert "nequip" in keys

    def test_unknown_key_raises(self) -> None:
        """Test CalculatorFactory raises for unregistered calculator."""
        from goal.md.core.calculator_factory import CalculatorFactory

        with pytest.raises(ValueError, match="not registered"):
            CalculatorFactory.create("nonexistent_calc_xyz")

    def test_custom_registration(self) -> None:
        """Test custom calculators can be registered with decorator."""
        from goal.md.core.calculator_factory import (
            CalculatorBuilder,
            CalculatorFactory,
            register_calculator,
        )

        @register_calculator("_test_emt")
        class _TestEMTBuilder(CalculatorBuilder):
            def build(self, **kwargs):  # type: ignore[override]
                return EMT()

        calc = CalculatorFactory.create("_test_emt")
        assert isinstance(calc, EMT)

    def test_flashmd_requires_import(self) -> None:
        """Test CalculatorFactory handles missing flashmd gracefully."""
        from goal.md.core.calculator_factory import CalculatorFactory

        try:
            calc = CalculatorFactory.create("flashmd")
        except ImportError:
            pass  # expected when flashmd not installed
        except Exception as exc:
            pytest.fail(f"Unexpected exception: {exc}")

    def test_xtb_requires_import(self) -> None:
        """Test CalculatorFactory handles missing xtb gracefully."""
        from goal.md.core.calculator_factory import CalculatorFactory

        try:
            calc = CalculatorFactory.create("xtb")
            from ase.calculators.calculator import Calculator

            assert isinstance(calc, Calculator)
        except ImportError:
            pass  # expected when xtb-python not installed


# ---------------------------------------------------------------------------
# DynamicsFactory
# ---------------------------------------------------------------------------


class TestDynamicsFactory:
    """Tests for DynamicsFactory."""

    def test_factory_is_importable(self) -> None:
        """Test DynamicsFactory is importable and has create method."""
        from goal.md.core.md_factory import DynamicsFactory

        assert hasattr(DynamicsFactory, "create")

    def test_registered_keys(self) -> None:
        """Test DynamicsFactory has all expected dynamics types registered."""
        from goal.md.core.md_factory import DynamicsFactory

        keys = list(DynamicsFactory._builders.keys())
        assert "langevin_ase" in keys
        assert "langevin_flashmd" in keys

    def test_langevin_ase_creation(self, water_atoms: Atoms) -> None:
        """Test DynamicsFactory creates Langevin dynamics successfully."""
        from ase import units

        from goal.md.core.md_factory import DynamicsFactory

        dyn = DynamicsFactory.create(
            "langevin_ase",
            atoms=water_atoms,
            timestep=1.0 * units.fs,
            temperature_K=300.0,
            friction=0.01,
        )
        assert hasattr(dyn, "run")
        assert dyn.atoms is water_atoms

    def test_unknown_key_raises(self, water_atoms: Atoms) -> None:
        """Test DynamicsFactory raises for unregistered dynamics type."""
        from goal.md.core.md_factory import DynamicsFactory

        with pytest.raises(ValueError, match="not registered"):
            DynamicsFactory.create("nonexistent_dynamics_xyz", atoms=water_atoms)
