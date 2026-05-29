"""Molecule factory for creating ASE Atoms objects from various sources."""

from __future__ import annotations

import abc
import typing

import ase
from ase import io as atoms_io

from goal.md.core.molecule_tools import generate_3d_coordinates_from_smiles


class MoleculeFactory:
    """Factory for creating molecules from different sources.

    Supports registration of custom builders via @register_molecule_set decorator.

    Examples
    --------
    >>> atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
    >>> atoms = MoleculeFactory.create("from_file", path="structure.xyz")
    """

    _builders: dict[str, MoleculeBuilder] = {}

    @classmethod
    def register(cls, key: str, builder: MoleculeBuilder) -> None:
        """Register a molecule builder."""
        cls._builders[key] = builder

    @classmethod
    def create(
        cls, key: str, *args: typing.Any, **kwargs: typing.Any
    ) -> ase.Atoms | list[ase.Atoms]:
        """Create molecule(s) using registered builder.

        Parameters
        ----------
        key : str
            Builder identifier (e.g., "from_smiles", "from_file")
        *args, **kwargs
            Arguments passed to builder

        Returns
        -------
        ase.Atoms or list[ase.Atoms]
            Created molecule(s)

        Raises
        ------
        ValueError
            If builder not found
        """
        builder = cls._builders.get(key)
        if builder is None:
            available = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Molecule builder '{key}' not registered. " f"Available: {available}"
            )
        return builder.build(*args, **kwargs)


class MoleculeBuilder(abc.ABC):
    """Abstract base class for molecule builders."""

    @abc.abstractmethod
    def build(self, *args: typing.Any, **kwargs: typing.Any) -> ase.Atoms | list[ase.Atoms]:
        """Build molecule(s) from parameters."""
        raise NotImplementedError


def register_molecule_set(
    key: str,
) -> typing.Callable[[type[MoleculeBuilder]], type[MoleculeBuilder]]:
    """Decorator to register molecule builder.

    Parameters
    ----------
    key : str
        Identifier for the builder

    Examples
    --------
    >>> @register_molecule_set("my_builder")
    ... class MyBuilder(MoleculeBuilder):
    ...     def build(self, **kwargs):
    ...         return ase.Atoms(...)
    """

    def decorator(
        builder_cls: type[MoleculeBuilder],
    ) -> type[MoleculeBuilder]:
        instance = builder_cls()
        MoleculeFactory.register(key, instance)
        return builder_cls

    return decorator


@register_molecule_set("from_smiles")
class FromSmilesBuilder(MoleculeBuilder):
    """Build molecule from SMILES string."""

    def build(self, smiles: str) -> ase.Atoms:
        """Create molecule from SMILES.

        Parameters
        ----------
        smiles : str
            SMILES string

        Returns
        -------
        ase.Atoms
            Molecule with 3D coordinates
        """
        symbols, positions = generate_3d_coordinates_from_smiles(smiles)
        return ase.Atoms(symbols, positions=positions)


@register_molecule_set("from_file")
class FromFileBuilder(MoleculeBuilder):
    """Build molecule(s) from file (XYZ, XSF, trajectory, etc)."""

    def build(self, path: str, **kwargs: typing.Any) -> ase.Atoms | list[ase.Atoms]:
        """Load molecule(s) from file.

        Parameters
        ----------
        path : str
            Path to structure file (supports ASE formats)
        **kwargs
            Additional arguments passed to ase.io.read

        Returns
        -------
        ase.Atoms or list[ase.Atoms]
            Loaded structure(s)
        """
        return atoms_io.read(path, **kwargs)


@register_molecule_set("from_db")
class FromDBBuilder(MoleculeBuilder):
    """Build molecule(s) from database."""

    def build(self, db_path: str, **kwargs: typing.Any) -> list[ase.Atoms]:
        """Load molecules from ASE database.

        Parameters
        ----------
        db_path : str
            Path to ASE database file
        **kwargs
            Additional query parameters

        Returns
        -------
        list[ase.Atoms]
            List of structures from database
        """
        raise NotImplementedError("Database loading not yet implemented")
