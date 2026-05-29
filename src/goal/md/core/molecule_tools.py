"""Molecular geometry and setup utilities."""

from __future__ import annotations

import typing

import numpy as np
from ase import Atoms

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def generate_3d_coordinates_from_smiles(
    smiles: str,
) -> tuple[list[str], np.ndarray]:
    """Generate 3D coordinates for a molecule from SMILES string.

    Uses RDKit to:
    1. Create molecule from SMILES
    2. Add implicit hydrogens
    3. Generate 3D conformation (ETKDG algorithm)
    4. Optimize geometry with MMFF94 force field

    Parameters
    ----------
    smiles : str
        SMILES string representation of molecule

    Returns
    -------
    tuple[list[str], np.ndarray]
        Atomic symbols and (N, 3) coordinate array in Ångströms

    Raises
    ------
    ValueError
        If 3D conformation generation fails
    ImportError
        If RDKit not installed
    """
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit not installed. Install with: pip install rdkit or pixi run command"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    conformer_id = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    if conformer_id == -1:
        raise ValueError(f"Could not generate 3D conformation for SMILES: {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, confId=0)

    conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return symbols, positions


def box_molecule(
    molecule: Atoms,
    box_scale: float = 10.0,
    pbc: bool = True,
    **kwargs: typing.Any,
) -> None:
    """Set a cubic periodic box around molecule.

    Sizes the box based on molecular diameter and optionally applies
    periodic boundary conditions.

    Parameters
    ----------
    molecule : ase.Atoms
        Molecular structure to box
    box_scale : float
        Box size = box_scale × max_molecular_span (default: 10)
    pbc : bool
        Apply periodic boundary conditions (default: True)
    **kwargs
        Additional arguments passed to set_cell/center methods
    """
    positions = molecule.get_positions()
    diameter = np.max(np.ptp(positions, axis=0))
    box_size = box_scale * diameter

    molecule.set_cell([box_size, box_size, box_size], **kwargs)
    molecule.center(**kwargs)
    molecule.set_pbc(pbc)
