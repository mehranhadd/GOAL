"""Bridges and adapters between goal.ml and ASE/MD infrastructure.

Provides seamless conversion and integration between:
- goal.ml data structures (AtomicGraph) ↔ ASE Atoms
- goal.ml trained models ↔ ASE calculators
- goal.ml trajectories ↔ ASE trajectory files
"""

from __future__ import annotations

import typing

import torch
from ase import Atoms

if typing.TYPE_CHECKING:
    from goal.ml.data.graph import AtomicGraph


def atomic_graph_from_ase(
    atoms: Atoms,
    cutoff: float,
    energy: float | None = None,
    forces: list | None = None,
    stress: list | None = None,
    dtype: torch.dtype = torch.float64,
    head: str | None = None,
) -> AtomicGraph:
    """Convert ASE Atoms to goal.ml AtomicGraph.

    This is the primary bridge for data conversion from ASE (MD) to goal.ml
    (ML training/analysis).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to convert
    cutoff : float
        Neighbor list cutoff in Ångströms
    energy : float, optional
        Potential energy (for training data)
    forces : array-like, optional
        Atomic forces (for training data)
    stress : array-like, optional
        Stress tensor (for training data)
    dtype : torch.dtype
        Tensor precision (default: torch.float64)
    head : str, optional
        Multi-head identifier for training

    Returns
    -------
    goal.ml.data.graph.AtomicGraph
        Converted structure in goal.ml format

    Examples
    --------
    >>> from goal.md.adapters.ase_converter import atomic_graph_from_ase
    >>> from ase.build import molecule
    >>> atoms = molecule('H2O')
    >>> graph = atomic_graph_from_ase(atoms, cutoff=5.0)
    """
    from goal.ml.data.graph import AtomicGraph

    return AtomicGraph.from_ase(
        atoms,
        cutoff=cutoff,
        energy=energy,
        forces=forces,
        stress=stress,
        dtype=dtype,
        head=head,
        neighbor_list_backend="ase",
    )


def ase_atoms_from_atomic_graph(graph: AtomicGraph) -> Atoms:
    """Convert goal.ml AtomicGraph to ASE Atoms.

    This bridge allows using goal.ml structures in ASE MD simulations.

    Parameters
    ----------
    graph : goal.ml.data.graph.AtomicGraph
        Structure in goal.ml format

    Returns
    -------
    ase.Atoms
        Converted structure in ASE format

    Examples
    --------
    >>> from goal.md.adapters.ase_converter import ase_atoms_from_atomic_graph
    >>> from goal.ml.data.graph import AtomicGraph
    >>> graph = AtomicGraph(...)
    >>> atoms = ase_atoms_from_atomic_graph(graph)
    """
    import numpy as np

    positions = graph.pos.cpu().numpy()
    atomic_numbers = graph.z.cpu().numpy()
    cell = graph.cell.squeeze(0).cpu().numpy() if graph.cell is not None else np.zeros((3, 3))
    pbc = graph.pbc.cpu().numpy() if graph.pbc is not None else np.zeros(3, dtype=bool)

    atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
    return atoms
