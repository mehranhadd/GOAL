"""Shared fixtures for SIMURGH tests.

Provides hand-crafted molecules (methane, water, methylamine) and optional
real GMD trajectory data, all wrapped as ``AtomicGraph`` / PyG ``Batch``
so that the model tests can run without touching any dataset on disk by
default.  Float64 is used throughout because GOAL's default dtype is
``torch.float64`` and several equivariant primitives (Bessel basis,
polynomial envelope) register their constants in float64.

Real GMD fixtures are conditionally loaded from::

    data/GMD/FragmentDuplication/Training/O2C4H8_PBE.traj

Tests that depend on them are automatically skipped when the file is absent.
"""

from __future__ import annotations

import math
import typing
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph

# ---------------------------------------------------------------------------
# GMD trajectory path — relative to the project root
# ---------------------------------------------------------------------------

_GMD_TRAJ: Path = (
    Path(__file__).parents[3]  # …/tests/ml/simurgh → project root
    / "data"
    / "GMD"
    / "FragmentDuplication"
    / "Training"
    / "O2C4H8_PBE.traj"
)

_GMD_AVAILABLE: bool = _GMD_TRAJ.exists()

requires_gmd = pytest.mark.skipif(
    not _GMD_AVAILABLE,
    reason=f"GMD trajectory not found at {_GMD_TRAJ}",
)


def _build_graph(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cutoff: float = 5.0,
) -> AtomicGraph:
    """Build an ``AtomicGraph`` with a brute-force radius cutoff."""
    n: int = positions.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vecs: list[torch.Tensor] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r: torch.Tensor = positions[j] - positions[i]
            d: torch.Tensor = r.norm()
            if d.item() < cutoff:
                rows.append(i)
                cols.append(j)
                vecs.append(r)
    edge_index: torch.Tensor = torch.tensor([rows, cols], dtype=torch.long)
    edge_vectors: torch.Tensor = (
        torch.stack(vecs) if vecs else torch.zeros(0, 3, dtype=positions.dtype)
    )
    edge_lengths: torch.Tensor = (
        edge_vectors.norm(dim=-1) if vecs else torch.zeros(0, dtype=positions.dtype)
    )
    cell: torch.Tensor = torch.zeros(3, 3, dtype=positions.dtype)
    pbc: torch.Tensor = torch.zeros(3, dtype=torch.bool)
    return AtomicGraph(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
    )


@pytest.fixture(scope="function")
def methane_batch() -> Batch:
    """PyG batch holding a single methane molecule (1 C, 4 H)."""
    a: float = 1.09 / math.sqrt(3.0)
    positions: torch.Tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a],
        ],
        dtype=torch.float64,
    )
    atomic_numbers: torch.Tensor = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    graph: AtomicGraph = _build_graph(positions, atomic_numbers, cutoff=5.0)
    return Batch.from_data_list([graph])


@pytest.fixture(scope="function")
def water_batch() -> Batch:
    """PyG batch holding a single water molecule (1 O, 2 H)."""
    positions: torch.Tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.240, 0.927, 0.0],
        ],
        dtype=torch.float64,
    )
    atomic_numbers: torch.Tensor = torch.tensor([8, 1, 1], dtype=torch.long)
    graph: AtomicGraph = _build_graph(positions, atomic_numbers, cutoff=5.0)
    return Batch.from_data_list([graph])


@pytest.fixture(scope="function")
def methylamine_batch() -> Batch:
    """PyG batch holding methylamine (1 C, 1 N, 5 H)."""
    positions: torch.Tensor = torch.tensor(
        [
            [0.000, 0.000, 0.000],  # C
            [1.470, 0.000, 0.000],  # N
            [-0.500, 1.000, 0.000],  # H on C
            [-0.500, -0.500, 0.870],  # H on C
            [-0.500, -0.500, -0.870],  # H on C
            [1.900, 0.700, 0.500],  # H on N
            [1.900, -0.700, 0.500],  # H on N
        ],
        dtype=torch.float64,
    )
    atomic_numbers: torch.Tensor = torch.tensor([6, 7, 1, 1, 1, 1, 1], dtype=torch.long)
    graph: AtomicGraph = _build_graph(positions, atomic_numbers, cutoff=5.0)
    return Batch.from_data_list([graph])


@pytest.fixture(scope="function")
def carbon_dimer_batch() -> Batch:
    """PyG batch holding a single C-C dimer (only one pair type present).

    Used by ``test_experts.py`` to verify that — with SIMURGH configured
    for H/C/N/O — running on this batch contributes exactly ``0.0`` from
    the 9 non-CC experts, even though they all execute every step.
    """
    positions: torch.Tensor = torch.tensor(
        [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]],
        dtype=torch.float64,
    )
    atomic_numbers: torch.Tensor = torch.tensor([6, 6], dtype=torch.long)
    graph: AtomicGraph = _build_graph(positions, atomic_numbers, cutoff=5.0)
    return Batch.from_data_list([graph])


def random_so3() -> torch.Tensor:
    """Random rotation matrix from a uniform distribution on SO(3)."""
    from scipy.spatial.transform import Rotation as R

    rot: torch.Tensor = torch.tensor(R.random().as_matrix(), dtype=torch.float64)
    return rot


@pytest.fixture(scope="module")
def gmd_batch() -> Batch:
    """PyG Batch built from the first 4 frames of the GMD O2C4H8 trajectory.

    Skipped automatically when the trajectory file is not present on disk.
    All graphs are built with explicit ``edge_vectors`` / ``edge_lengths``
    (the same convention used by the hand-crafted fixtures above) so they
    can be passed directly to ``EnvironmentDressing`` and the interaction
    block without recomputation.
    """
    if not _GMD_AVAILABLE:
        pytest.skip(f"GMD trajectory not found at {_GMD_TRAJ}")

    try:
        from ase.io import read as ase_read
    except ImportError:
        pytest.skip("ase not installed")

    frames = ase_read(str(_GMD_TRAJ), index=slice(0, 4))
    graphs: list[AtomicGraph] = []
    for atoms in frames:
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        graphs.append(_build_graph(pos, z, cutoff=5.0))
    return Batch.from_data_list(graphs)
