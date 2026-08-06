"""Fixtures for foundation-model fine-tuning tests.

Everything is mocked — no mace-torch / fairchem / peft required.  A tiny
``FakeMACE`` nn.Module stands in for a real MACE model: it exposes the same
surface the wrapper touches (``atomic_numbers`` z-table, ``r_max``,
``interactions``, ``readouts``, ``atomic_energies_fn.atomic_energies``) and a
``forward(data, training, compute_force)`` that returns
``{"energy", "forces"}`` with forces derived from positions so autograd works.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from goal.ml.data.graph import AtomicGraph


class _AtomicEnergies(nn.Module):
    def __init__(self, n_elements: int) -> None:
        super().__init__()
        self.register_buffer("atomic_energies", torch.zeros(1, n_elements, dtype=torch.float64))


class FakeMACE(nn.Module):
    """Minimal stand-in for a pre-trained MACE model."""

    def __init__(self, z_list: tuple[int, ...] = (1, 6, 7, 8), r_max: float = 5.0) -> None:
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(list(z_list), dtype=torch.long))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        n = len(z_list)
        self.interactions = nn.ModuleList([nn.Linear(n, n).double()])
        self.readouts = nn.ModuleList([nn.Linear(n, 1).double()])
        self.atomic_energies_fn = _AtomicEnergies(n)

    def forward(self, data, training: bool = False, compute_force: bool = True):
        pos = data["positions"]
        pos.requires_grad_(True)
        h = self.interactions[0](data["node_attrs"])
        node_e = self.readouts[0](h).squeeze(-1) + 0.01 * (pos**2).sum(-1)
        batch = data["batch"]
        n_graphs = int(batch.max().item()) + 1
        energy = torch.zeros(n_graphs, dtype=node_e.dtype).index_add(0, batch, node_e)
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=training)[0]
        return {"energy": energy, "forces": forces, "node_energy": node_e, "node_feats": h}


@pytest.fixture()
def fake_mace() -> FakeMACE:
    return FakeMACE()


@pytest.fixture()
def methane_graph() -> AtomicGraph:
    """Methane with energy + force labels (so the loss has targets)."""
    from ase.build import molecule

    atoms = molecule("CH4")
    n = len(atoms)
    return AtomicGraph.from_ase(
        atoms,
        cutoff=5.0,
        energy=-42.0,
        forces=np.zeros((n, 3)),
    )
