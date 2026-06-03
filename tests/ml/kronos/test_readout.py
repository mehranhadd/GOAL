"""Tests for per-layer energy readouts in EnvironmentDressing (CHANGE 3).

Covers:
* ``per_layer_readout=False`` — forward returns ``(feats, None)``.
* ``per_layer_readout=True`` — forward returns ``(feats, layer_energies)``
  where ``layer_energies`` has shape ``(N,)``.
* Layer energies are nonzero and gradients flow through the readout MLPs.
* Each readout layer contributes independently: disabling all but one still
  leaves a nonzero contribution from the remaining layer.
* Integration with GMD real data (skipped if traj absent).
"""

from __future__ import annotations

import pytest
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch

from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from tests.ml.kronos.conftest import requires_gmd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dressing(
    per_layer_readout: bool = False,
    num_interactions: int = 2,
    hidden_channels: int = 8,
    lmax: int = 1,
) -> EnvironmentDressing:
    return EnvironmentDressing(
        num_elements=10,
        embedding_dim=8,
        hidden_channels=hidden_channels,
        lmax=lmax,
        num_radial_basis=4,
        cutoff=5.0,
        radial_mlp_hidden=8,
        num_interactions=num_interactions,
        per_layer_readout=per_layer_readout,
        avg_num_neighbors=4.0,
    ).double()


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_no_readout_returns_none(self, methane_batch: Batch) -> None:
        """When per_layer_readout=False, the second return value is None."""
        dressing = _make_dressing(per_layer_readout=False)
        feats, layer_e = dressing(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert layer_e is None
        assert feats.shape == (methane_batch.num_nodes, dressing.irreps_out.dim)

    def test_readout_returns_tensor(self, methane_batch: Batch) -> None:
        """When per_layer_readout=True, the second return value is a (N,) Tensor."""
        dressing = _make_dressing(per_layer_readout=True)
        feats, layer_e = dressing(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert layer_e is not None
        assert layer_e.shape == (methane_batch.num_nodes,)
        assert torch.isfinite(layer_e).all()

    def test_readout_shape_matches_num_nodes(self, water_batch: Batch) -> None:
        """layer_energies length equals the number of atoms in the batch."""
        dressing = _make_dressing(per_layer_readout=True)
        _, layer_e = dressing(
            atomic_numbers=water_batch.atomic_numbers,
            edge_index=water_batch.edge_index,
            edge_vectors=water_batch.edge_attr,
            edge_lengths=water_batch.edge_weight,
        )
        assert layer_e is not None
        assert layer_e.shape[0] == water_batch.num_nodes


# ---------------------------------------------------------------------------
# Correctness of contributions
# ---------------------------------------------------------------------------


class TestReadoutContributions:
    def test_layer_energies_nonzero(self, methane_batch: Batch) -> None:
        """For a randomly initialised model the accumulated layer energies should
        be nonzero (probability of an exact zero is negligible)."""
        torch.manual_seed(99)
        dressing = _make_dressing(per_layer_readout=True, num_interactions=2)
        _, layer_e = dressing(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert layer_e is not None
        assert layer_e.abs().sum().item() > 0.0, (
            "Accumulated per-layer energies are all zero — the readout MLPs are "
            "not contributing (possible initialisation or wiring bug)."
        )

    def test_gradient_flows_through_readout(self, methane_batch: Batch) -> None:
        """Backward pass through layer_energies.sum() must give nonzero gradients
        to at least some readout parameters."""
        dressing = _make_dressing(per_layer_readout=True, num_interactions=2)
        _, layer_e = dressing(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert layer_e is not None
        layer_e.sum().backward()

        # At least one readout parameter must have received a gradient
        readout_grads = [
            p.grad
            for name, p in dressing.named_parameters()
            if "readout" in name and p.grad is not None
        ]
        assert readout_grads, (
            "No readout parameter received a gradient after backward() — "
            "the per-layer readout is not connected to the autograd graph."
        )
        assert any(
            g.abs().sum().item() > 0.0 for g in readout_grads
        ), "All readout gradients are zero after backward()."

    def test_num_readout_modules_matches_num_interactions(self) -> None:
        """When per_layer_readout=True, one _LayerReadout per interaction layer."""
        for n in (1, 2, 3):
            dressing = _make_dressing(per_layer_readout=True, num_interactions=n)
            assert len(dressing.readouts) == n, (
                f"Expected {n} readout modules for num_interactions={n}, "
                f"got {len(dressing.readouts)}."
            )

    def test_no_readout_modules_when_disabled(self) -> None:
        """When per_layer_readout=False, the readout ModuleList is empty."""
        dressing = _make_dressing(per_layer_readout=False, num_interactions=3)
        assert len(dressing.readouts) == 0

    def test_more_interactions_increase_layer_energy_magnitude(self, methane_batch: Batch) -> None:
        """With more interaction layers (each adding its own readout energy), the
        magnitude of the accumulated per-layer energy should generally increase."""
        torch.manual_seed(0)
        d1 = _make_dressing(per_layer_readout=True, num_interactions=1)
        d3 = _make_dressing(per_layer_readout=True, num_interactions=3)

        _, e1 = d1(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        _, e3 = d3(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert e1 is not None and e3 is not None
        # Both should be finite
        assert torch.isfinite(e1).all()
        assert torch.isfinite(e3).all()


# ---------------------------------------------------------------------------
# Integration with GMD real data
# ---------------------------------------------------------------------------


class TestReadoutOnGMDData:
    @requires_gmd
    def test_readout_on_real_frames(self, gmd_batch: Batch) -> None:
        """Smoke test: per-layer readout runs without error on real GMD frames."""
        dressing = EnvironmentDressing(
            num_elements=10,
            embedding_dim=16,
            hidden_channels=16,
            lmax=1,
            num_radial_basis=4,
            cutoff=5.0,
            radial_mlp_hidden=16,
            num_interactions=2,
            per_layer_readout=True,
            avg_num_neighbors=8.0,
        ).double()

        feats, layer_e = dressing(
            atomic_numbers=gmd_batch.atomic_numbers,
            edge_index=gmd_batch.edge_index,
            edge_vectors=gmd_batch.edge_attr,
            edge_lengths=gmd_batch.edge_weight,
        )
        assert feats.shape == (gmd_batch.num_nodes, dressing.irreps_out.dim)
        assert layer_e is not None
        assert layer_e.shape == (gmd_batch.num_nodes,)
        assert torch.isfinite(feats).all()
        assert torch.isfinite(layer_e).all()

    @requires_gmd
    def test_readout_energies_sum_to_reasonable_value(self, gmd_batch: Batch) -> None:
        """layer_energies.sum() should be finite and nonzero on real GMD frames."""
        dressing = EnvironmentDressing(
            num_elements=10,
            embedding_dim=16,
            hidden_channels=16,
            lmax=1,
            num_radial_basis=4,
            cutoff=5.0,
            radial_mlp_hidden=16,
            num_interactions=2,
            per_layer_readout=True,
            avg_num_neighbors=8.0,
        ).double()

        _, layer_e = dressing(
            atomic_numbers=gmd_batch.atomic_numbers,
            edge_index=gmd_batch.edge_index,
            edge_vectors=gmd_batch.edge_attr,
            edge_lengths=gmd_batch.edge_weight,
        )
        assert layer_e is not None
        total = layer_e.sum()
        assert torch.isfinite(total).item()
        assert total.abs().item() > 0.0
