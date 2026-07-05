"""Tests for the MACE-style per-element body-order contraction.

Covers the two contracts laid out in
``goal.ml.nn.blocks.symmetric_contraction``:

1. **Per-element learnable couplings** — different atoms (different Z)
   receive different weights, so the same equivariant input ``A`` yields
   different ``B`` when its element identity is changed.

2. **SO(3) equivariance** — the contraction commutes with rotations of
   the input features.  Identical contract as the existing body-order
   tests in ``test_body_order.py`` but now exercised through the new
   :class:`SymmetricContraction` path.
"""

from __future__ import annotations

import pytest
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from goal.ml.nn.blocks.symmetric_contraction import (
    PerElementLinear,
    PerElementWeightedTensorProduct,
    SymmetricContraction,
)
from tests.ml.simurgh.conftest import random_so3

# ---------------------------------------------------------------------------
# Unit tests for the per-element primitives
# ---------------------------------------------------------------------------


class TestPerElementLinear:
    """Per-element gather should make outputs depend on Z."""

    def test_output_shape_matches_irreps_out(self) -> None:
        torch.manual_seed(0)
        irreps_in: Irreps = Irreps("4x0e + 4x1o")
        irreps_out: Irreps = Irreps("2x0e + 2x1o")
        lin: PerElementLinear = PerElementLinear(irreps_in, irreps_out, num_elements=10).double()
        x: torch.Tensor = torch.randn(7, irreps_in.dim, dtype=torch.float64)
        z: torch.Tensor = torch.randint(0, 10, (7,))
        y: torch.Tensor = lin(x, z)
        assert y.shape == (7, irreps_out.dim)

    def test_different_z_yields_different_output(self) -> None:
        """Same x, different Z → different y (provided weights initialised non-zero)."""
        torch.manual_seed(0)
        irreps: Irreps = Irreps("4x0e + 4x1o")
        lin: PerElementLinear = PerElementLinear(irreps, irreps, num_elements=5).double()
        # Reuse identical x for two different element indices
        x: torch.Tensor = torch.randn(1, irreps.dim, dtype=torch.float64)
        y0: torch.Tensor = lin(x, torch.tensor([0]))
        y1: torch.Tensor = lin(x, torch.tensor([1]))
        assert not torch.allclose(y0, y1), (
            "Per-element linear gave identical outputs for distinct Z — "
            "weight gather is not element-specific"
        )


class TestPerElementWeightedTensorProduct:
    """Per-element TP must depend on Z and respect the FCTP interface."""

    def test_output_shape_matches_irreps_out(self) -> None:
        torch.manual_seed(0)
        irreps_a: Irreps = Irreps("4x0e + 4x1o")
        irreps_out: Irreps = Irreps("2x0e + 2x1o + 1x2e")
        tp: PerElementWeightedTensorProduct = PerElementWeightedTensorProduct(
            irreps_a, irreps_a, irreps_out, num_elements=10
        ).double()
        x1: torch.Tensor = torch.randn(7, irreps_a.dim, dtype=torch.float64)
        x2: torch.Tensor = torch.randn(7, irreps_a.dim, dtype=torch.float64)
        z: torch.Tensor = torch.randint(0, 10, (7,))
        y: torch.Tensor = tp(x1, x2, z)
        assert y.shape == (7, irreps_out.dim)

    def test_different_z_yields_different_output(self) -> None:
        torch.manual_seed(0)
        irreps: Irreps = Irreps("4x0e + 4x1o")
        tp: PerElementWeightedTensorProduct = PerElementWeightedTensorProduct(
            irreps, irreps, irreps, num_elements=5
        ).double()
        x1: torch.Tensor = torch.randn(1, irreps.dim, dtype=torch.float64)
        x2: torch.Tensor = torch.randn(1, irreps.dim, dtype=torch.float64)
        y0: torch.Tensor = tp(x1, x2, torch.tensor([0]))
        y1: torch.Tensor = tp(x1, x2, torch.tensor([1]))
        assert not torch.allclose(y0, y1)


# ---------------------------------------------------------------------------
# Tests for SymmetricContraction itself
# ---------------------------------------------------------------------------


class TestSymmetricContractionConstruction:
    def test_invalid_correlation_raises(self) -> None:
        with pytest.raises(ValueError, match="correlation"):
            SymmetricContraction(
                irreps_in="4x0e",
                irreps_out="4x0e",
                correlation=4,  # > 3 not supported
                num_elements=10,
            )

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_supported_correlations_construct(self, correlation: int) -> None:
        sc: SymmetricContraction = SymmetricContraction(
            irreps_in="4x0e + 4x1o",
            irreps_out="4x0e + 4x1o",
            correlation=correlation,
            num_elements=10,
        )
        assert sc.correlation == correlation
        # Body-1 always present.  Body-2 / Body-3 present iff correlation hits them.
        assert sc.body_1 is not None
        assert (sc.body_2 is not None) == (correlation >= 2)
        assert (sc.body_3 is not None) == (correlation >= 3)


class TestSymmetricContractionEquivariance:
    """The whole point of SymmetricContraction: rotation commutes with it."""

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_so3_equivariance(self, correlation: int) -> None:
        torch.manual_seed(7)
        irreps_in: Irreps = Irreps("4x0e + 4x1o + 4x2e")
        irreps_out: Irreps = Irreps("4x0e + 4x1o + 4x2e")
        sc: SymmetricContraction = SymmetricContraction(
            irreps_in, irreps_out, correlation=correlation, num_elements=10
        ).double()
        sc.eval()

        n: int = 5
        x: torch.Tensor = torch.randn(n, irreps_in.dim, dtype=torch.float64)
        z: torch.Tensor = torch.randint(0, 10, (n,))

        # Rotate the input features by D(R).
        R: torch.Tensor = random_so3()
        D_in: torch.Tensor = irreps_in.D_from_matrix(R).to(torch.float64)
        D_out: torch.Tensor = irreps_out.D_from_matrix(R).to(torch.float64)
        x_rot: torch.Tensor = x @ D_in.T

        y_orig: torch.Tensor = sc(x, z)
        y_rot: torch.Tensor = sc(x_rot, z)

        # Equivariance: y_rot == y_orig @ D_out.T
        expected: torch.Tensor = y_orig @ D_out.T
        diff: float = (y_rot - expected).abs().max().item()
        assert diff < 1e-7, f"correlation={correlation}: equivariance broken, max diff {diff:.3e}"


# ---------------------------------------------------------------------------
# Wiring through EnvironmentDressing
# ---------------------------------------------------------------------------


def _make_dressing(
    body_order: int,
    symmetric_contraction: bool,
    seed: int = 0,
) -> EnvironmentDressing:
    """Build a small dressing block on the symmetric-contraction path."""
    torch.manual_seed(seed)
    return EnvironmentDressing(
        num_elements=9,
        embedding_dim=16,
        hidden_channels=16,
        lmax=2,
        num_radial_basis=4,
        cutoff=5.0,
        radial_mlp_hidden=16,
        num_message_passing=1,
        body_order=body_order,
        symmetric_contraction=symmetric_contraction,
    ).double()


class TestEnvironmentDressingSymmetricPath:
    """End-to-end: the dressing block must stay equivariant on the new path."""

    @pytest.mark.parametrize("body_order", [2, 3])
    def test_default_path_unchanged_when_flag_off(self, body_order: int) -> None:
        """``symmetric_contraction=False`` must leave the FCTP modules in place."""
        d: EnvironmentDressing = _make_dressing(body_order=body_order, symmetric_contraction=False)
        assert d.symmetric_contraction is False
        assert d.sym_contraction is None
        assert d.tp_b2 is not None
        if body_order >= 3:
            assert d.tp_b3 is not None

    @pytest.mark.parametrize("body_order", [2, 3])
    def test_symmetric_path_replaces_fctp(self, body_order: int) -> None:
        d: EnvironmentDressing = _make_dressing(body_order=body_order, symmetric_contraction=True)
        assert d.symmetric_contraction is True
        assert d.sym_contraction is not None
        assert d.tp_b2 is None
        assert d.tp_b3 is None

    @pytest.mark.parametrize("body_order", [2, 3])
    def test_irreps_out_unchanged_by_symmetric_flag(self, body_order: int) -> None:
        d_off: EnvironmentDressing = _make_dressing(
            body_order=body_order, symmetric_contraction=False
        )
        d_on: EnvironmentDressing = _make_dressing(
            body_order=body_order, symmetric_contraction=True
        )
        assert d_off.irreps_out == d_on.irreps_out

    @pytest.mark.parametrize("body_order", [2, 3])
    def test_symmetric_path_equivariant_on_methane(
        self, body_order: int, methane_batch: Batch
    ) -> None:
        """Acceptance criterion: SO(3) equivariance with the new path active."""
        d: EnvironmentDressing = _make_dressing(
            body_order=body_order, symmetric_contraction=True, seed=42
        )
        d.eval()

        kwargs = dict(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        f_orig: torch.Tensor
        f_orig, _ = d(**kwargs)

        R: torch.Tensor = random_so3()
        rotated_positions: torch.Tensor = methane_batch.pos.detach() @ R.T
        rotated_graph: AtomicGraph = AtomicGraph(
            positions=rotated_positions,
            atomic_numbers=methane_batch.atomic_numbers,
            cell=torch.zeros(3, 3, dtype=methane_batch.pos.dtype),
            pbc=torch.zeros(3, dtype=torch.bool),
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr.detach() @ R.T,
            edge_lengths=methane_batch.edge_weight.detach(),
        )
        rotated_batch: Batch = Batch.from_data_list([rotated_graph])
        f_rot: torch.Tensor
        f_rot, _ = d(
            atomic_numbers=rotated_batch.atomic_numbers,
            edge_index=rotated_batch.edge_index,
            edge_vectors=rotated_batch.edge_attr,
            edge_lengths=rotated_batch.edge_weight,
        )

        D: torch.Tensor = d.irreps_out.D_from_matrix(R).to(f_orig.dtype)
        expected: torch.Tensor = f_orig @ D.T
        diff: float = (f_rot - expected).abs().max().item()
        assert diff < 1e-6, (
            f"body_order={body_order} symmetric_contraction=True: "
            f"equivariance broken, max diff {diff:.3e}"
        )

    @pytest.mark.parametrize("body_order", [2, 3])
    def test_symmetric_path_yields_different_features_than_fctp(
        self, body_order: int, methane_batch: Batch
    ) -> None:
        """Sanity: the new path is doing something different from the old one."""
        d_off: EnvironmentDressing = _make_dressing(
            body_order=body_order, symmetric_contraction=False, seed=42
        )
        d_on: EnvironmentDressing = _make_dressing(
            body_order=body_order, symmetric_contraction=True, seed=42
        )
        kwargs = dict(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        f_off: torch.Tensor
        f_off, _ = d_off(**kwargs)
        f_on: torch.Tensor
        f_on, _ = d_on(**kwargs)
        assert f_off.shape == f_on.shape
        assert not torch.allclose(f_off, f_on), (
            "Symmetric-contraction path produced identical features to the "
            "shared-weight FCTP path — the new path is a no-op"
        )
