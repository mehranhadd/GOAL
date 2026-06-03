"""ACE body-order expansion tests for KRONOS.

Validates the two acceptance criteria from CHANGE 4:

1. ACE¹ and ACE² produce **different** dressed features on the same
   molecule (the body-order expansion is doing something non-trivial).
2. ACE² and ACE³ dressed features remain **E(3)-equivariant** under
   an SO(3) rotation of the input positions.

Equivariance is checked using e3nn's ``Irreps.D_from_matrix`` to build
the representation matrix that should act on the irrep features after
a Cartesian rotation of the atoms.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from tests.ml.kronos.conftest import random_so3


def _make_dressing(body_order: int, seed: int = 0) -> EnvironmentDressing:
    """Build a small but non-trivial dressing block at a fixed seed."""
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
    ).double()


def _rotate_batch(batch: Batch, R: torch.Tensor) -> Batch:
    """Return a fresh ``Batch`` with positions rotated by ``R``."""
    rotated_positions = batch.pos.detach() @ R.T
    graph = AtomicGraph(
        positions=rotated_positions,
        atomic_numbers=batch.atomic_numbers,
        cell=torch.zeros(3, 3, dtype=batch.pos.dtype),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=batch.edge_index,
        edge_vectors=batch.edge_attr.detach() @ R.T,
        edge_lengths=batch.edge_weight.detach(),
    )
    return Batch.from_data_list([graph])


class TestBodyOrderInstantiation:
    """The constructor must accept body_order in {1, 2, 3} and reject
    anything else; introspection must report the right value."""

    @pytest.mark.parametrize("body_order", [1, 2, 3])
    def test_supported_orders_construct(self, body_order: int) -> None:
        dressing = _make_dressing(body_order=body_order)
        assert dressing.body_order == body_order

    @pytest.mark.parametrize("body_order", [0, 4, -1])
    def test_invalid_orders_raise(self, body_order: int) -> None:
        with pytest.raises(ValueError, match="body_order"):
            _make_dressing(body_order=body_order)

    def test_body_order_does_not_change_irreps_out(self) -> None:
        """The output irreps shape stays the same so the expert
        interface is unaffected by body order."""
        out1 = _make_dressing(body_order=1).irreps_out
        out2 = _make_dressing(body_order=2).irreps_out
        out3 = _make_dressing(body_order=3).irreps_out
        assert out1 == out2 == out3

    def test_b2_irreps_are_derived_programmatically(self) -> None:
        """``irreps_b2_full`` (the pre-compression CG product irreps)
        must respect ``l ≤ lmax`` and exist only when ``body_order ≥ 2``."""
        assert _make_dressing(body_order=1).irreps_b2_full is None

        d2 = _make_dressing(body_order=2)
        assert d2.irreps_b2_full is not None
        assert all(ir.l <= d2.lmax for _, ir in d2.irreps_b2_full)
        # The CG product of ``16x0e + 16x1o + 16x2e`` with itself must
        # contain at least one block of every l in 0..lmax.
        ls = {ir.l for _, ir in d2.irreps_b2_full}
        assert ls == set(range(d2.lmax + 1))

    def test_b3_only_when_body_order_three(self) -> None:
        assert _make_dressing(body_order=1).irreps_b3_full is None
        assert _make_dressing(body_order=2).irreps_b3_full is None
        d3 = _make_dressing(body_order=3)
        assert d3.irreps_b3_full is not None


class TestAceOrdersDiffer:
    """Acceptance criterion 1: ACE¹ and ACE² produce different dressed
    features on the same molecule."""

    def test_ace1_ace2_differ_on_methane(self, methane_batch) -> None:  # noqa: ANN001
        d1 = _make_dressing(body_order=1, seed=42)
        d2 = _make_dressing(body_order=2, seed=42)

        kwargs = dict(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        f1, _ = d1(**kwargs)
        f2, _ = d2(**kwargs)
        assert f1.shape == f2.shape
        assert not torch.allclose(f1, f2), (
            "ACE¹ and ACE² produced identical dressed features — "
            "the body-order expansion is a no-op"
        )

    def test_ace2_ace3_differ_on_methane(self, methane_batch) -> None:  # noqa: ANN001
        d2 = _make_dressing(body_order=2, seed=42)
        d3 = _make_dressing(body_order=3, seed=42)

        kwargs = dict(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        f2, _ = d2(**kwargs)
        f3, _ = d3(**kwargs)
        assert not torch.allclose(f2, f3), (
            "ACE² and ACE³ produced identical dressed features — "
            "the B³ tensor product is a no-op"
        )


class TestAceOrdersEquivariant:
    """Acceptance criterion 2: ACE² and ACE³ dressed features are
    E(3)-equivariant under SO(3) rotation."""

    def _check_equivariant(
        self,
        body_order: int,
        batch: Batch,
        atol: float = 1e-5,
    ) -> None:
        dressing = _make_dressing(body_order=body_order, seed=7)
        dressing.eval()

        # Run on the original molecule
        f_orig: torch.Tensor
        f_orig, _ = dressing(
            atomic_numbers=batch.atomic_numbers,
            edge_index=batch.edge_index,
            edge_vectors=batch.edge_attr,
            edge_lengths=batch.edge_weight,
        )

        # Run on a rotated copy
        R: torch.Tensor = random_so3()
        rotated: Batch = _rotate_batch(batch, R)
        f_rot: torch.Tensor
        f_rot, _ = dressing(
            atomic_numbers=rotated.atomic_numbers,
            edge_index=rotated.edge_index,
            edge_vectors=rotated.edge_attr,
            edge_lengths=rotated.edge_weight,
        )

        # Build the irrep representation matrix D(R) and apply it to the
        # original features.  We expect ``f_rot == f_orig @ D.T``.
        irreps = dressing.irreps_out
        D: torch.Tensor = irreps.D_from_matrix(R).to(f_orig.dtype)
        expected: torch.Tensor = f_orig @ D.T

        diff: float = (f_rot - expected).abs().max().item()
        assert diff < atol, f"body_order={body_order}: equivariance broken, max diff {diff:.3e}"

    def test_ace1_equivariant(self, methane_batch) -> None:  # noqa: ANN001
        # Sanity check that the baseline (body_order=1) is also equivariant.
        self._check_equivariant(body_order=1, batch=methane_batch)

    def test_ace2_equivariant(self, methane_batch) -> None:  # noqa: ANN001
        self._check_equivariant(body_order=2, batch=methane_batch)

    def test_ace3_equivariant(self, methane_batch) -> None:  # noqa: ANN001
        self._check_equivariant(body_order=3, batch=methane_batch)

    def test_ace2_equivariant_methylamine(self, methylamine_batch) -> None:  # noqa: ANN001
        # Larger molecule with all H/C/N/O pair types
        self._check_equivariant(body_order=2, batch=methylamine_batch)
