"""Tests for the ``EnvironmentDressing`` block."""

from __future__ import annotations

import torch
from e3nn.o3 import Irreps

from goal.ml.nn.blocks.env_dressing import (
    EnvironmentDressing,
    build_edge_irreps,
    build_hidden_irreps,
    cg_product_irreps,
)


class TestIrrepHelpers:
    def test_hidden_irreps_parity_pattern(self) -> None:
        irreps = build_hidden_irreps(hidden_channels=32, lmax=2)
        assert str(irreps) == "32x0e+32x1o+32x2e"

    def test_edge_irreps_single_channel(self) -> None:
        irreps = build_edge_irreps(lmax=2)
        assert str(irreps) == "1x0e+1x1o+1x2e"

    def test_cg_product_irreps_truncates_to_lmax(self) -> None:
        """CG product of ``32x0e + 32x1o + 32x2e`` with itself, truncated
        to ``l ≤ 2``, must contain no irreps of higher angular momentum."""
        a = build_hidden_irreps(hidden_channels=32, lmax=2)
        out = cg_product_irreps(a, a, lmax=2)
        assert all(ir.l <= 2 for _, ir in out)

    def test_cg_product_irreps_contains_each_l(self) -> None:
        """For ``A = 32x0e + 32x1o + 32x2e``, ``A ⊗ A`` truncated to lmax=2
        must contain at least one block of every ``l`` in 0..2."""
        a = build_hidden_irreps(hidden_channels=32, lmax=2)
        out = cg_product_irreps(a, a, lmax=2)
        ls = {ir.l for _, ir in out}
        assert ls == {0, 1, 2}


class TestEnvironmentDressing:
    def _make_dressing(self) -> EnvironmentDressing:
        return EnvironmentDressing(
            num_elements=9,
            embedding_dim=16,
            hidden_channels=16,
            lmax=2,
            num_radial_basis=4,
            cutoff=5.0,
            radial_mlp_hidden=16,
        ).double()

    def test_irreps_out(self) -> None:
        dressing = self._make_dressing()
        assert dressing.irreps_out == Irreps("16x0e+16x1o+16x2e")
        assert dressing.cutoff == 5.0
        assert dressing.lmax == 2
        assert dressing.hidden_channels == 16

    def test_forward_methane(self, methane_batch) -> None:  # noqa: ANN001
        dressing = self._make_dressing()
        feats = dressing(
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_vectors=methane_batch.edge_attr,
            edge_lengths=methane_batch.edge_weight,
        )
        assert feats.shape == (methane_batch.num_atoms, dressing.irreps_out.dim)
        assert torch.isfinite(feats).all()
