"""Per-block tests for KRONOS experts.

Validates:

* ``enumerate_element_pairs`` enumerates unordered pairs in a
  deterministic, sorted order.
* ``cosine_cutoff`` is exactly zero at and beyond the cutoff.
* ``PairwiseExpert`` produces a single scalar per pair for both the
  Linear and Transformer backbones.
* ``KronosMoE`` runs every expert exactly once per forward pass, even
  when some pair types are absent (zero-mask rule).
* Per-pair gates are registered learnable parameters named
  ``experts.z{a}_z{b}.gate``.
* **Static zero-masking schedule**: on a batch that only contains one
  pair type (C-C), every one of the nine absent experts still executes
  and contributes exactly ``0.0`` to the total energy.
"""

from __future__ import annotations

import typing

import pytest
import torch
from e3nn.o3 import Irreps

from goal.ml.nn.blocks.experts import (
    ExpertConfig,
    KronosMoE,
    PairwiseExpert,
    cosine_cutoff,
    enumerate_element_pairs,
    pair_label,
)


class TestEnumerateElementPairs:
    def test_hcno_yields_ten_pairs(self) -> None:
        pairs: list[tuple[int, int]] = enumerate_element_pairs([1, 6, 7, 8])
        assert len(pairs) == 10
        assert pairs == [
            (1, 1),
            (1, 6),
            (1, 7),
            (1, 8),
            (6, 6),
            (6, 7),
            (6, 8),
            (7, 7),
            (7, 8),
            (8, 8),
        ]

    def test_unsorted_input_is_sorted(self) -> None:
        pairs = enumerate_element_pairs([8, 1, 6, 7])
        assert pairs == enumerate_element_pairs([1, 6, 7, 8])

    def test_duplicates_collapsed(self) -> None:
        # KronosMoE collapses duplicates; helper itself does not, but check
        # K * (K + 1) / 2 formula
        pairs = enumerate_element_pairs([1, 6])
        assert len(pairs) == 3
        assert pairs == [(1, 1), (1, 6), (6, 6)]

    def test_pair_label(self) -> None:
        assert pair_label(1, 6) == "H-C"
        assert pair_label(6, 1) == "H-C"  # order-insensitive
        assert pair_label(8, 8) == "O-O"


class TestCosineCutoff:
    def test_zero_at_and_beyond_cutoff(self) -> None:
        d = torch.tensor([0.0, 2.5, 4.9999, 5.0, 5.01, 100.0], dtype=torch.float64)
        out = cosine_cutoff(d, cutoff=5.0)
        # At cutoff, the cosine envelope is 0.5 * (cos(pi) + 1) = 0 anyway,
        # but the mask should still kill anything >= cutoff.
        assert out[3].item() == 0.0
        assert out[4].item() == 0.0
        assert out[5].item() == 0.0
        # And strictly positive interior values:
        assert out[0].item() == pytest.approx(1.0, rel=1e-12)
        assert 0.0 < out[1].item() < 1.0

    def test_monotone_decreasing_inside_cutoff(self) -> None:
        d = torch.linspace(0.0, 4.99, 50, dtype=torch.float64)
        out = cosine_cutoff(d, cutoff=5.0)
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-12).all()


class TestPairwiseExpert:
    @pytest.mark.parametrize("expert_type", ["linear", "transformer"])
    def test_outputs_scalar_per_pair(self, expert_type: str) -> None:
        irreps_in = Irreps("8x0e+8x1o+8x2e")
        cfg = ExpertConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type=expert_type,
            transformer_heads=2,
            transformer_layers=1,
        )
        expert = PairwiseExpert(irreps_in, cfg).double()
        # Random scalars for 5 pairs
        scalars_a = torch.randn(5, 8, dtype=torch.float64)
        scalars_b = torch.randn(5, 8, dtype=torch.float64)
        dist = torch.linspace(0.5, 4.5, 5, dtype=torch.float64)
        out = expert(scalars_a, scalars_b, dist)
        assert out.shape == (5,)

    def test_gate_is_a_parameter(self) -> None:
        irreps_in = Irreps("4x0e")
        cfg = ExpertConfig(scalar_channels=4, hidden_dims=(8,), expert_type="linear")
        expert = PairwiseExpert(irreps_in, cfg)
        assert isinstance(expert.gate, torch.nn.Parameter)
        assert expert.gate.requires_grad is True
        # Initial value should be 1.0
        assert expert.gate.item() == pytest.approx(1.0, abs=1e-12)


class TestKronosMoE:
    def _make_moe(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        irreps_in: str = "16x0e+16x1o+16x2e",
        cutoff: float = 5.0,
    ) -> KronosMoE:
        cfg = ExpertConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type="linear",
        )
        return KronosMoE(
            elements=elements,
            irreps_in=irreps_in,
            expert_config=cfg,
            cutoff=cutoff,
        ).double()

    def test_num_experts_matches_pair_count(self) -> None:
        moe = self._make_moe(elements=(1, 6, 7, 8))
        # K = 4 → 4 * 5 / 2 = 10
        assert moe.num_experts == 10
        # Three-element set
        moe3 = self._make_moe(elements=(1, 6, 8))
        assert moe3.num_experts == 6

    def test_every_pair_has_a_gate(self) -> None:
        moe = self._make_moe()
        gate_names: list[str] = [n for n, _ in moe.named_parameters() if n.endswith(".gate")]
        # 10 unordered pairs in HCNO
        assert len(gate_names) == 10
        # Pair keys use ``z{lo}_z{hi}``
        assert "experts.z1_z6.gate" in gate_names
        assert "experts.z8_z8.gate" in gate_names

    def test_forward_runs_with_all_pair_types(self, methylamine_batch) -> None:  # noqa: ANN001
        moe = self._make_moe()
        n_atoms = methylamine_batch.num_atoms
        feats = torch.randn(n_atoms, moe._irreps_in.dim, dtype=torch.float64)
        energies = moe(
            atom_features=feats,
            atomic_numbers=methylamine_batch.atomic_numbers,
            edge_index=methylamine_batch.edge_index,
            edge_lengths=methylamine_batch.edge_weight,
        )
        assert energies.shape == (n_atoms,)
        assert torch.isfinite(energies).all()

    def test_zero_mask_runs_absent_pairs(self, water_batch) -> None:  # noqa: ANN001
        """Water has only H and O — N-containing experts must still execute
        (their gate parameters must be in the autograd graph)."""
        moe = self._make_moe()
        n_atoms = water_batch.num_atoms
        feats = torch.randn(n_atoms, moe._irreps_in.dim, dtype=torch.float64)
        energies = moe(
            atom_features=feats,
            atomic_numbers=water_batch.atomic_numbers,
            edge_index=water_batch.edge_index,
            edge_lengths=water_batch.edge_weight,
        )
        loss = energies.sum()
        loss.backward()
        for name, p in moe.named_parameters():
            if name.endswith(".gate"):
                assert p.grad is not None, f"gate {name} got no gradient"


class TestZeroMaskCCOnly:
    """CHANGE 3 acceptance test.

    On a C-C-only batch with KRONOS configured for H/C/N/O:

    1. All 10 experts execute (their parameters are reachable from the
       loss / appear in the autograd graph).
    2. The total contribution of each of the 9 *absent* experts is
       exactly ``0.0`` — verified both via the ``return_per_pair``
       diagnostic and by checking that tampering with the gates of
       absent experts leaves the energy unchanged.
    """

    def _make_moe(self) -> KronosMoE:
        cfg = ExpertConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type="linear",
        )
        return KronosMoE(
            elements=(1, 6, 7, 8),
            irreps_in="16x0e+16x1o+16x2e",
            expert_config=cfg,
            cutoff=5.0,
        ).double()

    def test_absent_experts_contribute_exactly_zero(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        moe = self._make_moe()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(n_atoms, moe._irreps_in.dim, dtype=torch.float64)

        # Use the diagnostic return-per-pair path
        atom_energies, per_pair = moe(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
            return_per_pair=True,
        )

        # Sanity: 9 absent experts contribute exactly 0.0, C-C contributes
        # whatever the model says.
        assert set(per_pair.keys()) == {
            "H-H",
            "H-C",
            "H-N",
            "H-O",
            "C-C",
            "C-N",
            "C-O",
            "N-N",
            "N-O",
            "O-O",
        }
        for label, value in per_pair.items():
            if label == "C-C":
                # C-C is the only non-trivially-zero contribution.  Its
                # value depends on the random features; just check it is
                # finite and *can* be non-zero.
                assert torch.isfinite(value).item()
            else:
                assert value.item() == 0.0, (
                    f"Absent expert '{label}' contributed {value.item()!r} "
                    f"to the energy — zero-masking is broken"
                )

    def test_absent_expert_gates_have_no_effect_on_energy(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        """Tamper with every non-CC gate (set it to 1e10) and verify the
        total energy is unchanged.  If the mask leaked, the energy would
        explode."""
        moe = self._make_moe()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(n_atoms, moe._irreps_in.dim, dtype=torch.float64)

        e_before: torch.Tensor = moe(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )

        cc_key = KronosMoE._key(6, 6)
        with torch.no_grad():
            for key, expert in moe.experts.items():
                if key != cc_key:
                    expert.gate.fill_(1.0e10)

        e_after: torch.Tensor = moe(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )

        assert torch.allclose(e_before, e_after), (
            "Non-CC expert gates affected the energy on a C-C-only batch — "
            "zero-masking is broken"
        )

    def test_all_experts_are_in_autograd_graph(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        """On a C-C-only batch every one of the 10 experts must have a
        registered (possibly zero-valued) gradient after backward — the
        static-shape schedule keeps every expert in the autograd graph."""
        moe = self._make_moe()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(n_atoms, moe._irreps_in.dim, dtype=torch.float64, requires_grad=True)

        energies: torch.Tensor = moe(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )
        energies.sum().backward()

        for key in moe.experts.keys():
            expert: PairwiseExpert = typing.cast(PairwiseExpert, moe.experts[key])
            assert expert.gate.grad is not None, (
                f"Gate of expert {key} did not appear in the autograd graph — "
                f"the zero-mask schedule must keep every expert reachable"
            )
            # For absent experts the gradient should be exactly zero
            if key != KronosMoE._key(6, 6):
                assert expert.gate.grad.item() == 0.0, (
                    f"Absent expert {key} received a non-zero gradient "
                    f"({expert.gate.grad.item()!r}) — zero-masking is broken"
                )
