"""Modular KRONOS backbone.

KRONOS = **K-order Routed Orthogonal Network of Symmetry with
Element-Pair Mixture-of-Experts.**

The backbone produces ``NodeFeatures`` where ``node_energies`` is
already populated with the per-atom MoE energy contributions.  Heads
that respect this (``energy``, ``energy_forces``, ``dual_forces``)
simply sum the populated channel; otherwise heads fall back to the
standard scalar-readout pathway.

This is the **default** variant of KRONOS in the project; a fully
self-contained monolithic counterpart lives in
``goal.ml.nn.models.kronos.monolithic``.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from goal.ml.nn.blocks.experts import ExpertConfig, KronosMoE
from goal.ml.nn.models.kronos.geometry import differentiable_edges
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("kronos")
@BACKBONE_REGISTRY.register("kronos")
class KronosBackbone(nn.Module):
    """KRONOS backbone — environment dressing + element-pair MoE.

    Two-stage architecture:

    1. **Environment dressing** — one or more rounds of ACE-style
       equivariant message passing produce the one-particle basis
       ``A_i``.  Optional body-order expansion (``B²``, ``B³``)
       captures bond angles / dihedrals.  Output irreps for body
       orders ≥ 2 are derived programmatically from the CG
       decomposition (see :func:`cg_product_irreps`) and compressed
       back to ``irreps_hidden`` so the expert interface is
       unchanged.
    2. **Element-pair experts** — one expert per unordered element
       pair contributes a scalar pair energy with a learnable gate
       ``P_{AB}`` and a cosine-cutoff envelope.  Static-shape
       zero-masking keeps the autograd graph identical across ranks
       (DDP-safe).

    The block emits ``NodeFeatures`` whose ``node_energies`` field
    already holds the per-atom MoE energy contributions, so downstream
    heads can either pick them up directly (``EnergyHead`` /
    ``EnergyForcesHead`` / ``DualForcesHead`` recognise the field) or
    apply their own readout to the dressed features.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.  Determines the number
        and identity of expert modules.
    dressing_kwargs : dict
        Forwarded to :class:`EnvironmentDressing` — see its docstring.
        Includes ``body_order`` for the ACE body-order expansion.
    expert_config : dict
        Forwarded to :class:`ExpertConfig` — see its docstring.
    cutoff : float
        Cosine-cutoff radius for the pairwise experts (Angstrom).
        Defaults to the dressing cutoff.
    """

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        dressing_kwargs: dict[str, typing.Any] | None = None,
        expert_config: dict[str, typing.Any] | None = None,
        cutoff: float | None = None,
    ) -> None:
        super().__init__()
        dressing_cfg: dict[str, typing.Any] = dict(dressing_kwargs or {})
        expert_cfg: dict[str, typing.Any] = dict(expert_config or {})

        self.dressing: EnvironmentDressing = EnvironmentDressing(**dressing_cfg)
        moe_cutoff: float = cutoff if cutoff is not None else self.dressing.cutoff

        # Normalise tuple-typed config entries that may arrive as ListConfig
        if "hidden_dims" in expert_cfg:
            expert_cfg["hidden_dims"] = tuple(int(x) for x in expert_cfg["hidden_dims"])
        self.moe: KronosMoE = KronosMoE(
            elements=elements,
            irreps_in=self.dressing.irreps_out,
            expert_config=ExpertConfig(**expert_cfg),
            cutoff=moe_cutoff,
        )

        self._elements: tuple[int, ...] = self.moe.elements
        self._num_interactions: int = len(self.dressing.interactions)
        self._irreps_out: Irreps = self.dressing.irreps_out

    # ------------------------------------------------------------------
    # EquivariantBackbone protocol
    # ------------------------------------------------------------------

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_out

    @property
    def num_interactions(self) -> int:
        return self._num_interactions

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_experts(self) -> int:
        return self.moe.num_experts

    @property
    def body_order(self) -> int:
        """ACE body-order parameter inherited from the dressing block."""
        return self.dressing.body_order

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every expert's gate parameter."""
        return self.moe.gates()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        # Re-derive edge geometry from positions so the autograd graph
        # links energy back to ``graph.pos`` (required by autograd-based
        # force heads).  When ``graph.pos.requires_grad`` is unset we
        # still recompute — the cost is small and the path stays uniform.
        positions: torch.Tensor = graph.pos
        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)

        # Dressed equivariant features (one or more ACE-style MP rounds,
        # optionally followed by a body-order expansion).
        dressed: torch.Tensor = self.dressing(
            atomic_numbers=graph.atomic_numbers,
            edge_index=graph.edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )  # (N, irreps_out.dim)

        # Pairwise expert energies → per-atom scalars
        node_energies: torch.Tensor = self.moe(
            atom_features=dressed,
            atomic_numbers=graph.atomic_numbers,
            edge_index=graph.edge_index,
            edge_lengths=edge_lengths,
        )  # (N,)

        return NodeFeatures(
            node_feats=dressed,
            irreps=str(self._irreps_out),
            node_energies=node_energies,
        )
