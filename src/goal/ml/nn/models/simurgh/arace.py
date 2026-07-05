"""SIMURGH ARACE model variant.

**ARACE = ARtisan + Atomic Cluster Expansion** — a monolithic SIMURGH
variant where potential **AR**\\ tisans and **ACE** message passing
alternate as equals, rather than the standard pipeline of ACE dressing
first and artisans as an auxiliary readout last.

The standard SIMURGH design runs ACE-style environment dressing first
and reads out pair energies through the artisan bank *last* (an
auxiliary readout).  This variant reverses the philosophy:

* **Potential artisans are the primary learning component at every
  stage** — each round starts with an equivariant element-pair artisan
  layer that builds per-edge representations and contributes a per-layer
  pair energy.
* **ACE message passing updates nodes between artisan stages** — the
  per-edge artisan features are aggregated back onto nodes so artisan
  layer ``L + 1`` sees richer context than layer ``L``.
* **Everything stays equivariant throughout**; the only transition to
  invariants is the scalar readout inside each artisan layer.

The core architectural rhythm is the node/edge alternation::

    nodes → edges (artisan) → nodes (ACE) → edges (artisan) → ...

One full round (stacked ``num_rounds`` times)::

    ┌───────────────────────────────────────────────────┐
    │ ARTISAN LAYER L                                   │
    │   h_AB   = L(h[row]) + L(h[col])      (symmetric) │
    │   h_pair = TP(h_AB, Y^l(r̂); w(d))    (E, irreps) │
    │   E_L    = MLP(scalars(h_pair))       → (E,)      │
    │   E_atom += index_add(row, 0.5 × E_L)             │
    └───────────────────────────────────────────────────┘
              ↓ h_pair (edge-level, equivariant)
    ┌───────────────────────────────────────────────────┐
    │ ACE MESSAGE PASSING BLOCK L                       │
    │   agg_i  = Σ_j h_pair_{j→i} / N̄                  │
    │   h_new  = RMSNorm(EquivLinear(agg) + h)          │
    └───────────────────────────────────────────────────┘
              ↓ repeat, then
    E_total = E_atomic + Σ_L E_artisan_L,  F = -∂E/∂r (autograd)

This module is completely isolated from the standard SIMURGH classes —
it reuses only the shared primitives and helpers, and registers the
monolithic model as ``"monolithic_arace"``.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import (
    _PairRMSNorm,
    build_artisan_edge_irreps,
    cosine_cutoff,
    enumerate_element_pairs,
    pair_label,
)
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


class _AracePairArtisan(nn.Module):
    """One equivariant element-pair artisan for the ARACE variant.

    Identical in spirit to the equivariant
    :class:`~goal.ml.nn.blocks.artisans._EquivariantArtisanCore`, but in
    addition to the per-edge scalar energy it also *returns the
    equivariant pair representation* ``h_pair`` so the ACE block can
    aggregate it back onto nodes.  All learnable maps are bias-free so
    zero-masked edges contribute exactly zero.

    Parameters mirror the ``artisan.equivariant`` sub-config; see
    :class:`MonolithicArace`.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        cutoff: float,
        hidden_irreps: Irreps | str = "16x0e + 16x1o + 16x2e",
        num_layers: int = 1,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        n_scalar_out: int = 16,
        final_hidden: int = 16,
        element_conditioned: bool = True,
        n_elements: int = 120,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.irreps_node: Irreps = Irreps(irreps_node)
        self.irreps_hidden: Irreps = Irreps(hidden_irreps)
        lmax: int = max((ir.l for _, ir in self.irreps_hidden), default=0)
        self.irreps_edge: Irreps = build_artisan_edge_irreps(lmax)
        self._element_conditioned: bool = bool(element_conditioned)
        self._n_elements: int = int(n_elements)

        # Shared symmetric node embedding — one map applied to both
        # endpoints guarantees E(A, B) = E(B, A) under the sum.
        self.node_embed: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_hidden, biases=False
        )

        # CG tensor product with bond geometry.
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_in1=self.irreps_hidden,
            irreps_in2=self.irreps_edge,
            irreps_out=self.irreps_hidden,
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=num_rbf, cutoff=cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=cutoff)
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=num_rbf,
            hidden_dim=radial_hidden,
            num_out=self.tp.weight_numel,
        )
        self.element_linear: nn.Linear | None = (
            nn.Linear(self._n_elements, self.tp.weight_numel, bias=False)
            if element_conditioned
            else None
        )

        # Optional deeper equivariant layers.
        self.layers: nn.ModuleList = nn.ModuleList(
            EquivariantLinear(self.irreps_hidden, self.irreps_hidden, biases=False)
            for _ in range(num_layers - 1)
        )
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(self.irreps_hidden) for _ in range(num_layers - 1)
        )

        # Invariant scalar readout (bias-free).
        scalar_irreps: Irreps = Irreps(f"{int(n_scalar_out)}x0e")
        self.to_scalars: EquivariantLinear = EquivariantLinear(
            self.irreps_hidden, scalar_irreps, biases=False
        )
        self.energy_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar_out), int(final_hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(final_hidden), 1, bias=False),
        )

        # Learnable gate, initialised to 1.0.
        self.gate: nn.Parameter = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        edge_sh: torch.Tensor,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-edge pair representation and gated scalar energy.

        Parameters
        ----------
        feats_a, feats_b : Tensor ``(E, irreps_node.dim)``
            Equivariant node features of source / destination atom.
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Spherical harmonics of the edge direction.
        distances : Tensor ``(E,)``
            Bond lengths (zeros on masked edges are clamped internally).
        z_dst : Tensor ``(E,)``, optional
            Destination atomic numbers (element conditioning).

        Returns
        -------
        tuple of Tensor
            ``(h_pair, e_pair)`` with shapes ``(E, irreps_hidden.dim)``
            and ``(E,)``; ``e_pair`` is already multiplied by the gate.
        """
        h_ab: torch.Tensor = self.node_embed(feats_a) + self.node_embed(feats_b)

        d_safe: torch.Tensor = distances.clamp_min(1e-6)
        rbf: torch.Tensor = self.radial_basis(d_safe)  # (E, num_rbf)
        env: torch.Tensor = self.envelope(d_safe).to(h_ab.dtype).unsqueeze(-1)  # (E, 1)
        radial_w: torch.Tensor = self.radial_mlp(rbf) * env  # (E, weight_numel)

        if self._element_conditioned and self.element_linear is not None:
            if z_dst is None:
                raise ValueError("element_conditioned=True requires z_dst in forward()")
            one_hot: torch.Tensor = F.one_hot(
                z_dst.clamp(0, self._n_elements - 1),
                num_classes=self._n_elements,
            ).to(radial_w.dtype)
            radial_w = radial_w * self.element_linear(one_hot)

        h_pair: torch.Tensor = self.tp(h_ab, edge_sh.to(h_ab.dtype), radial_w)
        for lin, norm in zip(self.layers, self.norms):
            h_pair = norm(lin(h_pair) + h_pair)

        scalars: torch.Tensor = self.to_scalars(h_pair)  # (E, n_scalar_out)
        e_pair: torch.Tensor = self.gate * self.energy_mlp(scalars).squeeze(-1)  # (E,)
        return h_pair, e_pair


class AraceBlock(nn.Module):
    """One artisan layer + one ACE message-passing block.

    The artisan layer routes every edge to its element-pair artisan with
    the same static-shape zero-masking schedule as the standard SIMURGH
    bank (inputs masked before the forward, outputs masked after — every
    artisan runs every step, DDP-safe).  The masked per-edge features of
    all artisans are summed into a single ``h_pair`` tensor (each edge
    matches exactly one pair type, so the sum is a routed select).

    The ACE block aggregates ``h_pair`` onto the *destination* nodes
    (``agg_i = Σ_j h_pair_{j→i} / N̄``) and updates the node features
    with an equivariant linear, a residual connection and a smooth
    RMS norm.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (defines the pair routing).
    irreps_node : Irreps or str
        Irreps of the node features entering and leaving the block.
        Must equal the artisans' ``hidden_irreps`` so the edge → node
        aggregation and the residual connection are well-typed.
    cutoff : float
        Cutoff radius (Angstrom) for the radial basis and the cosine
        taper on pair energies.
    artisan_kwargs : dict
        Forwarded to every :class:`_AracePairArtisan`.
    artisans : nn.ModuleDict, optional
        Pre-built artisan bank to **share** across blocks
        (``share_artisan_weights=True``).  When ``None`` the block
        builds its own independent bank.
    avg_num_neighbors : float, optional
        Mean neighbour count used to normalise the edge → node
        aggregation.  ``None`` disables normalisation.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_node: Irreps | str,
        cutoff: float,
        artisan_kwargs: dict[str, typing.Any] | None = None,
        artisans: nn.ModuleDict | None = None,
        avg_num_neighbors: float | None = None,
    ) -> None:
        super().__init__()
        self.irreps_node: Irreps = Irreps(irreps_node)
        self._cutoff: float = float(cutoff)
        self._pairs: tuple[tuple[int, int], ...] = tuple(enumerate_element_pairs(elements))
        self._pair_keys: list[str] = [f"z{a}_z{b}" for a, b in self._pairs]

        kwargs: dict[str, typing.Any] = dict(artisan_kwargs or {})
        if artisans is not None:
            self.artisans: nn.ModuleDict = artisans
        else:
            self.artisans = nn.ModuleDict(
                {
                    key: _AracePairArtisan(
                        irreps_node=self.irreps_node, cutoff=cutoff, **kwargs
                    )
                    for key in self._pair_keys
                }
            )

        first = typing.cast(_AracePairArtisan, self.artisans[self._pair_keys[0]])
        if first.irreps_hidden != self.irreps_node:
            raise ValueError(
                f"Artisan hidden_irreps ({first.irreps_hidden}) must equal the "
                f"node irreps ({self.irreps_node}) so the ACE aggregation and "
                "residual are well-typed."
            )
        self.irreps_edge: Irreps = first.irreps_edge

        # ACE block: aggregate edges → nodes, then linear + residual + norm.
        self.ace_linear: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_node, biases=False
        )
        self.ace_norm: _PairRMSNorm = _PairRMSNorm(self.irreps_node)

        norm_scale: float = (
            1.0 / float(avg_num_neighbors)
            if avg_num_neighbors is not None and float(avg_num_neighbors) > 0.0
            else 1.0
        )
        self.register_buffer(
            "agg_norm_scale",
            torch.tensor(norm_scale, dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one artisan + ACE round.

        Parameters
        ----------
        h : Tensor ``(N, irreps_node.dim)``
            Current equivariant node features.
        atomic_numbers : Tensor ``(N,)``
        edge_index : Tensor ``(2, E)``
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Pre-computed edge spherical harmonics (geometry only, shared
            across all blocks).
        edge_lengths : Tensor ``(E,)``

        Returns
        -------
        tuple of Tensor
            ``(h_new, e_atom)`` — updated node features
            ``(N, irreps_node.dim)`` and this layer's per-atom energy
            contribution ``(N,)``.
        """
        num_atoms: int = h.shape[0]
        device: torch.device = h.device
        dtype: torch.dtype = h.dtype
        row, col = edge_index
        E: int = int(edge_lengths.shape[0])

        e_atom: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)

        # ----- Degenerate case: no edges — keep every artisan reachable.
        if E == 0:
            dummy_feats: torch.Tensor = torch.zeros(
                1, self.irreps_node.dim, device=device, dtype=dtype
            )
            dummy_sh: torch.Tensor = torch.zeros(
                1, self.irreps_edge.dim, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            dummy_z: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
            dummy_sum: torch.Tensor = torch.zeros((), device=device, dtype=dtype)
            for key in self._pair_keys:
                artisan = typing.cast(_AracePairArtisan, self.artisans[key])
                h_p, e_p = artisan(dummy_feats, dummy_feats, dummy_sh, dummy_dist, dummy_z)
                dummy_sum = dummy_sum + 0.0 * (h_p.sum() + e_p.sum())
            agg: torch.Tensor = torch.zeros_like(h) + dummy_sum
            h_new: torch.Tensor = self.ace_norm(self.ace_linear(agg) + h)
            return h_new, e_atom + dummy_sum

        edge_lengths = edge_lengths.to(dtype)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # ----- Artisan layer: route every edge to its pair artisan.
        h_pair_total: torch.Tensor = torch.zeros(
            E, self.irreps_node.dim, device=device, dtype=dtype
        )
        for (a, b), key in zip(self._pairs, self._pair_keys):
            artisan = typing.cast(_AracePairArtisan, self.artisans[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_col: torch.Tensor = mask.unsqueeze(-1)
            feats_a: torch.Tensor = h[row] * mask_col
            feats_b: torch.Tensor = h[col] * mask_col
            sh_in: torch.Tensor = edge_sh.to(dtype) * mask_col
            dist_in: torch.Tensor = edge_lengths * mask
            h_p, e_p = artisan(feats_a, feats_b, sh_in, dist_in, z_col)
            h_pair_total = h_pair_total + h_p * mask_col
            tapered: torch.Tensor = e_p * mask * cut_env
            e_atom = e_atom.index_add(0, row, 0.5 * tapered)

        # ----- ACE block: aggregate edge features onto destination nodes.
        agg = torch.zeros_like(h)
        agg = agg.index_add(0, col, h_pair_total)
        agg = agg * self.agg_norm_scale.to(dtype)
        h_new = self.ace_norm(self.ace_linear(agg) + h)
        return h_new, e_atom


class AraceBackbone(nn.Module):
    """Stacks ``num_rounds`` :class:`AraceBlock` rounds.

    Node features are initialised from the atomic-number embedding table
    (no separate environment-dressing phase) and projected into the
    artisans' equivariant space.  Edge spherical harmonics are computed
    once and shared across all rounds.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    num_rounds : int
        Number of artisan + ACE rounds.
    share_artisan_weights : bool
        ``True`` → one artisan bank shared by every round;
        ``False`` (default) → an independent bank per round.
    artisan : dict, optional
        Artisan sub-config.  ``architecture`` must be ``"equivariant"``
        (the ARACE variant is equivariant throughout); the
        remaining keys are forwarded to
        :class:`_AracePairArtisan` (``hidden_irreps``,
        ``num_layers``, ``num_rbf``, ``radial_hidden``,
        ``n_scalar_out``, ``final_hidden``, ``element_conditioned``).
    cutoff : float
        Neighbour cutoff radius (Angstrom).
    embedding_dim : int
        Width of the initial scalar embedding.
    num_elements : int
        Size of the embedding table (must exceed the largest Z).
    avg_num_neighbors : float, optional
        Edge → node aggregation normaliser for every ACE block.
    """

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        num_rounds: int = 2,
        share_artisan_weights: bool = False,
        artisan: dict[str, typing.Any] | None = None,
        cutoff: float = 5.0,
        embedding_dim: int = 32,
        num_elements: int = 120,
        avg_num_neighbors: float | None = None,
    ) -> None:
        super().__init__()
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._num_rounds: int = int(num_rounds)
        self._share_artisan_weights: bool = bool(share_artisan_weights)
        self._cutoff: float = float(cutoff)

        artisan_cfg: dict[str, typing.Any] = dict(artisan or {})
        architecture: str = str(artisan_cfg.pop("architecture", "equivariant"))
        if architecture != "equivariant":
            raise ValueError(
                "The ARACE variant is equivariant throughout; "
                f"artisan.architecture must be 'equivariant', got '{architecture}'."
            )
        self._artisan_kwargs: dict[str, typing.Any] = artisan_cfg

        hidden_irreps: Irreps = Irreps(
            artisan_cfg.get("hidden_irreps", "16x0e + 16x1o + 16x2e")
        )
        self._irreps_hidden: Irreps = hidden_irreps

        # Initial node features: embedding lookup → equivariant space.
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        self.input_linear: EquivariantLinear = EquivariantLinear(
            Irreps(f"{embedding_dim}x0e"), hidden_irreps
        )

        # Shared bank (or None → each block builds its own).
        shared_bank: nn.ModuleDict | None = None
        if share_artisan_weights:
            shared_bank = nn.ModuleDict(
                {
                    f"z{a}_z{b}": _AracePairArtisan(
                        irreps_node=hidden_irreps, cutoff=cutoff, **artisan_cfg
                    )
                    for a, b in enumerate_element_pairs(self._elements)
                }
            )

        self.blocks: nn.ModuleList = nn.ModuleList(
            AraceBlock(
                elements=self._elements,
                irreps_node=hidden_irreps,
                cutoff=cutoff,
                artisan_kwargs=artisan_cfg,
                artisans=shared_bank,
                avg_num_neighbors=avg_num_neighbors,
            )
            for _ in range(self._num_rounds)
        )
        self.irreps_edge: Irreps = typing.cast(
            AraceBlock, self.blocks[0]
        ).irreps_edge

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_rounds(self) -> int:
        return self._num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self._share_artisan_weights

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_hidden

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every artisan gate, keyed ``layer{L}/{pair}``."""
        result: dict[str, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks):
            blk = typing.cast(AraceBlock, block)
            for (a, b), key in zip(blk._pairs, blk._pair_keys):
                artisan = typing.cast(_AracePairArtisan, blk.artisans[key])
                result[f"layer{layer_idx}/{pair_label(a, b)}"] = artisan.gate.detach()
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all rounds.

        Returns
        -------
        tuple of Tensor
            ``(node_energies, layer_energies)`` where ``node_energies``
            is the summed per-atom artisan energy ``(N,)`` and
            ``layer_energies`` stacks the per-round per-atom
            contributions ``(num_rounds, N)``.
        """
        h: torch.Tensor = self.input_linear(self.embedding(atomic_numbers))

        # Edge SH — geometry only, computed once and shared by every round.
        edge_sh: torch.Tensor = spherical_harmonics(
            self.irreps_edge,
            edge_vectors.to(h.dtype),
            normalize=True,
            normalization="component",
        )

        per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            h, e_layer = block(h, atomic_numbers, edge_index, edge_sh, edge_lengths)
            per_layer.append(e_layer)

        layer_energies: torch.Tensor = torch.stack(per_layer, dim=0)  # (L, N)
        return layer_energies.sum(dim=0), layer_energies


@MODEL_REGISTRY.register("monolithic_arace")
@BACKBONE_REGISTRY.register("monolithic_arace")
class MonolithicArace(nn.Module):
    """Self-contained ARACE SIMURGH model.

    Satisfies the ``MonolithicModel`` protocol — returns a property
    dict with ``"energy"``, ``"forces"``, ``"num_atoms"`` and
    ``"layer_energies"`` directly.  Configure with ``head: null``.

    ``E_total = E_atomic + scale × Σ_L E_artisan_L`` and
    ``F = -∂E_total/∂r`` via autograd.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (injected by ``train.py``
        from the training set).
    num_rounds : int
        Number of artisan + ACE rounds.
    share_artisan_weights : bool
        Share one artisan bank across all rounds (default ``False``).
    artisan : dict, optional
        Artisan sub-config — see :class:`AraceBackbone`.
    cutoff : float
        Neighbour cutoff radius (Angstrom); must match ``data.cutoff``.
    embedding_dim, num_elements :
        Initial embedding table configuration.
    avg_num_neighbors : float, optional
        ACE aggregation normaliser.
    atomic_energies : dict, optional
        Per-element baseline with the same three modes as the standard
        SIMURGH backbone: ``learned`` (default, ``nn.Parameter``),
        ``dataset`` / ``provided`` (fixed buffer seeded from
        ``values``).
    scale : float, optional
        Multiplicative gain on the artisan interaction energy.
    num_elements_table : int
        Size of the Z-indexed atomic-energy parameter/buffer.
    """

    atomic_energies: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        num_rounds: int = 2,
        share_artisan_weights: bool = False,
        artisan: dict[str, typing.Any] | None = None,
        cutoff: float = 5.0,
        embedding_dim: int = 32,
        num_elements: int = 120,
        avg_num_neighbors: float | None = None,
        atomic_energies: dict[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
    ) -> None:
        super().__init__()
        artisan_cfg: dict[str, typing.Any] | None = (
            {str(k): v for k, v in dict(artisan).items()} if artisan is not None else None
        )
        self.backbone: AraceBackbone = AraceBackbone(
            elements=elements,
            num_rounds=num_rounds,
            share_artisan_weights=share_artisan_weights,
            artisan=artisan_cfg,
            cutoff=cutoff,
            embedding_dim=embedding_dim,
            num_elements=num_elements,
            avg_num_neighbors=(
                float(avg_num_neighbors) if avg_num_neighbors is not None else None
            ),
        )

        # ----- Per-element atomic-energy baseline (3-mode contract) -----
        ae_cfg: dict[str, typing.Any] = dict(atomic_energies or {})
        mode: str = str(ae_cfg.get("mode", "learned"))
        if ae_cfg.get("compute_from_dataset", False) and mode == "learned":
            mode = "dataset"
        if mode not in ("learned", "dataset", "provided"):
            raise ValueError(
                f"atomic_energies.mode must be 'learned', 'dataset' or "
                f"'provided', got {mode!r}."
            )
        if mode == "learned":
            self.atomic_energies = nn.Parameter(
                torch.zeros(num_elements_table, dtype=torch.get_default_dtype())
            )
        else:
            values: typing.Any = ae_cfg.get("values")
            if values is None:
                raise ValueError(
                    f"atomic_energies.mode={mode!r} requires 'values' to be a "
                    "{Z: e_Z} dict (train.py injects it for mode='dataset')."
                )
            buf: torch.Tensor = torch.zeros(
                num_elements_table, dtype=torch.get_default_dtype()
            )
            for z_key, e_val in dict(values).items():
                buf[int(z_key)] = float(e_val)
            self.register_buffer("atomic_energies", buf)
        self._atomic_energies_mode: str = mode

        scale_val: float = float(scale) if scale is not None else 1.0
        self.register_buffer(
            "scale",
            torch.tensor(scale_val, dtype=torch.get_default_dtype()),
        )

    # ------------------------------------------------------------------
    # MonolithicModel protocol
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        """Return list of output keys produced by the model."""
        return ["energy", "forces"]

    @property
    def elements(self) -> tuple[int, ...]:
        return self.backbone.elements

    @property
    def num_rounds(self) -> int:
        return self.backbone.num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self.backbone.share_artisan_weights

    @property
    def atomic_energies_mode(self) -> str:
        return self._atomic_energies_mode

    def gates(self) -> dict[str, torch.Tensor]:
        """Return artisan gating weights keyed by layer and pair type."""
        return self.backbone.gates()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Forward pass: dict with 'energy', 'forces', 'num_atoms', 'layer_energies'."""
        positions: torch.Tensor = graph.pos
        positions.requires_grad_(True)

        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)

        node_energies: torch.Tensor
        layer_energies: torch.Tensor
        node_energies, layer_energies = self.backbone(
            atomic_numbers=graph.atomic_numbers,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )

        scale_v: torch.Tensor = self.scale.to(node_energies.dtype)
        baseline: torch.Tensor = self.atomic_energies[graph.atomic_numbers].to(
            node_energies.dtype
        )
        node_total: torch.Tensor = scale_v * node_energies + baseline  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_total.device)
        )
        energy: torch.Tensor = scatter(node_total, batch, dim=0, reduce="sum")  # (B,)
        num_atoms: torch.Tensor = scatter(
            torch.ones_like(node_total), batch, dim=0, reduce="sum"
        )  # (B,)

        # Per-round energy totals (L, B) — diagnostic for layer
        # contributions; detached so it never interferes with training.
        layer_totals: torch.Tensor = scatter(
            scale_v * layer_energies, batch, dim=1, reduce="sum"
        ).detach()  # (L, B)

        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=positions,
            create_graph=self.training,
            retain_graph=True,
        )
        forces: torch.Tensor = -grad_outputs[0]  # (N, 3)

        return {
            "energy": energy,
            "forces": forces,
            "num_atoms": num_atoms,
            "layer_energies": layer_totals,
        }
