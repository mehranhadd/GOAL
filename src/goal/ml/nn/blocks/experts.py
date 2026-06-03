"""KRONOS Element-Pair Mixture-of-Experts block.

For a configurable set of chemical elements ``E = {Z_1, ..., Z_K}`` this
module instantiates one expert ``E_{AB}`` for every *unordered* pair of
elements ``(A, B)`` (so ``K * (K + 1) // 2`` experts in total).  Each
expert reads:

* the per-atom invariant scalars of atoms ``A`` and ``B`` (projected from
  the dressed equivariant features via ``o3.Linear`` to a fixed
  ``scalar_channels`` width — preserving equivariance because we keep
  only ``l = 0`` outputs);
* the scalar interatomic distance ``d_{ij}``.

Each expert outputs a single scalar energy contribution per pair which
is multiplied by a learnable scalar gate ``P_{AB}`` and smoothly tapered
by a cosine cutoff envelope ``f_cut(d)`` so forces are continuous at the
cutoff.

Two expert backbones are configurable from Hydra:

* ``"linear"`` — the spec described in the design document
  (``Linear → SiLU → Linear → SiLU → Linear``).
* ``"transformer"`` — an *equivariance-safe* mini-transformer that
  operates exclusively on the **invariant scalar** branch, so the
  E(3) symmetry of the model is preserved.

Static-shape zero-masking schedule
----------------------------------
The block follows a strict **input-masking, output-masking** schedule
for DDP safety:

1. For every expert ``E_{AB}`` we build a per-edge mask
   ``m[e] ∈ {0.0, 1.0}`` that selects edges of pair type ``(A, B)``.
2. The *inputs* (scalars_A, scalars_B, distance) are multiplied by the
   mask **before** the expert forward pass — there is no ``if/else``
   that skips an expert.
3. The expert output ``E_{AB}(masked_inputs)`` is multiplied by the
   mask **again** before being added to ``atom_energies``, so the
   contribution from non-matching edges is exactly ``0.0``.

This keeps the autograd graph perfectly static across all ranks (no
DDP deadlock, no ``find_unused_parameters``, no stale AdamW moments),
and every expert touches a deterministic number of edges every step.
The only branch left is the (very rare) ``E == 0`` degenerate case
where the whole batch carries no edges at all; we then run each
expert once on a 1-edge dummy with mask = 0 so its parameters remain
reachable from the loss.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.nn.primitives.linear import EquivariantLinear

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def enumerate_element_pairs(elements: typing.Sequence[int]) -> list[tuple[int, int]]:
    """Return every unordered pair ``(A, B)`` for the given elements, including
    self-pairs.  Order is deterministic: ``(A, B)`` with ``A <= B``.

    Example
    -------
    >>> enumerate_element_pairs([1, 6, 7, 8])
    [(1, 1), (1, 6), (1, 7), (1, 8), (6, 6), (6, 7), (6, 8), (7, 7), (7, 8), (8, 8)]
    """
    sorted_el: list[int] = sorted(elements)
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(sorted_el):
        for b in sorted_el[i:]:
            pairs.append((a, b))
    return pairs


def pair_label(a: int, b: int, symbols: typing.Mapping[int, str] | None = None) -> str:
    """Pretty label for a pair, e.g. ``"H-C"`` (used in logging)."""
    sym: typing.Mapping[int, str] = symbols or _DEFAULT_SYMBOLS
    lo, hi = sorted((a, b))
    return f"{sym.get(lo, str(lo))}-{sym.get(hi, str(hi))}"


_DEFAULT_SYMBOLS: dict[int, str] = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cosine cutoff envelope, ``f(d) = 0.5 (cos(pi d / r_c) + 1)``.

    Zero outside ``cutoff``.  ``distances`` of shape ``(...,)``; output
    same shape.
    """
    mask: torch.Tensor = (distances < cutoff).to(distances.dtype)
    arg: torch.Tensor = math.pi * (distances / cutoff)
    return 0.5 * (torch.cos(arg) + 1.0) * mask


# ---------------------------------------------------------------------------
# Per-expert backbones
# ---------------------------------------------------------------------------


@dataclass
class ExpertConfig:
    """Static configuration carried by every pairwise expert.

    Attributes
    ----------
    scalar_channels : int
        Width of the invariant projection of each atom's features.
    hidden_dims : tuple of int
        Hidden widths of the expert MLP (excluding input + output).
    expert_type : str
        ``"linear"`` (default) or ``"transformer"``.
    transformer_heads : int
        Multi-head attention heads when ``expert_type == "transformer"``.
    transformer_layers : int
        Number of transformer encoder layers.
    dropout_rate : float
        Dropout probability inserted between every ``Linear`` layer in
        the ``"linear"`` backbone MLP.  ``0.0`` disables dropout (default).
        ``nn.Dropout`` is used so dropout is automatically disabled during
        ``model.eval()`` / validation and inference.
    """

    scalar_channels: int = 16
    hidden_dims: tuple[int, ...] = (64, 32)
    expert_type: str = "linear"
    transformer_heads: int = 2
    transformer_layers: int = 1
    dropout_rate: float = 0.0
    # Rare-pair expert (CHANGE 4)
    rare_pair_embed_dim: int = 16


class _LinearExpert(nn.Module):
    """Standard MLP expert ``[in] → SiLU → [Dropout] → ... → 1``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: typing.Sequence[int],
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev: int = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """``(P, in_dim) → (P,)``."""
        return self.net(pair_features).squeeze(-1)


class _TransformerExpert(nn.Module):
    """E(3)-safe transformer expert.

    The transformer operates purely on the *invariant scalar* track —
    we treat the per-pair feature vector as a sequence of three tokens
    ``[scalars_A, scalars_B, distance_embedding]`` (each projected to
    ``token_dim``).  Because every input token is already an invariant
    scalar, the entire encoder is automatically invariant under
    rotations of the input positions, and thus the larger network
    composed of dressing → invariant projection → transformer expert
    remains E(3)-equivariant.
    """

    def __init__(
        self,
        scalar_channels: int,
        hidden_dims: typing.Sequence[int],
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        token_dim: int = hidden_dims[0] if len(hidden_dims) > 0 else 64

        # Project per-atom scalars and the distance scalar to the same
        # token dimension so they can be concatenated as a sequence.
        self.atom_proj: nn.Linear = nn.Linear(scalar_channels, token_dim)
        self.distance_proj: nn.Linear = nn.Linear(1, token_dim)

        # Adjust num_heads so that token_dim is divisible
        adjusted_heads: int = max(1, num_heads)
        while token_dim % adjusted_heads != 0 and adjusted_heads > 1:
            adjusted_heads -= 1

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=adjusted_heads,
            dim_feedforward=hidden_dims[-1] if len(hidden_dims) > 0 else token_dim,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final readout: average tokens + scalar head
        self.readout: nn.Linear = nn.Linear(token_dim, 1)

    def forward(
        self,
        pair_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``([P, S], [P, S], [P, 1]) → (P,)`` — invariant-only inputs."""
        scalars_a, scalars_b, distance = pair_features
        token_a: torch.Tensor = self.atom_proj(scalars_a)  # (P, token_dim)
        token_b: torch.Tensor = self.atom_proj(scalars_b)  # (P, token_dim)
        token_d: torch.Tensor = self.distance_proj(distance)  # (P, token_dim)

        # Stack into sequence (P, 3, token_dim)
        sequence: torch.Tensor = torch.stack([token_a, token_b, token_d], dim=1)
        encoded: torch.Tensor = self.encoder(sequence)  # (P, 3, token_dim)
        pooled: torch.Tensor = encoded.mean(dim=1)  # (P, token_dim)
        return self.readout(pooled).squeeze(-1)  # (P,)


# ---------------------------------------------------------------------------
# Single expert wrapper
# ---------------------------------------------------------------------------


class PairwiseExpert(nn.Module):
    """One ``E_{AB}`` expert for a single element pair.

    Holds:

    * an invariant projection ``o3.Linear`` mapping atom features to
      ``scalar_channels x 0e`` scalars (E(3)-safe because only ``l = 0``
      outputs are retained);
    * the actual expert backbone (Linear MLP or Transformer);
    * a learnable scalar gate ``P_{AB}``.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        config: ExpertConfig,
    ) -> None:
        super().__init__()
        self.scalar_channels: int = config.scalar_channels
        scalar_irreps: Irreps = Irreps(f"{config.scalar_channels}x0e")

        # Project arbitrary equivariant atom features → scalars (l=0)
        self.scalar_proj: EquivariantLinear = EquivariantLinear(irreps_in, scalar_irreps)

        # Learnable gate, initialised to 1.0
        self.gate: nn.Parameter = nn.Parameter(torch.tensor(1.0))

        # Backbone
        in_dim: int = 2 * config.scalar_channels + 1
        if config.expert_type == "linear":
            self._backbone_type: str = "linear"
            self.backbone: nn.Module = _LinearExpert(
                in_dim=in_dim,
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
            )
        elif config.expert_type == "transformer":
            self._backbone_type = "transformer"
            self.backbone = _TransformerExpert(
                scalar_channels=config.scalar_channels,
                hidden_dims=config.hidden_dims,
                num_heads=config.transformer_heads,
                num_layers=config.transformer_layers,
            )
        else:
            raise ValueError(
                f"Unknown expert_type '{config.expert_type}'. Use 'linear' or 'transformer'."
            )

    def project(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Invariant projection ``(N, irreps_in.dim) → (N, scalar_channels)``."""
        return self.scalar_proj(atom_features)

    def forward(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated pair energies (un-tapered).

        Parameters
        ----------
        scalars_a : Tensor
            Invariant scalars of atom ``A``, shape ``(P, scalar_channels)``.
        scalars_b : Tensor
            Invariant scalars of atom ``B``, shape ``(P, scalar_channels)``.
        distances : Tensor
            Pair distances, shape ``(P,)``.

        Returns
        -------
        Tensor
            ``(P,)`` per-pair scalar energy (gate × backbone output).
        """
        if self._backbone_type == "linear":
            pair_features: torch.Tensor = torch.cat(  # (P, 2 * S + 1)
                [scalars_a, scalars_b, distances.unsqueeze(-1)], dim=-1
            )
            raw: torch.Tensor = self.backbone(pair_features)  # (P,)
        else:  # transformer
            raw = self.backbone((scalars_a, scalars_b, distances.unsqueeze(-1)))  # (P,)
        return self.gate * raw

    def forward_pairwise(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(E_ij, F_ij)`` for the pairwise force mode.

        ``F_ij = -∂E_ij/∂r_ij`` is taken via ``torch.autograd.grad`` so
        Newton's third law is satisfied per directed edge when the
        caller scatters ``F_ij`` to source and ``-F_ij`` to destination
        atoms.

        Parameters
        ----------
        scalars_a, scalars_b : Tensor
            Invariant scalars of source / destination atom, shape
            ``(P, scalar_channels)``.
        edge_vectors : Tensor
            Per-edge displacement vectors ``r_ij = pos[j] - pos[i]``
            of shape ``(P, 3)``.  Treated as the leaf of the local
            autograd graph — its ``requires_grad`` is enabled inside
            this method without affecting the caller's tensor.

        Returns
        -------
        tuple of Tensor
            ``(E_ij, F_ij)`` with shapes ``(P,)`` and ``(P, 3)``.
        """
        # Use a local leaf so the gradient stops at edge_vectors (we
        # only want the per-pair derivative, not a derivative back
        # through positions — the head can still take a separate
        # autograd path on ``graph.pos`` if it wants combined mode).
        r_ij: torch.Tensor = edge_vectors.detach().requires_grad_(True)
        distances: torch.Tensor = r_ij.norm(dim=-1)  # (P,)

        if self._backbone_type == "linear":
            pair_features: torch.Tensor = torch.cat(
                [scalars_a, scalars_b, distances.unsqueeze(-1)], dim=-1
            )
            raw: torch.Tensor = self.backbone(pair_features)  # (P,)
        else:  # transformer
            raw = self.backbone((scalars_a, scalars_b, distances.unsqueeze(-1)))  # (P,)
        e_ij: torch.Tensor = self.gate * raw  # (P,)

        # F_ij = -dE_ij/dr_ij
        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=e_ij.sum(),
            inputs=r_ij,
            create_graph=self.training,
            retain_graph=True,
        )
        f_ij: torch.Tensor = -grad_outputs[0]  # (P, 3)
        return e_ij, f_ij


# ---------------------------------------------------------------------------
# Rare-pair expert (CHANGE 4)
# ---------------------------------------------------------------------------


class RarePairExpert(nn.Module):
    """Shared expert for rare/infrequent element pairs.

    Unlike :class:`PairwiseExpert` — which dedicates one full MLP per pair —
    this module handles *all* rare pairs in a single network by injecting a
    per-pair learned embedding.  Each rare pair gets its own
    ``pair_embed_dim``-dimensional embedding vector and its own scalar gate,
    so the network can still specialize per pair.

    The ``n_rare`` learned gate parameters and embedding rows remain in the
    autograd graph regardless of whether the corresponding pair appears in
    the current batch (same zero-masking contract as :class:`KronosMoE`).

    Parameters
    ----------
    irreps_in : Irreps
        Irreps of the atom features fed into this expert.
    pairs : list of (int, int)
        The rare unordered element pairs this expert handles, in canonical
        ``(lo, hi)`` order.
    config : ExpertConfig
        Shared expert config; ``rare_pair_embed_dim`` controls the embedding width.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        pairs: list[tuple[int, int]],
        config: ExpertConfig,
    ) -> None:
        super().__init__()
        if not pairs:
            raise ValueError("RarePairExpert requires at least one pair.")
        self._pairs: tuple[tuple[int, int], ...] = tuple(pairs)
        self.scalar_channels: int = config.scalar_channels
        self._embed_dim: int = config.rare_pair_embed_dim

        scalar_irreps: Irreps = Irreps(f"{config.scalar_channels}x0e")
        self.scalar_proj: EquivariantLinear = EquivariantLinear(irreps_in, scalar_irreps)

        # Per-pair embedding and gates
        n: int = len(pairs)
        self.pair_embed: nn.Embedding = nn.Embedding(n, config.rare_pair_embed_dim)
        self.gates: nn.Parameter = nn.Parameter(torch.ones(n))

        # Shared MLP: [s_A, s_B, pair_embed, dist] → 1
        in_dim: int = 2 * config.scalar_channels + config.rare_pair_embed_dim + 1
        self.backbone: _LinearExpert = _LinearExpert(
            in_dim=in_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )

    def project(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Invariant projection ``(N, irreps_in.dim) → (N, scalar_channels)``."""
        return self.scalar_proj(atom_features)

    def forward(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        distances: torch.Tensor,
        pair_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated pair energies for a batch of edges (same rare pair type).

        Parameters
        ----------
        scalars_a, scalars_b : Tensor ``(E, scalar_channels)``
            Masked atom scalars for source / destination.
        distances : Tensor ``(E,)``
            Masked pair distances.
        pair_local_idx : Tensor ``(E,)`` int
            Local index into this expert's pair list (same value for all
            edges of the same rare pair type; varies when batching multiple
            rare pairs, but caller loops over pairs separately).

        Returns
        -------
        Tensor ``(E,)``
            Gated per-pair energy contributions.
        """
        embed: torch.Tensor = self.pair_embed(
            pair_local_idx.clamp(0, len(self._pairs) - 1)
        )  # (E, embed_dim)
        feats: torch.Tensor = torch.cat(
            [scalars_a, scalars_b, embed, distances.unsqueeze(-1)], dim=-1
        )  # (E, 2*S + D + 1)
        raw: torch.Tensor = self.backbone(feats)  # (E,)
        gate: torch.Tensor = self.gates[pair_local_idx]  # (E,)
        return gate * raw


# ---------------------------------------------------------------------------
# MoE container
# ---------------------------------------------------------------------------


class KronosMoE(nn.Module):
    """KRONOS Element-Pair Mixture-of-Experts with optional data-driven routing.

    By default, instantiates one :class:`PairwiseExpert` per unordered element
    pair (original behaviour, ``rare_pair_enabled=False``).

    When ``rare_pair_enabled=True`` and ``pair_counts`` is provided, pairs whose
    frequency (fraction of total edges) falls below ``min_pair_frequency`` are
    routed to a single shared :class:`RarePairExpert` that uses a learned
    pair-type embedding to distinguish them.  Common pairs still get their own
    dedicated :class:`PairwiseExpert`.  A routing summary is logged at init.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    irreps_in : Irreps or str
        Irreps of the per-atom features fed into the MoE.
    expert_config : ExpertConfig
        Config shared by all expert modules.
    cutoff : float
        Cosine-cutoff radius (Angstrom) applied to each pair energy.
    pair_counts : dict, optional
        ``{(Z_A, Z_B): edge_count}`` over the training set.  Required for
        data-driven routing (used when ``rare_pair_enabled=True``).
    rare_pair_enabled : bool
        When ``True`` and ``pair_counts`` is provided, apply frequency-based
        routing.  Pairs below ``min_pair_frequency`` go to
        :class:`RarePairExpert`.
    min_pair_frequency : float
        Minimum fraction of total edges for a pair to get a dedicated expert.
        Pairs below this threshold are handled by :class:`RarePairExpert`.
    pair_symbols : optional mapping
        Optional override for the pretty pair labels.  Purely cosmetic.

    Notes
    -----
    The block emits **per-atom** scalar energies.  Neighbour lists are
    bidirectional (both ``(i, j)`` and ``(j, i)`` edges are present),
    so the standard MLIP convention is followed: a directed edge
    ``(i, j)`` contributes ``E_pair / 2`` to atom ``i`` only.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_in: Irreps | str,
        expert_config: ExpertConfig,
        cutoff: float = 5.0,
        pair_counts: typing.Mapping[tuple[int, int], int] | None = None,
        rare_pair_enabled: bool = False,
        min_pair_frequency: float = 0.01,
        pair_symbols: typing.Mapping[int, str] | None = None,
    ) -> None:
        super().__init__()
        if len(elements) == 0:
            raise ValueError("`elements` must contain at least one atomic number.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._irreps_in: Irreps = Irreps(irreps_in)
        self._cutoff: float = cutoff
        self._symbols: typing.Mapping[int, str] = pair_symbols or _DEFAULT_SYMBOLS
        self._rare_pair_enabled: bool = bool(rare_pair_enabled)

        # Enumerate all pairs deterministically
        pairs: list[tuple[int, int]] = enumerate_element_pairs(self._elements)
        self._pairs: tuple[tuple[int, int], ...] = tuple(pairs)

        # Data-driven routing (CHANGE 4)
        dedicated_pairs: list[tuple[int, int]] = []
        rare_pairs: list[tuple[int, int]] = []

        if rare_pair_enabled and pair_counts is not None:
            total_edges: int = max(1, sum(pair_counts.values()))
            for p in pairs:
                cnt: int = int(pair_counts.get(p, pair_counts.get((p[1], p[0]), 0)))
                freq: float = cnt / total_edges
                if freq >= min_pair_frequency:
                    dedicated_pairs.append(p)
                else:
                    rare_pairs.append(p)
        else:
            dedicated_pairs = list(pairs)

        self._dedicated_pairs: tuple[tuple[int, int], ...] = tuple(dedicated_pairs)
        self._rare_pairs: tuple[tuple[int, int], ...] = tuple(rare_pairs)

        # Build dedicated experts
        experts: dict[str, PairwiseExpert] = {}
        self._pair_keys: list[str] = []
        for a, b in self._dedicated_pairs:
            key: str = self._key(a, b)
            self._pair_keys.append(key)
            experts[key] = PairwiseExpert(self._irreps_in, expert_config)
        self.experts: nn.ModuleDict = nn.ModuleDict(experts)

        # Build shared rare-pair expert (None when no rare pairs)
        self.rare_expert: RarePairExpert | None = (
            RarePairExpert(self._irreps_in, list(rare_pairs), expert_config)
            if rare_pairs
            else None
        )

        # Log routing table at construction time
        self._log_routing_table(pair_counts, min_pair_frequency)

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(a: int, b: int) -> str:
        lo, hi = sorted((a, b))
        return f"z{lo}_z{hi}"

    def pair_label(self, a: int, b: int) -> str:
        return pair_label(a, b, self._symbols)

    # ------------------------------------------------------------------
    # Routing-table logger
    # ------------------------------------------------------------------

    def _log_routing_table(
        self,
        pair_counts: typing.Mapping[tuple[int, int], int] | None,
        min_pair_frequency: float,
    ) -> None:
        """Print expert routing summary using rich (falls back to plain text)."""
        from lightning.pytorch.utilities.rank_zero import rank_zero_info

        total_edges: int = max(1, sum(pair_counts.values())) if pair_counts is not None else 0
        rare_labels: list[str] = [self.pair_label(a, b) for a, b in self._rare_pairs]

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title="[bold cyan]KRONOS Expert Routing[/bold cyan]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Pair", style="cyan", no_wrap=True)
            table.add_column("Count", justify="right")
            table.add_column("Frequency", justify="right")
            table.add_column("Routing", justify="left")

            for a, b in self._pairs:
                lbl = self.pair_label(a, b)
                cnt: int = 0
                freq_str: str = "n/a"
                if pair_counts is not None:
                    cnt = int(pair_counts.get((a, b), pair_counts.get((b, a), 0)))
                    freq = cnt / total_edges
                    freq_str = f"{freq*100:.2f}%"

                is_rare = (a, b) in self._rare_pairs
                if is_rare:
                    routing = f"[yellow]→ RarePairExpert[/yellow]"
                else:
                    routing = "[green]PairwiseExpert (dedicated)[/green]"
                table.add_row(lbl, str(cnt), freq_str, routing)

            import io

            buf = io.StringIO()
            Console(file=buf, no_color=True, width=90).print(table)
            rank_zero_info("\n" + buf.getvalue())

        except ImportError:
            # Plain-text fallback
            lines: list[str] = ["KRONOS Expert Routing:"]
            lines.append(f"  {'Pair':<10} {'Count':>8} {'Freq':>8}  Routing")
            lines.append("  " + "-" * 50)
            for a, b in self._pairs:
                lbl = self.pair_label(a, b)
                cnt = 0
                freq_str = "n/a"
                if pair_counts is not None:
                    cnt = int(pair_counts.get((a, b), pair_counts.get((b, a), 0)))
                    freq = cnt / total_edges
                    freq_str = f"{freq*100:.2f}%"
                is_rare = (a, b) in self._rare_pairs
                routing = "→ RarePairExpert" if is_rare else "PairwiseExpert (dedicated)"
                lines.append(f"  {lbl:<10} {cnt:>8} {freq_str:>8}  {routing}")
            lines.append("")
            rank_zero_info("\n".join(lines))

        # Summary line
        n_dedicated = len(self._dedicated_pairs)
        n_rare = len(self._rare_pairs)
        if n_rare > 0:
            rank_zero_info(
                f"[KRONOS MoE] {n_dedicated} dedicated PairwiseExperts | "
                f"{n_rare} rare pair(s) handled by RarePairExpert: "
                f"{', '.join(rare_labels)}"
            )
        else:
            rank_zero_info(
                f"[KRONOS MoE] {n_dedicated} dedicated PairwiseExperts "
                f"(no rare-pair routing active)"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_experts(self) -> int:
        """Total number of expert modules (dedicated + 1 shared if rare pairs exist)."""
        return len(self._dedicated_pairs) + (1 if self.rare_expert is not None else 0)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def gates(self) -> dict[str, torch.Tensor]:
        """Return a dict ``{pair_label: gate_value}`` for logging."""
        result: dict[str, torch.Tensor] = {
            self.pair_label(a, b): self.experts[self._key(a, b)].gate.detach()
            for a, b in self._dedicated_pairs
        }
        if self.rare_expert is not None:
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                result[self.pair_label(a, b)] = self.rare_expert.gates[local_idx].detach()
        return result

    @torch.no_grad()
    def compute_expert_loads(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute mean absolute energy contribution per expert over a batch.

        For each expert ``E_{AB}``:
            ``load_AB = mean(|P_AB * E_AB_output(masked_inputs)|)``

        averaged over all edges in the batch (masked edges contribute 0 and
        are included in the denominator, consistent with the forward pass).

        Returns
        -------
        dict
            ``{pair_label: scalar_tensor}`` of per-expert loads (detached).
            Also includes ``"load_variance"`` — the coefficient of variation
            ``std(loads) / mean(loads)`` across all experts.  A value > 2.0
            indicates expert collapse.  When E == 0 all loads are 0 and
            ``load_variance`` is 0.
        """
        dtype: torch.dtype = atom_features.dtype
        device: torch.device = atom_features.device
        E: int = int(edge_lengths.shape[0])

        result: dict[str, torch.Tensor] = {}

        if E == 0:
            for a, b in self._pairs:
                result[self.pair_label(a, b)] = torch.zeros((), device=device, dtype=dtype)
            result["load_variance"] = torch.zeros((), device=device, dtype=dtype)
            return result

        edge_lengths_dt = edge_lengths.to(dtype)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths_dt, self._cutoff)
        row, col = edge_index
        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        load_values: list[torch.Tensor] = []
        for (a, b), key in zip(self._pairs, self._pair_keys):
            expert = typing.cast(PairwiseExpert, self.experts[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_col = mask.unsqueeze(-1)

            scalars: torch.Tensor = expert.project(atom_features)
            scalars_a: torch.Tensor = scalars[row] * mask_col
            scalars_b: torch.Tensor = scalars[col] * mask_col
            dist_in: torch.Tensor = edge_lengths_dt * mask

            pair_e: torch.Tensor = expert(scalars_a, scalars_b, dist_in)
            tapered: torch.Tensor = pair_e * mask * cut_env
            load: torch.Tensor = tapered.abs().mean()
            result[self.pair_label(a, b)] = load
            load_values.append(load)

        loads_t: torch.Tensor = torch.stack(load_values)
        mean_load: torch.Tensor = loads_t.mean().clamp_min(1e-12)
        result["load_variance"] = loads_t.std() / mean_load
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        return_per_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-atom scalar energy contributions.

        Static-shape schedule:

        1. For every expert build a per-edge **float** mask selecting
           edges of its pair type.
        2. Multiply the (scalar, distance) inputs by the mask *before*
           the forward.
        3. Multiply the expert output by the mask *again* before adding
           to ``atom_energies`` — non-matching edges contribute
           **exactly** ``0.0`` to the total energy.

        Parameters
        ----------
        atom_features : Tensor
            Dressed per-atom equivariant features, shape
            ``(N, irreps_in.dim)``.
        atomic_numbers : Tensor
            Atomic numbers ``(N,)`` int64.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_lengths : Tensor
            Pair distances ``(E,)``.
        return_per_pair : bool
            If ``True``, return ``(atom_energies, per_pair_total_energy)``
            where ``per_pair_total_energy`` is a dict keyed by
            ``pair_label`` whose values are detached scalar tensors —
            useful for testing the zero-mask correctness.

        Returns
        -------
        Tensor or tuple
            Per-atom scalar energy ``(N,)``, optionally followed by a
            dict of detached per-pair total contributions.
        """
        num_atoms: int = atom_features.shape[0]
        device: torch.device = atom_features.device
        dtype: torch.dtype = atom_features.dtype

        row, col = edge_index  # both shape (E,)
        E: int = int(edge_lengths.shape[0])

        # Boundary cast: align edge_lengths with the per-atom feature dtype
        # so the downstream concat in ``_LinearExpert`` (scalars ⊕ distance)
        # and the per-edge mask multiplication stay in a single precision.
        # Same MACE / NequIP pattern as ``radial.RadialMLP`` / interaction.
        edge_lengths = edge_lengths.to(dtype)

        # Accumulator for per-atom interaction energy.  The per-element
        # baseline lives on the ``KronosBackbone`` as a fixed buffer
        # (``atomic_energies``, computed from data via least-squares
        # regression) and is added there — the MoE only models the
        # local interaction residual.
        atom_energies: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)
        per_pair_total: dict[str, torch.Tensor] = {}

        # ----- Degenerate case: no edges at all -----
        if E == 0:
            # Keep every expert in the autograd graph via a zero-contribution dummy.
            first_key: str = self._pair_keys[0] if self._pair_keys else ""
            scalar_channels: int = (
                self.experts[first_key].scalar_channels
                if first_key
                else (self.rare_expert.scalar_channels if self.rare_expert else 1)
            )
            dummy_scalars: torch.Tensor = torch.zeros(
                1, scalar_channels, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
                expert = typing.cast(PairwiseExpert, self.experts[key])
                _ = expert.project(atom_features)
                dummy_out: torch.Tensor = expert(dummy_scalars, dummy_scalars, dummy_dist)
                atom_energies = atom_energies + 0.0 * dummy_out.sum()
                if return_per_pair:
                    per_pair_total[self.pair_label(a, b)] = torch.zeros(
                        (), device=device, dtype=dtype
                    )
            if self.rare_expert is not None:
                _ = self.rare_expert.project(atom_features)
                dummy_idx: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
                dummy_re: torch.Tensor = self.rare_expert(
                    dummy_scalars, dummy_scalars, dummy_dist, dummy_idx
                )
                atom_energies = atom_energies + 0.0 * dummy_re.sum()
                for a, b in self._rare_pairs:
                    if return_per_pair:
                        per_pair_total[self.pair_label(a, b)] = torch.zeros(
                            (), device=device, dtype=dtype
                        )
            return (atom_energies, per_pair_total) if return_per_pair else atom_energies

        # ----- Normal case: E > 0 -----
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # ---- Dedicated pair loop ----
        for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
            expert = typing.cast(PairwiseExpert, self.experts[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_col: torch.Tensor = mask.unsqueeze(-1)
            scalars: torch.Tensor = expert.project(atom_features)
            scalars_a: torch.Tensor = scalars[row] * mask_col
            scalars_b: torch.Tensor = scalars[col] * mask_col
            dist_in: torch.Tensor = edge_lengths * mask
            pair_e: torch.Tensor = expert(scalars_a, scalars_b, dist_in)
            tapered: torch.Tensor = pair_e * mask * cut_env
            atom_energies = atom_energies.index_add(0, row, 0.5 * tapered)
            if return_per_pair:
                per_pair_total[self.pair_label(a, b)] = 0.5 * tapered.sum().detach()

        # ---- Rare pairs via shared RarePairExpert ----
        if self.rare_expert is not None:
            rare_scalars: torch.Tensor = self.rare_expert.project(atom_features)  # (N, S)
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                mask = ((z_lo == a) & (z_hi == b)).to(dtype)
                mask_col = mask.unsqueeze(-1)
                sa: torch.Tensor = rare_scalars[row] * mask_col
                sb: torch.Tensor = rare_scalars[col] * mask_col
                di: torch.Tensor = edge_lengths * mask
                pidx: torch.Tensor = torch.full((E,), local_idx, dtype=torch.long, device=device)
                pair_e = self.rare_expert(sa, sb, di, pidx)
                tapered = pair_e * mask * cut_env
                atom_energies = atom_energies.index_add(0, row, 0.5 * tapered)
                if return_per_pair:
                    per_pair_total[self.pair_label(a, b)] = 0.5 * tapered.sum().detach()

        return (atom_energies, per_pair_total) if return_per_pair else atom_energies

    # ------------------------------------------------------------------
    # Pairwise force mode (TASK 4)
    # ------------------------------------------------------------------

    def forward_pairwise(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-atom energies AND per-atom forces via pairwise mode.

        Each :class:`PairwiseExpert` produces ``(E_ij, F_ij)`` with
        ``F_ij = -∂E_ij/∂r_ij`` via autograd; the MoE applies the
        per-pair mask and cosine-cutoff envelope and scatter-sums to
        atoms with Newton's third law (``+0.5 F`` to row, ``-0.5 F``
        to col).  The factor of ``0.5`` accounts for the bidirectional
        edge convention — each undirected pair appears twice as a
        directed edge.

        Returns
        -------
        tuple of Tensor
            ``(atom_energies, atom_forces)`` with shapes ``(N,)`` and
            ``(N, 3)``.  Newton's third law: ``sum_i F_i ≡ 0`` per
            molecule by construction.
        """
        num_atoms: int = atom_features.shape[0]
        device: torch.device = atom_features.device
        dtype: torch.dtype = atom_features.dtype

        row, col = edge_index
        E: int = int(edge_vectors.shape[0])

        atom_energies: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)
        atom_forces: torch.Tensor = torch.zeros((num_atoms, 3), device=device, dtype=dtype)

        # Degenerate case: keep all experts in the autograd graph.
        if E == 0:
            first_key: str = self._pair_keys[0] if self._pair_keys else ""
            scalar_channels: int = (
                self.experts[first_key].scalar_channels
                if first_key
                else (self.rare_expert.scalar_channels if self.rare_expert else 1)
            )
            dummy_scalars: torch.Tensor = torch.zeros(
                1, scalar_channels, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            for _, key in zip(self._dedicated_pairs, self._pair_keys):
                expert = typing.cast(PairwiseExpert, self.experts[key])
                _ = expert.project(atom_features)
                dummy_out: torch.Tensor = expert(dummy_scalars, dummy_scalars, dummy_dist)
                atom_energies = atom_energies + 0.0 * dummy_out.sum()
            if self.rare_expert is not None:
                _ = self.rare_expert.project(atom_features)
                dummy_idx_re: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
                dummy_re: torch.Tensor = self.rare_expert(
                    dummy_scalars, dummy_scalars, dummy_dist, dummy_idx_re
                )
                atom_energies = atom_energies + 0.0 * dummy_re.sum()
            return atom_energies, atom_forces

        edge_vectors_dt: torch.Tensor = edge_vectors.to(dtype)
        edge_lengths: torch.Tensor = edge_vectors_dt.norm(dim=-1)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)
        cut_env_v: torch.Tensor = cut_env.unsqueeze(-1)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # ---- Dedicated pair loop ----
        for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
            expert = typing.cast(PairwiseExpert, self.experts[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_v: torch.Tensor = mask.unsqueeze(-1)
            scalars: torch.Tensor = expert.project(atom_features)
            scalars_a: torch.Tensor = scalars[row] * mask_v
            scalars_b: torch.Tensor = scalars[col] * mask_v
            e_ij, f_ij = expert.forward_pairwise(scalars_a, scalars_b, edge_vectors_dt)
            tapered_e: torch.Tensor = e_ij * mask * cut_env
            tapered_f: torch.Tensor = f_ij * mask_v * cut_env_v
            atom_energies = atom_energies.index_add(0, row, 0.5 * tapered_e)
            atom_forces = atom_forces.index_add(0, row, 0.5 * tapered_f)
            atom_forces = atom_forces.index_add(0, col, -0.5 * tapered_f)

        # ---- Rare pairs via shared RarePairExpert (energy only, no pairwise forces) ----
        # RarePairExpert doesn't implement forward_pairwise; use regular forward
        # and let the backbone's autograd head derive forces from positions.
        if self.rare_expert is not None:
            rare_scalars: torch.Tensor = self.rare_expert.project(atom_features)
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                mask = ((z_lo == a) & (z_hi == b)).to(dtype)
                mask_col_re = mask.unsqueeze(-1)
                sa_re: torch.Tensor = rare_scalars[row] * mask_col_re
                sb_re: torch.Tensor = rare_scalars[col] * mask_col_re
                di_re: torch.Tensor = edge_lengths * mask
                pidx_re: torch.Tensor = torch.full(
                    (E,), local_idx, dtype=torch.long, device=device
                )
                pair_e_re: torch.Tensor = self.rare_expert(sa_re, sb_re, di_re, pidx_re)
                tapered_re: torch.Tensor = pair_e_re * mask * cut_env
                atom_energies = atom_energies.index_add(0, row, 0.5 * tapered_re)

        return atom_energies, atom_forces
