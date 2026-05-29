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
    """

    scalar_channels: int = 16
    hidden_dims: tuple[int, ...] = (64, 32)
    expert_type: str = "linear"
    transformer_heads: int = 2
    transformer_layers: int = 1


class _LinearExpert(nn.Module):
    """Standard MLP expert ``[in] → SiLU → ... → 1``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: typing.Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev: int = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
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


# ---------------------------------------------------------------------------
# MoE container
# ---------------------------------------------------------------------------


class KronosMoE(nn.Module):
    """KRONOS Element-Pair Mixture-of-Experts.

    Instantiates exactly one ``PairwiseExpert`` per unordered element
    pair derived from the ``elements`` list.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.  For example
        ``(1, 6, 7, 8)`` gives 10 unique unordered pairs:
        ``H-H, H-C, H-N, H-O, C-C, C-N, C-O, N-N, N-O, O-O``.
    irreps_in : Irreps or str
        Irreps of the per-atom features fed into the MoE.
    expert_config : ExpertConfig
        Static configuration for every expert (shared across pairs).
    cutoff : float
        Cosine-cutoff radius (Angstrom) applied to each pair energy.
    pair_symbols : optional mapping
        Optional override for the pretty pair labels (default uses
        H/C/N/O/...).  Purely cosmetic.

    Notes
    -----
    The block emits **per-atom** scalar energies.  Neighbour lists are
    bidirectional (both ``(i, j)`` and ``(j, i)`` edges are present),
    so the standard MLIP convention is followed: a directed edge
    ``(i, j)`` contributes ``E_pair / 2`` to atom ``i`` only.  Summing
    over all directed edges then recovers each undirected pair energy
    exactly once.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_in: Irreps | str,
        expert_config: ExpertConfig,
        cutoff: float = 5.0,
        pair_symbols: typing.Mapping[int, str] | None = None,
    ) -> None:
        super().__init__()
        if len(elements) == 0:
            raise ValueError("`elements` must contain at least one atomic number.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._irreps_in: Irreps = Irreps(irreps_in)
        self._cutoff: float = cutoff
        self._symbols: typing.Mapping[int, str] = pair_symbols or _DEFAULT_SYMBOLS

        # Enumerate pairs deterministically
        pairs: list[tuple[int, int]] = enumerate_element_pairs(self._elements)
        self._pairs: tuple[tuple[int, int], ...] = tuple(pairs)

        # Map pair → expert index (ModuleDict keyed by safe string name)
        experts: dict[str, PairwiseExpert] = {}
        self._pair_keys: list[str] = []
        for a, b in self._pairs:
            key: str = self._key(a, b)
            self._pair_keys.append(key)
            experts[key] = PairwiseExpert(self._irreps_in, expert_config)
        self.experts: nn.ModuleDict = nn.ModuleDict(experts)

        # Learnable per-element atomic-energy reference.  Added to every
        # atom's energy *outside* the cosine-cutoff envelope so the model
        # can represent the large constant per-Z offset present in raw
        # DFT labels (cf. MACE's AtomicEnergiesBlock, NequIP's
        # PerSpeciesShift).  Without this term every per-atom
        # contribution decays to zero at the cutoff and the loss
        # plateaus at the dataset's |E_target/n_atoms|.
        self.atomic_shift: nn.Parameter = nn.Parameter(torch.zeros(len(self._elements)))
        max_z: int = max(self._elements)
        z_to_idx: torch.Tensor = torch.full((max_z + 1,), -1, dtype=torch.long)
        for idx, z in enumerate(self._elements):
            z_to_idx[z] = idx
        self.register_buffer("_z_to_idx", z_to_idx, persistent=False)

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(a: int, b: int) -> str:
        """Deterministic, ModuleDict-safe key for an unordered pair."""
        lo, hi = sorted((a, b))
        return f"z{lo}_z{hi}"

    def pair_label(self, a: int, b: int) -> str:
        return pair_label(a, b, self._symbols)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_experts(self) -> int:
        return len(self._pairs)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def gates(self) -> dict[str, torch.Tensor]:
        """Return a dict ``{pair_label: gate_value}`` for logging."""
        return {
            self.pair_label(a, b): self.experts[self._key(a, b)].gate.detach()
            for a, b in self._pairs
        }

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

        # Accumulator for per-atom energies, seeded with the learnable
        # per-element shift (the only path that can express a non-zero
        # constant per-atom energy — every pair contribution downstream
        # is multiplied by ``cos_env(d)`` and therefore vanishes at the
        # cutoff).  Out-of-vocab atomic numbers (Z not in ``elements``)
        # map to ``-1`` in ``_z_to_idx`` and contribute zero shift.
        z_idx: torch.Tensor = self._z_to_idx[atomic_numbers]  # (N,)
        in_vocab: torch.Tensor = z_idx >= 0  # (N,)
        gathered: torch.Tensor = self.atomic_shift[z_idx.clamp(min=0)]  # (N,)
        shift_per_atom: torch.Tensor = (gathered * in_vocab.to(gathered.dtype)).to(
            device=device, dtype=dtype
        )
        atom_energies: torch.Tensor = shift_per_atom
        per_pair_total: dict[str, torch.Tensor] = {}

        # ----- Degenerate case: no edges at all -----
        if E == 0:
            first_key: str = self._pair_keys[0]
            scalar_channels: int = self.experts[first_key].scalar_channels
            dummy_scalars: torch.Tensor = torch.zeros(
                1, scalar_channels, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            for (a, b), key in zip(self._pairs, self._pair_keys):
                expert = typing.cast(PairwiseExpert, self.experts[key])
                _ = expert.project(atom_features)  # keep proj in the graph
                dummy_out: torch.Tensor = expert(dummy_scalars, dummy_scalars, dummy_dist)
                atom_energies = atom_energies + 0.0 * dummy_out.sum()
                if return_per_pair:
                    per_pair_total[self.pair_label(a, b)] = torch.zeros(
                        (), device=device, dtype=dtype
                    )
            return (atom_energies, per_pair_total) if return_per_pair else atom_energies

        # ----- Normal case: E > 0 -----
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)  # (E,)

        # Edge types (unordered): identify the pair by (min, max) atomic Z.
        z_row: torch.Tensor = atomic_numbers[row]  # (E,)
        z_col: torch.Tensor = atomic_numbers[col]  # (E,)
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)  # (E,)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)  # (E,)

        for (a, b), key in zip(self._pairs, self._pair_keys):
            expert = typing.cast(PairwiseExpert, self.experts[key])

            # Per-edge float mask: 1.0 for matching pair, 0.0 otherwise.
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)  # (E,)
            mask_col: torch.Tensor = mask.unsqueeze(-1)  # (E, 1) for broadcasting

            # Project EVERY atom's features to scalars — same for all experts'
            # input lookup; the scalar projection itself is per-expert.
            scalars: torch.Tensor = expert.project(atom_features)  # (N, S)

            # Apply input mask BEFORE the forward — never an ``if/else`` that
            # skips the expert.
            scalars_a: torch.Tensor = scalars[row] * mask_col  # (E, S)
            scalars_b: torch.Tensor = scalars[col] * mask_col  # (E, S)
            dist_in: torch.Tensor = edge_lengths * mask  # (E,)

            # Run expert on all E edges — static shape, no branching.
            pair_e: torch.Tensor = expert(scalars_a, scalars_b, dist_in)  # (E,)

            # Apply output mask so non-matching edges contribute EXACTLY 0.
            tapered: torch.Tensor = pair_e * mask * cut_env  # (E,)

            # Bidirectional edges → half-energy to row endpoint only.
            half: torch.Tensor = 0.5 * tapered
            atom_energies = atom_energies.index_add(0, row, half)

            if return_per_pair:
                # Each undirected pair contributes via two directed
                # edges; each edge adds ``0.5 * tapered`` to one
                # endpoint, so the pair's total contribution to the
                # graph energy is ``0.5 * tapered.sum()``.
                per_pair_total[self.pair_label(a, b)] = 0.5 * tapered.sum().detach()

        return (atom_energies, per_pair_total) if return_per_pair else atom_energies
