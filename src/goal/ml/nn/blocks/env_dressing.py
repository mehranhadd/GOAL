"""Environment dressing block for the SIMURGH native model.

Produces "dressed" per-atom features that already carry information from
each atom's local neighbourhood, before they are fed into downstream
blocks (the SIMURGH pairwise experts, scalar readouts, etc.).

Pipeline
--------
1. Look up an initial scalar embedding from the atomic number.
2. Project the scalars into an equivariant feature space (irreps
   ``hidden_channels x 0e + 1o + 2e ...``).
3. **Compute edge spherical harmonics once** from the edge vectors —
   these are shared across all interaction layers (they depend only on
   geometry, not on learnable parameters).
4. Compute the radial basis + envelope per edge — also geometry-only
   and reused across layers.
5. Run ``num_interactions`` GNN-style message-passing layers:
   each layer uses the **previous layer's output** as neighbour features
   (multi-hop, not single-hop ACE).  Layer ``l`` receives the pre-computed
   SH and produces updated node features with a residual connection.
   Optionally, the TP weights in each layer are modulated by the central
   atom's element type (``element_conditioned=True``).
6. Optional per-layer energy readout: attach a small MLP after each
   interaction layer that maps node features → per-node scalar energy.
   Summing all per-layer contributions gives a body-order decomposition
   of the energy (MACE-style).
7. Optional ACE body-order expansion on the *final* layer's features:

   * ``body_order = 1`` — output is the final ``h_L`` directly.
   * ``body_order = 2`` — output is ``B² = h_L ⊗_CG h_L``.
   * ``body_order = 3`` — output is ``B³ = B² ⊗_CG h_L``.

All sub-modules are taken from ``goal.ml.nn.{primitives,blocks}`` so the
dressing block reuses the equivariant primitives already shipped with
the project.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, spherical_harmonics

from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.interaction import EquivariantInteractionBlock
from goal.ml.nn.blocks.symmetric_contraction import SymmetricContraction
from goal.ml.nn.primitives.linear import EquivariantLinear


def build_hidden_irreps(hidden_channels: int, lmax: int) -> Irreps:
    """Build the standard SIMURGH hidden irreps specification.

    Convention: even ``l`` → parity ``e``; odd ``l`` → parity ``o``.

    Parameters
    ----------
    hidden_channels : int
        Multiplicity of every angular-momentum block.
    lmax : int
        Largest angular momentum (inclusive).
    """
    parts: list[str] = []
    for l_val in range(lmax + 1):
        parity: str = "e" if l_val % 2 == 0 else "o"
        parts.append(f"{hidden_channels}x{l_val}{parity}")
    return Irreps("+".join(parts))


def build_edge_irreps(lmax: int) -> Irreps:
    """Spherical-harmonics irreps for edges, one channel per ``l``."""
    parts: list[str] = []
    for l_val in range(lmax + 1):
        parity: str = "e" if l_val % 2 == 0 else "o"
        parts.append(f"1x{l_val}{parity}")
    return Irreps("+".join(parts))


def cg_product_irreps(
    irreps1: Irreps,
    irreps2: Irreps,
    lmax: int,
) -> Irreps:
    """Irreps of the Clebsch-Gordan product ``irreps1 ⊗ irreps2``, truncated to ``l ≤ lmax``.

    Parameters
    ----------
    irreps1, irreps2 : Irreps
        Input irreps.
    lmax : int
        Drop output irreps with ``l > lmax`` to keep tensor widths bounded.

    Returns
    -------
    Irreps
        Simplified irreps of the truncated CG product.
    """
    out_list: list[tuple[int, tuple[int, int]]] = []
    for mul1, ir1 in irreps1:
        for mul2, ir2 in irreps2:
            for ir_out in ir1 * ir2:
                if ir_out.l <= lmax:
                    out_list.append((mul1 * mul2, (ir_out.l, ir_out.p)))
    return Irreps(out_list).simplify()


class _LayerReadout(nn.Module):
    """Per-layer scalar energy readout.

    Extracts the ``l=0`` scalars from equivariant node features and passes
    them through a 2-layer MLP to produce a per-node scalar energy.

    Architecture (mirrors MACE's LinearReadoutBlock for early layers):
        EquivariantLinear(irreps → scalars) → Linear(S, S//2) → SiLU → Linear(S//2, 1)
    """

    def __init__(self, irreps_in: Irreps) -> None:
        super().__init__()
        num_scalars: int = sum(mul for mul, ir in irreps_in if ir.l == 0)
        scalar_irreps: Irreps = Irreps(f"{num_scalars}x0e")
        self.to_scalars = EquivariantLinear(irreps_in, scalar_irreps)
        hidden: int = max(1, num_scalars // 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """``(N, irreps.dim) → (N,)`` per-node scalar energy."""
        scalars = self.to_scalars(node_feats)  # (N, num_scalars)
        return self.mlp(scalars).squeeze(-1)  # (N,)


class EnvironmentDressing(nn.Module):
    """Equivariant environment-dressing block with GNN-style multi-layer MP.

    Parameters
    ----------
    num_elements : int
        Size of the atomic-number embedding table.
    embedding_dim : int
        Width of the initial scalar embedding.
    hidden_channels : int
        Multiplicity per ``l`` block.
    lmax : int
        Highest spherical-harmonic order ``l`` to include.
    num_radial_basis : int
        Number of Bessel basis functions.
    cutoff : float
        Cutoff radius (Angstrom).
    radial_mlp_hidden : int
        Width of hidden layers in the radial MLP.
    num_interactions : int
        Number of GNN-style message-passing layers.  Each layer uses the
        previous layer's output as neighbour features (multi-hop).
        Also accepted as ``num_message_passing`` for backward compatibility.
    body_order : int
        ACE body-order parameter applied to the *final* layer's output
        (``1``, ``2`` or ``3``).
    avg_num_neighbors : float, optional
        Mean degree at ``cutoff`` over the training set.
    agg_norm_exponent : float, optional
        Exponent for the aggregation normalisation (see
        :class:`EquivariantInteractionBlock`).
    element_conditioned : bool, optional
        When ``True``, each interaction layer conditions its TP weights on
        the destination atom's element type via a
        ``Linear(n_elements, weight_numel, bias=False)`` map.  This is the
        CHANGE-1 element-conditioning described in the SIMURGH design doc.
        Backward-compatible default is ``False``.
    per_layer_readout : bool, optional
        When ``True``, attach a :class:`_LayerReadout` after each
        interaction layer and accumulate per-node energy contributions.
        The accumulated sum is returned alongside the final equivariant
        features; it is then added to the backbone's total energy on top
        of the artisan bank contribution.  Default ``False`` (original behaviour).
    symmetric_contraction : bool, optional
        When ``True`` and ``body_order >= 2``, use MACE-style
        :class:`SymmetricContraction` for the body-order expansion.
    """

    SUPPORTED_BODY_ORDERS: tuple[int, ...] = (1, 2, 3)

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 32,
        hidden_channels: int = 32,
        lmax: int = 2,
        num_radial_basis: int = 8,
        cutoff: float = 5.0,
        radial_mlp_hidden: int = 32,
        num_interactions: int = 2,
        # backward-compat alias — takes priority only when num_interactions is
        # at its default value and the old key is explicitly passed.
        num_message_passing: int | None = None,
        body_order: int = 1,
        avg_num_neighbors: float | None = None,
        agg_norm_exponent: float = 1.0,
        element_conditioned: bool = False,
        per_layer_readout: bool = False,
        symmetric_contraction: bool = False,
    ) -> None:
        super().__init__()
        if body_order not in self.SUPPORTED_BODY_ORDERS:
            raise ValueError(
                f"body_order must be one of {self.SUPPORTED_BODY_ORDERS}, got {body_order}."
            )

        # Honour the legacy ``num_message_passing`` key when present.
        if num_message_passing is not None:
            num_interactions = int(num_message_passing)

        self._cutoff: float = cutoff
        self._hidden_channels: int = hidden_channels
        self._lmax: int = lmax
        self._body_order: int = body_order
        self._irreps_hidden: Irreps = build_hidden_irreps(hidden_channels, lmax)
        self._irreps_edge: Irreps = build_edge_irreps(lmax)
        self._element_conditioned: bool = bool(element_conditioned)
        self._per_layer_readout: bool = bool(per_layer_readout)
        self._num_interactions: int = int(num_interactions)

        # Initial scalar embedding → equivariant feature space
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        scalar_irreps: Irreps = Irreps(f"{embedding_dim}x0e")
        self.input_linear: EquivariantLinear = EquivariantLinear(
            scalar_irreps, self._irreps_hidden
        )

        # GNN-style interaction layers (CHANGE 2: SH computed once outside)
        self._avg_num_neighbors: float | None = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )
        self._agg_norm_exponent: float = float(agg_norm_exponent)
        self.interactions: nn.ModuleList = nn.ModuleList(
            [
                EquivariantInteractionBlock(
                    irreps_node=self._irreps_hidden,
                    irreps_edge=self._irreps_edge,
                    num_basis=num_radial_basis,
                    cutoff=cutoff,
                    hidden_dim=radial_mlp_hidden,
                    avg_num_neighbors=self._avg_num_neighbors,
                    agg_norm_exponent=self._agg_norm_exponent,
                    element_conditioned=element_conditioned,
                    n_elements=num_elements,
                )
                for _ in range(self._num_interactions)
            ]
        )

        # Per-layer readouts (CHANGE 3)
        self.readouts: nn.ModuleList = nn.ModuleList(
            [_LayerReadout(self._irreps_hidden) for _ in range(self._num_interactions)]
            if per_layer_readout
            else []
        )

        # Body-order expansion on the final layer's output
        self._symmetric_contraction: bool = bool(symmetric_contraction)
        self.tp_b2: FullyConnectedTensorProduct | None = None
        self.proj_b2: EquivariantLinear | None = None
        self.tp_b3: FullyConnectedTensorProduct | None = None
        self.proj_b3: EquivariantLinear | None = None
        self.sym_contraction: SymmetricContraction | None = None
        self._irreps_b2_full: Irreps | None = None
        self._irreps_b3_full: Irreps | None = None

        if body_order >= 2 and self._symmetric_contraction:
            self.sym_contraction = SymmetricContraction(
                irreps_in=self._irreps_hidden,
                irreps_out=self._irreps_hidden,
                correlation=body_order,
                num_elements=num_elements,
            )

        if body_order >= 2 and not self._symmetric_contraction:
            self._irreps_b2_full = cg_product_irreps(
                self._irreps_hidden, self._irreps_hidden, lmax
            )
            self.tp_b2 = FullyConnectedTensorProduct(
                irreps_in1=self._irreps_hidden,
                irreps_in2=self._irreps_hidden,
                irreps_out=self._irreps_b2_full,
            )
            self.proj_b2 = EquivariantLinear(self._irreps_b2_full, self._irreps_hidden)

        if body_order >= 3 and not self._symmetric_contraction:
            self._irreps_b3_full = cg_product_irreps(
                self._irreps_hidden, self._irreps_hidden, lmax
            )
            self.tp_b3 = FullyConnectedTensorProduct(
                irreps_in1=self._irreps_hidden,
                irreps_in2=self._irreps_hidden,
                irreps_out=self._irreps_b3_full,
            )
            self.proj_b3 = EquivariantLinear(self._irreps_b3_full, self._irreps_hidden)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_hidden

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def hidden_channels(self) -> int:
        return self._hidden_channels

    @property
    def body_order(self) -> int:
        return self._body_order

    @property
    def symmetric_contraction(self) -> bool:
        return self._symmetric_contraction

    @property
    def element_conditioned(self) -> bool:
        return self._element_conditioned

    @property
    def per_layer_readout(self) -> bool:
        return self._per_layer_readout

    @property
    def irreps_b2_full(self) -> Irreps | None:
        return self._irreps_b2_full

    @property
    def irreps_b3_full(self) -> Irreps | None:
        return self._irreps_b3_full

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run dressing: GNN-style multi-layer MP + optional body-order expansion.

        Parameters
        ----------
        atomic_numbers : Tensor  ``(N,)``
        edge_index : Tensor      ``(2, E)``
        edge_vectors : Tensor    ``(E, 3)``
        edge_lengths : Tensor    ``(E,)``

        Returns
        -------
        tuple
            ``(node_feats, layer_energies)`` where:

            * ``node_feats`` — dressed equivariant features
              ``(N, irreps_out.dim)`` (after body-order expansion if any).
            * ``layer_energies`` — summed per-layer per-node scalar energies
              ``(N,)`` when ``per_layer_readout=True``, else ``None``.
        """
        h: torch.Tensor = self.embedding(atomic_numbers)  # (N, embedding_dim)
        h = self.input_linear(h)  # (N, irreps_hidden.dim)

        # ---- Compute edge SH once, reuse across all layers (CHANGE 2) ----
        ev = edge_vectors.to(h.dtype)
        edge_sh: torch.Tensor = spherical_harmonics(  # (E, irreps_edge.dim)
            self._irreps_edge,
            ev,
            normalize=True,
            normalization="component",
        )

        # ---- GNN-style interaction loop ----
        layer_energies: torch.Tensor | None = None

        for idx, interaction in enumerate(self.interactions):
            h = interaction(
                node_feats=h,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                edge_sh=edge_sh,  # pre-computed SH (CHANGE 2)
                atomic_numbers=atomic_numbers if self._element_conditioned else None,
            )

            # Per-layer readout contribution (CHANGE 3)
            if self._per_layer_readout and len(self.readouts) > 0:
                readout: _LayerReadout = typing.cast(_LayerReadout, self.readouts[idx])
                layer_e: torch.Tensor = readout(h)  # (N,)
                if layer_energies is None:
                    layer_energies = layer_e
                else:
                    layer_energies = layer_energies + layer_e

        # ---- Body-order expansion on final layer output ----
        if self._body_order == 1:
            return h, layer_energies

        if self._symmetric_contraction:
            assert self.sym_contraction is not None
            return self.sym_contraction(h, atomic_numbers), layer_energies

        assert self.tp_b2 is not None and self.proj_b2 is not None
        b2_full: torch.Tensor = self.tp_b2(h, h)
        b2: torch.Tensor = self.proj_b2(b2_full)

        if self._body_order == 2:
            return b2, layer_energies

        assert self.tp_b3 is not None and self.proj_b3 is not None
        b3_full: torch.Tensor = self.tp_b3(b2, h)
        return self.proj_b3(b3_full), layer_energies
