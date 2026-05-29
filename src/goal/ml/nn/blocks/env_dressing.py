"""Environment dressing block for the KRONOS native model.

Produces "dressed" per-atom features that already carry information from
each atom's local neighbourhood, before they are fed into downstream
blocks (the KRONOS pairwise experts, scalar readouts, etc.).

Pipeline:

1. Look up an initial scalar embedding from the atomic number.
2. Project the scalars into an equivariant feature space (irreps
   ``hidden_channels x 0e + 1o + 2e ...``).
3. Build edge spherical harmonics ``Y^l(r_hat)`` and a radial expansion
   ``MLP(Bessel(d))`` weighted by a smooth envelope.
4. Run one (or several) ACE-style message-passing layer(s): a
   fully-connected, weighted tensor product between sender node
   features and edge spherical harmonics, aggregated by sum.  The
   output is the one-particle basis ``A_i`` of ACE.
5. Optional ACE body-order expansion:

   * ``body_order = 1`` (default) — output is ``A_i`` (3-body
     interactions via the message-passing tensor product).
   * ``body_order = 2`` — output is ``B²_i = A_i ⊗_CG A_i``
     (captures bond *angles*; 4-body in total).
   * ``body_order = 3`` — output is ``B³_i = B²_i ⊗_CG A_i``
     (captures *dihedrals*; 5-body in total).

The CG-product output irreps for each body order are derived
**programmatically** from the angular-momentum decomposition (see
:func:`cg_product_irreps`); they are *not* hard-coded.  An equivariant
``Linear`` then compresses the result back to the standard
``irreps_hidden`` shape so the downstream expert interface is
unchanged regardless of body order.

All sub-modules are taken from ``goal.ml.nn.{primitives,blocks}`` so the
dressing block reuses the equivariant primitives already shipped with
the project.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.interaction import EquivariantInteractionBlock
from goal.ml.nn.primitives.linear import EquivariantLinear


def build_hidden_irreps(hidden_channels: int, lmax: int) -> Irreps:
    """Build the standard KRONOS hidden irreps specification.

    Convention:

    - Even angular momenta ``l`` get parity ``e``.
    - Odd angular momenta ``l`` get parity ``o``.

    Parameters
    ----------
    hidden_channels : int
        Multiplicity of every angular-momentum block (e.g. 32 → ``32x0e``).
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

    Used by KRONOS to derive the ACE body-order output irreps
    programmatically — never hard-coded — so the shape of ``B²`` and
    ``B³`` is always exactly what the angular-momentum algebra dictates.

    Parameters
    ----------
    irreps1, irreps2 : Irreps
        Input irreps.
    lmax : int
        Drop output irreps with ``l > lmax`` to keep tensor widths
        bounded.

    Returns
    -------
    Irreps
        Simplified irreps of the truncated CG product.  Multiplicities
        coming from different ``(mul1 × mul2)`` contributions are
        summed by ``Irreps.simplify``.
    """
    out_list: list[tuple[int, tuple[int, int]]] = []
    for mul1, ir1 in irreps1:
        for mul2, ir2 in irreps2:
            for ir_out in ir1 * ir2:
                if ir_out.l <= lmax:
                    out_list.append((mul1 * mul2, (ir_out.l, ir_out.p)))
    return Irreps(out_list).simplify()


class EnvironmentDressing(nn.Module):
    """Equivariant environment-dressing block with optional ACE body-order expansion.

    Parameters
    ----------
    num_elements : int
        Size of the atomic-number embedding table.  Must be large enough
        to cover the maximum ``Z`` in the dataset (e.g. ``9`` is enough
        for H/C/N/O, ``120`` covers the whole periodic table).
    embedding_dim : int
        Width of the initial scalar embedding.
    hidden_channels : int
        Multiplicity used for every ``l`` block in the hidden features
        and (after compression) in the output features.
    lmax : int
        Highest spherical-harmonic order ``l`` to include.
    num_radial_basis : int
        Number of Bessel basis functions used by the radial MLP.
    cutoff : float
        Cutoff radius (Angstrom).  Shared between the radial basis and
        the polynomial envelope inside the interaction block.
    radial_mlp_hidden : int
        Width of the hidden layer inside the radial MLP that produces
        tensor-product weights.
    num_message_passing : int
        Number of message-passing rounds producing the one-particle
        basis ``A_i``.  Defaults to ``1``.
    body_order : int
        ACE body-order parameter (``1``, ``2`` or ``3``).  See module
        docstring for the geometric interpretation.
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
        num_message_passing: int = 1,
        body_order: int = 1,
    ) -> None:
        super().__init__()
        if body_order not in self.SUPPORTED_BODY_ORDERS:
            raise ValueError(
                f"body_order must be one of {self.SUPPORTED_BODY_ORDERS}, got {body_order}."
            )

        self._cutoff: float = cutoff
        self._hidden_channels: int = hidden_channels
        self._lmax: int = lmax
        self._body_order: int = body_order
        self._irreps_hidden: Irreps = build_hidden_irreps(hidden_channels, lmax)
        self._irreps_edge: Irreps = build_edge_irreps(lmax)

        # Scalar atomic embedding → equivariant features
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        scalar_irreps: Irreps = Irreps(f"{embedding_dim}x0e")
        self.input_linear: EquivariantLinear = EquivariantLinear(
            scalar_irreps, self._irreps_hidden
        )

        # Stack of ACE-style interaction blocks producing the one-particle
        # basis A_i.
        self.interactions: nn.ModuleList = nn.ModuleList(
            [
                EquivariantInteractionBlock(
                    irreps_node=self._irreps_hidden,
                    irreps_edge=self._irreps_edge,
                    num_basis=num_radial_basis,
                    cutoff=cutoff,
                    hidden_dim=radial_mlp_hidden,
                )
                for _ in range(num_message_passing)
            ]
        )

        # ----- Body-order expansion -----
        #
        # B² and B³ are built by tensor-producting A with itself / with
        # the previous B; the output irreps come from the CG algebra
        # (no hard-coded shapes) and are then compressed back to
        # ``irreps_hidden`` so the expert interface stays unchanged.
        self.tp_b2: FullyConnectedTensorProduct | None = None
        self.proj_b2: EquivariantLinear | None = None
        self.tp_b3: FullyConnectedTensorProduct | None = None
        self.proj_b3: EquivariantLinear | None = None
        self._irreps_b2_full: Irreps | None = None
        self._irreps_b3_full: Irreps | None = None

        if body_order >= 2:
            self._irreps_b2_full = cg_product_irreps(
                self._irreps_hidden, self._irreps_hidden, lmax
            )
            self.tp_b2 = FullyConnectedTensorProduct(
                irreps_in1=self._irreps_hidden,
                irreps_in2=self._irreps_hidden,
                irreps_out=self._irreps_b2_full,
            )
            self.proj_b2 = EquivariantLinear(self._irreps_b2_full, self._irreps_hidden)

        if body_order >= 3:
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
        """Output irreps of the dressed atom features (unchanged by body order)."""
        return self._irreps_hidden

    @property
    def cutoff(self) -> float:
        """Effective cutoff radius (Angstrom)."""
        return self._cutoff

    @property
    def lmax(self) -> int:
        """Highest spherical-harmonic order used."""
        return self._lmax

    @property
    def hidden_channels(self) -> int:
        """Multiplicity per ``l`` block."""
        return self._hidden_channels

    @property
    def body_order(self) -> int:
        """ACE body-order parameter (``1``, ``2`` or ``3``)."""
        return self._body_order

    @property
    def irreps_b2_full(self) -> Irreps | None:
        """Internal (pre-compression) irreps of the B² tensor; ``None`` when ``body_order < 2``."""
        return self._irreps_b2_full

    @property
    def irreps_b3_full(self) -> Irreps | None:
        """Internal (pre-compression) irreps of the B³ tensor; ``None`` when ``body_order < 3``."""
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
    ) -> torch.Tensor:
        """Run dressing including the ACE body-order expansion.

        Parameters
        ----------
        atomic_numbers : Tensor
            Integer atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_vectors : Tensor
            Edge displacement vectors ``(E, 3)``.
        edge_lengths : Tensor
            Edge lengths ``(E,)``.

        Returns
        -------
        Tensor
            Dressed equivariant node features ``(N, irreps_out.dim)``.
        """
        h: torch.Tensor = self.embedding(atomic_numbers)  # (N, embedding_dim)
        h = self.input_linear(h)  # (N, irreps_hidden.dim)
        for interaction in self.interactions:
            h = interaction(
                node_feats=h,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
            )
        # ``h`` is now A_i, the one-particle basis.

        if self._body_order == 1:
            return h

        # B² = A ⊗_CG A, then compress to irreps_hidden
        b2_full: torch.Tensor = self.tp_b2(h, h)  # (N, irreps_b2_full.dim)
        b2: torch.Tensor = self.proj_b2(b2_full)  # (N, irreps_hidden.dim)

        if self._body_order == 2:
            return b2

        # B³ = B² ⊗_CG A, then compress
        b3_full: torch.Tensor = self.tp_b3(b2, h)  # (N, irreps_b3_full.dim)
        b3: torch.Tensor = self.proj_b3(b3_full)  # (N, irreps_hidden.dim)
        return b3
