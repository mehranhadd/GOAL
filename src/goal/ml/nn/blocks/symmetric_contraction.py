"""Per-element body-order contraction (MACE-style ``SymmetricContraction``).

This module is the equivariant heart of the body-order expansion.  It
replaces the shared-weight ``FullyConnectedTensorProduct`` previously
used in :mod:`goal.ml.nn.blocks.env_dressing` for the ``B²`` / ``B³``
stages.  Two contracts:

1. **Per-element learnable couplings.**  Each body order ``n`` carries
   its own weight tensor of shape ``(num_elements, weight_numel_n)``,
   gathered by ``atomic_numbers`` at forward time.  This is the
   single most important architectural difference between MACE and
   the original ACE: it lets the network express element-specific
   coupling tensors for each body order, instead of forcing one
   shared tensor to fit every element.

2. **True body-order summation.**  The output is
   ``Σ_{n=1}^{N} W_n[Z] · CG_n(A)`` where ``CG_n(A)`` is the
   ``n``-fold Clebsch–Gordan contraction of the one-particle basis
   ``A`` (provided by the upstream interaction blocks).  ``N`` is the
   ``correlation`` argument; we support ``N ∈ {1, 2, 3}`` which
   covers MACE's standard ``correlation = 3`` setting.

The "symmetric" qualifier in the MACE name refers to the use of a
fully symmetrised CG basis for the ``n``-fold tensor power.  Our
implementation uses :class:`e3nn.o3.FullyConnectedTensorProduct`
iteratively, which is *not* the maximally symmetric construction but
**is** equivariance-correct and captures the same expressivity for a
modest weight-count overhead.  The equivariance tests in
``tests/ml/simurgh/test_symmetric_contraction.py`` enforce SO(3)
covariance over the dressing block's full output irreps.
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.o3 import Linear as O3Linear


class PerElementLinear(nn.Module):
    """Equivariant linear layer with per-element learnable weights.

    Wraps :class:`e3nn.o3.Linear` with ``internal_weights=False`` and
    ``shared_weights=False``, and stores a ``(num_elements,
    weight_numel)`` parameter table that is gathered per atom at
    forward time.  Initialisation uses ``1/sqrt(weight_numel)`` to
    keep output variance ``O(1)``.
    """

    def __init__(
        self,
        irreps_in: str | Irreps,
        irreps_out: str | Irreps,
        num_elements: int,
    ) -> None:
        super().__init__()
        self.irreps_in: Irreps = Irreps(irreps_in)
        self.irreps_out: Irreps = Irreps(irreps_out)
        self._lin: O3Linear = O3Linear(
            self.irreps_in,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        scale: float = 1.0 / math.sqrt(max(self._lin.weight_numel, 1))
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(num_elements, self._lin.weight_numel) * scale
        )

    @property
    def weight_numel(self) -> int:
        return self._lin.weight_numel

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the per-element linear.

        Parameters
        ----------
        x : Tensor
            Equivariant features, shape ``(N, irreps_in.dim)``.
        atomic_numbers : Tensor
            Integer atomic numbers, shape ``(N,)``.  Used to gather
            weights from the per-element table.
        """
        w: torch.Tensor = self.weight[atomic_numbers]
        return self._lin(x, w)


class PerElementWeightedTensorProduct(nn.Module):
    """:class:`FullyConnectedTensorProduct` with per-element weights.

    Same construction as :class:`WeightedTensorProduct` in
    :mod:`goal.ml.nn.primitives.tp` (``shared_weights=False,
    internal_weights=False``) but the per-edge weight tensor is
    replaced with a per-atom gather from a learnable
    ``(num_elements, weight_numel)`` parameter.
    """

    def __init__(
        self,
        irreps_in1: str | Irreps,
        irreps_in2: str | Irreps,
        irreps_out: str | Irreps,
        num_elements: int,
    ) -> None:
        super().__init__()
        self.irreps_in1: Irreps = Irreps(irreps_in1)
        self.irreps_in2: Irreps = Irreps(irreps_in2)
        self.irreps_out: Irreps = Irreps(irreps_out)
        self._tp: FullyConnectedTensorProduct = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            shared_weights=False,
            internal_weights=False,
        )
        scale: float = 1.0 / math.sqrt(max(self._tp.weight_numel, 1))
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(num_elements, self._tp.weight_numel) * scale
        )

    @property
    def weight_numel(self) -> int:
        return self._tp.weight_numel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom tensor product.

        Parameters
        ----------
        x1, x2 : Tensor
            Equivariant inputs, shapes ``(N, irreps_in*.dim)``.
        atomic_numbers : Tensor
            Integer atomic numbers, shape ``(N,)``.  Used to gather
            weights from the per-element table.
        """
        w: torch.Tensor = self.weight[atomic_numbers]
        return self._tp(x1, x2, w)


class SymmetricContraction(nn.Module):
    """Per-element body-order contraction over the one-particle basis.

    Computes::

        B(A_i, Z_i) = Σ_{n=1}^{N} W_n[Z_i] · CG_n(A_i)

    where:

    * ``A_i`` is the per-atom one-particle basis (equivariant features
      coming out of the upstream interaction block).
    * ``CG_n(·)`` is the ``n``-fold Clebsch–Gordan contraction.  For
      ``n = 1`` this is the identity (a per-element linear).  For
      ``n = 2`` it is ``A ⊗ A``; for ``n = 3`` it is ``(A ⊗ A) ⊗ A``,
      both implemented via :class:`PerElementWeightedTensorProduct`.
    * ``W_n[Z_i]`` is a per-element learnable weight tensor for body
      order ``n``.

    All outputs share the same target irreps (``irreps_out``) so the
    contributions can be summed directly without a separate
    projection — a cleaner factoring than the
    ``FullyConnectedTensorProduct → EquivariantLinear`` pair used by
    the earlier shared-weight implementation.

    Parameters
    ----------
    irreps_in : str or Irreps
        Irreps of the one-particle basis ``A_i``.
    irreps_out : str or Irreps
        Irreps of the output ``B``.
    correlation : int
        Maximum body order ``N``.  Must be in ``{1, 2, 3}``.
    num_elements : int
        Size of the per-element weight table.  Must cover every Z
        that can appear in the data (``120`` covers the whole
        periodic table).
    """

    SUPPORTED_CORRELATIONS: typing.ClassVar[tuple[int, ...]] = (1, 2, 3)

    def __init__(
        self,
        irreps_in: str | Irreps,
        irreps_out: str | Irreps,
        correlation: int,
        num_elements: int = 120,
    ) -> None:
        super().__init__()
        if correlation not in self.SUPPORTED_CORRELATIONS:
            raise ValueError(
                f"correlation must be one of {self.SUPPORTED_CORRELATIONS}, " f"got {correlation}."
            )
        self._correlation: int = correlation
        self._num_elements: int = num_elements
        self.irreps_in: Irreps = Irreps(irreps_in)
        self.irreps_out: Irreps = Irreps(irreps_out)

        # Body order 1 — per-element linear ``A → irreps_out``.
        self.body_1: PerElementLinear = PerElementLinear(
            self.irreps_in, self.irreps_out, num_elements
        )

        # Body order 2 — per-element TP ``A ⊗ A → irreps_out``.
        self.body_2: PerElementWeightedTensorProduct | None = None
        if correlation >= 2:
            self.body_2 = PerElementWeightedTensorProduct(
                self.irreps_in, self.irreps_in, self.irreps_out, num_elements
            )

        # Body order 3 — per-element TP ``B² ⊗ A → irreps_out``.  We
        # iterate on the body-2 output rather than computing a 3-fold
        # CG product directly so the implementation stays expressible
        # in the e3nn primitives we already use; equivariance is still
        # exact (the test suite verifies this).
        self.body_3: PerElementWeightedTensorProduct | None = None
        if correlation >= 3:
            self.body_3 = PerElementWeightedTensorProduct(
                self.irreps_out, self.irreps_in, self.irreps_out, num_elements
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def correlation(self) -> int:
        return self._correlation

    @property
    def num_elements(self) -> int:
        return self._num_elements

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Run the body-order contraction.

        Parameters
        ----------
        x : Tensor
            One-particle basis ``A_i``, shape ``(N, irreps_in.dim)``.
        atomic_numbers : Tensor
            Integer atomic numbers, shape ``(N,)``.

        Returns
        -------
        Tensor
            Contracted features, shape ``(N, irreps_out.dim)``.
        """
        out: torch.Tensor = self.body_1(x, atomic_numbers)

        if self._correlation >= 2:
            # mypy / pyright: body_2 is set whenever correlation >= 2.
            assert self.body_2 is not None
            b2: torch.Tensor = self.body_2(x, x, atomic_numbers)
            out = out + b2
        else:
            b2 = out  # never used; placeholder for the b3 path below

        if self._correlation >= 3:
            assert self.body_3 is not None
            # ``b2`` is in irreps_out space; ``x`` in irreps_in.
            b3: torch.Tensor = self.body_3(b2, x, atomic_numbers)
            out = out + b3

        return out
