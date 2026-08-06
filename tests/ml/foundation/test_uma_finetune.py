"""Unit tests for the UMA fine-tune backbone (fairchem not installed)."""

from __future__ import annotations

import pytest
import torch.nn as nn

from goal.md.calculators.base import UnsupportedOperationError
from goal.ml.nn.models.foundation.uma import UMAFinetune


def test_fairchem_missing_raises_unsupported() -> None:
    """Without fairchem, loading raises UnsupportedOperationError w/ docs link."""
    pytest.importorskip  # noqa: B018 - documents intent; we assert fairchem absent below
    try:
        import fairchem  # noqa: F401

        pytest.skip("fairchem is installed; this test covers the absent case.")
    except ImportError:
        pass

    with pytest.raises(UnsupportedOperationError) as exc:
        UMAFinetune(checkpoint="uma-s-1")
    assert "fair-chem" in str(exc.value)


def test_strategy_applies_with_stub_model() -> None:
    """With a stub model (bypassing load), strategies still apply generically."""

    class Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 1)

    bb = UMAFinetune(model=Stub(), head="omol", strategy="full")
    assert all(p.requires_grad for p in bb._model.parameters())
    assert bb.head_name == "omol"
    assert bb.output_keys == ["energy", "forces"]
