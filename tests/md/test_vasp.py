"""Unit tests for the VASP builder (mocked — commercial, not installed)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from goal.md.calculators.base import GOALCalculatorNotFound
from goal.md.core import calculator_factory as cf


def test_missing_pp_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_VASP", True)
    monkeypatch.delenv("VASP_PP_PATH", raising=False)
    with pytest.raises(GOALCalculatorNotFound, match="POTCAR|VASP_PP_PATH"):
        cf.CalculatorFactory.create("vasp")


def test_vasp_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_VASP", False)
    with pytest.raises(GOALCalculatorNotFound, match="VASP"):
        cf.CalculatorFactory.create("vasp")


def test_bulk_preset_params(monkeypatch: pytest.MonkeyPatch) -> None:
    vasp = MagicMock()
    monkeypatch.setattr(cf, "HAS_VASP", True)
    monkeypatch.setattr(cf, "Vasp", vasp)
    monkeypatch.setenv("VASP_PP_PATH", "/data/potcars")

    cf.CalculatorFactory.create("vasp", preset="bulk")
    kwargs = vasp.call_args.kwargs
    assert kwargs["kpts"] == [4, 4, 4]
    assert kwargs["ismear"] == 1
    assert kwargs["sigma"] == 0.2
    assert kwargs["encut"] == 520
