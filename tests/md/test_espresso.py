"""Unit tests for the Quantum ESPRESSO builder (mocked — no pw.x)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from goal.md.calculators.base import GOALCalculatorNotFound
from goal.md.core import calculator_factory as cf


def test_missing_pseudopotentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_ESPRESSO", True)
    with pytest.raises(GOALCalculatorNotFound, match="pseudopotential"):
        cf.CalculatorFactory.create("espresso", pseudopotentials={})


def test_missing_element_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_ESPRESSO", True)
    with pytest.raises(GOALCalculatorNotFound) as exc:
        cf.CalculatorFactory.create(
            "espresso",
            pseudopotentials={"H": "H.UPF"},
            elements=["H", "O"],
        )
    assert "O" in str(exc.value)


def test_molecular_preset_sets_assume_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    espresso = MagicMock()
    profile = MagicMock()
    monkeypatch.setattr(cf, "HAS_ESPRESSO", True)
    monkeypatch.setattr(cf, "Espresso", espresso)
    monkeypatch.setattr(cf, "EspressoProfile", profile)

    cf.CalculatorFactory.create(
        "espresso",
        command="pw.x",
        pseudo_dir="/tmp/pseudo",
        pseudopotentials={"H": "H.UPF", "O": "O.UPF"},
        elements=["H", "O"],
        preset="molecular",
    )
    input_data = espresso.call_args.kwargs["input_data"]
    assert input_data["system"]["assume_isolated"] == "mt"


def test_espresso_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "HAS_ESPRESSO", False)
    with pytest.raises(GOALCalculatorNotFound, match="ASE"):
        cf.CalculatorFactory.create("espresso", pseudopotentials={"H": "H.UPF"})
