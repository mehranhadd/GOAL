"""Unit tests for the CP2K calculator builder (mocked — no CP2K binary)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from goal.md.calculators.base import GOALCalculatorNotFound
from goal.md.core import calculator_factory as cf


def test_set_pos_file_true_with_mpi(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_pos_file must default True and n_mpi must build an mpirun command."""
    captured = MagicMock()
    monkeypatch.setattr(cf, "HAS_CP2K", True)
    monkeypatch.setattr(cf, "CP2K", captured)
    # Pretend the shell binary resolves.
    monkeypatch.setattr(cf.shutil, "which", lambda _p: "/usr/bin/cp2k_shell.psmp")

    cf.CalculatorFactory.create(
        "cp2k", command="cp2k_shell.psmp", n_mpi=4, omp_threads=2, preset="molecular"
    )

    kwargs = captured.call_args.kwargs
    assert kwargs["set_pos_file"] is True
    assert "mpirun -n 4" in kwargs["command"]
    # molecular preset → Martyna–Tuckerman, no stress
    assert kwargs["poisson_solver"] == "MT"
    assert kwargs["stress_tensor"] is False
    import os

    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_cp2k_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear GOALCalculatorNotFound is raised when CP2K is unavailable."""
    monkeypatch.setattr(cf, "HAS_CP2K", False)
    with pytest.raises(GOALCalculatorNotFound, match="CP2K"):
        cf.CalculatorFactory.create("cp2k")


def test_bulk_preset_uses_stress(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = MagicMock()
    monkeypatch.setattr(cf, "HAS_CP2K", True)
    monkeypatch.setattr(cf, "CP2K", captured)
    monkeypatch.setattr(cf.shutil, "which", lambda _p: "/usr/bin/cp2k_shell.psmp")

    cf.CalculatorFactory.create("cp2k", command="cp2k_shell.psmp", preset="bulk")
    assert captured.call_args.kwargs["stress_tensor"] is True
