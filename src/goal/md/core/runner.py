"""SimulationRunner — orchestrates multi-step MD runs with progress tracking.

Provides ``run_dynamics`` for executing a dynamics object with:
- Chunked execution (to allow thermostat restarts, callbacks, etc.)
- Rich progress bar
- Optional Maxwell–Boltzmann velocity re-initialisation per turn
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclasses.dataclass
class RunResult:
    """Summary returned after a simulation run.

    Attributes
    ----------
    steps_completed : int
        Total number of MD steps executed.
    final_energy : float or None
        Potential energy (eV) at the last step.
    final_temperature : float or None
        Kinetic temperature (K) at the last step.
    energy_history : list[float]
        Potential energies at each logged step (empty if no collector attached).
    """

    steps_completed: int
    final_energy: float | None = None
    final_temperature: float | None = None
    energy_history: list[float] = dataclasses.field(default_factory=list)


def run_dynamics(
    dynamics: object,
    steps: int,
    chunk_size: int = 100,
    restart_thermostat: bool = False,
    temperature_K: float = 300.0,
    progress: bool = True,
    label: str = "MD",
    energy_collector: typing.Any | None = None,
) -> RunResult:
    """Run a dynamics object for *steps* total steps.

    Parameters
    ----------
    dynamics : object
        ASE dynamics instance (must have ``.run(n)`` and ``.atoms``).
    steps : int
        Total number of integration steps to perform.
    chunk_size : int
        Number of steps per inner ``dynamics.run()`` call.  Smaller chunks
        allow thermostat restarts and finer progress updates.
    restart_thermostat : bool
        Re-sample Maxwell–Boltzmann velocities after each chunk.
    temperature_K : float
        Temperature for thermostat restart (used only when
        ``restart_thermostat=True``).
    progress : bool
        Show a Rich progress bar (requires ``rich``).
    label : str
        Label shown on the progress bar.
    energy_collector : EnergyCollector, optional
        If provided, its ``.energies`` list is copied to the result.

    Returns
    -------
    RunResult
        Summary of the completed run.
    """
    atoms: Atoms = dynamics.atoms  # type: ignore[attr-defined]

    def _run_loop() -> None:
        remaining = steps
        while remaining > 0:
            n = min(chunk_size, remaining)
            dynamics.run(n)  # type: ignore[attr-defined]
            remaining -= n
            if restart_thermostat and remaining > 0:
                MaxwellBoltzmannDistribution(atoms=atoms, temperature_K=temperature_K)

    if progress and HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as prog:
            task = prog.add_task(label, total=steps)

            remaining = steps
            while remaining > 0:
                n = min(chunk_size, remaining)
                dynamics.run(n)  # type: ignore[attr-defined]
                remaining -= n
                prog.update(task, advance=n)
                if restart_thermostat and remaining > 0:
                    MaxwellBoltzmannDistribution(atoms=atoms, temperature_K=temperature_K)
    else:
        _run_loop()

    final_energy: float | None = None
    final_temperature: float | None = None
    try:
        final_energy = float(atoms.get_potential_energy())
    except Exception:
        pass
    try:
        # kinetic temperature: T = 2*KE / (3*N*kB)
        ke = float(atoms.get_kinetic_energy())
        n_atoms = len(atoms)
        final_temperature = 2.0 * ke / (3.0 * n_atoms * units.kB)
    except Exception:
        pass

    energy_history: list[float] = []
    if energy_collector is not None:
        energy_history = list(energy_collector.energies)

    return RunResult(
        steps_completed=steps,
        final_energy=final_energy,
        final_temperature=final_temperature,
        energy_history=energy_history,
    )
