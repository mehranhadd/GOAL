"""MTS (Multi-Time-Step) submodule for goal.md.

Exposes the RESPA integrator, i-pi compatibility layer, and the
adaptive QM/ML online-learning dual calculator.

Quick start
-----------
Simple MTS with two force levels::

    from goal.md.mts.ipi_adapter import MTSSimulation, MTSConfig

    cfg = MTSConfig(mts_ratio=4, timestep_fs=1.0, total_steps=10000)
    sim = MTSSimulation(atoms, fast_calc, config=cfg, slow_calc=qm_calc)
    result = sim.run()

Adaptive online-learning MTS::

    from goal.md.mts.dual_calculator.qm_ml_learner import (
        QMLearnerCalculator, ThresholdPolicy, GOALOnlineFinetuner
    )

    finetuner = GOALOnlineFinetuner(ml_calc, cutoff=5.0)
    learner = QMLearnerCalculator(
        base_calculator=qm_calc,
        ml_calculator=ml_calc,
        policy=ThresholdPolicy(force_rmse=0.05),
        fine_tuner=finetuner,
    )
    atoms.calc = learner
    dyn.run(steps)
"""

from goal.md.mts.dual_calculator.base import DualCalculator, DualCalculatorState
from goal.md.mts.dual_calculator.qm_ml_learner import (
    AlwaysBasePolicy,
    GOALOnlineFinetuner,
    QMLearnerCalculator,
    StepFractionPolicy,
    ThresholdPolicy,
)
from goal.md.mts.integrators.respa import RespaIntegrator
from goal.md.mts.ipi_adapter import IPICompatibleDriver, MTSConfig, MTSSimulation

__all__ = [
    "MTSConfig",
    "MTSSimulation",
    "IPICompatibleDriver",
    "RespaIntegrator",
    "DualCalculator",
    "DualCalculatorState",
    "QMLearnerCalculator",
    "ThresholdPolicy",
    "AlwaysBasePolicy",
    "StepFractionPolicy",
    "GOALOnlineFinetuner",
]
