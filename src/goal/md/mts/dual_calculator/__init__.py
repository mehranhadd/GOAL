"""Dual-calculator framework for adaptive MTS simulations.

Provides:

- :class:`~goal.md.mts.dual_calculator.base.DualCalculator` — abstract base.
- :class:`~goal.md.mts.dual_calculator.qm_ml_learner.QMLearnerCalculator` —
  QM reference + on-the-fly ML fine-tuning with configurable switch policies.
"""

from goal.md.mts.dual_calculator.base import DualCalculator, DualCalculatorState
from goal.md.mts.dual_calculator.qm_ml_learner import (
    AlwaysBasePolicy,
    GOALOnlineFinetuner,
    QMLearnerCalculator,
    StepFractionPolicy,
    ThresholdPolicy,
)

__all__ = [
    "DualCalculator",
    "DualCalculatorState",
    "QMLearnerCalculator",
    "ThresholdPolicy",
    "AlwaysBasePolicy",
    "StepFractionPolicy",
    "GOALOnlineFinetuner",
]
