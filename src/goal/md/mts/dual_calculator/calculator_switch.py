"""Calculator switching strategies and logic."""

from __future__ import annotations

import abc
import typing

import numpy as np


class SwitchingStrategy(abc.ABC):
    """Abstract base for calculator switching strategies."""

    @abc.abstractmethod
    def decide(
        self,
        step: int,
        base_forces: np.ndarray,
        ml_forces: np.ndarray,
        base_energy: float,
        ml_energy: float,
        **kwargs: typing.Any,
    ) -> str:
        """Decide which calculator to use.

        Returns
        -------
        str
            "base" or "ml"
        """
        pass


class AccuracyBasedSwitcher(SwitchingStrategy):
    """Switch based on force/energy error thresholds.

    Switches to ML once:
    - Force RMSE < threshold
    - Energy MAE < threshold
    - Minimum training steps completed
    """

    def __init__(
        self,
        force_threshold: float = 0.01,
        energy_threshold: float = 0.001,
        min_steps: int = 500,
    ) -> None:
        self.f_thresh = force_threshold
        self.e_thresh = energy_threshold
        self.min_steps = min_steps
        self.training_steps = 0

    def decide(
        self,
        step: int,
        base_forces: np.ndarray,
        ml_forces: np.ndarray,
        base_energy: float,
        ml_energy: float,
        **kwargs: typing.Any,
    ) -> str:
        """Accuracy-based switching."""
        self.training_steps += 1

        if self.training_steps < self.min_steps:
            return "base"

        force_rmse = np.sqrt(np.mean((ml_forces - base_forces) ** 2))
        energy_mae = np.abs(ml_energy - base_energy)

        if force_rmse < self.f_thresh and energy_mae < self.e_thresh:
            return "ml"

        return "base"


class TimeBasedSwitcher(SwitchingStrategy):
    """Switch after fixed number of steps."""

    def __init__(self, switch_at_step: int = 1000) -> None:
        self.switch_step = switch_at_step

    def decide(
        self,
        step: int,
        base_forces: np.ndarray,
        ml_forces: np.ndarray,
        base_energy: float,
        ml_energy: float,
        **kwargs: typing.Any,
    ) -> str:
        """Time-based switching."""
        if step >= self.switch_step:
            return "ml"
        return "base"


class HybridSwitcher(SwitchingStrategy):
    """Hybrid strategy: ensemble blending or conditional switching."""

    def __init__(
        self,
        accuracy_threshold: float = 0.01,
        blend_mode: bool = False,
    ) -> None:
        self.accuracy_thresh = accuracy_threshold
        self.blend_mode = blend_mode

    def decide(
        self,
        step: int,
        base_forces: np.ndarray,
        ml_forces: np.ndarray,
        base_energy: float,
        ml_energy: float,
        uncertainty: float | None = None,
        **kwargs: typing.Any,
    ) -> str:
        """Hybrid switching with optional blending."""
        force_rmse = np.sqrt(np.mean((ml_forces - base_forces) ** 2))

        if self.blend_mode and uncertainty is not None:
            # Could implement ensemble blending here
            # For now, just use standard switching
            if force_rmse < self.accuracy_thresh:
                return "ml"
            return "base"

        if force_rmse < self.accuracy_thresh:
            return "ml"
        return "base"


class FallbackSwitcher(SwitchingStrategy):
    """Switch back to base if ML fails."""

    def __init__(self, base_switcher: SwitchingStrategy) -> None:
        self.base_switcher = base_switcher
        self.fallback_active = False

    def decide(
        self,
        step: int,
        base_forces: np.ndarray,
        ml_forces: np.ndarray,
        base_energy: float,
        ml_energy: float,
        **kwargs: typing.Any,
    ) -> str:
        """Switching with fallback to base."""
        # Check for NaN/Inf (ML failure)
        if np.any(~np.isfinite(ml_forces)) or not np.isfinite(ml_energy):
            self.fallback_active = True
            return "base"

        # Try to recover from fallback
        if self.fallback_active:
            force_rmse = np.sqrt(np.mean((ml_forces - base_forces) ** 2))
            if force_rmse < 0.05:  # Stricter threshold
                self.fallback_active = False
            else:
                return "base"

        # Normal switching
        return self.base_switcher.decide(
            step, base_forces, ml_forces, base_energy, ml_energy, **kwargs
        )
