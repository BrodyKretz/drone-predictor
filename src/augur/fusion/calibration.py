"""Conformal calibration of posterior intervals.

After fusion, the stated X% intervals must actually cover X% on held-out drones.
Split conformal: on a calibration set, find the multiplicative width factor that
makes empirical coverage match nominal, then apply it to future intervals.

This is honest about its dependency: it needs a real calibration set (the golden
`calib` split). The algorithm here is exercised on synthetic data; the factor it
learns is only meaningful once fit on real characterized drones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntervalCalibrator:
    """Learns a single multiplicative widening factor per variable so that the
    nominal central interval achieves nominal empirical coverage."""

    level: float = 0.9
    factor: float = 1.0

    def fit(self, medians: np.ndarray, half_widths: np.ndarray, truths: np.ndarray) -> IntervalCalibrator:
        """Fit on calibration data.

        Nonconformity score = |truth - median| / half_width. The (level)-quantile
        of these scores is the factor by which the nominal half-width must scale
        to reach `level` coverage. factor < 1 means the raw intervals were too
        wide; > 1 means overconfident.
        """
        medians = np.asarray(medians, float)
        half_widths = np.asarray(half_widths, float)
        truths = np.asarray(truths, float)
        if not (medians.shape == half_widths.shape == truths.shape):
            raise ValueError("medians, half_widths, truths must have equal shape")
        if np.any(half_widths <= 0):
            raise ValueError("half_widths must be positive")

        scores = np.abs(truths - medians) / half_widths
        # Finite-sample conformal quantile.
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * self.level) / n)
        self.factor = float(np.quantile(scores, q_level))
        return self

    def apply(self, median: float, interval: tuple[float, float]) -> tuple[float, float]:
        lo, hi = interval
        half = (hi - lo) / 2.0
        return (median - self.factor * half, median + self.factor * half)


def empirical_coverage(medians: np.ndarray, half_widths: np.ndarray, truths: np.ndarray,
                       factor: float = 1.0) -> float:
    """Fraction of truths inside median +/- factor*half_width."""
    medians = np.asarray(medians, float)
    half_widths = np.asarray(half_widths, float) * factor
    truths = np.asarray(truths, float)
    inside = np.abs(truths - medians) <= half_widths
    return float(np.mean(inside))
