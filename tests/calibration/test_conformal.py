"""Conformal calibration: an overconfident interval set is corrected to nominal
coverage on a held-out split."""

import numpy as np

from augur.fusion.calibration import IntervalCalibrator, empirical_coverage


def _make_data(n, true_sigma, stated_half_width, seed):
    rng = np.random.default_rng(seed)
    medians = rng.uniform(0, 100, size=n)
    truths = medians + rng.normal(0, true_sigma, size=n)
    half = np.full(n, stated_half_width)
    return medians, half, truths


def test_overconfident_intervals_get_widened():
    # Stated half-width 1.0 but true noise sigma 3.0 -> badly overconfident.
    m, h, y = _make_data(2000, true_sigma=3.0, stated_half_width=1.0, seed=0)
    raw_cov = empirical_coverage(m, h, y, factor=1.0)
    cal = IntervalCalibrator(level=0.9).fit(m, h, y)
    assert cal.factor > 1.0  # must widen
    cal_cov = empirical_coverage(m, h, y, factor=cal.factor)
    assert cal_cov > raw_cov
    assert abs(cal_cov - 0.9) < 0.03


def test_calibration_generalizes_to_holdout():
    m, h, y = _make_data(2000, true_sigma=2.0, stated_half_width=0.8, seed=1)
    cal = IntervalCalibrator(level=0.8).fit(m, h, y)
    m2, h2, y2 = _make_data(2000, true_sigma=2.0, stated_half_width=0.8, seed=99)
    cov = empirical_coverage(m2, h2, y2, factor=cal.factor)
    assert abs(cov - 0.8) < 0.05  # within tolerance of nominal on fresh data


def test_overwide_intervals_get_tightened():
    # Stated half-width 10 but true sigma 1 -> too wide; factor should shrink.
    m, h, y = _make_data(2000, true_sigma=1.0, stated_half_width=10.0, seed=2)
    cal = IntervalCalibrator(level=0.9).fit(m, h, y)
    assert cal.factor < 1.0
