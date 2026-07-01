"""Golden-set eval harness: leakage-free splits + §10 metrics + calibrator fit."""

import numpy as np
import pytest

from augur import eval as ev
from augur.types import Distribution, PredictionReport


def test_assign_splits_drone_level_disjoint():
    drones = [f"d{i}" for i in range(20)]
    assignment = ev.assign_splits(drones, seed=1)
    # every drone assigned exactly one split
    assert set(assignment) == set(drones)
    assert set(assignment.values()) <= {"train", "calib", "test"}


def test_assign_splits_repeats_collapse_to_one_split():
    # the same drone appearing many times still lands in a single split
    drones = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    assignment = ev.assign_splits(drones)
    assert set(assignment) == {"a", "b", "c"}


def test_assign_splits_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        ev.assign_splits(["a", "b"], fractions=(0.5, 0.2, 0.2))


def test_assign_splits_deterministic():
    d = [f"d{i}" for i in range(30)]
    assert ev.assign_splits(d, seed=7) == ev.assign_splits(d, seed=7)


def _report(var: str, samples) -> PredictionReport:
    r = PredictionReport()
    r.variables[var] = Distribution(samples).to_summary(level=0.9)
    return r


def test_score_variable_perfect_point_estimate():
    # Tight distributions centered exactly on truth -> ~0 MAPE, full coverage.
    reports = [_report("rpm", np.full(500, t) + np.random.default_rng(i).normal(0, 1, 500))
               for i, t in enumerate([10000, 12000, 14000])]
    truths = [{"rpm": t} for t in (10000, 12000, 14000)]
    res = ev.evaluate(reports, truths)
    assert res["rpm"].mape < 0.01
    assert res["rpm"].coverage == pytest.approx(1.0)


def test_score_variable_detects_undercoverage():
    # Intervals centered far from truth -> coverage well below nominal.
    rng = np.random.default_rng(0)
    reports = [_report("mass_kg", rng.normal(1.0, 0.02, 500)) for _ in range(10)]
    truths = [{"mass_kg": 2.0} for _ in range(10)]  # truth nowhere near the tight interval
    res = ev.evaluate(reports, truths)
    assert res["mass_kg"].coverage == 0.0
    assert not res["mass_kg"].is_calibrated


def test_evaluate_skips_absent_variables():
    reports = [_report("rpm", np.full(100, 9000.0))]
    truths = [{"rpm": 9000.0, "mass_kg": 1.0}]  # mass not predicted
    res = ev.evaluate(reports, truths)
    assert "rpm" in res and "mass_kg" not in res


def test_fit_calibrator_widens_overconfident_intervals():
    # Truths scatter ~3x wider than the stated half-widths -> factor > 1.
    rng = np.random.default_rng(0)
    reports, truths = [], []
    for _ in range(200):
        center = 100.0
        reports.append(_report("thrust", rng.normal(center, 1.0, 800)))  # narrow interval
        truths.append({"thrust": center + rng.normal(0, 3.0)})           # actually wide
    cal = ev.fit_calibrator(reports, truths, "thrust", level=0.9)
    assert cal.factor > 1.0


def test_format_metrics_table_runs():
    reports = [_report("rpm", np.full(100, 9000.0) + np.random.default_rng(i).normal(0, 50, 100))
               for i in range(5)]
    truths = [{"rpm": 9000.0} for _ in range(5)]
    table = ev.format_metrics_table(ev.evaluate(reports, truths))
    assert "rpm" in table and "MAPE" in table
