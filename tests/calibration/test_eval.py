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


def test_reliability_curve_calibrated_model_tracks_diagonal():
    # Truths drawn from the same distribution the samples describe -> empirical
    # coverage matches the nominal level at every point on the curve.
    rng = np.random.default_rng(0)
    dists, truths = [], []
    for _ in range(400):
        mu = rng.uniform(-5.0, 5.0)
        dists.append(Distribution(rng.normal(mu, 1.0, 2000)))
        truths.append(float(rng.normal(mu, 1.0)))
    levels, coverage = ev.reliability_curve(dists, truths)
    assert levels.shape == coverage.shape
    assert np.all(np.abs(coverage - levels) < 0.07)


def test_reliability_curve_overconfident_model_sits_below_diagonal():
    # Stated spread 3x too narrow -> coverage under nominal everywhere.
    rng = np.random.default_rng(1)
    dists, truths = [], []
    for _ in range(300):
        mu = rng.uniform(-5.0, 5.0)
        dists.append(Distribution(rng.normal(mu, 0.5, 2000)))
        truths.append(float(rng.normal(mu, 1.5)))
    levels, coverage = ev.reliability_curve(dists, truths)
    assert np.all(coverage < levels)


def test_reliability_curve_validates_inputs():
    with pytest.raises(ValueError):
        ev.reliability_curve([], [])
    with pytest.raises(ValueError):
        ev.reliability_curve([Distribution([1.0, 2.0])], [1.0, 2.0])
    levels, coverage = ev.reliability_curve(
        [Distribution(np.linspace(0, 1, 100))], [0.5], levels=(0.5, 0.9))
    assert list(levels) == [0.5, 0.9]
    assert coverage.shape == (2,)


def test_fuse_distributions_matches_fuse_summaries():
    from augur.fusion.posterior import FusionInputs, fuse, fuse_distributions
    from augur.types import VerbalSpec

    inputs = FusionInputs(verbal=VerbalSpec(num_motors=4, prop_diameter_inch=5.0,
                                            blade_count=2, drone_class="racing"))
    dists = fuse_distributions(inputs, n=2000, seed=5)
    report = fuse(inputs, n=2000, seed=5)
    assert set(dists) == set(report.variables)
    for var, dist in dists.items():
        assert isinstance(dist, Distribution)
        assert dist.median == pytest.approx(report.variables[var].median)
