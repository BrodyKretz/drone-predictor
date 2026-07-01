"""Evaluation harness for the golden set.

Two jobs:

1. Assign train / calib / test splits at the DRONE level, so no physical drone
   leaks across splits (the manifest enforces this; here we produce a clean
   assignment to begin with).
2. Score predictions against measured truth — point error, interval coverage,
   and sharpness (relative width) per variable. These are the numbers behind the
   spec §10 metric table, plus a helper to fit the conformal calibrator on a
   calib split.

The harness runs today on any (prediction, truth) pairs; it only becomes
*meaningful* once a real golden set exists. See docs/GOLDEN_SET.md for how to
capture that data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from augur.data_manifest import Split
from augur.fusion.calibration import IntervalCalibrator
from augur.types import PredictionReport, VariableSummary


def assign_splits(drone_ids, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
                  seed: int = 0) -> dict[str, str]:
    """Map each unique drone to a split, disjoint at the drone level.

    fractions are (train, calib, test) and must sum to ~1. Assignment is
    deterministic given the seed."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    uniq = sorted({str(d) for d in drone_ids})
    rng = np.random.default_rng(seed)
    uniq = [uniq[i] for i in rng.permutation(len(uniq))]

    n = len(uniq)
    n_train = int(round(fractions[0] * n))
    n_calib = int(round(fractions[1] * n))
    assignment: dict[str, str] = {}
    for i, drone in enumerate(uniq):
        if i < n_train:
            assignment[drone] = Split.train.value
        elif i < n_train + n_calib:
            assignment[drone] = Split.calib.value
        else:
            assignment[drone] = Split.test.value
    return assignment


@dataclass
class VariableMetrics:
    variable: str
    n: int
    mape: float            # median absolute percentage error of the point estimate
    coverage: float        # fraction of truths inside the stated interval
    nominal_level: float   # the interval level those intervals claimed
    median_rel_width: float  # sharpness: median interval width / |median|

    @property
    def is_calibrated(self) -> bool:
        """Coverage should meet or slightly exceed the nominal level."""
        return self.coverage >= self.nominal_level - 0.05


def score_variable(summaries: list[VariableSummary], truths: list[float],
                   variable: str = "") -> VariableMetrics:
    if len(summaries) != len(truths):
        raise ValueError("summaries and truths must be equal length")
    if not summaries:
        raise ValueError("need at least one (summary, truth) pair")

    abs_pct, rel_widths, covered = [], [], []
    level = summaries[0].interval_level
    for s, truth in zip(summaries, truths, strict=True):
        if truth != 0:
            abs_pct.append(abs(s.median - truth) / abs(truth))
        denom = abs(s.median) if abs(s.median) > 1e-12 else 1e-12
        rel_widths.append((s.interval_high - s.interval_low) / denom)
        covered.append(s.interval_low <= truth <= s.interval_high)

    return VariableMetrics(
        variable=variable,
        n=len(summaries),
        mape=float(np.median(abs_pct)) if abs_pct else float("nan"),
        coverage=float(np.mean(covered)),
        nominal_level=level,
        median_rel_width=float(np.median(rel_widths)),
    )


def evaluate(reports: list[PredictionReport], truths: list[dict],
             variables: list[str] | None = None) -> dict[str, VariableMetrics]:
    """Score every variable that appears in both the reports and the truth dicts."""
    if len(reports) != len(truths):
        raise ValueError("reports and truths must be equal length")

    if variables is None:
        variables = sorted({v for r in reports for v in r.variables})

    results: dict[str, VariableMetrics] = {}
    for var in variables:
        pairs = [(r.variables[var], t[var]) for r, t in zip(reports, truths, strict=True)
                 if var in r.variables and var in t and t[var] is not None]
        if pairs:
            summaries, tvals = zip(*pairs, strict=True)
            results[var] = score_variable(list(summaries), list(tvals), variable=var)
    return results


def fit_calibrator(reports: list[PredictionReport], truths: list[dict], variable: str,
                   level: float = 0.9) -> IntervalCalibrator:
    """Fit a conformal width factor for one variable from a calib split."""
    medians, half_widths, tvals = [], [], []
    for r, t in zip(reports, truths, strict=True):
        if variable in r.variables and t.get(variable) is not None:
            s = r.variables[variable]
            medians.append(s.median)
            half_widths.append((s.interval_high - s.interval_low) / 2.0)
            tvals.append(t[variable])
    if not medians:
        raise ValueError(f"no calibration pairs for {variable!r}")
    return IntervalCalibrator(level=level).fit(
        np.array(medians), np.array(half_widths), np.array(tvals))


def format_metrics_table(results: dict[str, VariableMetrics]) -> str:
    """Human-readable §10-style metric table."""
    lines = [f"{'variable':<22}{'n':>4}{'MAPE':>9}{'cover':>8}{'nom':>6}{'rel_w':>8}  cal",
             "-" * 64]
    for var, m in results.items():
        lines.append(
            f"{var:<22}{m.n:>4}{m.mape * 100:>8.1f}%{m.coverage * 100:>7.0f}%"
            f"{m.nominal_level * 100:>5.0f}%{m.median_rel_width:>8.2f}  {'✓' if m.is_calibrated else '✗'}"
        )
    return "\n".join(lines)
