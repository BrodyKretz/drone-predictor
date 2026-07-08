"""Reliability-diagram plotting (viz extra)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="viz extra not installed")

from augur import eval as ev  # noqa: E402
from augur import viz  # noqa: E402
from augur.types import Distribution  # noqa: E402


def _curve(seed: int, spread: float):
    rng = np.random.default_rng(seed)
    dists, truths = [], []
    for _ in range(60):
        mu = rng.uniform(0.0, 5.0)
        dists.append(Distribution(rng.normal(mu, 1.0, 500)))
        truths.append(float(rng.normal(mu, spread)))
    return ev.reliability_curve(dists, truths)


def test_plot_reliability_writes_figure(tmp_path):
    curves = {"rpm": _curve(0, 1.0), "mass_kg": _curve(1, 2.0)}
    out = tmp_path / "reliability.png"
    fig = viz.plot_reliability(curves, path=out)
    assert out.exists() and out.stat().st_size > 0
    ax = fig.axes[0]
    assert len(ax.get_lines()) == len(curves) + 1  # one per variable + diagonal


def test_plot_reliability_requires_curves():
    with pytest.raises(ValueError):
        viz.plot_reliability({})
