"""Reliability-diagram plotting. Requires the optional `viz` extra
(`pip install -e ".[viz]"`).

The curve math lives in `augur.eval.reliability_curve` (numpy-only); this module
only renders. Uses matplotlib's object API directly (no pyplot) so importing it
never touches global state or needs a GUI backend.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def plot_reliability(
    curves: dict[str, tuple],
    path: str | Path | None = None,
    title: str = "Reliability diagram",
) -> Figure:
    """Plot empirical coverage vs nominal level, one curve per variable.

    `curves` maps variable name -> (levels, coverage) as returned by
    `eval.reliability_curve`. The dashed diagonal is perfect calibration;
    points below it are overconfident intervals."""
    if not curves:
        raise ValueError("curves must contain at least one variable")

    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="perfect calibration")
    for variable, (levels, coverage) in curves.items():
        ax.plot(levels, coverage, marker="o", markersize=4, linewidth=1.5,
                label=variable)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal interval level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
