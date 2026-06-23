"""Human-readable rendering of a PredictionReport.

Shows, per variable: median + interval + confidence label + which modalities
tightened it, plus the explicit assumptions used. Never prints a bare scalar for
a predicted quantity."""

from __future__ import annotations

from augur.types import PredictionReport

_PRETTY = {
    "rpm": "RPM",
    "thrust_per_motor_n": "Thrust / motor",
    "total_thrust_n": "Total thrust",
    "mass_kg": "Mass",
    "weight_n": "Weight",
    "shaft_power_w": "Shaft power",
    "electrical_power_w": "Electrical power",
    "disk_loading_n_m2": "Disk loading",
    "min_frame_diagonal_m": "Frame diagonal (min)",
    "thrust_to_weight": "Thrust-to-weight",
    "battery_wh": "Battery capacity",
    "endurance_s": "Endurance",
    "num_motors": "Motor count",
    "prop_diameter_inch": "Prop diameter",
}


def _fmt(x: float) -> str:
    if abs(x) >= 1000 or (x != 0 and abs(x) < 0.01):
        return f"{x:.3g}"
    return f"{x:.3g}"


def render(report: PredictionReport, level: float = 0.9) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("AUGUR — drone property inference")
    lines.append(f"inputs: {', '.join(report.inputs_used) or 'none'}   "
                 f"config v{report.config_version}")
    lines.append("=" * 64)

    # Which modality narrowed each variable most.
    best_source: dict[str, str] = {}
    best_gain: dict[str, float] = {}
    for a in report.attribution:
        gain = a.fractional_narrowing
        if gain > best_gain.get(a.variable, 0.0):
            best_gain[a.variable] = gain
            best_source[a.variable] = a.source

    pct = int(level * 100)
    for var, s in report.variables.items():
        name = _PRETTY.get(var, var)
        unit = f" {s.unit}" if s.unit else ""
        tightened = best_source.get(var)
        tag = ""
        if tightened and best_gain.get(var, 0) > 0.05:
            tag = f"  ← {tightened} (-{best_gain[var]*100:.0f}% width)"
        lines.append(
            f"{name:<22} {_fmt(s.median)}{unit:<7} "
            f"[{pct}%: {_fmt(s.interval_low)}–{_fmt(s.interval_high)}]  "
            f"<{s.confidence}>{tag}"
        )

    if report.assumptions:
        lines.append("-" * 64)
        lines.append("assumptions:")
        for a in report.assumptions:
            lines.append(f"  • {a}")

    if report.notes:
        lines.append("-" * 64)
        lines.append("notes:")
        for nline in report.notes:
            lines.append(f"  ! {nline}")

    lines.append("=" * 64)
    return "\n".join(lines)
