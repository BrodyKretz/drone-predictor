"""Core types for Augur.

The central rule of the system: every *predicted* quantity is a distribution,
never a bare scalar. `Distribution` is the carrier for that. Verbal/known facts
(things the user asserts) can still be scalars on input, but anything the system
infers comes back wrapped.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, field_validator

ConfidenceLabel = str  # "high" | "medium" | "low" | "unconstrained"


class Distribution:
    """A predicted quantity represented by Monte Carlo samples.

    Backed by a 1-D numpy array. All summary statistics derive from the samples
    so that arbitrary (non-Gaussian, multimodal) posteriors are representable.
    """

    __slots__ = ("samples", "unit")

    def __init__(self, samples: np.ndarray | list[float], unit: str | None = None):
        arr = np.asarray(samples, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("Distribution requires at least one sample")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Distribution samples must all be finite")
        self.samples = arr
        self.unit = unit

    @classmethod
    def from_scalar(cls, value: float, unit: str | None = None) -> Distribution:
        """A degenerate (known-exactly) distribution. Used for asserted facts."""
        return cls(np.full(1, float(value)), unit=unit)

    @classmethod
    def from_uniform(cls, low: float, high: float, n: int, unit: str | None = None) -> Distribution:
        if high < low:
            raise ValueError(f"from_uniform requires high >= low, got {low}..{high}")
        rng = np.random.default_rng(0)
        return cls(rng.uniform(low, high, size=n), unit=unit)

    @property
    def n(self) -> int:
        return self.samples.size

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """Central credible interval at the given level (e.g. 0.9 → 5th/95th pct)."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.samples, [tail, 1.0 - tail])
        return float(lo), float(hi)

    @property
    def relative_width(self) -> float:
        """Width of the 90% interval relative to the median (a unitless spread)."""
        lo, hi = self.interval(0.9)
        denom = abs(self.median)
        if denom < 1e-12:
            return float("inf") if (hi - lo) > 1e-12 else 0.0
        return (hi - lo) / denom

    @property
    def confidence_label(self) -> ConfidenceLabel:
        if self.n == 1 or self.std == 0.0:
            return "high"
        rw = self.relative_width
        if rw < 0.15:
            return "high"
        if rw < 0.40:
            return "medium"
        if rw < 1.5:
            return "low"
        return "unconstrained"

    def to_summary(self, level: float = 0.9) -> VariableSummary:
        lo, hi = self.interval(level)
        return VariableSummary(
            median=self.median,
            mean=self.mean,
            interval_low=lo,
            interval_high=hi,
            interval_level=level,
            confidence=self.confidence_label,
            unit=self.unit,
        )

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        lo, hi = self.interval(0.9)
        u = f" {self.unit}" if self.unit else ""
        return f"Distribution(median={self.median:.4g}{u}, 90%=[{lo:.4g}, {hi:.4g}], {self.confidence_label})"


class DroneClass(str, Enum):
    racing = "racing"
    cinematic = "cinematic"
    survey = "survey"
    unknown = "unknown"


class VerbalSpec(BaseModel):
    """Structured facts a user can assert. Every field optional; missing fields
    simply contribute no constraint."""

    num_motors: int | None = None
    props_per_motor: int = 1
    prop_diameter_inch: float | None = None
    prop_pitch_inch: float | None = None
    blade_count: int | None = None
    voltage: float | None = None
    cell_count: int | None = None
    drone_class: DroneClass | None = None
    payload_mass_kg: float | None = None
    notes: str | None = None

    @field_validator("num_motors", "blade_count", "cell_count")
    @classmethod
    def _positive_int(cls, v):
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer")
        return v


class FlightState(str, Enum):
    hover = "hover"
    climb = "climb"
    cruise = "cruise"
    coast = "coast"
    mixed = "mixed"
    unknown = "unknown"


class VariableSummary(BaseModel):
    """Serializable summary of a Distribution for reports / API responses."""

    median: float
    mean: float
    interval_low: float
    interval_high: float
    interval_level: float
    confidence: str
    unit: str | None = None


class Attribution(BaseModel):
    """How much a single modality narrowed a variable's interval, in order of
    application. width_before/after are absolute 90%-interval widths;
    median_before/after let consumers compute relative (information) narrowing,
    which is the monotone quantity when an input also relocates the estimate."""

    variable: str
    source: str
    width_before: float
    width_after: float
    median_before: float
    median_after: float

    @property
    def fractional_narrowing(self) -> float:
        """Absolute-width narrowing (for human-readable 'what tightened what')."""
        if self.width_before <= 0:
            return 0.0
        return max(0.0, (self.width_before - self.width_after) / self.width_before)

    @staticmethod
    def _relative(width: float, median: float) -> float:
        denom = abs(median)
        if denom < 1e-12:
            return float("inf") if width > 1e-12 else 0.0
        return width / denom

    @property
    def relative_width_before(self) -> float:
        return self._relative(self.width_before, self.median_before)

    @property
    def relative_width_after(self) -> float:
        return self._relative(self.width_after, self.median_after)


class PredictionReport(BaseModel):
    """Top-level output. One summary per predicted variable, the assumptions
    used, and per-modality attribution of interval narrowing."""

    variables: dict[str, VariableSummary] = Field(default_factory=dict)
    attribution: list[Attribution] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    inputs_used: list[str] = Field(default_factory=list)
    config_version: int | None = None
    notes: list[str] = Field(default_factory=list)
