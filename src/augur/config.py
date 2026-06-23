"""Versioned assumptions registry loader.

Every physical assumption (efficiency bands, energy density, drag coefficients,
mass fractions, pitch bands) lives in config/priors.yaml and config/classes.yaml,
never hardcoded in logic. Predictions record the config version used.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Band(BaseModel):
    """A [low, high] prior band. Sampled uniformly unless a module says otherwise."""

    low: float
    high: float

    def clamp(self, x: float) -> float:
        return max(self.low, min(self.high, x))

    @property
    def mid(self) -> float:
        return 0.5 * (self.low + self.high)


def _band(d: dict) -> Band:
    return Band(low=d["low"], high=d["high"])


class Priors:
    """Typed accessor over priors.yaml. Raw dict kept on `.raw` for anything not
    yet promoted to a typed property."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def version(self) -> int:
        return int(self.raw["version"])

    @property
    def air_density(self) -> float:
        return float(self.raw["air_density_kg_m3"])

    @property
    def gravity(self) -> float:
        return float(self.raw["gravity_m_s2"])

    @property
    def efficiency(self) -> Band:
        return _band(self.raw["motor_esc_efficiency"])

    def energy_density(self, chemistry: str = "lipo") -> Band:
        return _band(self.raw["battery_energy_density_wh_kg"][chemistry])

    @property
    def drag_coefficient(self) -> Band:
        return _band(self.raw["drag_coefficient"])

    def pitch_to_diameter(self, drone_class: str) -> Band:
        table = self.raw["pitch_to_diameter_by_class"]
        return _band(table.get(drone_class, table["unknown"]))

    def battery_mass_fraction(self, drone_class: str) -> Band:
        table = self.raw["battery_mass_fraction_by_class"]
        return _band(table.get(drone_class, table["unknown"]))

    @property
    def c_t_fallback(self) -> Band:
        return _band(self.raw["prop_coefficient_fallback"]["c_t"])

    @property
    def c_p_fallback(self) -> Band:
        return _band(self.raw["prop_coefficient_fallback"]["c_p"])

    @property
    def per_motor_rpm_spread(self) -> Band:
        return _band(self.raw["per_motor_rpm_spread_fraction"])


class Classes:
    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def version(self) -> int:
        return int(self.raw["version"])

    def get(self, name: str) -> dict:
        return self.raw["classes"].get(name, self.raw["classes"]["unknown"])

    @property
    def names(self) -> list[str]:
        return list(self.raw["classes"].keys())


@cache
def load_priors(path: str | Path | None = None) -> Priors:
    p = Path(path) if path else _CONFIG_DIR / "priors.yaml"
    with open(p) as f:
        return Priors(yaml.safe_load(f))


@cache
def load_classes(path: str | Path | None = None) -> Classes:
    p = Path(path) if path else _CONFIG_DIR / "classes.yaml"
    with open(p) as f:
        return Classes(yaml.safe_load(f))
