"""Common observation schema.

Every input module emits `Observation`s: a soft constraint on one output
variable, with a Gaussian likelihood width. The fusion layer weights Monte Carlo
samples by how well they match all available observations. Verbal facts and the
audio RPM estimate are handled specially (as priors/proposals) rather than as
generic observations — see posterior.py.
"""

from __future__ import annotations

from pydantic import BaseModel


class Observation(BaseModel):
    """A soft constraint: `variable` is observed to be `value` with Gaussian
    width `sigma` (same units as the variable). `source` is the modality."""

    variable: str
    value: float
    sigma: float
    source: str
    note: str | None = None

    def model_post_init(self, _ctx) -> None:
        if self.sigma <= 0:
            raise ValueError(f"Observation sigma must be > 0, got {self.sigma}")
