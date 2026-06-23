"""Propeller coefficient lookup.

Keyed by diameter and pitch. Pitch is usually unknown, so the public entry point
returns *distributions* over C_T / C_P rather than scalars. When
config/prop_db.parquet exists (built by scripts/ingest_uiuc.py from UIUC/APC
data) it is used; otherwise we fall back to coarse class bands from priors.yaml.

This keeps the whole pipeline runnable before any external data is ingested —
the cost is wider intervals, which is the honest behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from augur.config import load_priors

_PROP_DB_PATH = Path(__file__).resolve().parents[3] / "config" / "prop_db.parquet"


def _load_db() -> pd.DataFrame | None:
    if _PROP_DB_PATH.exists():
        return pd.read_parquet(_PROP_DB_PATH)
    return None


def sample_coefficients(
    diameter_inch: float | None,
    n: int,
    drone_class: str = "unknown",
    pitch_inch: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (C_T, C_P) sample arrays of length n.

    Currently uses the config fallback bands (uniform). Once prop_db.parquet is
    populated this should interpolate the measured C_T/C_P curves at the static /
    near-static advance ratio, conditioned on diameter and the pitch distribution.
    """
    rng = rng or np.random.default_rng()
    priors = load_priors()

    db = _load_db()
    if db is not None and diameter_inch is not None:
        c_t, c_p = _sample_from_db(db, diameter_inch, pitch_inch, n, rng)
        if c_t is not None:
            return c_t, c_p

    ct_band = priors.c_t_fallback
    cp_band = priors.c_p_fallback
    c_t = rng.uniform(ct_band.low, ct_band.high, size=n)
    c_p = rng.uniform(cp_band.low, cp_band.high, size=n)
    return c_t, c_p


def _sample_from_db(
    db: pd.DataFrame,
    diameter_inch: float,
    pitch_inch: float | None,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample C_T/C_P from the ingested database near the requested diameter.

    Expects columns: diameter_inch, pitch_inch, c_t_static, c_p_static. Selects
    rows within a diameter tolerance (and pitch tolerance if given), then draws
    with replacement so the spread of the matched rows becomes the prior width.
    """
    tol = 0.75  # inches
    mask = (db["diameter_inch"] - diameter_inch).abs() <= tol
    if pitch_inch is not None and "pitch_inch" in db:
        mask &= (db["pitch_inch"] - pitch_inch).abs() <= 1.0
    matched = db[mask]
    if matched.empty:
        return None, None
    idx = rng.integers(0, len(matched), size=n)
    return (
        matched["c_t_static"].to_numpy()[idx],
        matched["c_p_static"].to_numpy()[idx],
    )
