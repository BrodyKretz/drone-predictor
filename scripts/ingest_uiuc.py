"""Build config/prop_db.parquet from UIUC / APC propeller data.

STATUS: stub. Needs the raw data downloaded + license verified (record provenance
in data/public/SOURCES.md before ingest).

Target output schema (consumed by augur.physics.prop_db):
    diameter_inch: float
    pitch_inch: float
    c_t_static: float   # thrust coefficient near static/zero advance ratio
    c_p_static: float   # power coefficient likewise
    source: str         # "UIUC" | "APC"

The UIUC Propeller Data Site publishes per-prop performance files (C_T, C_P, eta
vs advance ratio J). For each prop, take the J->0 (static) end, or the lowest-J
measured point, as c_t_static / c_p_static. APC performance files give similar
curves. Once written, prop_db.sample_coefficients() automatically prefers this
over the config fallback bands.
"""

from __future__ import annotations


def main():
    raise NotImplementedError(
        "ingest_uiuc is a stub. Download UIUC/APC prop data, verify licenses, "
        "record provenance in data/public/SOURCES.md, then parse each prop's "
        "low-advance-ratio C_T/C_P into config/prop_db.parquet with columns "
        "[diameter_inch, pitch_inch, c_t_static, c_p_static, source]."
    )


if __name__ == "__main__":
    main()
