"""Parse UIUC/APC propeller static-test files into config/prop_db.parquet.

The UIUC Propeller Data Site publishes per-propeller static-test files named like
``<maker>_<D>x<P>_static_<rig>.txt``, each a table of RPM vs C_T vs C_P at zero
advance ratio. We aggregate each prop to a single representative static (C_T,
C_P) and emit one row per prop; `prop_db.sample_coefficients` then draws across
props of similar diameter, so the natural spread between props becomes the prior
width.

This module only parses local files. Downloading the data and confirming its
license is a separate, deliberate step — record provenance in
``data/public/SOURCES.md`` before committing any derived database.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROP_DB_COLUMNS = ["diameter_inch", "pitch_inch", "c_t_static", "c_p_static", "source"]

# `10x4.7`, `4.2x4` — diameter x pitch, in inches. Case-insensitive on the 'x'.
_NAME_RE = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", re.IGNORECASE)

# Low RPM is Reynolds-sensitive and noisy; prefer the high-RPM plateau.
_RPM_FLOOR = 2000.0


def parse_prop_name(name: str) -> tuple[float, float] | None:
    """Extract (diameter_inch, pitch_inch) from a UIUC-style file or prop name."""
    m = _NAME_RE.search(name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _find_header(lines: list[str]) -> tuple[int, tuple[int, int, int] | None]:
    """Locate the column header and the indices of RPM, Ct, Cp (order-agnostic)."""
    for idx, line in enumerate(lines):
        toks = [t.lower() for t in line.split()]
        if "rpm" in toks and "ct" in toks and "cp" in toks:
            return idx, (toks.index("rpm"), toks.index("ct"), toks.index("cp"))
    return -1, None


def parse_static_file(path: Path) -> tuple[float, float] | None:
    """Aggregate one static-test file to a representative (c_t, c_p).

    Returns the median Ct/Cp over rows above the low-RPM floor (or all rows if
    none clear it). None if the file has no recognizable RPM/Ct/Cp table."""
    lines = Path(path).read_text().splitlines()
    header_idx, cols = _find_header(lines)
    if cols is None:
        return None
    i_rpm, i_ct, i_cp = cols

    rpms: list[float] = []
    cts: list[float] = []
    cps: list[float] = []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) <= max(cols):
            continue
        try:
            rpm = float(parts[i_rpm])
            ct = float(parts[i_ct])
            cp = float(parts[i_cp])
        except ValueError:
            continue
        rpms.append(rpm)
        cts.append(ct)
        cps.append(cp)

    if not cts:
        return None
    df = pd.DataFrame({"rpm": rpms, "c_t": cts, "c_p": cps})
    plateau = df[df["rpm"] >= _RPM_FLOOR]
    use = plateau if not plateau.empty else df
    return float(use["c_t"].median()), float(use["c_p"].median())


def build_prop_db(src_dir: str | Path, source: str = "UIUC",
                  pattern: str = "*static*.txt") -> pd.DataFrame:
    """Walk a directory of static-test files into the prop-db DataFrame.

    Files whose name lacks a DxP token, or that have no parseable table, are
    skipped. One row per prop."""
    src = Path(src_dir)
    rows = []
    for path in sorted(src.rglob(pattern)):
        dims = parse_prop_name(path.name)
        coeffs = parse_static_file(path)
        if dims is None or coeffs is None:
            continue
        rows.append({
            "diameter_inch": dims[0],
            "pitch_inch": dims[1],
            "c_t_static": coeffs[0],
            "c_p_static": coeffs[1],
            "source": source,
        })
    return pd.DataFrame(rows, columns=PROP_DB_COLUMNS)


def save_prop_db(df: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
