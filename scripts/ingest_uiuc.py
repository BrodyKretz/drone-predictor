"""CLI: build config/prop_db.parquet from a local UIUC/APC prop-data directory.

Download the data and verify its license first; record provenance in
data/public/SOURCES.md. Parsing logic and the expected file format live in
augur.prop_ingest.

Usage: python scripts/ingest_uiuc.py --src path/to/uiuc_data [--source UIUC]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from augur.prop_ingest import build_prop_db, save_prop_db

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "config" / "prop_db.parquet"


def main():
    ap = argparse.ArgumentParser(description="Build prop_db.parquet from UIUC/APC data.")
    ap.add_argument("--src", required=True, help="Directory of static-test files")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output parquet path")
    ap.add_argument("--source", default="UIUC", help="Provenance label for these props")
    args = ap.parse_args()

    df = build_prop_db(args.src, source=args.source)
    if df.empty:
        raise SystemExit(f"No parseable prop files found under {args.src}")
    save_prop_db(df, args.out)
    print(f"wrote {len(df)} props to {args.out}")


if __name__ == "__main__":
    main()
