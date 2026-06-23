"""Dataset manifest schema (data/manifest.parquet).

One row per sample, linking every sample to its modalities + ground truth.
Ground truth lives only on golden/synthetic rows. Splits are kept disjoint at
the DRONE level (never split one physical drone across train and test)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = {
    "sample_id": "string",
    "drone_id": "string",
    "audio_path": "string",
    "image_paths": "object",   # list[str]
    "video_path": "string",
    "verbal": "object",        # json dict
    "flight_state": "string",  # hover/climb/cruise/coast/mixed
    "truth": "object",         # json dict, golden/synthetic only
    "source": "string",        # self / public name / synthetic
    "split": "string",         # train / calib / test
    "license": "string",
}


class Split(str, Enum):
    train = "train"
    calib = "calib"
    test = "test"


def empty_manifest() -> pd.DataFrame:
    """An empty manifest with the correct columns and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in MANIFEST_COLUMNS.items()})
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_manifest(df)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "split" in df and len(df):
        bad = set(df["split"].dropna()) - {s.value for s in Split}
        if bad:
            raise ValueError(f"invalid split values: {bad}")
    # Drone-level split disjointness: no drone in more than one split.
    if len(df) and df["drone_id"].notna().any():
        per_drone_splits = df.dropna(subset=["drone_id"]).groupby("drone_id")["split"].nunique()
        leaked = per_drone_splits[per_drone_splits > 1]
        if len(leaked):
            raise ValueError(f"drones split across multiple splits (leakage): {list(leaked.index)}")


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    validate_manifest(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
