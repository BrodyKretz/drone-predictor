"""Manifest schema, round-trip, and drone-level split-leakage guard."""

import pandas as pd
import pytest

from augur.data_manifest import (
    MANIFEST_COLUMNS,
    empty_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def test_empty_manifest_has_columns():
    df = empty_manifest()
    assert set(df.columns) == set(MANIFEST_COLUMNS)


def test_round_trip(tmp_path):
    df = empty_manifest()
    df.loc[0] = {
        "sample_id": "s1", "drone_id": "d1", "audio_path": "a.wav",
        "image_paths": ["i.jpg"], "video_path": None, "verbal": {"num_motors": 4},
        "flight_state": "hover", "truth": {"mass_kg": 0.5}, "source": "synthetic",
        "split": "train", "license": "self",
    }
    p = tmp_path / "manifest.parquet"
    save_manifest(df, p)
    loaded = load_manifest(p)
    assert loaded.loc[0, "sample_id"] == "s1"
    assert loaded.loc[0, "verbal"]["num_motors"] == 4


def test_invalid_split_rejected():
    df = empty_manifest()
    df.loc[0] = {c: None for c in MANIFEST_COLUMNS}
    df.loc[0, "split"] = "bogus"
    with pytest.raises(ValueError, match="invalid split"):
        validate_manifest(df)


def test_drone_split_leakage_rejected():
    df = empty_manifest()
    df.loc[0] = {c: None for c in MANIFEST_COLUMNS}
    df.loc[1] = {c: None for c in MANIFEST_COLUMNS}
    df.loc[:, "drone_id"] = "d1"
    df.loc[0, "split"] = "train"
    df.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(df)


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        validate_manifest(pd.DataFrame({"sample_id": ["x"]}))
