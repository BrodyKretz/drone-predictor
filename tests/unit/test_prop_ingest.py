"""UIUC/APC prop-data ingest. Validated against fixtures authored in the
documented file format (not redistributed UIUC data)."""

import numpy as np
import pandas as pd
import pytest

from augur import prop_ingest
from augur.physics import prop_db

# A static-test file as published: a header then RPM / Ct / Cp columns. Low-RPM
# rows are deliberately off-plateau to confirm they're excluded from the median.
_FILE_10x47 = """\
Static Test
          RPM            Ct            Cp
          800        0.0900        0.0400
         3000        0.1120        0.0500
         4500        0.1140        0.0506
"""

_FILE_5x3 = """\
          RPM            Ct            Cp
         3000        0.1300        0.0700
         6000        0.1320        0.0710
"""


@pytest.fixture
def uiuc_dir(tmp_path):
    (tmp_path / "apcsf_10x4.7_static_rd.txt").write_text(_FILE_10x47)
    (tmp_path / "gwsdd_5x3_static_kt.txt").write_text(_FILE_5x3)
    (tmp_path / "readme_no_dims.txt").write_text("not a prop file")
    return tmp_path


def test_parse_prop_name():
    assert prop_ingest.parse_prop_name("apcsf_10x4.7_static_rd.txt") == (10.0, 4.7)
    assert prop_ingest.parse_prop_name("nr640_4.2x4_static.txt") == (4.2, 4.0)
    assert prop_ingest.parse_prop_name("readme.txt") is None


def test_parse_static_file_uses_plateau_median(tmp_path):
    f = tmp_path / "apcsf_10x4.7_static_rd.txt"
    f.write_text(_FILE_10x47)
    c_t, c_p = prop_ingest.parse_static_file(f)
    # Median of the >=2000 RPM rows only: Ct {0.1120, 0.1140}, Cp {0.0500, 0.0506}.
    assert c_t == pytest.approx(0.1130)
    assert c_p == pytest.approx(0.0503)


def test_build_prop_db(uiuc_dir):
    df = prop_ingest.build_prop_db(uiuc_dir)
    assert list(df.columns) == prop_ingest.PROP_DB_COLUMNS
    assert len(df) == 2  # the no-dims file is skipped
    row = df[df["diameter_inch"] == 10.0].iloc[0]
    assert row["pitch_inch"] == pytest.approx(4.7)
    assert row["c_t_static"] == pytest.approx(0.1130)
    assert (df["source"] == "UIUC").all()


def test_build_then_save_roundtrip(uiuc_dir, tmp_path):
    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "config" / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    assert out.exists()
    reloaded = pd.read_parquet(out)
    assert len(reloaded) == 2


def test_prop_db_prefers_ingested_data(uiuc_dir, tmp_path, monkeypatch):
    """Once a parquet exists, sample_coefficients draws from it (tight), not the
    wide config fallback band."""
    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)

    c_t, c_p = prop_db.sample_coefficients(10.0, n=2000, rng=np.random.default_rng(0))
    # All samples come from the single matched 10x4.7 prop row.
    assert np.allclose(c_t, 0.1130)
    assert np.allclose(c_p, 0.0503)


def test_prop_db_falls_back_when_no_match(uiuc_dir, tmp_path, monkeypatch):
    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)

    # No prop near 30" in the DB -> fall back to the wide config band.
    c_t, _ = prop_db.sample_coefficients(30.0, n=2000, rng=np.random.default_rng(0))
    assert c_t.min() < 0.1130 or c_t.max() > 0.1130  # spread of the fallback band
