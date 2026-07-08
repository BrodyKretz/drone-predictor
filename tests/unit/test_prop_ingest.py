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


@pytest.fixture
def uiuc_dir_with_blades(uiuc_dir):
    (uiuc_dir / "da4022_10x4.7_3b_static_md.txt").write_text(_FILE_5x3)
    return uiuc_dir


def test_parse_prop_name():
    assert prop_ingest.parse_prop_name("apcsf_10x4.7_static_rd.txt") == (10.0, 4.7)
    assert prop_ingest.parse_prop_name("nr640_4.2x4_static.txt") == (4.2, 4.0)
    assert prop_ingest.parse_prop_name("readme.txt") is None


def test_parse_prop_name_mm_makers_convert_to_inches():
    # Volume-2 micro props are published in mm (site: "140 mm X 45 mm").
    d, p = prop_ingest.parse_prop_name("vp_140x45_static_0590rd.txt")
    assert d == pytest.approx(140 / 25.4)
    assert p == pytest.approx(45 / 25.4)
    d, p = prop_ingest.parse_prop_name("kpf_96x70_static_0382rd.txt")
    assert d == pytest.approx(96 / 25.4)
    assert p == pytest.approx(70 / 25.4)


def test_parse_prop_name_restores_dropped_decimals():
    # Site lists these as 12.5x9 and 12x6.5; filenames drop the '.'.
    assert prop_ingest.parse_prop_name("ancf_125x9_static_0905od.txt") == (12.5, 9.0)
    assert prop_ingest.parse_prop_name("ancf_12x65_static_0806od.txt") == (12.0, 6.5)
    assert prop_ingest.parse_prop_name("apce_14x85_static_1014od.txt") == (14.0, 8.5)
    assert prop_ingest.parse_prop_name("ancf_125x75_static_0914od.txt") == (12.5, 7.5)


def test_parse_prop_name_keeps_genuine_high_pitch():
    # Real Aeronaut CAM designations with pitch > diameter — must not be rescaled.
    assert prop_ingest.parse_prop_name("ancf_12x13_static_0780od.txt") == (12.0, 13.0)
    assert prop_ingest.parse_prop_name("apcsp_7x9_static_2897cm.txt") == (7.0, 9.0)


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


def test_blade_count_parsed_from_filename(uiuc_dir_with_blades):
    """UIUC marks 3/4-blade variants with a _3b_/_4b_ suffix; unmarked files are
    the standard 2-blade props."""
    df = prop_ingest.build_prop_db(uiuc_dir_with_blades)
    assert "blade_count" in df.columns
    three = df[df["blade_count"] == 3]
    assert len(three) == 1
    assert three.iloc[0]["diameter_inch"] == pytest.approx(10.0)
    assert (df[df["blade_count"] == 2].shape[0]) == 2


def test_matching_filters_on_blade_count(uiuc_dir_with_blades, tmp_path, monkeypatch):
    df = prop_ingest.build_prop_db(uiuc_dir_with_blades)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)

    # Two 10-inch rows exist (2-blade at Ct 0.1130, 3-blade at Ct 0.1310);
    # asserting blade count must select only the matching one.
    u = np.linspace(0.0, 0.999, 100)
    c_t, _ = prop_db.coefficients_from_streams(10.0, u, blade_count=2)
    assert np.allclose(c_t, 0.1130)
    c_t, _ = prop_db.coefficients_from_streams(10.0, u, blade_count=3)
    assert np.allclose(c_t, 0.1310)
    assert prop_db.matched_prop_count(10.0) == 2
    assert prop_db.matched_prop_count(10.0, blade_count=2) == 1


def test_coefficients_from_streams_uses_matched_rows(uiuc_dir, tmp_path, monkeypatch):
    """The CRN path: pre-drawn uniforms index the matched rows, both coefficients
    correlated from the same prop."""
    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)

    u = np.linspace(0.0, 0.999, 200)
    coeffs = prop_db.coefficients_from_streams(10.0, u)
    assert coeffs is not None
    c_t, c_p = coeffs
    assert np.allclose(c_t, 0.1130)  # single matched 10x4.7 row
    assert np.allclose(c_p, 0.0503)

    assert prop_db.coefficients_from_streams(30.0, u) is None
    assert prop_db.coefficients_from_streams(None, u) is None
    assert prop_db.matched_prop_count(10.0) == 1
    assert prop_db.matched_prop_count(30.0) == 0


def test_posterior_uses_prop_db_when_diameter_asserted(uiuc_dir, tmp_path, monkeypatch):
    from augur.fusion.posterior import FusionInputs, fuse
    from augur.types import VerbalSpec

    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    inputs = FusionInputs(verbal=VerbalSpec(num_motors=4, prop_diameter_inch=10.0,
                                            drone_class="racing"))

    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)
    with_db = fuse(inputs, n=4000, seed=3)
    assert any("measured prop" in a for a in with_db.assumptions)

    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", tmp_path / "missing.parquet")
    fallback = fuse(inputs, n=4000, seed=3)
    assert any("fallback" in a for a in fallback.assumptions)

    # One matched prop pins C_T/C_P -> thrust and power tighten vs the wide band.
    for var in ("total_thrust_n", "electrical_power_w"):
        w_db = with_db.variables[var].interval_high - with_db.variables[var].interval_low
        w_fb = fallback.variables[var].interval_high - fallback.variables[var].interval_low
        assert w_db < w_fb, f"prop_db did not tighten {var}"


def test_prop_db_falls_back_when_no_match(uiuc_dir, tmp_path, monkeypatch):
    df = prop_ingest.build_prop_db(uiuc_dir)
    out = tmp_path / "prop_db.parquet"
    prop_ingest.save_prop_db(df, out)
    monkeypatch.setattr(prop_db, "_PROP_DB_PATH", out)

    # No prop near 30" in the DB -> fall back to the wide config band.
    c_t, _ = prop_db.sample_coefficients(30.0, n=2000, rng=np.random.default_rng(0))
    assert c_t.min() < 0.1130 or c_t.max() > 0.1130  # spread of the fallback band
