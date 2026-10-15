"""Live identification: a class posterior from audio, gated on hover (rule 6),
marginalizing blade count, sharpened by asserted facts, and honest when there's
no tone or the tone is mains hum."""

import numpy as np
import pytest

from augur.fusion.identify import identify
from augur.fusion.posterior import OUTPUT_UNITS
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import VerbalSpec

RACING = dict(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
              c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=0.55,
              battery_wh=55.5, drone_class="racing")


def _hover_rpm(spec: DroneSpec) -> float:
    d_m = spec.prop_diameter_inch * 0.0254
    return float(np.sqrt(spec.mass_kg * 9.80665 / (spec.num_motors * spec.c_t * 1.225 * d_m**4)) * 60.0)


def _clip(sample_rate: int = 44100, drift: float = 0.0):
    spec = DroneSpec(**RACING)
    return synth_waveform(spec, FlightCondition(rpm=_hover_rpm(spec), state="hover"),
                          sample_rate=sample_rate, duration_s=2.0, snr_db=25.0,
                          rpm_drift_fraction=drift, rng=np.random.default_rng(0))


def _hum(sample_rate: int = 44100):
    t = np.arange(2 * sample_rate) / sample_rate
    hum = sum(np.sin(2 * np.pi * 60.0 * k * t) / k for k in range(1, 8))
    return hum + np.random.default_rng(0).normal(0.0, 0.02, t.size), sample_rate


def _p(report, name: str) -> float:
    return next(c.probability for c in report.classes if c.name == name)


@pytest.fixture(scope="module")
def racing_report():
    sig, sr = _clip()
    return identify(sig, sr, n=2000)


def test_class_and_blade_posteriors_sum_to_one(racing_report):
    assert racing_report.status == "ok"
    assert sum(c.probability for c in racing_report.classes) == pytest.approx(1.0)
    assert sum(b.probability for b in racing_report.blades) == pytest.approx(1.0)


def test_racing_clip_ranks_racing_first(racing_report):
    assert racing_report.classes[0].name == "racing"


def test_unknown_blade_count_is_marginalized(racing_report):
    assert sorted(b.blades for b in racing_report.blades) == [2, 3]
    f0 = racing_report.acoustics.fundamental_hz
    rpm = racing_report.variables["rpm"]
    assert rpm.interval_low < f0 * 60 / 3 * 1.05
    assert rpm.interval_high > f0 * 60 / 2 * 0.95


def test_given_prop_size_sharpens_the_guess():
    sig, sr = _clip()
    assert _p(identify(sig, sr, VerbalSpec(prop_diameter_inch=5.0), n=2000), "racing") >= 0.9


def test_given_class_is_the_only_hypothesis():
    sig, sr = _clip()
    report = identify(sig, sr, VerbalSpec(drone_class="racing"), n=2000)
    assert [c.name for c in report.classes] == ["racing"]
    assert report.classes[0].probability == pytest.approx(1.0)


def test_throttle_ramp_is_not_hover_so_classes_stay_at_prior():
    sig, sr = _clip(drift=0.4)
    report = identify(sig, sr, n=2000)
    assert report.status == "not_hovering"
    assert all(c.probability == pytest.approx(0.25) for c in report.classes)
    assert any("not hover" in note for note in report.notes)


@pytest.mark.parametrize("kind", ["noise", "silence"])
def test_no_tone_reports_status_and_prior(kind):
    x = np.zeros(88200) if kind == "silence" else np.random.default_rng(0).normal(0.0, 0.3, 88200)
    report = identify(x, 44100, n=2000)
    assert report.status == "no_tone"
    assert report.acoustics.fundamental_hz is None
    assert all(c.probability == pytest.approx(0.25) for c in report.classes)


def test_mains_hum_is_flagged():
    hum, sr = _hum()
    assert any("mains hum" in note for note in identify(hum, sr, n=2000).notes)


def test_48k_audio_still_ranks_racing_first():
    sig, sr = _clip(sample_rate=48000)
    assert identify(sig, sr, n=2000).classes[0].name == "racing"


def test_prop_size_outside_every_class_still_returns_a_report():
    sig, sr = _clip()
    report = identify(sig, sr, VerbalSpec(prop_diameter_inch=40.0), n=2000)
    assert report.status == "no_class_fits"
    assert {"mass_kg", "rpm"} <= set(report.variables)


@pytest.mark.parametrize("kind", ["noise", "ramp", "hover"])
def test_facts_that_rule_out_every_class_say_so(kind):
    if kind == "noise":
        sig, sr = np.random.default_rng(0).normal(0.0, 0.3, 88200), 44100
    else:
        sig, sr = _clip(drift=0.4 if kind == "ramp" else 0.0)
    report = identify(sig, sr, VerbalSpec(prop_diameter_inch=40.0), n=2000)
    assert report.status == "no_class_fits"
    assert all(c.probability == 0.0 for c in report.classes)
    assert any("don't match" in note for note in report.notes)
    assert not any("doesn't fit" in note for note in report.notes)
    assert any("No steady blade-pass tone" in note for note in report.notes) == (kind == "noise")


def test_tone_that_fits_no_class_says_so():
    hum, sr = _hum()
    report = identify(hum, sr, VerbalSpec(drone_class="racing"), n=2000)
    assert report.status == "no_class_fits"
    assert all(c.probability == 0.0 for c in report.classes)
    assert any("doesn't fit" in note for note in report.notes)
    assert not any("don't match" in note for note in report.notes)


def test_one_sample_still_gives_finite_estimates():
    sig, sr = _clip()
    report = identify(sig, sr, n=1)
    assert report.status == "ok"
    assert set(report.variables) == set(OUTPUT_UNITS)
    assert all(np.isfinite(v.median) for v in report.variables.values())


def test_fewer_samples_than_hypotheses_still_reports():
    sig, sr = _clip(drift=0.4)
    assert identify(sig, sr, n=3).status == "not_hovering"


def test_identify_rejects_bad_arguments():
    sig, sr = _clip()
    with pytest.raises(ValueError, match="samples"):
        identify(np.array([]), sr)
    with pytest.raises(ValueError, match="sample_rate"):
        identify(sig, 0)
    with pytest.raises(ValueError, match="n must be at least 1"):
        identify(sig, sr, n=0)


def test_window_shorter_than_two_seconds_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        identify(np.random.default_rng(0).normal(0.0, 0.3, 44100), 44100)
