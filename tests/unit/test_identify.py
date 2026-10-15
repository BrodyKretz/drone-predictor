"""Live identification: a class posterior from audio, gated on hover (rule 6),
marginalizing blade count, sharpened by asserted facts, and honest when there's
no tone or the tone is mains hum."""

import numpy as np
import pytest

from augur.config import load_classes
from augur.fusion.identify import candidate_classes, identify
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


@pytest.mark.parametrize("freq_hz", [440.0, 60.0])
def test_lone_sine_is_no_tone_without_a_fundamental_or_hum_note(freq_hz):
    t = np.arange(88200) / 44100
    report = identify(np.sin(2 * np.pi * freq_hz * t), 44100, n=2000)
    assert report.status == "no_tone"
    assert report.acoustics.fundamental_hz is None
    assert not any("mains hum" in note for note in report.notes)


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


@pytest.mark.parametrize("kind, status", [("silence", "no_tone"), ("prop40", "no_class_fits")])
def test_prior_width_mass_stays_in_the_unknown_class_band(kind, status):
    if kind == "silence":
        report = identify(np.zeros(88200), 44100, n=2000)
    else:
        sig, sr = _clip()
        report = identify(sig, sr, VerbalSpec(prop_diameter_inch=40.0), n=2000)
    band = load_classes().get("unknown")["all_up_mass_kg"]
    mass = report.variables["mass_kg"]
    assert report.status == status
    assert band["low"] <= mass.interval_low and mass.interval_high <= band["high"]


def test_not_hovering_mass_stays_within_the_class_bands():
    sig, sr = _clip(drift=0.4)
    report = identify(sig, sr, n=2000)
    bands = [load_classes().get(c)["all_up_mass_kg"] for c in candidate_classes(None)]
    mass = report.variables["mass_kg"]
    assert report.status == "not_hovering"
    assert min(b["low"] for b in bands) <= mass.interval_low
    assert mass.interval_high <= max(b["high"] for b in bands)


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
    with pytest.raises(ValueError, match="n must be an integer"):
        identify(sig, sr, n=2000.0)
    with pytest.raises(ValueError, match="n must be an integer"):
        identify(sig, sr, n=True)


def test_identify_rejects_samples_that_are_not_mono_arrays():
    with pytest.raises(ValueError, match="1-D"):
        identify(np.zeros((88200, 2)), 44100)
    with pytest.raises(ValueError, match="numpy array"):
        identify([0.0] * 88200, 44100)


@pytest.mark.parametrize("sample_rate, seconds", [(44100, 1.0), (16000, 2.0), (22050, 2.0)])
def test_window_under_the_minimum_sample_count_is_rejected(sample_rate, seconds):
    noise = np.random.default_rng(0).normal(0.0, 0.3, int(seconds * sample_rate))
    with pytest.raises(ValueError, match="samples of audio"):
        identify(noise, sample_rate)


def test_minimum_window_of_noise_at_a_low_sample_rate_has_no_tone():
    noise = np.random.default_rng(0).normal(0.0, 0.3, 88200)
    assert identify(noise, 22050, n=2000).status == "no_tone"
