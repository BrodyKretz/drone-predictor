"""Acoustic diagnostics for the UI: comb matching, flight-state evidence, and the
display spectrum. Nothing here may change an estimate, so the comb score and the
flight-state label must stay identical to what the RPM path computes."""

import numpy as np
import pytest

from augur.audio import diagnostics, hover, rpm, spectral
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import FlightState


def _spec() -> DroneSpec:
    return DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
                     c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=0.55,
                     battery_wh=55.5, drone_class="racing")


def _clip(drift: float = 0.0):
    return synth_waveform(_spec(), FlightCondition(rpm=12000.0, state="hover"), duration_s=2.0,
                          snr_db=30.0, rpm_drift_fraction=drift, rng=np.random.default_rng(0))


def _noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.3, 88200)


def test_match_harmonics_hits_and_misses():
    peaks = np.array([100.0, 200.0, 401.0])
    assert rpm.match_harmonics(100.0, peaks, 4, tol_hz=5.0) == [100.0, 200.0, None, 401.0]


def test_comb_score_counts_matched_harmonics():
    assert rpm._comb_score(100.0, np.array([100.0, 200.0, 401.0]), 4, tol_hz=5.0) == 0.75


def test_assess_flight_state_hover_evidence():
    sig, sr = _clip()
    ev = hover.assess_flight_state(sig, sr)
    assert ev.state == FlightState.hover
    assert ev.pitch_variation is not None and ev.pitch_variation <= ev.steady_threshold
    assert hover.classify_state(sig, sr) == ev.state


def test_assess_flight_state_ramp_evidence():
    sig, sr = _clip(drift=0.4)
    ev = hover.assess_flight_state(sig, sr)
    assert ev.state != FlightState.hover
    assert ev.pitch_variation > ev.steady_threshold
    assert hover.classify_state(sig, sr) == ev.state


def test_assess_flight_state_noise_is_unknown():
    ev = hover.assess_flight_state(_noise(), 44100)
    assert ev.state == FlightState.unknown
    assert ev.pitch_variation is None and ev.pitch_drift is None


def test_max_pool_keeps_every_spike():
    freqs = np.linspace(0.0, 5000.0, 2001)
    spec_db = np.full(freqs.size, -60.0)
    spikes = [137, 911, 1500]
    spec_db[spikes] = [-5.0, -10.0, 0.0]
    f, d = diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz=5000.0, bins=256)
    assert len(f) == len(d) == 256
    for i in spikes:
        assert d[f.index(float(freqs[i]))] == pytest.approx(spec_db[i])


def test_max_pool_passes_small_windows_through():
    freqs = np.linspace(0.0, 1000.0, 101)
    f, d = diagnostics.max_pool_spectrum(freqs, -np.linspace(0.0, 50.0, 101), f_max_hz=1000.0)
    assert f == [float(x) for x in freqs]
    assert max(d) == 0.0


def test_build_acoustics_covers_the_comb():
    sig, sr = _clip()
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    f0, conf, peaks = rpm.detect_fundamental(freqs, spec_db)
    ac = diagnostics.build_acoustics(freqs, spec_db, f0, peaks, conf, hover.assess_flight_state(sig, sr))
    assert ac.fundamental_hz == pytest.approx(12000.0 / 60.0 * 2, rel=0.02)
    assert len(ac.harmonics) == rpm.N_HARMONICS
    assert all(h.matched_hz is not None for h in ac.harmonics)
    assert ac.spectrum_freq_hz[-1] >= rpm.N_HARMONICS * ac.fundamental_hz
    assert max(ac.spectrum_db) == 0.0
    assert min(ac.spectrum_db) >= diagnostics.SPECTRUM_DB_FLOOR


def test_build_acoustics_without_tone():
    noise = _noise(1)
    freqs, spec_db = spectral.average_spectrum(noise, 44100)
    ac = diagnostics.build_acoustics(freqs, spec_db, 0.0, np.array([]), 0.0,
                                     hover.assess_flight_state(noise, 44100))
    assert ac.fundamental_hz is None and ac.harmonics == []
    assert ac.spectrum_freq_hz[-1] <= rpm.DETECTION_FMAX_HZ


@pytest.mark.parametrize("fill", [0.0, np.nan], ids=["digital-silence", "nan-samples"])
def test_silent_or_corrupt_audio_draws_a_flat_spectrum_at_the_floor(fill):
    samples = np.full(88200, fill)
    freqs, spec_db = spectral.average_spectrum(samples, 44100)
    ac = diagnostics.build_acoustics(freqs, spec_db, 0.0, np.array([]), 0.0,
                                     hover.assess_flight_state(samples, 44100))
    assert all(np.isfinite(ac.spectrum_db))
    assert set(ac.spectrum_db) == {diagnostics.SPECTRUM_DB_FLOOR}


@pytest.mark.parametrize("bad_db", [np.nan, np.inf, -np.inf])
def test_max_pool_never_returns_a_non_finite_point(bad_db):
    freqs = np.linspace(0.0, 1000.0, 101)
    spec_db = -np.linspace(0.0, 50.0, 101)
    spec_db[40] = bad_db
    _, d = diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz=1000.0)
    assert all(np.isfinite(d))


def test_quiet_audio_keeps_its_peak_relative_shape():
    tone = np.sin(2 * np.pi * 440.0 * np.arange(88200) / 44100)
    loud = spectral.average_spectrum(tone, 44100)
    quiet = spectral.average_spectrum(10 ** (-90 / 20) * tone, 44100)
    _, loud_db = diagnostics.max_pool_spectrum(*loud, f_max_hz=5000.0)
    _, quiet_db = diagnostics.max_pool_spectrum(*quiet, f_max_hz=5000.0)
    assert max(quiet_db) == 0.0
    assert quiet_db == pytest.approx(loud_db, abs=0.01)


def test_comb_tolerance_rejects_fewer_than_two_bins():
    with pytest.raises(ValueError, match="bins"):
        rpm.comb_tolerance_hz(np.array([0.0]))


def test_match_harmonics_rejects_bad_arguments():
    peaks = np.array([100.0, 200.0])
    with pytest.raises(ValueError, match="n_harmonics"):
        rpm.match_harmonics(100.0, peaks, 0, tol_hz=5.0)
    with pytest.raises(ValueError, match="tol_hz"):
        rpm.match_harmonics(100.0, peaks, 4, tol_hz=-1.0)


def test_assess_flight_state_rejects_bad_arguments():
    with pytest.raises(ValueError, match="sample_rate"):
        hover.assess_flight_state(_noise(), 0)
    with pytest.raises(ValueError, match="steady_threshold"):
        hover.assess_flight_state(_noise(), 44100, steady_threshold=0.0)


def test_max_pool_rejects_bad_arguments():
    freqs = np.linspace(0.0, 1000.0, 101)
    spec_db = -np.linspace(0.0, 50.0, 101)
    with pytest.raises(ValueError, match="length"):
        diagnostics.max_pool_spectrum(freqs, spec_db[:-1], f_max_hz=1000.0)
    with pytest.raises(ValueError, match="bins"):
        diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz=1000.0, bins=0)
    with pytest.raises(ValueError, match="f_max_hz"):
        diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz=-1.0)
