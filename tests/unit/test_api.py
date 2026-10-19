"""API helpers: the framework-free functions behind the HTTP routes
(`predict_from_uploads`, `identify_from_upload`, `reference_info`). They run without
the serve extra; the middleware and the routes are tested in test_api_middleware.py
and test_api_routes.py."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from augur.api import MAX_MC_SAMPLES, identify_from_upload, predict_from_uploads, reference_info
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform

RACING_SAMPLE = Path(__file__).resolve().parents[2] / "web" / "public" / "templates" / "racing.wav"


def _wav_bytes(rpm: float = 12000.0, seed: int = 0) -> bytes:
    spec = DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0,
                     blade_count=2, c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8,
                     mass_kg=0.55, battery_wh=55.5, drone_class="racing")
    cond = FlightCondition(rpm=rpm, state="hover")
    sig, sr = synth_waveform(spec, cond, duration_s=2.0, snr_db=30.0,
                             rng=np.random.default_rng(seed))
    pcm = np.int16(sig / np.max(np.abs(sig)) * 32767)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return buf.getvalue()


VERBAL = json.dumps({"num_motors": 4, "prop_diameter_inch": 5.0, "blade_count": 2,
                     "drone_class": "racing"})
NON_OBJECT_JSON = ["[1, 2]", "null"]
BAD_SAMPLE_COUNTS = [0, MAX_MC_SAMPLES + 1, True, 2000.0]


def test_helper_audio_plus_verbal():
    report = predict_from_uploads(_wav_bytes(), VERBAL, n=4000, seed=0)
    assert "rpm" in report.variables
    assert report.variables["rpm"].confidence in {"high", "medium"}
    assert set(report.inputs_used) >= {"audio", "verbal"}


def test_helper_verbal_only():
    report = predict_from_uploads(None, VERBAL, n=2000, seed=0)
    assert "rpm" in report.variables
    assert "audio" not in report.inputs_used


def test_helper_requires_an_input():
    with pytest.raises(ValueError):
        predict_from_uploads(None, None)


def test_helper_rejects_malformed_verbal():
    with pytest.raises(json.JSONDecodeError):
        predict_from_uploads(None, "{not json")


def test_identify_helper_ranks_classes():
    report = identify_from_upload(_wav_bytes(), n=2000)
    assert report.status == "ok"
    assert sum(c.probability for c in report.classes) == pytest.approx(1.0)
    assert report.acoustics.fundamental_hz is not None


def test_identify_helper_reads_stereo_float_wav_like_mono_int16():
    mono_wav = _wav_bytes()
    rate, mono = wavfile.read(io.BytesIO(mono_wav))
    stereo = (np.stack([mono, mono], axis=1) / 32767).astype(np.float32)
    buf = io.BytesIO()
    wavfile.write(buf, rate, stereo)
    from_stereo = identify_from_upload(buf.getvalue(), n=2000)
    from_mono = identify_from_upload(mono_wav, n=2000)
    assert from_stereo.status == from_mono.status == "ok"
    assert from_stereo.classes == from_mono.classes
    assert from_stereo.variables["mass_kg"] == from_mono.variables["mass_kg"]


def test_identify_helper_applies_the_known_facts_in_the_spec():
    report = identify_from_upload(RACING_SAMPLE.read_bytes(), json.dumps({"prop_diameter_inch": 5}),
                                  n=2000)
    prop = report.variables["prop_diameter_inch"]
    assert report.classes[0].name == "racing"
    assert report.classes[0].probability >= 0.9
    assert prop.interval_low == prop.median == prop.interval_high == 5.0


def test_identify_helper_rejects_non_wav():
    with pytest.raises(ValueError):
        identify_from_upload(b"ID3\x04\x00\x00 not a wav", n=2000)


def test_identify_helper_rejects_too_short_audio():
    rate, mono = wavfile.read(io.BytesIO(_wav_bytes()))
    buf = io.BytesIO()
    wavfile.write(buf, rate, mono[: rate // 10])
    with pytest.raises(ValueError, match="at least"):
        identify_from_upload(buf.getvalue(), n=2000)


def test_identify_helper_rejects_non_bytes_audio():
    with pytest.raises(ValueError, match="audio_bytes"):
        identify_from_upload("not a wav", n=2000)
    with pytest.raises(ValueError, match="audio_bytes"):
        identify_from_upload(None, n=2000)


def test_reference_info_bands_and_classes():
    info = reference_info()
    low, high = info["bands"]["mass_kg"]
    assert 0 < low < high
    assert set(info["classes"]) == {"racing", "cinematic", "survey", "military"}


def test_reference_info_rejects_bad_sample_count():
    with pytest.raises(ValueError, match="positive integer"):
        reference_info(0)
    with pytest.raises(ValueError, match="positive integer"):
        reference_info(2000.0)
    with pytest.raises(ValueError, match="positive integer"):
        reference_info(True)


@pytest.mark.parametrize("raw", [b"RIFF", b"RIFF\x04\x00\x00\x00WAVE"])
def test_identify_helper_rejects_truncated_wav(raw):
    with pytest.raises(ValueError, match="not a valid WAV file"):
        identify_from_upload(raw, n=2000)


@pytest.mark.parametrize("spec", NON_OBJECT_JSON)
def test_identify_helper_rejects_non_object_spec(spec):
    with pytest.raises(ValueError, match="spec must be a JSON object"):
        identify_from_upload(_wav_bytes(), spec, n=2000)


@pytest.mark.parametrize("verbal", NON_OBJECT_JSON)
def test_helper_rejects_non_object_verbal(verbal):
    with pytest.raises(ValueError, match="verbal spec must be a JSON object"):
        predict_from_uploads(None, verbal, n=2000)


@pytest.mark.parametrize("n", BAD_SAMPLE_COUNTS)
def test_identify_helper_rejects_bad_sample_count(n):
    with pytest.raises(ValueError, match=f"samples must be between 1 and {MAX_MC_SAMPLES}"):
        identify_from_upload(_wav_bytes(), n=n)


@pytest.mark.parametrize("n", BAD_SAMPLE_COUNTS)
def test_helper_rejects_bad_sample_count(n):
    with pytest.raises(ValueError, match=f"samples must be between 1 and {MAX_MC_SAMPLES}"):
        predict_from_uploads(None, VERBAL, n=n)


def test_helpers_accept_the_maximum_sample_count():
    assert identify_from_upload(_wav_bytes(), n=MAX_MC_SAMPLES).status == "ok"
    assert "rpm" in predict_from_uploads(None, VERBAL, n=MAX_MC_SAMPLES).variables


@pytest.mark.parametrize("channels", [0, 3])
def test_identify_helper_rejects_wav_header_with_impossible_channel_count(channels):
    raw = bytearray(_wav_bytes())
    raw[22:24] = channels.to_bytes(2, "little")
    with pytest.raises(ValueError, match="not a valid WAV file"):
        identify_from_upload(bytes(raw), n=2000)
