"""API layer: the framework-free helper always runs; the HTTP routes run only
when the optional `serve` extra (FastAPI + an httpx-backed TestClient) is present."""

import io
import json

import numpy as np
import pytest
from scipy.io import wavfile

from augur.api import predict_from_uploads
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform


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


# --- HTTP layer: skipped unless the serve extra is installed ----------------

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from augur.api import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_route_audio_and_verbal(client):
    r = client.post(
        "/predict",
        files={"audio": ("drone.wav", _wav_bytes(), "audio/wav")},
        data={"verbal_json": VERBAL, "samples": "4000", "seed": "0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "rpm" in body["variables"]
    assert "audio" in body["inputs_used"] and "verbal" in body["inputs_used"]


def test_predict_route_verbal_file(client):
    r = client.post(
        "/predict",
        files={"verbal": ("spec.json", VERBAL, "application/json")},
        data={"samples": "2000"},
    )
    assert r.status_code == 200, r.text
    assert "rpm" in r.json()["variables"]


def test_predict_route_no_input_is_422(client):
    r = client.post("/predict", data={"samples": "2000"})
    assert r.status_code == 422


def test_predict_route_bad_verbal_is_422(client):
    r = client.post("/predict", data={"verbal_json": "{nope", "samples": "2000"})
    assert r.status_code == 422
