"""HTTP routes through FastAPI's TestClient. They run only when the serve extra and
httpx (which TestClient needs) are installed; the helpers they wrap are tested in
test_api.py without either."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from augur.api import MAX_MC_SAMPLES, create_app
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

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


def test_identify_route(client):
    r = client.post("/identify", files={"audio": ("w.wav", _wav_bytes(), "audio/wav")},
                    data={"samples": "2000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["classes"][0]["name"] in {"racing", "cinematic", "survey", "military"}
    assert len(body["acoustics"]["spectrum_freq_hz"]) == len(body["acoustics"]["spectrum_db"])


def test_identify_route_applies_the_known_facts_in_the_spec_field(client):
    wav = RACING_SAMPLE.read_bytes()
    r = client.post("/identify", files={"audio": ("racing.wav", wav, "audio/wav")},
                    data={"spec": json.dumps({"prop_diameter_inch": 5}), "samples": "2000"})
    assert r.status_code == 200, r.text
    body = r.json()
    prop = body["variables"]["prop_diameter_inch"]
    assert body["classes"][0]["name"] == "racing"
    assert body["classes"][0]["probability"] >= 0.9
    assert prop["interval_low"] == prop["median"] == prop["interval_high"] == 5.0


def test_identify_route_non_wav_is_422(client):
    r = client.post("/identify", files={"audio": ("a.mp3", b"ID3\x04 nope", "audio/mpeg")})
    assert r.status_code == 422


def test_reference_route(client):
    r = client.get("/reference")
    assert r.status_code == 200
    assert "mass_kg" in r.json()["bands"]


@pytest.mark.parametrize("path", ["/identify", "/predict"])
def test_route_truncated_wav_is_422(client, path):
    r = client.post(path, files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 422
    assert "not a valid WAV file" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/identify", "/predict"])
def test_route_sample_count_over_the_cap_is_422(client, path):
    r = client.post(path, files={"audio": ("w.wav", _wav_bytes(), "audio/wav")},
                    data={"samples": str(MAX_MC_SAMPLES + 1)})
    assert r.status_code == 422
    assert "samples must be between" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/identify", "/predict"])
def test_route_oversized_audio_upload_is_413(client, monkeypatch, path):
    wav = _wav_bytes()
    monkeypatch.setattr("augur.api.MAX_UPLOAD_BYTES", len(wav) - 1)
    r = client.post(path, files={"audio": ("w.wav", wav, "audio/wav")}, data={"samples": "2000"})
    assert r.status_code == 413
    assert r.json()["detail"] == "audio upload larger than 16 MiB"


def test_identify_route_accepts_audio_at_the_upload_limit(client, monkeypatch):
    wav = _wav_bytes()
    monkeypatch.setattr("augur.api.MAX_UPLOAD_BYTES", len(wav))
    r = client.post("/identify", files={"audio": ("w.wav", wav, "audio/wav")},
                    data={"samples": "2000"})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path, field", [("/identify", "audio"), ("/predict", "audio"),
                                         ("/predict", "verbal")])
def test_route_request_body_over_the_limit_is_413(client, monkeypatch, path, field):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)
    r = client.post(path, files={field: ("f.bin", b"x" * 2000, "application/octet-stream")},
                    headers={"origin": "http://localhost:5173"})
    assert r.status_code == 413
    assert r.json()["detail"] == "request body too large"
    assert r.headers["access-control-allow-origin"] == "*"


def test_route_request_at_the_size_limit_is_accepted(client, monkeypatch):
    request = client.build_request("POST", "/identify",
                                   files={"audio": ("w.wav", _wav_bytes(), "audio/wav")},
                                   data={"samples": "2000"})
    request_bytes = int(request.headers["content-length"])
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", request_bytes - 1)
    assert client.send(request).status_code == 413
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", request_bytes)
    assert client.send(request).status_code == 200


def test_route_chunked_body_over_the_limit_is_413(client, monkeypatch):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)

    def chunked_upload():
        yield b'--x\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n\r\n'
        for _ in range(20):
            yield b"A" * 100

    r = client.post("/identify", content=chunked_upload(),
                    headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
    assert r.json()["detail"] == "request body too large"


@pytest.mark.parametrize("content_length", ["abc", "-1", "1.5"])
def test_route_malformed_content_length_is_400(client, content_length):
    r = client.post("/identify", content=b"x", headers={"content-length": content_length})
    assert r.status_code == 400
    assert r.json()["detail"] == "invalid Content-Length"
