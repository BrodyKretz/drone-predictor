"""API layer: the framework-free helper always runs; the HTTP routes run only
when the optional `serve` extra (FastAPI + an httpx-backed TestClient) is present."""

import asyncio
import io
import json

import numpy as np
import pytest
from scipy.io import wavfile

from augur.api import MAX_MC_SAMPLES, identify_from_upload, predict_from_uploads, reference_info
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


def test_identify_helper_accepts_stereo_float_wav():
    rate, mono = wavfile.read(io.BytesIO(_wav_bytes()))
    stereo = (np.stack([mono, mono], axis=1) / 32767).astype(np.float32)
    buf = io.BytesIO()
    wavfile.write(buf, rate, stereo)
    assert identify_from_upload(buf.getvalue(), n=2000).status == "ok"


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


# --- HTTP layer: skipped unless the serve extra is installed ----------------

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from augur.api import RequestSizeLimitMiddleware, create_app  # noqa: E402


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


def _run_middleware(inner_app, headers, messages=(), scope_type="http"):
    sent = []
    pending = list(messages)

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": scope_type, "headers": headers}
    asyncio.run(RequestSizeLimitMiddleware(inner_app)(scope, receive, send))
    return sent


@pytest.mark.parametrize("declared_length", [b"5000", b"9" * 5000],
                         ids=["just-over", "5000-digits"])
def test_size_limit_rejects_declared_oversize_before_reading(monkeypatch, declared_length):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)

    async def inner_app(scope, receive, send):
        raise AssertionError("the app should never run")

    sent = _run_middleware(inner_app, [(b"content-length", declared_length)])
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("chunk_sizes, is_rejected", [([400, 400, 200], False),
                                                      ([400, 400, 201], True)])
def test_size_limit_counts_streamed_bytes(monkeypatch, chunk_sizes, is_rejected):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)
    messages = [{"type": "http.request", "body": b"x" * size, "more_body": i < len(chunk_sizes) - 1}
                for i, size in enumerate(chunk_sizes)]

    async def draining_app(scope, receive, send):
        while (await receive())["more_body"]:
            pass

    sent = _run_middleware(draining_app, [], messages)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == (
        [413] if is_rejected else [])


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


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
def test_size_limit_only_applies_to_http(monkeypatch, scope_type):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)
    calls = []

    async def inner_app(scope, receive, send):
        calls.append(scope["type"])

    sent = _run_middleware(inner_app, [(b"content-length", b"5000")], scope_type=scope_type)
    assert calls == [scope_type]
    assert sent == []

