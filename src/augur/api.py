"""HTTP API for Augur — a thin FastAPI wrapper over the prediction pipeline.

Requires the optional `serve` extra (`pip install -e ".[serve]"`). The core
logic lives in `predict_from_uploads`, which is framework-free and unit-testable
without FastAPI installed; the FastAPI layer is just request plumbing on top.

Run it: `augur serve` (or `uvicorn augur.api:app`).

Note: this module deliberately does NOT use `from __future__ import annotations`.
FastAPI resolves parameter type hints at runtime, and stringized annotations turn
the locally-imported `UploadFile` into an unresolvable forward reference.
"""

import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np

from augur import __version__
from augur.audio import spectral
from augur.config import load_classes
from augur.fusion.identify import candidate_classes, identify
from augur.fusion.posterior import FusionInputs, fuse_distributions
from augur.pipeline import predict
from augur.types import IdentifyReport, PredictionReport, VerbalSpec

MAX_MC_SAMPLES = 20000
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
_TOO_LARGE = (413, "request body too large")


def _check_sample_count(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= MAX_MC_SAMPLES:
        raise ValueError(f"samples must be between 1 and {MAX_MC_SAMPLES}")


def predict_from_uploads(
    audio_bytes: bytes | None = None,
    verbal_json: str | bytes | None = None,
    n: int = 8000,
    seed: int = 0,
) -> PredictionReport:
    """Run a prediction from in-memory inputs.

    The pipeline reads from file paths, so audio bytes are written to a temp WAV
    for the duration of the call. Verbal JSON is validated up front so malformed
    input fails clearly here rather than deep inside fusion.
    """
    if audio_bytes is None and verbal_json is None:
        raise ValueError("provide at least one of audio or verbal")
    _check_sample_count(n)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        audio_path = None
        verbal_path = None

        if audio_bytes is not None:
            audio_path = tmpdir / "audio.wav"
            audio_path.write_bytes(audio_bytes)

        if verbal_json is not None:
            data = json.loads(verbal_json)
            if not isinstance(data, dict):
                raise ValueError("verbal spec must be a JSON object")
            VerbalSpec(**data)  # validate shape before writing
            verbal_path = tmpdir / "verbal.json"
            verbal_path.write_text(json.dumps(data))

        return predict(audio=audio_path, verbal=verbal_path, n=n, seed=seed)


def identify_from_upload(audio_bytes: bytes, spec_json: str | bytes | None = None,
                         n: int = 4000, seed: int = 0) -> IdentifyReport:
    """Identify the drone class from one in-memory WAV window (the live UI sends
    the latest few seconds). Raises ValueError on non-WAV or too-short audio, a sample
    count outside 1..MAX_MC_SAMPLES, or a spec that isn't a JSON object."""
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise ValueError(f"audio_bytes must be bytes, got {type(audio_bytes).__name__}")
    _check_sample_count(n)
    samples, sample_rate = spectral.read_wav_bytes(audio_bytes)
    verbal = None
    if spec_json:
        spec = json.loads(spec_json)
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        verbal = VerbalSpec(**spec)
    return identify(samples, sample_rate, verbal, n=n, seed=seed)


@cache
def reference_info(n: int = 4000) -> dict:
    """The no-input 99% band per variable (the UI's display scale) and the class
    descriptions from classes.yaml."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    out = fuse_distributions(FusionInputs(assume_hover_when_unknown=False), n=n, seed=0)
    bands = {v: [float(x) for x in np.quantile(d.samples, [0.005, 0.995])] for v, d in out.items()}
    classes = load_classes()
    return {"bands": bands,
            "classes": {c: classes.get(c).get("description", "") for c in candidate_classes(None)}}


class _RequestTooLarge(BaseException):
    """Not an Exception: FastAPI turns any Exception raised while it reads the body into a 400."""


def _declared_length_rejection(headers, limit):
    """The (status, detail) to reject with, judged from Content-Length alone, or None."""
    for name, value in headers:
        if name == b"content-length":
            if not value.isdigit():
                return 400, "invalid Content-Length"
            # 19+ digits is far past any limit, and int() refuses very long digit strings
            if len(value) > 18 or int(value) > limit:
                return _TOO_LARGE
    return None


async def _send_error(scope, receive, send, status_code, detail):
    from starlette.responses import JSONResponse  # lazy: starlette comes with the serve extra

    await JSONResponse({"detail": detail}, status_code=status_code)(scope, receive, send)


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies before the framework parses them: from the
    declared Content-Length when there is one, otherwise by counting bytes as they
    stream in (chunked uploads have no Content-Length)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = MAX_REQUEST_BYTES
        rejection = _declared_length_rejection(scope.get("headers", []), limit)
        if rejection:
            await _send_error(scope, receive, send, *rejection)
            return
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _RequestTooLarge:
            await _send_error(scope, receive, send, *_TOO_LARGE)


def create_app():
    """Build the FastAPI app. Imported lazily so the package works without the
    `serve` extra installed."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Augur",
        description="Multimodal drone property inference — calibrated distributions.",
        version=__version__,
    )

    app.add_middleware(RequestSizeLimitMiddleware)

    # Dev-permissive CORS so the local web UI (Vite, another port) can call the API.
    # Tighten allow_origins before any non-local deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/predict", response_model=PredictionReport)
    async def predict_route(
        audio: UploadFile | None = File(default=None, description="WAV recording"),
        verbal: UploadFile | None = File(default=None, description="JSON verbal spec"),
        verbal_json: str | None = Form(default=None, description="Verbal spec as inline JSON"),
        samples: int = Form(default=8000),
        seed: int = Form(default=0),
    ):
        audio_bytes = None
        if audio is not None:
            audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
            if len(audio_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="audio upload larger than 16 MiB")
        verbal_payload: str | bytes | None = None
        if verbal is not None:
            verbal_payload = await verbal.read()
        elif verbal_json is not None:
            verbal_payload = verbal_json

        try:
            return predict_from_uploads(audio_bytes, verbal_payload, n=samples, seed=seed)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/identify", response_model=IdentifyReport)
    async def identify_route(
        audio: UploadFile = File(..., description="WAV window, mono or stereo"),
        spec: str | None = Form(default=None, description="Known facts as JSON (optional)"),
        samples: int = Form(default=4000),
        seed: int = Form(default=0),
    ):
        audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="audio upload larger than 16 MiB")
        try:
            return identify_from_upload(audio_bytes, spec, n=samples, seed=seed)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/reference")
    def reference():
        return reference_info()

    return app


def __getattr__(name: str):
    # Lazy module-level `app` so `uvicorn augur.api:app` works without forcing a
    # FastAPI import on every `import augur.api`.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
