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
from pathlib import Path

from augur import __version__
from augur.pipeline import predict
from augur.types import PredictionReport, VerbalSpec


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

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        audio_path = None
        verbal_path = None

        if audio_bytes is not None:
            audio_path = tmpdir / "audio.wav"
            audio_path.write_bytes(audio_bytes)

        if verbal_json is not None:
            data = json.loads(verbal_json)
            VerbalSpec(**data)  # validate shape before writing
            verbal_path = tmpdir / "verbal.json"
            verbal_path.write_text(json.dumps(data))

        return predict(audio=audio_path, verbal=verbal_path, n=n, seed=seed)


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
        audio_bytes = await audio.read() if audio is not None else None
        verbal_payload: str | bytes | None = None
        if verbal is not None:
            verbal_payload = await verbal.read()
        elif verbal_json is not None:
            verbal_payload = verbal_json

        try:
            return predict_from_uploads(audio_bytes, verbal_payload, n=samples, seed=seed)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


def __getattr__(name: str):
    # Lazy module-level `app` so `uvicorn augur.api:app` works without forcing a
    # FastAPI import on every `import augur.api`.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
