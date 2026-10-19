# Augur web UI

A Vite + React frontend for Augur's live identification. Drop a recording (WAV, MP3
or OGG — decoded in the browser), play one of the 8 s simulated samples, or record
from the mic. Every 600 ms the latest 3 s are sent to `POST /identify`; the page
shows the best-guess drone class with its probability and the physical estimates as
90% intervals that update live. The guess names a *class*, not a model, and can be
fooled by other tonal sounds (fans, mains hum) — see `docs/design/2026-10-14-live-identify.md`.

Facts you know (blades per prop, prop size, motors) are sent with each window;
changing them after a clip ends re-analyzes its last window.

## Run

Two processes: the API backend and this dev server.

```bash
# 1. backend (from the repo root)
./.venv/bin/pip install -e ".[serve]"
./.venv/bin/augur serve   # or: ./.venv/bin/uvicorn augur.api:app --reload --reload-dir src

# 2. frontend (from web/)
npm install
npm run dev                             # http://localhost:5173
npm test                                # node:test for src/lib
```

Open http://localhost:5173.

The API base URL defaults to `http://127.0.0.1:8000`; override with `VITE_API_URL`.
