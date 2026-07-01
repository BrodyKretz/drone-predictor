# Augur web UI

A small Vite + React frontend for the Augur prediction API. Upload an audio clip
and/or a verbal spec, and see the calibrated distributions rendered as interval
bars — band width encodes relative uncertainty, color encodes the confidence
label, and each row shows which input tightened it most.

## Run

Two processes: the API backend and this dev server.

```bash
# 1. backend (from the repo root)
./.venv/bin/pip install -e ".[serve]"
./.venv/bin/augur serve                 # http://127.0.0.1:8000

# 2. frontend (from web/)
npm install
npm run dev                             # http://localhost:5173
```

Open http://localhost:5173. Generate demo inputs first with
`python scripts/make_demo_sample.py` (writes `data/demo/drone.wav` + `spec.json`).

The API base URL defaults to `http://127.0.0.1:8000`; override with `VITE_API_URL`.
