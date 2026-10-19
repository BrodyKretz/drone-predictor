# drone-predictor

**Augur — Passive Multimodal Drone Detection & Classification** (working name)

A counter-UAS research concept: characterize and classify an unknown drone from
its **passive signature** — sound, a verbal description, an image, or video —
without any cooperation from the target. From any combination of those inputs it
infers the drone's physical/performance properties and class (racing, cinematic,
survey, military), so different airframes can be told apart by what they emit.

This is a *detection/identification* concept in the spirit of counter-UAS work
(e.g. the kind of passive-sensing efforts pursued by defense primes and the U.S.
government) — **not** a tool for building or operating drones. Every output is a
*calibrated distribution*, never a point estimate, and confidence increases
strictly as more inputs are supplied.

## What it estimates

RPM, per-motor RPM spread, thrust per motor, total thrust, weight, shaft power,
electrical power, disk loading, frame size, thrust-to-weight ratio, drone class,
battery capacity (Wh, and mAh when cell count is known), battery mass fraction,
feature inventory, and flight endurance.

## Core principle

Each input is an independent constraint. Inputs are fused as likelihoods, so
adding one can only narrow the posterior, never widen it (given consistent
evidence). Every prediction reports *which inputs tightened which variables*.

When a quantity is genuinely unknowable from the given inputs, the interval stays
wide — the system does not invent a number to fill the gap.

## Status

**The sound + verbal → calibrated-distributions pipeline works end to end**
(225 tests). Image, video, and calibration logic are implemented and tested
against synthetic data and fake clients; what remains is external — real
footage, one live VLM run, and the golden-set measurements. See
**[`PLAN.md`](PLAN.md)** for live phase status and next steps, and
**[`docs/SPEC.md`](docs/SPEC.md)** for the full spec.

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q

./.venv/bin/pip install -e ".[serve]" && ./.venv/bin/augur serve   # HTTP API on :8000
```

A web UI (Vite + React) lives in [`web/`](web/): drop a recording, pick a sample or
record from the mic, and watch the drone-class guess and the interval estimates
update live. Run `augur serve`, then `npm install && npm run dev` in `web/`.

## Prop database (optional, local-only)

Thrust/power priors tighten when `config/prop_db.parquet` exists — a summary
(one static C_T/C_P per prop) derived from the
[UIUC Propeller Data Site](https://m-selig.ae.illinois.edu/props/propDB.html).
That data is citation-only (no redistribution license), so the parquet is **not
committed**; without it the pipeline honestly falls back to wide coefficient
bands. To build it locally:

```bash
curl -LO https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip
unzip UIUC-propDB.zip -d data/external/uiuc
./.venv/bin/python scripts/ingest_uiuc.py --src data/external/uiuc/UIUC-propDB
```

Provenance and citation live in [`data/public/SOURCES.md`](data/public/SOURCES.md).

| phase | status |
|---|---|
| 0 Scaffold · 1 Simulator · 2 Audio→RPM · 3 Physics+verbal · 4 Fusion | ✅ done + tested |
| CLI · HTTP API (`/predict`, `/identify`, `/reference`, `/health`) · property-based round-trip suite | ✅ done + tested |
| 5 Image (geometry + VLM extraction done; needs one real API run) | 🟡 near-complete |
| 6 Video (tracking + maneuver seg + coast-down mass done; needs real footage) | 🟡 near-complete |
| 7 Calibration (conformal + eval harness done; needs golden set for fit + metric table) | 🟡 partial |
| Prop database (UIUC ingest + fusion wiring done; parquet is local-only) | ✅ done + tested |

## Stack

Python 3.11+ · numpy/scipy/librosa · pydantic/pyyaml · opencv/norfair ·
anthropic (VLM) · fastapi/typer · pytest/hypothesis
