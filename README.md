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

**The sound + verbal → calibrated-distributions pipeline works end to end.**
Phases 0–4 are complete and tested (67 tests). Image, video, and calibration are
scaffolded: the self-contained logic is implemented, and the parts that need
external data or a live API are honestly stubbed (they raise rather than fake a
result). See **[`PLAN.md`](PLAN.md)** for live phase status and next steps, and
**[`docs/SPEC.md`](docs/SPEC.md)** for the full spec.

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q                      # 83 tests

./.venv/bin/pip install -e ".[serve]" && ./.venv/bin/augur serve   # HTTP API on :8000
```

A web UI (Vite + React) lives in [`web/`](web/) — upload audio + a verbal spec and
see the distributions as interval bars with confidence colors and per-input
attribution. Run `augur serve`, then `npm install && npm run dev` in `web/`.

| phase | status |
|---|---|
| 0 Scaffold · 1 Simulator · 2 Audio→RPM · 3 Physics+verbal · 4 Fusion | ✅ done + tested |
| CLI · HTTP API (`/predict`, `/health`) · property-based round-trip suite | ✅ done + tested |
| 5 Image (geometry + VLM parser done; live VLM call stubbed) | 🟡 partial |
| 6 Video (maneuver seg + coast-down mass done; pixel tracking stubbed) | 🟡 partial |
| 7 Calibration (conformal done; needs golden set for fit + metric table) | 🟡 partial |

## Stack

Python 3.11+ · numpy/scipy/librosa · pydantic/pyyaml · opencv/norfair ·
anthropic (VLM) · fastapi/typer · pytest/hypothesis
