# drone-predictor

**Augur — Multimodal Drone Property Inference** (working name)

Predicts a drone's physical and performance properties from any combination of
four inputs — **sound, verbal spec, image, video** — and gets strictly more
confident as more inputs are supplied. Every output is a *calibrated
distribution*, never a point estimate.

## What it estimates

RPM, per-motor RPM spread, thrust per motor, total thrust, weight, shaft power,
electrical power, disk loading, frame size, thrust-to-weight ratio, drone class,
battery capacity (Wh, and mAh when cell count is known), battery mass fraction,
feature inventory, and flight endurance.

## Core principle

Each input is an independent constraint. Inputs are fused as likelihoods so that
adding an input can only narrow the posterior, never widen it (given consistent
evidence). Per prediction, the system reports *which inputs tightened which
variables*.

When a quantity is genuinely unknowable from the given inputs, the system widens
the interval — it does not invent a number.

## Status

**Working sound + verbal → calibrated distributions pipeline.** Phases 0–4 are
complete and tested; image/video/calibration are scaffolded with the
self-contained logic implemented and the external-data/API parts honestly
stubbed. See **[`PLAN.md`](PLAN.md)** for live phase status and next steps, and
**[`docs/SPEC.md`](docs/SPEC.md)** for the full spec.

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python scripts/make_demo_sample.py
./.venv/bin/augur predict --audio data/demo/drone.wav --verbal data/demo/spec.json
./.venv/bin/pytest tests/ -q
```

| phase | status |
|---|---|
| 0 Scaffold · 1 Simulator · 2 Audio→RPM · 3 Physics+verbal · 4 Fusion | ✅ done + tested |
| 5 Image (geometry + VLM parser done; live VLM call stubbed) | 🟡 partial |
| 6 Video (maneuver seg + coast-down mass done; pixel tracking stubbed) | 🟡 partial |
| 7 Calibration (conformal done; needs golden set for fit + metric table) | 🟡 partial |

## Stack

Python 3.11+ · numpy/scipy/librosa · pydantic/pyyaml · opencv/norfair ·
anthropic (VLM) · fastapi/typer · pytest/hypothesis
