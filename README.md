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

🚧 Scaffolding. Build order follows the phased plan:

0. Scaffold — repo, types (`Distribution` + pydantic models), config, manifest
1. Forward simulator (the physics test oracle — built first)
2. Audio → RPM
3. Physics core + priors + verbal
4. Fusion skeleton
5. Image module (VLM extraction)
6. Video module (tracking, coast-down mass)
7. Calibration + eval harness + hardening

See the project spec for the full physics reference, module specs, and
acceptance criteria.

## Stack

Python 3.11+ · numpy/scipy/librosa · pydantic/pyyaml · opencv/norfair ·
anthropic (VLM) · fastapi/typer · pytest/hypothesis
