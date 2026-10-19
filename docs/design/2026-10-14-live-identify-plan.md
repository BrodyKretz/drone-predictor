# Live Drone Identification Implementation Plan

> **Superseded during execution** (see PLAN.md and the branch history): the minimum analysis window is `MIN_WINDOW_SAMPLES = 88200` samples (≈2 s at 44.1 kHz), not 0.25 s, and the test counts grew beyond the 164 / 9 quoted here.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-screen web UI that takes a dropped file, a demo clip or the mic, and live-updates a big "what kind of drone is this" guess plus interval estimates, backed by a new `/identify` endpoint.

**Architecture:** A new `augur/fusion/identify.py` scores each (class, blade count) hypothesis with hover-physics evidence by reusing the existing plug-in fusion (`fuse_distributions`), then mixes the per-hypothesis posteriors. FastAPI gains `POST /identify` and `GET /reference`. The React app is rebuilt around a live session hook that posts the latest 3 s of audio every 600 ms.

**Tech Stack:** Python 3.12 (numpy, scipy, pydantic 2, FastAPI via the `serve` extra), pytest + Hypothesis; React 18 + Vite 5; `node --test`.

**Spec:** [`docs/design/2026-10-14-live-identify.md`](2026-10-14-live-identify.md)

## Global Constraints

- Run Python tools from `./.venv/bin/` (never the user's conda env); Python ≥ 3.11.
- No new Python or npm dependencies. The `serve` extra already has `fastapi`, `uvicorn`, `python-multipart`.
- Every predicted quantity stays a distribution/interval; the class output is a probability over classes, never a bare label (design rule 1).
- Physical assumptions live in `config/priors.yaml` / `config/classes.yaml` (rule 3): blade-count options move to `priors.yaml`; class bands come from `classes.yaml`.
- Mass only from hover force balance; class evidence only when hover is detected (rules 5–6).
- `/predict` responses and `augur predict` CLI output are unchanged.
- No code comments unless they explain a non-obvious WHY. f-strings, snake_case, ruff line length 100.
- Work on branch `feature/live-identify`; one commit per task; push only in Task 9.
- Baseline before Task 1: `./.venv/bin/pytest tests/ -q` → 131 passed; `./.venv/bin/ruff check src/ tests/` clean.

## Review Focus

- A clip shorter than one analysis frame (0.1 s) sent to `/identify` → 422 "need at least 0.25 s of audio", not a 500. Pinned in Task 4.
- A non-WAV upload (MP3 bytes) sent straight to `/identify` → 422, not a 500. Pinned in Task 4.
- Stereo float32 WAVs (common from recorders and DAWs) → analyzed exactly like mono int16. Pinned in Task 4.
- Silence (a muted mic) → status `no_tone`, no exception. Pinned in Task 3.
- 48 kHz audio (the browser mic default) → same class ranking as 44.1 kHz. Pinned in Task 3.

---

### Task 1: Carry motor-count uncertainty into frame diagonal and disk loading

**Files:**
- Modify: `src/augur/fusion/posterior.py:123` (delete `num_motors_mode`) and `:170-171`
- Test: `tests/unit/test_fusion_geometry.py` (create)

**Interfaces:**
- Consumes: `fuse(inputs, n, seed) -> PredictionReport`, `fuse_distributions(inputs, n, seed) -> dict[str, Distribution]` (existing).
- Produces: no new names. `min_frame_diagonal_m` and `disk_loading_n_m2` samples now use each sample's own motor count.

- [ ] **Step 1: Write the failing tests**

```python
"""Frame diagonal and disk loading must carry motor-count uncertainty, not a
modal motor count (design rule 4: widen, never invent)."""

import numpy as np
import pytest

from augur.fusion.posterior import FusionInputs, fuse, fuse_distributions
from augur.types import VerbalSpec

D_5IN_M = 5.0 * 0.0254


@pytest.mark.parametrize("seed", [0, 1])
def test_frame_diagonal_uncertain_when_motor_count_unknown(seed):
    report = fuse(FusionInputs(verbal=VerbalSpec(prop_diameter_inch=5.0)), n=4000, seed=seed)
    frame = report.variables["min_frame_diagonal_m"]
    assert frame.interval_low == pytest.approx(D_5IN_M / np.sin(np.pi / 3), rel=1e-6)
    assert frame.interval_high == pytest.approx(D_5IN_M / np.sin(np.pi / 8), rel=1e-6)


def test_frame_diagonal_exact_when_motor_count_given():
    verbal = VerbalSpec(num_motors=4, prop_diameter_inch=5.0)
    frame = fuse(FusionInputs(verbal=verbal), n=2000, seed=0).variables["min_frame_diagonal_m"]
    assert frame.interval_low == frame.interval_high
    assert frame.median == pytest.approx(D_5IN_M / np.sin(np.pi / 4), rel=1e-6)


def test_disk_loading_uses_each_samples_motor_count():
    out = fuse_distributions(FusionInputs(verbal=VerbalSpec(prop_diameter_inch=5.0)), n=4000, seed=0)
    motors = out["num_motors"].samples
    expected = out["total_thrust_n"].samples / (motors * np.pi * (D_5IN_M / 2) ** 2)
    assert np.allclose(out["disk_loading_n_m2"].samples, expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/pytest tests/unit/test_fusion_geometry.py -v`
Expected: `test_frame_diagonal_uncertain_when_motor_count_unknown` FAILS (low == high, a single modal value) and `test_disk_loading_uses_each_samples_motor_count` FAILS; the exact-count test passes.

- [ ] **Step 3: Implement**

In `src/augur/fusion/posterior.py`, delete the line `num_motors_mode = int(np.bincount(num_motors).argmax())`, then replace

```python
    disk = core.disk_loading(t_total, num_motors_mode, diameter_m)
    frame = core.min_frame_diagonal_m(diameter_m, num_motors_mode)
```

with

```python
    disk = np.empty(n)
    frame = np.empty(n)
    for motor_count in np.unique(num_motors):
        sel = num_motors == motor_count
        disk[sel] = core.disk_loading(t_total[sel], int(motor_count), diameter_m[sel])
        frame[sel] = core.min_frame_diagonal_m(diameter_m[sel], int(motor_count))
```

(`core.disk_loading` / `core.min_frame_diagonal_m` take a scalar `int` motor count; keep their signatures.)

- [ ] **Step 4: Run the tests and the full suite**

Run: `./.venv/bin/pytest tests/unit/test_fusion_geometry.py -v && ./.venv/bin/pytest tests/ -q && ./.venv/bin/ruff check src/ tests/`
Expected: 4 new tests PASS; full suite 135 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/augur/fusion/posterior.py tests/unit/test_fusion_geometry.py
git commit -m "Carry motor-count uncertainty into frame diagonal and disk loading"
```

---

### Task 2: Acoustic diagnostics — comb matches, flight-state evidence, display spectrum

**Files:**
- Modify: `src/augur/types.py` (append `Harmonic`, `FlightStateEvidence`, `AcousticsDiagnostics`)
- Modify: `src/augur/audio/rpm.py` (constants + `comb_tolerance_hz`, `match_harmonics`; `_comb_score` delegates)
- Modify: `src/augur/audio/hover.py` (`assess_flight_state`; `classify_state` delegates)
- Create: `src/augur/audio/diagnostics.py`
- Test: `tests/unit/test_audio_diagnostics.py` (create)

**Interfaces:**
- Consumes: `spectral.average_spectrum`, `spectral.dominant_freq_track`, `rpm.detect_fundamental` (existing).
- Produces:
  - `rpm.N_HARMONICS: int = 6`, `rpm.DETECTION_FMAX_HZ: float = 5000.0`
  - `rpm.comb_tolerance_hz(freqs: np.ndarray) -> float`
  - `rpm.match_harmonics(fundamental: float, peaks: np.ndarray, n_harmonics: int, tol_hz: float) -> list[float | None]`
  - `hover.assess_flight_state(samples, sample_rate, steady_threshold=0.04, nperseg=8192) -> FlightStateEvidence`
  - `diagnostics.SPECTRUM_DISPLAY_BINS = 256`, `diagnostics.SPECTRUM_DB_FLOOR = -80.0`
  - `diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz, bins=256) -> tuple[list[float], list[float]]`
  - `diagnostics.build_acoustics(freqs, spec_db, fundamental_hz: float, peaks: np.ndarray, comb_confidence: float, evidence: FlightStateEvidence) -> AcousticsDiagnostics`
  - types `Harmonic(order, expected_hz, matched_hz)`, `FlightStateEvidence(state, pitch_variation, pitch_drift, steady_threshold)`, `AcousticsDiagnostics(spectrum_freq_hz, spectrum_db, fundamental_hz, harmonics, comb_confidence, flight_state)`

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_audio_diagnostics.py`

```python
"""Acoustic diagnostics for the UI: comb matching, flight-state evidence, and the
display spectrum. Nothing here may change an estimate, so the comb score and the
flight-state label must stay identical to what the RPM path computes."""

import numpy as np
import pytest

from augur.audio import diagnostics, hover, rpm, spectral
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import FlightState


def _spec() -> DroneSpec:
    return DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
                     c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=0.55,
                     battery_wh=55.5, drone_class="racing")


def _clip(drift: float = 0.0):
    return synth_waveform(_spec(), FlightCondition(rpm=12000.0, state="hover"), duration_s=2.0,
                          snr_db=30.0, rpm_drift_fraction=drift, rng=np.random.default_rng(0))


def _noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.3, 88200)


def test_match_harmonics_hits_and_misses():
    peaks = np.array([100.0, 200.0, 401.0])
    assert rpm.match_harmonics(100.0, peaks, 4, tol_hz=5.0) == [100.0, 200.0, None, 401.0]


def test_comb_score_counts_matched_harmonics():
    assert rpm._comb_score(100.0, np.array([100.0, 200.0, 401.0]), 4, tol_hz=5.0) == 0.75


def test_assess_flight_state_hover_evidence():
    sig, sr = _clip()
    ev = hover.assess_flight_state(sig, sr)
    assert ev.state == FlightState.hover
    assert ev.pitch_variation is not None and ev.pitch_variation <= ev.steady_threshold
    assert hover.classify_state(sig, sr) == ev.state


def test_assess_flight_state_ramp_evidence():
    sig, sr = _clip(drift=0.4)
    ev = hover.assess_flight_state(sig, sr)
    assert ev.state != FlightState.hover
    assert ev.pitch_variation > ev.steady_threshold
    assert hover.classify_state(sig, sr) == ev.state


def test_assess_flight_state_noise_is_unknown():
    ev = hover.assess_flight_state(_noise(), 44100)
    assert ev.state == FlightState.unknown
    assert ev.pitch_variation is None and ev.pitch_drift is None


def test_max_pool_keeps_every_spike():
    freqs = np.linspace(0.0, 5000.0, 2001)
    spec_db = np.full(freqs.size, -60.0)
    spikes = [137, 911, 1500]
    spec_db[spikes] = [-5.0, -10.0, 0.0]
    f, d = diagnostics.max_pool_spectrum(freqs, spec_db, f_max_hz=5000.0, bins=256)
    assert len(f) == len(d) == 256
    for i in spikes:
        assert d[f.index(float(freqs[i]))] == pytest.approx(spec_db[i])


def test_max_pool_passes_small_windows_through():
    freqs = np.linspace(0.0, 1000.0, 101)
    f, d = diagnostics.max_pool_spectrum(freqs, -np.linspace(0.0, 50.0, 101), f_max_hz=1000.0)
    assert f == [float(x) for x in freqs]
    assert max(d) == 0.0


def test_build_acoustics_covers_the_comb():
    sig, sr = _clip()
    freqs, spec_db = spectral.average_spectrum(sig, sr)
    f0, conf, peaks = rpm.detect_fundamental(freqs, spec_db)
    ac = diagnostics.build_acoustics(freqs, spec_db, f0, peaks, conf, hover.assess_flight_state(sig, sr))
    assert ac.fundamental_hz == pytest.approx(12000.0 / 60.0 * 2, rel=0.02)
    assert len(ac.harmonics) == rpm.N_HARMONICS
    assert all(h.matched_hz is not None for h in ac.harmonics)
    assert ac.spectrum_freq_hz[-1] >= rpm.N_HARMONICS * ac.fundamental_hz
    assert max(ac.spectrum_db) == 0.0
    assert min(ac.spectrum_db) >= diagnostics.SPECTRUM_DB_FLOOR


def test_build_acoustics_without_tone():
    noise = _noise(1)
    freqs, spec_db = spectral.average_spectrum(noise, 44100)
    ac = diagnostics.build_acoustics(freqs, spec_db, 0.0, np.array([]), 0.0,
                                     hover.assess_flight_state(noise, 44100))
    assert ac.fundamental_hz is None and ac.harmonics == []
    assert ac.spectrum_freq_hz[-1] <= rpm.DETECTION_FMAX_HZ
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/pytest tests/unit/test_audio_diagnostics.py -v`
Expected: FAIL with `ImportError: cannot import name 'diagnostics'`.

- [ ] **Step 3: Add the types** — append to `src/augur/types.py`

```python
class Harmonic(BaseModel):
    """One tooth of the blade-pass comb: where it should be, and the detected
    peak that matched it (None when nothing matched within tolerance)."""

    order: int
    expected_hz: float
    matched_hz: float | None = None


class FlightStateEvidence(BaseModel):
    """The flight-state label plus the measurements behind it, so a reader can
    see how close the call was. pitch_* are None when no stable tone was found."""

    state: FlightState
    pitch_variation: float | None = None
    pitch_drift: float | None = None
    steady_threshold: float


class AcousticsDiagnostics(BaseModel):
    """What the audio front-end saw: a display spectrum (dB relative to its peak),
    the harmonic comb, and the flight-state evidence."""

    spectrum_freq_hz: list[float]
    spectrum_db: list[float]
    fundamental_hz: float | None = None
    harmonics: list[Harmonic] = Field(default_factory=list)
    comb_confidence: float
    flight_state: FlightStateEvidence
```

- [ ] **Step 4: Comb helpers** — in `src/augur/audio/rpm.py`, add below the imports:

```python
N_HARMONICS = 6
DETECTION_FMAX_HZ = 5000.0


def comb_tolerance_hz(freqs: np.ndarray) -> float:
    """How far a peak may sit from h*f0 and still count as that harmonic."""
    return max(3.0 * float(freqs[1] - freqs[0]), 5.0)


def match_harmonics(fundamental: float, peaks: np.ndarray, n_harmonics: int,
                    tol_hz: float) -> list[float | None]:
    """For each of the first n_harmonics, the nearest peak within tol_hz, else None."""
    matched: list[float | None] = []
    for h in range(1, n_harmonics + 1):
        target = fundamental * h
        if fundamental > 0 and peaks.size:
            nearest = float(peaks[np.argmin(np.abs(peaks - target))])
            matched.append(nearest if abs(nearest - target) <= tol_hz else None)
        else:
            matched.append(None)
    return matched
```

Replace the body of `_comb_score` with:

```python
    if fundamental <= 0:
        return 0.0
    matched = match_harmonics(fundamental, peaks, n_harmonics, tol_hz)
    return sum(m is not None for m in matched) / n_harmonics
```

In `detect_fundamental`, change the signature defaults to `fmax: float = DETECTION_FMAX_HZ, n_harmonics: int = N_HARMONICS` and replace

```python
    bin_hz = float(freqs[1] - freqs[0])
    tol_hz = max(3.0 * bin_hz, 5.0)
```

with `tol_hz = comb_tolerance_hz(freqs)`. In `estimate_rpm`, change `fmax: float = 5000.0` to `fmax: float = DETECTION_FMAX_HZ` and `n_harmonics=6` to `n_harmonics=N_HARMONICS`.

- [ ] **Step 5: Flight-state evidence** — in `src/augur/audio/hover.py`, change the import to `from augur.types import FlightState, FlightStateEvidence` and replace `classify_state` with:

```python
def assess_flight_state(samples: np.ndarray, sample_rate: int,
                        steady_threshold: float = 0.04,
                        nperseg: int = 8192) -> FlightStateEvidence:
    """Classify flight state from the stability of the blade-pass fundamental,
    and return the measurements behind the call.

    steady_threshold is the max robust coefficient of variation (IQR/median) of
    the fundamental track for the clip to count as steady hover. Per-motor control
    spread (a few percent) stays under this; a throttle ramp/climb exceeds it.
    """
    unknown = FlightStateEvidence(state=FlightState.unknown, steady_threshold=steady_threshold)
    freqs, spec_db = spectral.average_spectrum(samples, sample_rate, nperseg=nperseg)
    f0, conf, _ = rpm.detect_fundamental(freqs, spec_db)
    if f0 <= 0 or conf < 0.3:
        return unknown

    _, track = spectral.dominant_freq_track(samples, sample_rate, fmin=0.85 * f0,
                                            fmax=1.15 * f0, nperseg=nperseg)
    if track.size < 3:
        return unknown
    median_f = float(np.median(track))
    if median_f <= 0:
        return unknown

    q25, q75 = np.percentile(track, [25, 75])
    robust_cov = float((q75 - q25) / median_f)
    # A sustained monotonic drift in the fundamental is a throttle ramp (climb).
    slope = np.polyfit(np.arange(track.size), track, 1)[0]
    rel_slope = float(abs(slope) * track.size / median_f)
    if robust_cov <= steady_threshold:
        state = FlightState.hover
    elif rel_slope > steady_threshold:
        state = FlightState.climb
    else:
        state = FlightState.mixed
    return FlightStateEvidence(state=state, pitch_variation=robust_cov, pitch_drift=rel_slope,
                               steady_threshold=steady_threshold)


def classify_state(samples: np.ndarray, sample_rate: int,
                   steady_threshold: float = 0.04, nperseg: int = 8192) -> FlightState:
    """Flight-state label only; see assess_flight_state for the evidence."""
    return assess_flight_state(samples, sample_rate, steady_threshold, nperseg).state
```

- [ ] **Step 6: Display spectrum** — create `src/augur/audio/diagnostics.py`

```python
"""Display-side acoustics for the UI: a peak-preserving downsampled spectrum and
the harmonic comb. Nothing here changes an estimate."""

from __future__ import annotations

import numpy as np

from augur.audio import rpm
from augur.types import AcousticsDiagnostics, FlightStateEvidence, Harmonic

SPECTRUM_DISPLAY_BINS = 256
SPECTRUM_DB_FLOOR = -80.0


def max_pool_spectrum(freqs: np.ndarray, spec_db: np.ndarray, f_max_hz: float,
                      bins: int = SPECTRUM_DISPLAY_BINS) -> tuple[list[float], list[float]]:
    """Crop to [0, f_max_hz] and keep each group's loudest bin at its true
    frequency, so harmonic peaks survive downsampling. dB is relative to the
    peak (max 0) and floored at SPECTRUM_DB_FLOOR."""
    window = freqs <= f_max_hz
    f, d = freqs[window], spec_db[window]
    if f.size > bins:
        groups = np.array_split(np.arange(f.size), bins)
        idx = np.array([g[np.argmax(d[g])] for g in groups])
        f, d = f[idx], d[idx]
    d = np.maximum(d - d.max(), SPECTRUM_DB_FLOOR)
    return [float(x) for x in f], [float(x) for x in d]


def build_acoustics(freqs: np.ndarray, spec_db: np.ndarray, fundamental_hz: float,
                    peaks: np.ndarray, comb_confidence: float,
                    evidence: FlightStateEvidence) -> AcousticsDiagnostics:
    has_tone = fundamental_hz > 0
    top = float(freqs[-1])
    f_max = min(fundamental_hz * (rpm.N_HARMONICS + 1), top) if has_tone else min(rpm.DETECTION_FMAX_HZ, top)
    spectrum_f, spectrum_d = max_pool_spectrum(freqs, spec_db, f_max)
    matched = (rpm.match_harmonics(fundamental_hz, peaks, rpm.N_HARMONICS,
                                   rpm.comb_tolerance_hz(freqs)) if has_tone else [])
    harmonics = [Harmonic(order=i + 1, expected_hz=fundamental_hz * (i + 1), matched_hz=m)
                 for i, m in enumerate(matched)]
    return AcousticsDiagnostics(
        spectrum_freq_hz=spectrum_f, spectrum_db=spectrum_d,
        fundamental_hz=fundamental_hz if has_tone else None,
        harmonics=harmonics, comb_confidence=comb_confidence, flight_state=evidence,
    )
```

- [ ] **Step 7: Run tests**

Run: `./.venv/bin/pytest tests/unit/test_audio_diagnostics.py tests/unit/test_audio_rpm.py -v && ./.venv/bin/pytest tests/ -q && ./.venv/bin/ruff check src/ tests/`
Expected: 9 new tests PASS; existing `classify_state` tests still PASS; full suite 144 passed; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/augur/types.py src/augur/audio/rpm.py src/augur/audio/hover.py src/augur/audio/diagnostics.py tests/unit/test_audio_diagnostics.py
git commit -m "Expose acoustic diagnostics: comb matches, flight-state evidence, display spectrum"
```

---

### Task 3: Identify the drone class from audio

**Files:**
- Modify: `config/priors.yaml` (version 3, `blade_count_options`)
- Modify: `src/augur/config.py` (`Priors.blade_count_options`)
- Modify: `src/augur/fusion/posterior.py` (blade alignment draw reads the config)
- Modify: `src/augur/types.py` (append `ClassProbability`, `BladeProbability`, `IdentifyReport`; add `from typing import Literal`)
- Create: `src/augur/fusion/identify.py`
- Test: `tests/unit/test_identify.py`, `tests/calibration/test_identify_calibration.py` (create both)

**Interfaces:**
- Consumes: Task 2's `hover.assess_flight_state`, `diagnostics.build_acoustics`, `AcousticsDiagnostics`; existing `fuse_distributions`, `rpm.estimate_rpm`, `rpm.detect_fundamental`, `spectral.average_spectrum`, `load_classes().names/.get()`.
- Produces:
  - `Priors.blade_count_options -> list[int]`
  - `identify(samples: np.ndarray, sample_rate: int, verbal: VerbalSpec | None = None, n: int = 4000, seed: int = 0) -> IdentifyReport`
  - `candidate_classes(verbal) -> list[str]` (classes.yaml minus `unknown`, or the asserted class)
  - `IdentifyReport(status: Literal["ok","no_tone","not_hovering","no_class_fits"], duration_s, sample_rate, classes: list[ClassProbability], blades: list[BladeProbability], acoustics: AcousticsDiagnostics, variables: dict[str, VariableSummary], notes: list[str], config_version: int)` — `classes`/`blades` sorted by probability, descending.

- [ ] **Step 1: Write the failing unit tests** — `tests/unit/test_identify.py`

```python
"""Live identification: a class posterior from audio, gated on hover (rule 6),
marginalizing blade count, sharpened by asserted facts, and honest when there's
no tone or the tone is mains hum."""

import numpy as np
import pytest

from augur.fusion.identify import identify
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform
from augur.types import VerbalSpec

RACING = dict(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0, blade_count=2,
              c_t=0.11, c_p=0.05, pole_count=14, efficiency=0.8, mass_kg=0.55,
              battery_wh=55.5, drone_class="racing")


def _hover_rpm(spec: DroneSpec) -> float:
    d_m = spec.prop_diameter_inch * 0.0254
    return float(np.sqrt(spec.mass_kg * 9.80665 / (spec.num_motors * spec.c_t * 1.225 * d_m**4)) * 60.0)


def _clip(sample_rate: int = 44100, drift: float = 0.0):
    spec = DroneSpec(**RACING)
    return synth_waveform(spec, FlightCondition(rpm=_hover_rpm(spec), state="hover"),
                          sample_rate=sample_rate, duration_s=2.0, snr_db=25.0,
                          rpm_drift_fraction=drift, rng=np.random.default_rng(0))


def _p(report, name: str) -> float:
    return next(c.probability for c in report.classes if c.name == name)


@pytest.fixture(scope="module")
def racing_report():
    sig, sr = _clip()
    return identify(sig, sr, n=2000)


def test_class_and_blade_posteriors_sum_to_one(racing_report):
    assert racing_report.status == "ok"
    assert sum(c.probability for c in racing_report.classes) == pytest.approx(1.0)
    assert sum(b.probability for b in racing_report.blades) == pytest.approx(1.0)


def test_racing_clip_ranks_racing_first(racing_report):
    assert racing_report.classes[0].name == "racing"


def test_unknown_blade_count_is_marginalized(racing_report):
    assert sorted(b.blades for b in racing_report.blades) == [2, 3]
    f0 = racing_report.acoustics.fundamental_hz
    rpm = racing_report.variables["rpm"]
    assert rpm.interval_low < f0 * 60 / 3 * 1.05
    assert rpm.interval_high > f0 * 60 / 2 * 0.95


def test_given_prop_size_sharpens_the_guess():
    sig, sr = _clip()
    assert _p(identify(sig, sr, VerbalSpec(prop_diameter_inch=5.0), n=2000), "racing") >= 0.9


def test_given_class_is_the_only_hypothesis():
    sig, sr = _clip()
    report = identify(sig, sr, VerbalSpec(drone_class="racing"), n=2000)
    assert [c.name for c in report.classes] == ["racing"]
    assert report.classes[0].probability == pytest.approx(1.0)


def test_throttle_ramp_is_not_hover_so_classes_stay_at_prior():
    sig, sr = _clip(drift=0.4)
    report = identify(sig, sr, n=2000)
    assert report.status == "not_hovering"
    assert all(c.probability == pytest.approx(0.25) for c in report.classes)
    assert any("not hover" in note for note in report.notes)


@pytest.mark.parametrize("kind", ["noise", "silence"])
def test_no_tone_reports_status_and_prior(kind):
    x = np.zeros(88200) if kind == "silence" else np.random.default_rng(0).normal(0.0, 0.3, 88200)
    report = identify(x, 44100, n=2000)
    assert report.status == "no_tone"
    assert report.acoustics.fundamental_hz is None
    assert all(c.probability == pytest.approx(0.25) for c in report.classes)


def test_mains_hum_is_flagged():
    sr = 44100
    t = np.arange(2 * sr) / sr
    hum = sum(np.sin(2 * np.pi * 60.0 * k * t) / k for k in range(1, 8))
    hum = hum + np.random.default_rng(0).normal(0.0, 0.02, t.size)
    assert any("mains hum" in note for note in identify(hum, sr, n=2000).notes)


def test_48k_audio_still_ranks_racing_first():
    sig, sr = _clip(sample_rate=48000)
    assert identify(sig, sr, n=2000).classes[0].name == "racing"


def test_prop_size_outside_every_class_still_returns_a_report():
    sig, sr = _clip()
    report = identify(sig, sr, VerbalSpec(prop_diameter_inch=40.0), n=2000)
    assert report.status == "no_class_fits"
    assert {"mass_kg", "rpm"} <= set(report.variables)
```

- [ ] **Step 2: Write the failing calibration test** — `tests/calibration/test_identify_calibration.py`

```python
"""Simulator round-trip for the class guess (rule 2): drones drawn from each
class's own priors -> hover audio -> identify. A Bayes-consistent posterior beats
chance and its stated confidence matches its hit rate. Measured during design
(48 drones): accuracy 0.54 vs mean top probability 0.50; chance is 0.25."""

import numpy as np

from augur.config import load_classes, load_priors
from augur.fusion.identify import candidate_classes, identify
from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform

PER_CLASS = 12
DETECTABLE_BPF_HZ = (60.0, 4500.0)


def _drones(rng):
    classes, priors = load_classes(), load_priors()
    ct = priors.c_t_fallback
    for name in candidate_classes(None):
        cfg = classes.get(name)
        made = 0
        while made < PER_CLASS:
            n_motors = int(rng.choice(cfg["num_motors"]))
            d_in = rng.uniform(cfg["prop_diameter_inch"]["low"], cfg["prop_diameter_inch"]["high"])
            mass = rng.uniform(cfg["all_up_mass_kg"]["low"], cfg["all_up_mass_kg"]["high"])
            c_t = rng.uniform(ct.low, ct.high)
            blades = int(rng.choice(priors.blade_count_options))
            d_m = d_in * 0.0254
            rpm = float(np.sqrt(mass * priors.gravity
                                / (n_motors * c_t * priors.air_density * d_m**4)) * 60.0)
            if not DETECTABLE_BPF_HZ[0] <= rpm / 60.0 * blades <= DETECTABLE_BPF_HZ[1]:
                continue
            spec = DroneSpec(num_motors=n_motors, prop_diameter_inch=d_in, prop_pitch_inch=0.5 * d_in,
                             blade_count=blades, c_t=c_t, c_p=0.05, pole_count=14, efficiency=0.8,
                             mass_kg=mass, battery_wh=50.0, drone_class=name)
            yield name, spec, rpm
            made += 1


def test_class_guess_beats_chance_and_is_calibrated():
    hits, top_p = [], []
    for i, (name, spec, rpm) in enumerate(_drones(np.random.default_rng(42))):
        sig, sr = synth_waveform(spec, FlightCondition(rpm=rpm, state="hover"), duration_s=2.0,
                                 snr_db=25.0, rng=np.random.default_rng(i))
        report = identify(sig, sr, n=2000)
        assert report.status == "ok"
        hits.append(report.classes[0].name == name)
        top_p.append(report.classes[0].probability)
    accuracy, confidence = float(np.mean(hits)), float(np.mean(top_p))
    assert accuracy >= 0.40, f"top-1 accuracy {accuracy:.2f} (chance 0.25)"
    assert abs(accuracy - confidence) <= 0.15, f"accuracy {accuracy:.2f} vs confidence {confidence:.2f}"
```

- [ ] **Step 3: Run to verify they fail**

Run: `./.venv/bin/pytest tests/unit/test_identify.py tests/calibration/test_identify_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'augur.fusion.identify'`.

- [ ] **Step 4: Config** — in `config/priors.yaml` change `version: 2` to `version: 3` and append:

```yaml
# Blades per prop when the spec doesn't say. The audio measures blade-pass
# frequency = blades * rev/s, so this choice decides how a tone maps to RPM.
blade_count_options: [2, 3]
```

In `src/augur/config.py`, add to `Priors`:

```python
    @property
    def blade_count_options(self) -> list[int]:
        return [int(b) for b in self.raw["blade_count_options"]]
```

In `src/augur/fusion/posterior.py`, replace

```python
    if not (verbal and verbal.blade_count):
        _ = np.array([2, 3])[np.floor(u_blade * 2).astype(int)]
```

with

```python
    if not (verbal and verbal.blade_count):
        options = np.array(priors.blade_count_options)
        _ = options[np.floor(u_blade * len(options)).astype(int)]
```

- [ ] **Step 5: Types** — in `src/augur/types.py` add `from typing import Literal` to the imports and append:

```python
class ClassProbability(BaseModel):
    name: str
    probability: float


class BladeProbability(BaseModel):
    blades: int
    probability: float


class IdentifyReport(BaseModel):
    """Live identification of one audio window: a distribution over drone
    classes (never a bare label), the blade-count weights, what the audio
    front-end saw, and mixture estimates for every variable."""

    status: Literal["ok", "no_tone", "not_hovering", "no_class_fits"]
    duration_s: float
    sample_rate: int
    classes: list[ClassProbability]
    blades: list[BladeProbability]
    acoustics: AcousticsDiagnostics
    variables: dict[str, VariableSummary]
    notes: list[str] = Field(default_factory=list)
    config_version: int
```

- [ ] **Step 6: Implement** — create `src/augur/fusion/identify.py`

```python
"""Live identification: which kind of drone is this, from audio alone?

In hover, a class's priors (prop size, motor count, all-up mass) plus the prop
coefficients imply a range of blade-pass frequencies. Each hypothesis (class c,
blade count B) reuses the plug-in fusion: RPM = BPF * 60 / B, latents from c.
The class's all-up-mass band is a uniform prior on mass, and hover gives
mass = k * rpm^2, so the evidence for (c, B) is

    p(BPF | c, B) = (60 / B) * E[ 1{m in band} * 2 m / (rpm * (hi - lo)) ]

and the same term is the importance weight that turns the plug-in samples into
the posterior under (c, B). Facts the user asserts (prop size, motor count) add
their likelihood under each class's band. Estimates are the evidence-weighted
mixture. Valid only in hover (rule 6); otherwise only asserted facts move the
classes off their prior.
"""

from __future__ import annotations

import numpy as np

from augur.audio import diagnostics, hover, rpm, spectral
from augur.config import load_classes, load_priors
from augur.fusion.posterior import OUTPUT_UNITS, FusionInputs, fuse_distributions
from augur.types import (
    BladeProbability,
    ClassProbability,
    Distribution,
    FlightState,
    IdentifyReport,
    VariableSummary,
    VerbalSpec,
)

MIN_COMB_CONFIDENCE = 0.3
MAINS_HZ = (50.0, 60.0, 100.0, 120.0)
MAINS_TOLERANCE = 0.015


def candidate_classes(verbal: VerbalSpec | None) -> list[str]:
    if verbal and verbal.drone_class:
        return [verbal.drone_class.value]
    return [name for name in load_classes().names if name != "unknown"]


def candidate_blades(verbal: VerbalSpec | None) -> list[int]:
    if verbal and verbal.blade_count:
        return [verbal.blade_count]
    return load_priors().blade_count_options


def fact_likelihood(class_cfg: dict, verbal: VerbalSpec | None) -> float:
    """p(asserted prop size and motor count | class): uniform over the class's
    band or choices, zero outside them."""
    p = 1.0
    if verbal and verbal.prop_diameter_inch:
        band = class_cfg["prop_diameter_inch"]
        inside = band["low"] <= verbal.prop_diameter_inch <= band["high"]
        p *= 1.0 / (band["high"] - band["low"]) if inside else 0.0
    if verbal and verbal.num_motors:
        choices = class_cfg["num_motors"]
        p *= 1.0 / len(choices) if verbal.num_motors in choices else 0.0
    return p


def mains_hum_note(fundamental_hz: float, bin_hz: float) -> list[str]:
    for mains in MAINS_HZ:
        tolerance = max(MAINS_TOLERANCE * mains, bin_hz)
        if fundamental_hz > 0 and abs(fundamental_hz - mains) <= tolerance:
            return [f"The tone sits at {fundamental_hz:.0f} Hz, where mains hum lives "
                    "(50/60 Hz and multiples). It may not be a drone."]
    return []


def _summaries(draws: dict[str, np.ndarray]) -> dict[str, VariableSummary]:
    return {v: Distribution(a, unit=OUTPUT_UNITS[v] or None).to_summary() for v, a in draws.items()}


def _ranked_classes(weights: dict[str, float]) -> list[ClassProbability]:
    total = sum(weights.values())
    probs = {c: (w / total if total > 0 else 1.0 / len(weights)) for c, w in weights.items()}
    return sorted((ClassProbability(name=c, probability=p) for c, p in probs.items()),
                  key=lambda x: -x.probability)


def _ranked_blades(weights: dict[int, float]) -> list[BladeProbability]:
    total = sum(weights.values())
    probs = {b: (w / total if total > 0 else 1.0 / len(weights)) for b, w in weights.items()}
    return sorted((BladeProbability(blades=b, probability=p) for b, p in probs.items()),
                  key=lambda x: -x.probability)


def identify(samples: np.ndarray, sample_rate: int, verbal: VerbalSpec | None = None,
             n: int = 4000, seed: int = 0) -> IdentifyReport:
    evidence = hover.assess_flight_state(samples, sample_rate)
    freqs, spec_db = spectral.average_spectrum(samples, sample_rate)
    f0, conf, peaks = rpm.detect_fundamental(freqs, spec_db)
    class_cfg = load_classes()
    classes, blades = candidate_classes(verbal), candidate_blades(verbal)
    facts = {c: fact_likelihood(class_cfg.get(c), verbal) for c in classes}
    common = dict(duration_s=samples.size / sample_rate, sample_rate=sample_rate,
                  acoustics=diagnostics.build_acoustics(freqs, spec_db, f0, peaks, conf, evidence),
                  config_version=load_priors().version)
    notes = mains_hum_note(f0, float(freqs[1] - freqs[0]))

    if f0 <= 0 or conf < MIN_COMB_CONFIDENCE:
        out = fuse_distributions(FusionInputs(verbal=verbal), n=n, seed=seed)
        notes.append("No steady blade-pass tone yet, so there is nothing to classify. "
                     "Estimates are prior-width.")
        return IdentifyReport(**common, status="no_tone", classes=_ranked_classes(facts),
                              blades=_ranked_blades(dict.fromkeys(blades, 1.0)),
                              variables=_summaries({v: d.samples for v, d in out.items()}),
                              notes=notes)

    hovering = evidence.state == FlightState.hover
    components = []
    for c in classes:
        band = class_cfg.get(c)["all_up_mass_kg"]
        for b in blades:
            est = rpm.estimate_rpm(freqs, spec_db, num_blades=b, rng=np.random.default_rng(seed))
            spec_cb = VerbalSpec(**{**(verbal.model_dump() if verbal else {}),
                                    "drone_class": c, "blade_count": b})
            out = fuse_distributions(FusionInputs(verbal=spec_cb, rpm_estimate=est,
                                                  flight_state=evidence.state), n=n, seed=seed)
            draws = {v: d.samples for v, d in out.items()}
            if hovering:
                m, r = draws["mass_kg"], draws["rpm"]
                inside = (m >= band["low"]) & (m <= band["high"])
                weights = inside * (2.0 * m / r) / (band["high"] - band["low"])
                strength = facts[c] * (60.0 / b) * float(weights.mean())
            else:
                weights, strength = np.ones(n), facts[c]
            components.append((c, b, strength, weights, draws))

    total = sum(strength for _, _, strength, _, _ in components)
    if total <= 0:
        out = fuse_distributions(FusionInputs(verbal=verbal), n=n, seed=seed)
        notes.append("The tone doesn't fit any known class in hover. It may be a drone "
                     "type outside config/classes.yaml.")
        return IdentifyReport(**common, status="no_class_fits",
                              classes=[ClassProbability(name=c, probability=0.0) for c in classes],
                              blades=_ranked_blades(dict.fromkeys(blades, 1.0)),
                              variables=_summaries({v: d.samples for v, d in out.items()}),
                              notes=notes)

    rng = np.random.default_rng(seed + 1)
    parts: dict[str, list[np.ndarray]] = {v: [] for v in OUTPUT_UNITS}
    class_w = dict.fromkeys(classes, 0.0)
    blade_w = dict.fromkeys(blades, 0.0)
    for c, b, strength, weights, draws in components:
        p = strength / total
        class_w[c] += p
        blade_w[b] += p
        k = int(round(p * n))
        if k == 0:
            continue
        idx = rng.choice(n, size=k, replace=True, p=weights / weights.sum())
        for v in OUTPUT_UNITS:
            parts[v].append(draws[v][idx])

    if not hovering:
        notes.append(f"Flight state is {evidence.state.value}, not hover: the class guess "
                     "stays at the prior and mass comes from the class prior.")
    return IdentifyReport(**common, status="ok" if hovering else "not_hovering",
                          classes=_ranked_classes(class_w), blades=_ranked_blades(blade_w),
                          variables=_summaries({v: np.concatenate(p) for v, p in parts.items()}),
                          notes=notes)
```

- [ ] **Step 7: Run tests**

Run: `./.venv/bin/pytest tests/unit/test_identify.py tests/calibration/test_identify_calibration.py -v && ./.venv/bin/pytest tests/ -q && ./.venv/bin/ruff check src/ tests/`
Expected: 12 new tests PASS (the calibration test takes a few seconds); full suite 156 passed; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add config/priors.yaml src/augur/config.py src/augur/fusion/posterior.py src/augur/types.py src/augur/fusion/identify.py tests/unit/test_identify.py tests/calibration/test_identify_calibration.py
git commit -m "Identify drone class from audio via hover-physics evidence"
```

---

### Task 4: `/identify` and `/reference` endpoints

**Files:**
- Modify: `src/augur/audio/spectral.py` (shared mono/normalize helper + `read_wav_bytes`)
- Modify: `src/augur/api.py` (`identify_from_upload`, `reference_info`, two routes)
- Test: `tests/unit/test_api.py`

**Interfaces:**
- Consumes: Task 3's `identify`, `candidate_classes`, `IdentifyReport`.
- Produces:
  - `spectral.read_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]` (raises `ValueError` on non-WAV)
  - `api.MIN_IDENTIFY_SECONDS = 0.25`
  - `api.identify_from_upload(audio_bytes: bytes, spec_json: str | bytes | None = None, n: int = 4000, seed: int = 0) -> IdentifyReport` (raises `ValueError`)
  - `api.reference_info(n: int = 4000) -> dict` → `{"bands": {variable: [low, high]}, "classes": {name: description}}`
  - HTTP: `POST /identify` (multipart: file `audio`, optional form `spec` JSON, `samples`, `seed`) → `IdentifyReport` JSON or 422; `GET /reference` → `reference_info()`.

- [ ] **Step 1: Write the failing tests** — in `tests/unit/test_api.py`, change the import to `from augur.api import identify_from_upload, predict_from_uploads, reference_info`, add these above the `# --- HTTP layer` comment:

```python
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


def test_reference_info_bands_and_classes():
    info = reference_info()
    low, high = info["bands"]["mass_kg"]
    assert 0 < low < high
    assert set(info["classes"]) == {"racing", "cinematic", "survey", "military"}
```

and append at the end of the file:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/pytest tests/unit/test_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'identify_from_upload'`.

- [ ] **Step 3: WAV bytes reader** — in `src/augur/audio/spectral.py` add `import io` and replace `load_wav` with:

```python
def _to_mono_float(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    # Normalise integer PCM to [-1, 1].
    if data.size and np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    return data


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float64 in [-1, 1]. Returns (samples, sample_rate)."""
    sr, data = wavfile.read(str(path))
    return _to_mono_float(data), int(sr)


def read_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """load_wav for an in-memory WAV (an HTTP upload). Raises ValueError if the
    bytes aren't a WAV."""
    sr, data = wavfile.read(io.BytesIO(raw))
    return _to_mono_float(data), int(sr)
```

- [ ] **Step 4: API** — in `src/augur/api.py` extend the imports:

```python
from functools import cache

import numpy as np

from augur.audio import spectral
from augur.config import load_classes
from augur.fusion.identify import candidate_classes, identify
from augur.fusion.posterior import FusionInputs, fuse_distributions
from augur.types import IdentifyReport, PredictionReport, VerbalSpec
```

add after `predict_from_uploads`:

```python
MIN_IDENTIFY_SECONDS = 0.25


def identify_from_upload(audio_bytes: bytes, spec_json: str | bytes | None = None,
                         n: int = 4000, seed: int = 0) -> IdentifyReport:
    """Identify the drone class from one in-memory WAV window (the live UI sends
    the latest few seconds). Raises ValueError on non-WAV or too-short audio."""
    samples, sample_rate = spectral.read_wav_bytes(audio_bytes)
    if samples.size < MIN_IDENTIFY_SECONDS * sample_rate:
        raise ValueError(f"need at least {MIN_IDENTIFY_SECONDS} s of audio")
    verbal = VerbalSpec(**json.loads(spec_json)) if spec_json else None
    return identify(samples, sample_rate, verbal, n=n, seed=seed)


@cache
def reference_info(n: int = 4000) -> dict:
    """The no-input 99% band per variable (the UI's display scale) and the class
    descriptions from classes.yaml."""
    out = fuse_distributions(FusionInputs(), n=n, seed=0)
    bands = {v: [float(x) for x in np.quantile(d.samples, [0.005, 0.995])] for v, d in out.items()}
    classes = load_classes()
    return {"bands": bands,
            "classes": {c: classes.get(c).get("description", "") for c in candidate_classes(None)}}
```

and inside `create_app()`, after the `/predict` route:

```python
    @app.post("/identify", response_model=IdentifyReport)
    async def identify_route(
        audio: UploadFile = File(..., description="WAV window, mono or stereo"),
        spec: str | None = Form(default=None, description="Known facts as JSON (optional)"),
        samples: int = Form(default=4000),
        seed: int = Form(default=0),
    ):
        try:
            return identify_from_upload(await audio.read(), spec, n=samples, seed=seed)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/reference")
    def reference():
        return reference_info()
```

- [ ] **Step 5: Run tests**

Run: `./.venv/bin/pytest tests/unit/test_api.py -v && ./.venv/bin/pytest tests/ -q && ./.venv/bin/ruff check src/ tests/`
Expected: 8 new tests PASS; full suite 164 passed; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/augur/audio/spectral.py src/augur/api.py tests/unit/test_api.py
git commit -m "Add /identify and /reference endpoints for the live UI"
```

---

### Task 5: 8-second demo clips

**Files:**
- Modify: `scripts/make_templates.py`
- Regenerate: `web/public/templates/{racing,cinematic,survey,military}.wav` (`templates.json` must come out unchanged)

**Interfaces:**
- Produces: four 8.0 s mono 44.1 kHz WAVs at the same paths; `templates.json` unchanged (same `true_rpm`).

- [ ] **Step 1: Add the option** — in `scripts/make_templates.py` add `import argparse`, change the docstring usage line to `Usage: python scripts/make_templates.py [--duration 8]`, and make `main()` start with:

```python
def main():
    parser = argparse.ArgumentParser(description="Generate the web UI's demo clips.")
    parser.add_argument("--duration", type=float, default=8.0, help="clip length in seconds")
    args = parser.parse_args()
```

then change `duration_s=2.0` in the `synth_waveform` call to `duration_s=args.duration`.

- [ ] **Step 2: Regenerate and verify**

Run:
```bash
./.venv/bin/python scripts/make_templates.py
./.venv/bin/python -c "
from scipy.io import wavfile
for n in ('racing','cinematic','survey','military'):
    sr, x = wavfile.read(f'web/public/templates/{n}.wav'); print(n, sr, len(x) / sr)"
git diff --stat web/public/templates/templates.json
```
Expected: each clip `44100 8.0`; `templates.json` shows no diff.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_templates.py web/public/templates/*.wav
git commit -m "Lengthen demo clips to 8 s for live playback"
```

---

### Task 6: Web helpers — WAV encoding and formatting (node:test)

**Files:**
- Create: `web/src/lib/wav.js`, `web/src/lib/wav.test.js`, `web/src/lib/format.test.js`
- Modify: `web/src/lib/format.js` (replace `fmt`, add new helpers; keep the old exports until Task 7)
- Modify: `web/package.json` (`test` script)

**Interfaces:**
- Produces:
  - `encodeWav(samples: Float32Array, sampleRate: number) -> ArrayBuffer` (16-bit mono PCM, clipped to ±1)
  - `downmix(channels: Float32Array[]) -> Float32Array`
  - `latestWindow(samples: Float32Array, sampleRate: number, endIndex: number, seconds: number) -> Float32Array`
  - `fmt(x)`, `fmtK(x)`, `fmtRange(lo, hi)`, `displayFactor(key)`, `displayUnit(key, unit)`, `confidenceWord(p)` from `lib/format.js`

- [ ] **Step 1: Write the failing tests** — `web/src/lib/wav.test.js`

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { downmix, encodeWav, latestWindow } from "./wav.js";

test("encodeWav writes a 16-bit mono PCM header", () => {
  const buf = encodeWav(new Float32Array([0, 0.5, -1]), 48000);
  const view = new DataView(buf);
  const text = (offset, n) => String.fromCharCode(...new Uint8Array(buf, offset, n));
  assert.equal(buf.byteLength, 44 + 3 * 2);
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 4), "WAVE");
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 48000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 6);
});

test("encodeWav clips out-of-range samples", () => {
  const view = new DataView(encodeWav(new Float32Array([2, -2, 0.5]), 8000));
  assert.equal(view.getInt16(44, true), 32767);
  assert.equal(view.getInt16(46, true), -32767);
  assert.equal(view.getInt16(48, true), 16384);
});

test("downmix averages channels", () => {
  const mixed = downmix([new Float32Array([1, 0]), new Float32Array([0, 0])]);
  assert.deepEqual(Array.from(mixed), [0.5, 0]);
});

test("latestWindow returns the last N seconds before the end index", () => {
  const samples = Float32Array.from({ length: 10 }, (_, i) => i);
  assert.deepEqual(Array.from(latestWindow(samples, 2, 8, 1.5)), [5, 6, 7]);
  assert.deepEqual(Array.from(latestWindow(samples, 2, 2, 3)), [0, 1]);
});
```

and `web/src/lib/format.test.js`

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { confidenceWord, displayFactor, displayUnit, fmt, fmtK, fmtRange } from "./format.js";

test("fmt picks precision by magnitude", () => {
  assert.equal(fmt(0.6123), "0.61");
  assert.equal(fmt(7.994), "7.99");
  assert.equal(fmt(117.9), "118");
  assert.equal(fmt(2042.4), "2,042");
  assert.equal(fmt(46480), "46.5k");
  assert.equal(fmt(1.46e6), "1.46M");
  assert.equal(fmt(NaN), "—");
});

test("fmtK keeps rpm-style numbers compact", () => {
  assert.equal(fmtK(11787), "11.8k");
  assert.equal(fmtK(7852), "7.85k");
  assert.equal(fmtK(950), "950");
});

test("fmtRange shows full digits for narrow intervals", () => {
  assert.equal(fmtRange(11654, 11920), "11,654–11,920");
  assert.equal(fmtRange(0.29, 0.83), "0.29–0.83");
});

test("endurance displays in minutes and units are prettified", () => {
  assert.ok(Math.abs(displayFactor("endurance_s") * 600 - 10) < 1e-9);
  assert.equal(displayUnit("endurance_s", "s"), "min");
  assert.equal(displayUnit("disk_loading_n_m2", "N/m^2"), "N/m²");
  assert.equal(displayUnit("thrust_to_weight", null), "");
});

test("confidence words follow the probability bands", () => {
  assert.equal(confidenceWord(0.85), "Confident");
  assert.equal(confidenceWord(0.65), "Likely");
  assert.equal(confidenceWord(0.54), "Leaning");
  assert.equal(confidenceWord(0.3), "Unsure");
});
```

In `web/package.json` add to `"scripts"`: `"test": "node --test src/lib/*.test.js"`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npm test`
Expected: FAIL — `Cannot find module .../wav.js` and missing exports from `format.js`.

- [ ] **Step 3: Implement** — create `web/src/lib/wav.js`

```js
export function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  }
  return buffer;
}

export function downmix(channels) {
  const out = new Float32Array(channels[0].length);
  for (const channel of channels) {
    for (let i = 0; i < out.length; i++) out[i] += channel[i] / channels.length;
  }
  return out;
}

export function latestWindow(samples, sampleRate, endIndex, seconds) {
  const end = Math.min(endIndex, samples.length);
  return samples.subarray(Math.max(0, end - Math.round(seconds * sampleRate)), end);
}
```

In `web/src/lib/format.js`, replace the existing `fmt` with the version below and append the other functions (leave `PRETTY`, `ORDER`, `CONFIDENCE_COLORS`, `relativeWidth`, `bestAttribution`, `sortedVariableKeys` in place; Task 7 removes them):

```js
export function trimZeros(text) {
  return text.includes(".") && !text.includes("e") ? text.replace(/\.?0+$/, "") : text;
}

export function fmt(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1e6) return trimZeros((x / 1e6).toPrecision(3)) + "M";
  if (a >= 1e4) return trimZeros((x / 1e3).toPrecision(3)) + "k";
  if (a >= 1000) return Math.round(x).toLocaleString("en-US");
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  if (a >= 1) return x.toFixed(2);
  if (a === 0) return "0";
  return trimZeros(x.toPrecision(2));
}

export function fmtK(x) {
  return Math.abs(x) >= 1000 ? trimZeros((x / 1000).toPrecision(3)) + "k" : fmt(x);
}

export function fmtRange(lo, hi) {
  const mid = Math.abs((lo + hi) / 2) || 1;
  const full = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : fmt(v));
  return (hi - lo) / mid < 0.1 ? `${full(lo)}–${full(hi)}` : `${fmt(lo)}–${fmt(hi)}`;
}

const DISPLAY = { endurance_s: { factor: 1 / 60, unit: "min" } };

export function displayFactor(key) {
  return DISPLAY[key] ? DISPLAY[key].factor : 1;
}

export function displayUnit(key, unit) {
  return DISPLAY[key] ? DISPLAY[key].unit : (unit || "").replace("^2", "²");
}

export function confidenceWord(p) {
  if (p >= 0.8) return "Confident";
  if (p >= 0.6) return "Likely";
  if (p >= 0.4) return "Leaning";
  return "Unsure";
}
```

- [ ] **Step 4: Run tests and build**

Run: `cd web && npm test && ./node_modules/.bin/vite build`
Expected: 9 tests pass; build succeeds (the old UI still compiles).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/wav.js web/src/lib/wav.test.js web/src/lib/format.js web/src/lib/format.test.js web/package.json
git commit -m "Add WAV encoding and formatting helpers for the live UI"
```

---

### Task 7: Live session, input panel and spectrum

**Files:**
- Modify: `web/src/api.js` (replace `predict` with `identify`, add `reference`)
- Create: `web/src/lib/audioSession.js`, `web/src/lib/useLiveSession.js`
- Create: `web/src/components/chrome.jsx`, `web/src/components/InputPanel.jsx`, `web/src/components/Spectrum.jsx`
- Rewrite: `web/src/App.jsx`, `web/src/styles.css`
- Delete: `web/src/components/Report.jsx`, `web/src/components/VariableBar.jsx`; remove `PRETTY`, `ORDER`, `CONFIDENCE_COLORS`, `relativeWidth`, `bestAttribution`, `sortedVariableKeys` from `web/src/lib/format.js`

**Interfaces:**
- Consumes: Task 4's `POST /identify`, `GET /reference`, `GET /health`; Task 6's `encodeWav`, `downmix`, `latestWindow`.
- Produces:
  - `identify(wavBuffer: ArrayBuffer, spec: object | null) -> Promise<IdentifyReport JSON>`, `reference() -> Promise<{bands, classes}>`, `health()` in `api.js`
  - `useLiveSession(known) -> { mode: "idle"|"playing"|"recording"|"done", source: {label, sampleId, duration} | null, result, history: {[variable]: {median, low, high}[]}, error, playSample(sample), playFile(file), record(), stop(), clear(), elapsed(): number | null }`
  - `Icon({name, className})`, `PanelTab({icon, title})`, `TopBar({backendUp})` in `components/chrome.jsx`
  - `SAMPLES` constant in `App.jsx`: `[{id, label}]`; known-facts state `{blade_count, prop_diameter_inch, num_motors}` with string values (`""` = not given)

- [ ] **Step 1: Run the API with auto-reload for UI work**

The API started earlier (`augur serve`) has no reload and predates `/identify`. Stop whatever listens on :8000, then:

```bash
kill $(lsof -nP -tiTCP:8000 -sTCP:LISTEN) 2>/dev/null
./.venv/bin/uvicorn augur.api:app --reload --reload-dir src --port 8000   # leave running
curl -s http://127.0.0.1:8000/reference | head -c 200
```
Expected: JSON starting with `{"bands":`.

- [ ] **Step 2: API client** — `web/src/api.js`

```js
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function errorDetail(res) {
  try {
    const body = await res.json();
    if (body.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* non-JSON error body */
  }
  return `${res.status} ${res.statusText}`;
}

export async function identify(wavBuffer, spec) {
  const form = new FormData();
  form.append("audio", new Blob([wavBuffer], { type: "audio/wav" }), "window.wav");
  if (spec) form.append("spec", JSON.stringify(spec));
  const res = await fetch(`${API_URL}/identify`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function reference() {
  const res = await fetch(`${API_URL}/reference`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function health() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("backend unreachable");
  return res.json();
}
```

- [ ] **Step 3: Audio session** — `web/src/lib/audioSession.js`

```js
import { downmix, encodeWav, latestWindow } from "./wav.js";

export const WINDOW_SECONDS = 3;
export const MIN_SECONDS = 0.5;
export const TICK_MS = 600;
const PLAYBACK_GAIN = 0.25;
const MIC_BUFFER_SECONDS = 8;

let context = null;

function audioContext() {
  if (!context) context = new AudioContext();
  if (context.state === "suspended") context.resume();
  return context;
}

export async function decodeFile(file) {
  return audioContext().decodeAudioData(await file.arrayBuffer());
}

export async function decodeUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return audioContext().decodeAudioData(await res.arrayBuffer());
}

function windowWav(samples, sampleRate, endIndex) {
  const win = latestWindow(samples, sampleRate, endIndex, WINDOW_SECONDS);
  return win.length >= MIN_SECONDS * sampleRate ? encodeWav(win, sampleRate) : null;
}

export function playBuffer(buffer, { onWindow, onEnd }) {
  const ctx = audioContext();
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const mono = downmix(channels);
  const rate = buffer.sampleRate;
  const source = ctx.createBufferSource();
  const gain = ctx.createGain();
  gain.gain.value = PLAYBACK_GAIN;
  source.buffer = buffer;
  source.connect(gain).connect(ctx.destination);
  const startedAt = ctx.currentTime;
  let finished = false;
  const elapsed = () => Math.min(ctx.currentTime - startedAt, buffer.duration);
  const timer = setInterval(() => {
    const wav = windowWav(mono, rate, Math.floor(elapsed() * rate));
    if (wav) onWindow(wav);
  }, TICK_MS);
  const finish = (endIndex) => {
    if (finished) return;
    finished = true;
    clearInterval(timer);
    onEnd(windowWav(mono, rate, endIndex));
  };
  source.onended = () => finish(mono.length);
  source.start();
  return {
    elapsed,
    stop() {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
      finish(Math.floor(elapsed() * rate));
    },
  };
}

export async function startMic({ onWindow }) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const ctx = audioContext();
  const rate = ctx.sampleRate;
  const keep = MIC_BUFFER_SECONDS * rate;
  const input = ctx.createMediaStreamSource(stream);
  // ScriptProcessorNode is deprecated but works everywhere and needs no separate worklet module.
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  let ring = new Float32Array(0);
  processor.onaudioprocess = (event) => {
    const chunk = event.inputBuffer.getChannelData(0);
    const next = new Float32Array(Math.min(ring.length + chunk.length, keep));
    const kept = ring.subarray(Math.max(0, ring.length + chunk.length - keep));
    next.set(kept, 0);
    next.set(chunk.subarray(chunk.length - (next.length - kept.length)), kept.length);
    ring = next;
  };
  input.connect(processor);
  processor.connect(ctx.destination);
  const startedAt = performance.now();
  let stoppedAt = null;
  const timer = setInterval(() => {
    const wav = windowWav(ring, rate, ring.length);
    if (wav) onWindow(wav);
  }, TICK_MS);
  return {
    elapsed: () => ((stoppedAt ?? performance.now()) - startedAt) / 1000,
    stop() {
      if (stoppedAt !== null) return;
      stoppedAt = performance.now();
      clearInterval(timer);
      processor.disconnect();
      input.disconnect();
      stream.getTracks().forEach((track) => track.stop());
    },
  };
}
```

- [ ] **Step 4: Session hook** — `web/src/lib/useLiveSession.js`

```js
import { useCallback, useEffect, useRef, useState } from "react";
import { identify } from "../api.js";
import { decodeFile, decodeUrl, playBuffer, startMic } from "./audioSession.js";

const HISTORY_LENGTH = 40;

function knownSpec(known) {
  const spec = {};
  for (const [key, value] of Object.entries(known)) if (value !== "") spec[key] = Number(value);
  return Object.keys(spec).length ? spec : null;
}

function appendHistory(history, report) {
  const next = { ...history };
  for (const [key, v] of Object.entries(report.variables)) {
    const point = { median: v.median, low: v.interval_low, high: v.interval_high };
    next[key] = [...(history[key] || []), point].slice(-HISTORY_LENGTH);
  }
  return next;
}

export function useLiveSession(known) {
  const [mode, setMode] = useState("idle");
  const [source, setSource] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState({});
  const [error, setError] = useState(null);
  const session = useRef(null);
  const sessionId = useRef(0);
  const inflight = useRef(false);
  const pendingFinal = useRef(null);
  const knownRef = useRef(known);
  knownRef.current = known;

  const analyze = useCallback(async function run(wav, id, final = false) {
    if (!wav || id !== sessionId.current) return;
    if (inflight.current) {
      if (final) pendingFinal.current = { wav, id };
      return;
    }
    inflight.current = true;
    try {
      const report = await identify(wav, knownSpec(knownRef.current));
      if (id !== sessionId.current) return;
      setResult(report);
      setError(null);
      setHistory((h) => appendHistory(h, report));
    } catch (err) {
      if (id === sessionId.current) setError(`Analysis failed: ${err.message}`);
    } finally {
      inflight.current = false;
      const next = pendingFinal.current;
      pendingFinal.current = null;
      if (next) run(next.wav, next.id, true);
    }
  }, []);

  const begin = useCallback((nextMode, nextSource) => {
    const old = session.current;
    session.current = null;
    sessionId.current += 1;
    old?.stop();
    setMode(nextMode);
    setSource(nextSource);
    setResult(null);
    setHistory({});
    setError(null);
    return sessionId.current;
  }, []);

  const play = useCallback((buffer, label, sampleId = null) => {
    const id = begin("playing", { label, sampleId, duration: buffer.duration });
    session.current = playBuffer(buffer, {
      onWindow: (wav) => analyze(wav, id),
      onEnd: (wav) => {
        if (id !== sessionId.current) return;
        setMode("done");
        analyze(wav, id, true);
      },
    });
  }, [analyze, begin]);

  const playSample = useCallback(async (sample) => {
    try {
      play(await decodeUrl(`/templates/${sample.id}.wav`), `${sample.label} sample (simulated)`, sample.id);
    } catch (err) {
      setError(`Couldn't load the ${sample.label} sample: ${err.message}`);
    }
  }, [play]);

  const playFile = useCallback(async (file) => {
    try {
      play(await decodeFile(file), file.name);
    } catch {
      setError(`Couldn't decode ${file.name}. Try a WAV, MP3 or OGG file.`);
    }
  }, [play]);

  const record = useCallback(async () => {
    const id = begin("recording", { label: "Microphone", sampleId: null, duration: 0 });
    let mic;
    try {
      mic = await startMic({ onWindow: (wav) => analyze(wav, id) });
    } catch (err) {
      if (id === sessionId.current) {
        setMode("idle");
        setError(`Microphone unavailable: ${err.message}`);
      }
      return;
    }
    if (id !== sessionId.current) mic.stop();
    else session.current = mic;
  }, [analyze, begin]);

  const stop = useCallback(() => {
    if (!session.current) return;
    session.current.stop();
    setMode("done");
  }, []);

  const clear = useCallback(() => {
    begin("idle", null);
  }, [begin]);

  const elapsed = useCallback(() => (session.current ? session.current.elapsed() : null), []);

  useEffect(() => () => session.current?.stop(), []);

  return { mode, source, result, history, error, playSample, playFile, record, stop, clear, elapsed };
}
```

(For playback, `stop()` triggers `onEnd`, which runs the final analysis of the audio heard so far; for the mic it just freezes the clock.)

- [ ] **Step 5: Chrome components** — `web/src/components/chrome.jsx`

```jsx
const PATHS = {
  drone: (
    <>
      <circle cx="3.5" cy="3.5" r="2.2" />
      <circle cx="12.5" cy="3.5" r="2.2" />
      <circle cx="3.5" cy="12.5" r="2.2" />
      <circle cx="12.5" cy="12.5" r="2.2" />
      <path d="M5.2 5.2l5.6 5.6M10.8 5.2l-5.6 5.6" />
    </>
  ),
  wave: <path d="M1 8h2l1.5-4 2 8 2-10 2 10 1.5-4H15" />,
  upload: <path d="M8 10.5V2M4.5 5.5L8 2l3.5 3.5M2 10.5v3h12v-3" />,
  stop: <rect x="3.5" y="3.5" width="9" height="9" rx="1" fill="currentColor" stroke="none" />,
  target: (
    <>
      <circle cx="8" cy="8" r="6" />
      <circle cx="8" cy="8" r="2.5" />
    </>
  ),
  list: <path d="M2 4h12M2 8h12M2 12h8" />,
};

export function Icon({ name, className = "" }) {
  return (
    <svg className={`icon ${className}`} viewBox="0 0 16 16" fill="none" stroke="currentColor"
      strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {PATHS[name]}
    </svg>
  );
}

export function PanelTab({ icon, title }) {
  return (
    <div className="strip">
      <span><Icon name={icon} />{title}</span>
    </div>
  );
}

export function TopBar({ backendUp }) {
  const status = backendUp === null ? "connecting…"
    : backendUp ? "analysis server connected" : "analysis server offline: run augur serve";
  return (
    <header className="topbar">
      <div className="brand"><Icon name="drone" /><span className="word">AUGUR</span></div>
      <div className="vdiv" />
      <span className="tagline">Identify a drone from its sound, live</span>
      <div className="spacer" />
      <div className="status">
        <span className={`dot ${backendUp ? "ok" : backendUp === false ? "bad" : ""}`} />
        {status}
      </div>
    </header>
  );
}
```

- [ ] **Step 6: Spectrum** — `web/src/components/Spectrum.jsx`

```jsx
import { useLayoutEffect, useRef, useState } from "react";

const HEIGHT = 170;
const M = { left: 40, right: 12, top: 18, bottom: 22 };

function useWidth(ref) {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

export default function Spectrum({ acoustics }) {
  const host = useRef(null);
  const width = useWidth(host);
  const ready = acoustics && acoustics.spectrum_freq_hz.length > 0 && width > 0;
  return (
    <div className="spectrum" ref={host}>
      {ready ? <Plot acoustics={acoustics} width={width} /> : <div className="wait">Listening…</div>}
    </div>
  );
}

function Plot({ acoustics, width }) {
  const freqs = acoustics.spectrum_freq_hz;
  const db = acoustics.spectrum_db;
  const fmax = freqs[freqs.length - 1];
  const x = (f) => M.left + (f / fmax) * (width - M.left - M.right);
  const y = (d) => M.top + (-d / 80) * (HEIGHT - M.top - M.bottom);
  const points = freqs.map((f, i) => `${x(f).toFixed(1)},${y(db[i]).toFixed(1)}`).join(" L");
  const step = niceStep(fmax, Math.max(3, Math.floor((width - M.left - M.right) / 90)));
  const ticks = [];
  for (let t = 0; t <= fmax + 1e-9; t += step) ticks.push(t);
  const nearestDb = (f) => {
    let j = 0;
    for (let i = 1; i < freqs.length; i++) if (Math.abs(freqs[i] - f) < Math.abs(freqs[j] - f)) j = i;
    return db[j];
  };
  return (
    <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label="Live audio spectrum">
      <defs>
        <linearGradient id="spectrum-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#a986ec" stopOpacity="0.3" />
          <stop offset="1" stopColor="#a986ec" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, -40, -80].map((d) => (
        <g key={d}>
          <line className="grid" x1={M.left} x2={width - M.right} y1={y(d)} y2={y(d)} />
          <text className="ax-tick" x={M.left - 6} y={y(d) + 3.5} textAnchor="end">{d === 0 ? "0 dB" : `−${-d}`}</text>
        </g>
      ))}
      <path d={`M${x(freqs[0])},${y(-80)} L${points} L${x(fmax)},${y(-80)}Z`} fill="url(#spectrum-fill)" />
      <path className="spec-line" d={`M${points}`} />
      <line className="ax-line" x1={M.left} x2={width - M.right} y1={HEIGHT - M.bottom} y2={HEIGHT - M.bottom} />
      {ticks.map((t) => (
        <text key={t} className="ax-tick" x={x(t)} y={HEIGHT - M.bottom + 13} textAnchor="middle">
          {t >= 1000 ? Math.round(t).toLocaleString("en-US") : Math.round(t)}{t === 0 ? " Hz" : ""}
        </text>
      ))}
      {acoustics.harmonics.map((h) => {
        const label = h.order === 1 ? "BPF" : `${h.order}×`;
        if (h.matched_hz == null) {
          const hx = x(h.expected_hz);
          return (
            <g key={h.order}>
              <line className="h-miss" x1={hx} x2={hx} y1={M.top - 2} y2={HEIGHT - M.bottom} />
              <text className="h-miss-label" x={hx} y={M.top - 6} textAnchor="middle">{label}?</text>
            </g>
          );
        }
        const hx = x(h.matched_hz);
        const hy = y(nearestDb(h.matched_hz));
        return (
          <g key={h.order}>
            <line className="h-hit" x1={hx} x2={hx} y1={M.top - 2} y2={hy - 8} />
            <path className="h-hit-mark" d={`M${hx} ${hy - 2}l-4 -6h8z`} />
            <text className="h-hit-label" x={hx} y={M.top - 6} textAnchor="middle">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 7: Input panel** — `web/src/components/InputPanel.jsx`

```jsx
import { useEffect, useRef, useState } from "react";
import { Icon, PanelTab } from "./chrome.jsx";
import Spectrum from "./Spectrum.jsx";

const KNOWN_FIELDS = [
  { key: "blade_count", label: "Blades per prop", options: [["", "Any"], ["2", "2"], ["3", "3"], ["4", "4"]] },
  { key: "prop_diameter_inch", label: "Prop size",
    options: [["", "Any"], ["5", "5″"], ["7", "7″"], ["9", "9″"], ["10", "10″"], ["13", "13″"], ["15", "15″"], ["18", "18″"]] },
  { key: "num_motors", label: "Motors", options: [["", "Any"], ["4", "4"], ["6", "6"], ["8", "8"]] },
];
const BADGE = { playing: "PLAYING", recording: "LIVE", done: "DONE" };

function clock(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function useElapsed(elapsed, mode) {
  const [now, setNow] = useState(null);
  useEffect(() => {
    setNow(elapsed());
    if (mode !== "playing" && mode !== "recording") return undefined;
    const timer = setInterval(() => setNow(elapsed()), 100);
    return () => clearInterval(timer);
  }, [elapsed, mode]);
  return now;
}

export default function InputPanel({ session, known, onKnownChange, samples }) {
  const { mode, source, result, error, playSample, playFile, record, stop, elapsed } = session;
  const fileInput = useRef(null);
  const [dragging, setDragging] = useState(false);
  const now = useElapsed(elapsed, mode);
  const live = mode === "playing" || mode === "recording";
  const onFile = (event) => {
    const file = event.target.files?.[0];
    if (file) playFile(file);
    event.target.value = "";
  };
  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) playFile(file);
  };
  const actions = (
    <>
      <button type="button" className={`btn ${mode === "idle" ? "btn--primary" : ""}`} onClick={() => fileInput.current?.click()}>
        <Icon name="upload" />Choose file
      </button>
      <button type="button" className="btn btn--rec" onClick={record}>
        <span className="rec-dot" />{mode === "idle" ? "Record from mic" : "Record"}
      </button>
    </>
  );

  return (
    <section className="panel">
      <PanelTab icon="wave" title="Input" />
      <div className="body">
        <input ref={fileInput} type="file" accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac" hidden onChange={onFile} />
        {mode === "idle" ? (
          <div className={`drop ${dragging ? "over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <Icon name="wave" className="big-icon" />
            <h2>Drop a drone recording here</h2>
            <p>WAV, MP3 or OGG. Or record it live from your microphone.</p>
            <div className="actions">{actions}</div>
          </div>
        ) : (
          <>
            <div className="live-head">
              <span className={`live-badge live-badge--${mode}`}><span className="pulse" />{BADGE[mode]}</span>
              <span className="live-src">{source?.label}</span>
              {now != null && (
                <span className="live-time">
                  {source?.duration ? `${clock(Math.min(now, source.duration))} / ${clock(source.duration)}` : clock(now)}
                </span>
              )}
              <div className="live-actions">
                {live && <button type="button" className="btn btn--stop" onClick={stop}><Icon name="stop" />Stop</button>}
                {actions}
              </div>
            </div>
            {mode === "recording" && (
              <p className="mic-note">
                Listening on your mic. Augur assumes a drone is in earshot: fans, AC units and electrical hum
                make the same kind of tone and can fool it.
              </p>
            )}
            {source?.duration ? (
              <div className="progress"><i style={{ width: `${Math.min(1, (now ?? 0) / source.duration) * 100}%` }} /></div>
            ) : null}
            <Spectrum acoustics={result?.acoustics} />
          </>
        )}
        <div className="samples">
          <span className="lbl">Try a sample:</span>
          {samples.map((s) => (
            <button key={s.id} type="button" className="chip-btn" aria-pressed={source?.sampleId === s.id} onClick={() => playSample(s)}>
              {s.label}
            </button>
          ))}
        </div>
        <details className="known" open={Object.values(known).some(Boolean) || undefined}>
          <summary>Know anything about it? (optional)</summary>
          <div className="known-grid">
            {KNOWN_FIELDS.map((f) => (
              <label key={f.key}>
                {f.label}
                <select value={known[f.key]} onChange={(e) => onKnownChange(f.key, e.target.value)}>
                  {f.options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
                </select>
              </label>
            ))}
          </div>
        </details>
        {error && <p className="err">{error}</p>}
      </div>
    </section>
  );
}
```

- [ ] **Step 8: App shell (temporary main area)** — `web/src/App.jsx`

```jsx
import { useEffect, useState } from "react";
import { health } from "./api.js";
import { PanelTab, TopBar } from "./components/chrome.jsx";
import InputPanel from "./components/InputPanel.jsx";
import { useLiveSession } from "./lib/useLiveSession.js";

const SAMPLES = [
  { id: "racing", label: "Racing 5″" },
  { id: "cinematic", label: "Cinematic 9″" },
  { id: "survey", label: "Survey 15″ hex" },
  { id: "military", label: "Military 13″ · 3-blade" },
];
const NO_KNOWN = { blade_count: "", prop_diameter_inch: "", num_motors: "" };

export default function App() {
  const [backendUp, setBackendUp] = useState(null);
  const [known, setKnown] = useState(NO_KNOWN);
  const session = useLiveSession(known);

  useEffect(() => {
    health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
  }, []);

  return (
    <div className="app">
      <TopBar backendUp={backendUp} />
      <main className="workspace">
        <div className="col">
          <InputPanel session={session} known={known} samples={SAMPLES}
            onKnownChange={(key, value) => setKnown((k) => ({ ...k, [key]: value }))} />
          <section className="panel panel--grow">
            <PanelTab icon="target" title="What it is" />
            <div className="body">{session.result ? session.result.classes[0].name : "Waiting for audio"}</div>
          </section>
        </div>
        <section className="panel"><PanelTab icon="list" title="Live estimates" /></section>
      </main>
    </div>
  );
}
```

- [ ] **Step 9: Styles** — replace `web/src/styles.css` entirely:

```css
:root {
  color-scheme: dark;
  --bg-app: #0e0e0e;
  --bg-topbar: #1c1c1c;
  --bg-tabstrip: #000000;
  --bg-pane: #222222;
  --bg-raised: #2a2a2a;
  --bg-hover: #2e2e2e;
  --bg-input: #1a1a1a;
  --border-strong: #3a3a3a;
  --border-input: #757575;
  --divider: #3a3a3a;
  --text-primary: #ececec;
  --text-secondary: #bdbdbd;
  --text-muted: #a0a0a0;
  --accent-fill: #663399;
  --accent-fill-hover: #7442ad;
  --accent: #a986ec;
  --accent-strong: #b896f0;
  --accent-tint: rgba(102, 51, 153, 0.35);
  --series-contrast: #56c2d6;
  --track: #404040;
  --median: #f2f2f2;
  --grid: #2e2e2e;
  --axis: #737373;
  --tick: #a8a8a8;
  --conf-high: #3fbf7f;
  --conf-medium: #e0b34a;
  --conf-low: #f08a4b;
  --conf-wide: #8c8c8c;
  --ok: #3fbf7f;
  --error: #f2797a;
  --warn: #e8c35a;
  --rec: #f2797a;
  --font-ui: "Roboto", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --radius: 2px;
}

* { box-sizing: border-box; }
html, body, #root { height: 100%; }
body {
  margin: 0;
  background: var(--bg-app);
  color: var(--text-primary);
  font: 400 12.5px/1.4 var(--font-ui);
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
button, select { font: inherit; color: inherit; }
svg { display: block; }
:focus-visible { outline: 2px solid var(--accent-strong); outline-offset: 1px; }
.icon { width: 14px; height: 14px; flex: none; }

.app { height: 100%; display: flex; flex-direction: column; overflow: hidden; }
.topbar { height: 36px; flex: none; display: flex; align-items: center; gap: 12px; background: var(--bg-topbar); border-bottom: 1px solid #000; padding: 0 12px; white-space: nowrap; overflow: hidden; }
.brand { display: flex; align-items: center; gap: 7px; font-size: 15px; font-weight: 800; letter-spacing: .06em; }
.brand .word { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 3px; }
.brand .icon { width: 18px; height: 18px; color: var(--accent); }
.tagline { color: var(--text-secondary); }
.vdiv { width: 1px; height: 18px; background: var(--divider); }
.spacer { flex: 1; }
.status { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }
.dot.ok { background: var(--ok); }
.dot.bad { background: var(--error); }

.workspace { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 400px; gap: 4px; padding-top: 4px; }
.col { display: flex; flex-direction: column; gap: 4px; min-height: 0; min-width: 0; }
.panel { background: var(--bg-pane); display: flex; flex-direction: column; min-height: 0; min-width: 0; }
.panel--grow { flex: 1; }
.strip { height: 22px; flex: none; background: var(--bg-tabstrip); display: flex; align-items: flex-end; }
.strip span { height: 20px; display: flex; align-items: center; gap: 6px; padding: 0 10px; font-size: 11px; font-weight: 500; color: var(--text-secondary); background: var(--bg-pane); border-top: 2px solid var(--accent-fill); }
.strip .icon { width: 12px; height: 12px; }
.body { padding: 14px 16px 16px; min-height: 0; }
.body--scroll { overflow: auto; flex: 1; scrollbar-width: thin; scrollbar-color: #5a5a5a var(--bg-pane); }

.drop { border: 1.5px dashed var(--border-input); border-radius: 4px; background: var(--bg-input); padding: 26px 20px 22px; display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center; transition: border-color .15s, background .15s; }
.drop.over { border-color: var(--accent); background: #241d2e; }
.drop .big-icon { width: 34px; height: 34px; color: var(--accent); }
.drop h2 { margin: 0; font-size: 18px; font-weight: 600; }
.drop p { margin: 0; color: var(--text-muted); font-size: 12px; }
.actions { display: flex; gap: 10px; margin-top: 6px; flex-wrap: wrap; justify-content: center; }
.btn { display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 16px; border-radius: var(--radius); border: 1px solid var(--border-strong); background: var(--bg-raised); font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap; }
.btn:hover { background: var(--bg-hover); }
.btn--primary { background: var(--accent-fill); border-color: var(--accent-fill); color: #fff; }
.btn--primary:hover { background: var(--accent-fill-hover); }
.rec-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--rec); }
.btn--stop { background: #3a1f24; border-color: #6b2f37; }
.btn--stop:hover { background: #4a252c; }
.samples { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
.samples .lbl { color: var(--text-muted); font-size: 11.5px; margin-right: 4px; }
.chip-btn { height: 26px; padding: 0 10px; border-radius: var(--radius); border: 1px solid var(--border-strong); background: var(--bg-raised); font-size: 12px; cursor: pointer; }
.chip-btn:hover { background: var(--bg-hover); }
.chip-btn[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-tint); }
.known { margin-top: 12px; color: var(--text-muted); font-size: 11.5px; }
.known summary { cursor: pointer; width: max-content; }
.known-grid { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.known label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--text-secondary); }
.known select { height: 26px; min-width: 90px; padding: 0 22px 0 8px; background: var(--bg-input); border: 1px solid var(--border-strong); border-bottom-color: var(--border-input); border-radius: var(--radius); appearance: none; font-size: 12px; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23bdbdbd' stroke-width='1.5'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 7px center; }
.err { margin: 10px 0 0; color: var(--error); font-size: 12px; }

.live-head { display: flex; align-items: center; gap: 12px; }
.live-badge { display: inline-flex; align-items: center; gap: 7px; font-weight: 700; font-size: 12px; letter-spacing: .06em; }
.pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--rec); animation: pulse 1.2s infinite; }
.live-badge--playing .pulse { background: var(--accent); }
.live-badge--done .pulse { background: var(--ok); animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
@media (prefers-reduced-motion: reduce) { .pulse { animation: none; } }
.live-src { color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.live-time { color: var(--text-muted); }
.live-actions { margin-left: auto; display: flex; gap: 8px; }
.live-actions .btn { height: 30px; padding: 0 12px; font-size: 12.5px; }
.mic-note { margin: 8px 0 0; color: var(--warn); font-size: 11.5px; }
.progress { height: 3px; background: var(--track); margin-top: 10px; border-radius: 2px; overflow: hidden; }
.progress i { display: block; height: 100%; background: var(--accent); }
.spectrum { position: relative; height: 170px; margin-top: 8px; }
.spectrum svg { position: absolute; inset: 0; }
.spectrum .wait { position: absolute; inset: 0; display: grid; place-items: center; color: var(--text-muted); }
.ax-line { stroke: var(--axis); }
.ax-tick { fill: var(--tick); font-size: 10px; }
.grid { stroke: var(--grid); }
.spec-line { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-linejoin: round; }
.h-hit { stroke: var(--series-contrast); stroke-dasharray: 3 3; }
.h-hit-mark { fill: var(--series-contrast); }
.h-hit-label { fill: var(--series-contrast); font-size: 10px; font-weight: 600; }
.h-miss { stroke: var(--text-muted); stroke-dasharray: 2 3; }
.h-miss-label { fill: var(--text-muted); font-size: 10px; }

@media (max-width: 1000px) {
  .app { height: auto; overflow: visible; }
  .workspace { grid-template-columns: minmax(0, 1fr); }
  .tagline { display: none; }
}
@media (max-width: 640px) {
  .live-head { flex-wrap: wrap; row-gap: 8px; }
  .live-actions { margin-left: 0; width: 100%; }
  .live-actions .btn { flex: 1; justify-content: center; }
  .drop { padding: 20px 14px; }
}
```

- [ ] **Step 10: Delete the old report UI** — `git rm web/src/components/Report.jsx web/src/components/VariableBar.jsx`, then delete `PRETTY`, `ORDER`, `CONFIDENCE_COLORS`, `prettyName`, `relativeWidth`, `bestAttribution` and `sortedVariableKeys` from `web/src/lib/format.js` (keep everything Task 6 added).

- [ ] **Step 11: Build, test, and check it live**

Run: `cd web && npm test && ./node_modules/.bin/vite build`
Expected: 9 tests pass; build succeeds with no warnings about missing exports.

Then with the API on :8000 and `npm run dev` on :5173, run the real-time headless check (writes PNGs; prints the final page state):

```bash
P=/private/tmp/claude-501/-Users-brodyk-work-drone-predictor/5fb2ff8b-d40a-4f1f-83df-78935023fbc3/scratchpad/proto
CDP_PROFILE=$P/cdp-profile node $P/cdp_shots.mjs "http://localhost:5173/" $P/shots/task7 1440x900 1000,5000 'button.chip-btn'
```
(The script's final `state` probe targets the prototype's globals and will print `undefined` here; judge by the screenshots.) Expected in `task7-5000ms.png`: PLAYING header with the Racing sample, a spectrum with six cyan harmonic markers, the main area showing `racing`. Also click *Record* manually in a real browser at http://localhost:5173 and confirm the mic permission prompt, the LIVE badge and the mic note.

- [ ] **Step 12: Commit**

```bash
git add web/src
git commit -m "Rebuild the web UI around a live input session and spectrum"
```

---

### Task 8: Identification hero and live estimates panel

**Files:**
- Create: `web/src/components/Identification.jsx`, `web/src/components/EstimatesPanel.jsx`, `web/src/components/Sparkline.jsx`
- Modify: `web/src/App.jsx` (fetch `reference()`, render the new panels)
- Modify: `web/src/styles.css` (append identification + estimates styles)

**Interfaces:**
- Consumes: Task 7's `useLiveSession` (`result`, `history`), `reference()` → `{bands, classes}`, `PanelTab`; Task 6's `fmt`, `fmtK`, `fmtRange`, `displayFactor`, `displayUnit`, `confidenceWord`.
- Produces: `Identification({result, descriptions})`, `EstimatesPanel({result, history, bands, known})`, `Sparkline({points, factor})`.

- [ ] **Step 1: Sparkline** — `web/src/components/Sparkline.jsx`

```jsx
const W = 96;
const H = 20;

export default function Sparkline({ points, factor }) {
  if (points.length < 2) return <svg className="spark" aria-hidden="true" />;
  const lows = points.map((p) => p.low * factor);
  const highs = points.map((p) => p.high * factor);
  const medians = points.map((p) => p.median * factor);
  const lo = Math.min(...lows);
  const hi = Math.max(...highs);
  const useLog = lo > 0 && hi / lo > 20;
  const tr = (v) => (useLog ? Math.log10(v) : v);
  const a = tr(lo);
  const b = tr(hi) === a ? a + 1 : tr(hi);
  const x = (i) => (i / (points.length - 1)) * (W - 2) + 1;
  const y = (v) => H - 1 - ((tr(v) - a) / (b - a)) * (H - 2);
  const path = (values) => values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const top = path(highs);
  const bottom = path(lows).reverse();
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <path className="band" d={`M${top.join("L")}L${bottom.join("L")}Z`} />
      <path className="line" d={`M${path(medians).join("L")}`} />
    </svg>
  );
}
```

- [ ] **Step 2: Identification** — `web/src/components/Identification.jsx`

```jsx
import { confidenceWord } from "../lib/format.js";
import { PanelTab } from "./chrome.jsx";

const CLASS_NAMES = {
  racing: "Racing / FPV quad",
  cinematic: "Cinematic camera drone",
  survey: "Survey / mapping drone",
  military: "Military / ISR drone",
};
const displayName = (name) => CLASS_NAMES[name] || name;

const SPECIAL = {
  no_tone: () => ["Listening…", "No steady drone tone yet. Keep the drone in earshot, ideally hovering."],
  not_hovering: (r) => ["Can't tell yet", `The pitch is changing (${r.acoustics.flight_state.state}), so the hover physics behind the guess doesn't apply. Wait for a steady hover.`],
  no_class_fits: () => ["Unknown type", "The tone doesn't fit any of the four known classes in hover."],
};

function whyLines(result) {
  const ac = result.acoustics;
  const fs = ac.flight_state;
  const lines = [];
  if (ac.fundamental_hz) {
    const found = ac.harmonics.filter((h) => h.matched_hz != null).length;
    lines.push(<>Blade-pass tone <b>{ac.fundamental_hz.toFixed(0)} Hz</b> with <b>{found}/{ac.harmonics.length}</b> harmonics found</>);
  }
  if (fs.state === "hover" && fs.pitch_variation != null) {
    lines.push(<>Hovering: pitch steady (variation {(fs.pitch_variation * 100).toFixed(1)}% ≤ {(fs.steady_threshold * 100).toFixed(0)}%)</>);
  }
  if (result.blades.length > 1) {
    lines.push(
      <>
        Blades per prop:{" "}
        {result.blades.map((b, i) => (
          <span key={b.blades}>{i ? " or " : ""}<b>{b.blades}</b> ({Math.round(b.probability * 100)}%)</span>
        ))}
        . Not given, so both are considered.
      </>,
    );
  }
  return lines;
}

function Guess({ result, descriptions }) {
  const top = result.classes[0];
  const ok = result.status === "ok";
  const [specialName, specialSub] = ok ? [] : SPECIAL[result.status](result);
  const warnings = [...result.notes];
  if (ok && top.probability < 0.6) {
    warnings.push("Several classes sound alike from audio alone. Telling it the prop size (above) can narrow it down.");
  }
  return (
    <>
      <div className="guess">
        {ok ? (
          <>
            <span className="guess-name">{displayName(top.name)}</span>
            <span className="guess-p">{Math.round(top.probability * 100)}%</span>
          </>
        ) : (
          <span className="guess-name dim">{specialName}</span>
        )}
      </div>
      <div className="guess-sub">
        {ok ? `${confidenceWord(top.probability)} · from ${result.duration_s.toFixed(1)} s of audio` : specialSub}
      </div>
      {ok && descriptions?.[top.name] && <div className="guess-desc">{descriptions[top.name]}</div>}
      <div className="bars">
        {result.classes.map((c, i) => {
          const isTop = ok && i === 0;
          return (
            <div className="bar-row" key={c.name}>
              <span className={`name ${isTop ? "top" : ""}`}>{displayName(c.name)}</span>
              <div className={`bar ${isTop ? "top" : ""}`}><i style={{ width: `${(c.probability * 100).toFixed(1)}%` }} /></div>
              <span className="pct">{Math.round(c.probability * 100)}%</span>
            </div>
          );
        })}
      </div>
      <div className="why">
        {whyLines(result).map((line, i) => <div key={i}>{line}</div>)}
        {warnings.map((w) => <div key={w} className="warn">{w}</div>)}
      </div>
    </>
  );
}

export default function Identification({ result, descriptions }) {
  return (
    <section className="panel panel--grow">
      <PanelTab icon="target" title="What it is" />
      <div className="body body--scroll">
        <div className="eyebrow">Best guess</div>
        {result ? (
          <Guess result={result} descriptions={descriptions} />
        ) : (
          <>
            <div className="guess"><span className="guess-name dim">Waiting for audio</span></div>
            <div className="guess-sub">Drop a file, record from the mic, or try a sample above.</div>
          </>
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Estimates panel** — `web/src/components/EstimatesPanel.jsx`

```jsx
import { displayFactor, displayUnit, fmt, fmtK, fmtRange } from "../lib/format.js";
import { PanelTab } from "./chrome.jsx";
import Sparkline from "./Sparkline.jsx";

const ESTIMATES = [
  ["mass_kg", "Mass"],
  ["rpm", "Rotor speed"],
  ["electrical_power_w", "Hover power"],
  ["endurance_s", "Hover endurance"],
  ["battery_wh", "Battery"],
  ["total_thrust_n", "Total thrust"],
  ["disk_loading_n_m2", "Disk loading"],
  ["prop_diameter_inch", "Prop size"],
  ["num_motors", "Motors"],
  ["thrust_to_weight", "Thrust-to-weight"],
];

function logPosition(band) {
  const lo = Math.log10(Math.max(band[0], 1e-9));
  const hi = Math.log10(band[1]);
  return (x) => Math.min(1, Math.max(0, (Math.log10(Math.max(x, 1e-12)) - lo) / (hi - lo))) * 100;
}

function Bar({ left, width, wide, ticks }) {
  return (
    <div className="est-bar">
      <span className={`fill ${wide ? "wide" : ""}`} style={{ left: `${left}%`, width: `${width}%` }} />
      {ticks.map((t) => <span key={t} className="tick" style={{ left: `calc(${t}% - 1px)` }} />)}
    </div>
  );
}

function RpmRow({ label, summary, f0, blades, pos, history }) {
  const alts = [...blades].sort((a, b) => a.blades - b.blades).map((b) => ({ blades: b.blades, rpm: (f0 * 60) / b.blades }));
  const left = pos(summary.interval_low);
  return (
    <div className="est">
      <div className="est-top">
        <span className="est-name">{label}</span>
        <span className="est-val">{alts.map((a) => fmtK(a.rpm)).join(" or ")}<span className="u">rpm</span></span>
      </div>
      <Bar left={left} width={Math.max(pos(summary.interval_high) - left, 1.2)} wide={false} ticks={alts.map((a) => pos(a.rpm))} />
      <div className="est-foot">
        <span className="left">{alts.map((a) => `${a.blades}-blade`).join(" or ")} props · count not given</span>
        <Sparkline points={history} factor={1} />
      </div>
    </div>
  );
}

function EstimateRow({ varKey, label, summary, band, history, result, given }) {
  const pos = logPosition(band);
  const f0 = result.acoustics.fundamental_hz;
  if (varKey === "rpm" && f0 && result.blades.length > 1 && result.status !== "no_tone") {
    return <RpmRow label={label} summary={summary} f0={f0} blades={result.blades} pos={pos} history={history} />;
  }
  const f = displayFactor(varKey);
  const unit = displayUnit(varKey, summary.unit);
  const fixed = summary.interval_low === summary.interval_high;
  const prior = varKey === "thrust_to_weight" || (varKey === "rpm" && !f0);
  const wide = summary.confidence === "unconstrained" || prior;
  const isInt = varKey === "num_motors";
  const range = isInt
    ? `${Math.round(summary.interval_low)}–${Math.round(summary.interval_high)}`
    : fmtRange(summary.interval_low * f, summary.interval_high * f);
  const value = fixed ? (isInt ? Math.round(summary.median) : fmt(summary.median * f)) : wide || isInt ? range : fmt(summary.median * f);
  const left = pos(summary.interval_low);
  const chipText = fixed ? (given ? "given" : "fixed") : varKey === "thrust_to_weight" ? "class prior" : wide ? "rough" : summary.confidence;
  const chipTone = fixed ? "fixed" : wide ? "wide" : summary.confidence;
  return (
    <div className="est">
      <div className="est-top">
        <span className="est-name">{label}</span>
        <span className={`est-val ${wide && !fixed ? "wide" : ""}`}>{value}{unit && <span className="u">{unit}</span>}</span>
      </div>
      <Bar left={left} width={Math.max(pos(summary.interval_high) - left, 1.2)} wide={wide} ticks={wide || fixed ? [] : [pos(summary.median)]} />
      <div className="est-foot">
        <span className="left">
          {fixed ? "from your input" : `90% ${range}`} <span className={`chip chip--${chipTone}`}><i />{chipText}</span>
        </span>
        <Sparkline points={history} factor={f} />
      </div>
    </div>
  );
}

export default function EstimatesPanel({ result, history, bands, known }) {
  return (
    <section className="panel">
      <PanelTab icon="list" title="Live estimates" />
      <div className="body--scroll">
        {!result || !bands ? (
          <div className="placeholder">Estimates appear here and update as audio comes in.</div>
        ) : (
          <>
            {ESTIMATES.map(([key, label]) => (
              <EstimateRow key={key} varKey={key} label={label} summary={result.variables[key]} band={bands[key]}
                history={history[key] || []} result={result} given={Boolean(known[key])} />
            ))}
            <div className="side-note">
              <div>Every value is a 90% interval: the bar spans it, the white tick is the median. Grey means the inputs barely constrain it.</div>
              <div>Bars use a log scale covering what Augur would say with no input at all.</div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Wire the App** — in `web/src/App.jsx` change the imports and body:

```jsx
import { useEffect, useState } from "react";
import { health, reference } from "./api.js";
import EstimatesPanel from "./components/EstimatesPanel.jsx";
import Identification from "./components/Identification.jsx";
import InputPanel from "./components/InputPanel.jsx";
import { TopBar } from "./components/chrome.jsx";
import { useLiveSession } from "./lib/useLiveSession.js";
```

(keep `SAMPLES` and `NO_KNOWN`), then:

```jsx
export default function App() {
  const [backendUp, setBackendUp] = useState(null);
  const [ref, setRef] = useState(null);
  const [known, setKnown] = useState(NO_KNOWN);
  const session = useLiveSession(known);

  useEffect(() => {
    health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
    reference().then(setRef).catch(() => setRef(null));
  }, []);

  return (
    <div className="app">
      <TopBar backendUp={backendUp} />
      <main className="workspace">
        <div className="col">
          <InputPanel session={session} known={known} samples={SAMPLES}
            onKnownChange={(key, value) => setKnown((k) => ({ ...k, [key]: value }))} />
          <Identification result={session.result} descriptions={ref?.classes} />
        </div>
        <EstimatesPanel result={session.result} history={session.history} bands={ref?.bands} known={known} />
      </main>
    </div>
  );
}
```

- [ ] **Step 5: Styles** — append to `web/src/styles.css`:

```css
.eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--text-muted); }
.guess { display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; margin-top: 4px; }
.guess-name { font-size: 44px; font-weight: 800; letter-spacing: -.01em; line-height: 1.05; }
.guess-name.dim { color: var(--text-muted); font-weight: 600; }
.guess-p { font-size: 44px; font-weight: 300; color: var(--accent-strong); line-height: 1.05; }
.guess-sub { margin-top: 6px; color: var(--text-secondary); font-size: 14px; }
.guess-desc { margin-top: 2px; color: var(--text-muted); }
.bars { margin-top: 20px; display: flex; flex-direction: column; gap: 9px; max-width: 720px; }
.bar-row { display: grid; grid-template-columns: 170px minmax(0, 1fr) 48px; gap: 12px; align-items: center; }
.bar-row .name { font-size: 13px; color: var(--text-secondary); }
.bar-row .name.top { color: var(--text-primary); font-weight: 600; }
.bar { height: 12px; background: var(--track); border-radius: 2px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); transition: width .45s ease; }
.bar.top i { background: var(--accent-strong); }
.bar-row .pct { text-align: right; font-size: 13px; font-weight: 600; }
.why { margin-top: 20px; padding-top: 12px; border-top: 1px solid #333; color: var(--text-secondary); display: flex; flex-direction: column; gap: 6px; max-width: 760px; }
.why b { color: var(--text-primary); font-weight: 600; }
.why .warn { color: var(--warn); }

.est { padding: 12px 14px; border-bottom: 1px solid #2c2c2c; }
.est-top { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.est-name { color: var(--text-secondary); font-size: 12px; font-weight: 500; }
.est-val { font-size: 20px; font-weight: 700; white-space: nowrap; }
.est-val .u { font-size: 12px; font-weight: 400; color: var(--text-muted); margin-left: 3px; }
.est-val.wide { font-size: 15px; font-weight: 600; color: var(--text-secondary); }
.est-bar { position: relative; height: 8px; background: var(--track); border-radius: 2px; margin-top: 8px; }
.est-bar .fill { position: absolute; top: 0; bottom: 0; background: var(--accent); border-radius: 2px; transition: left .45s ease, width .45s ease; }
.est-bar .fill.wide { background: repeating-linear-gradient(90deg, var(--conf-wide) 0 4px, transparent 4px 7px); }
.est-bar .tick { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--median); box-shadow: 0 0 0 1.5px var(--bg-pane); transition: left .45s ease; }
.est-foot { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 6px; font-size: 11px; color: var(--text-muted); }
.est-foot .left { display: flex; gap: 10px; align-items: center; white-space: nowrap; overflow: hidden; }
.spark { width: 96px; height: 20px; flex: none; }
.spark .band { fill: rgba(169, 134, 236, .22); }
.spark .line { fill: none; stroke: var(--accent); stroke-width: 1.5; }
.chip { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.chip i { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.chip--high { color: var(--conf-high); }
.chip--medium { color: var(--conf-medium); }
.chip--low { color: var(--conf-low); }
.chip--unconstrained, .chip--wide { color: var(--conf-wide); }
.chip--fixed { color: var(--accent-strong); }
.side-note { padding: 12px 14px 16px; color: var(--text-muted); font-size: 11.5px; display: flex; flex-direction: column; gap: 6px; }
.placeholder { padding: 40px 18px; text-align: center; color: var(--text-muted); }

@media (max-width: 1000px) {
  .guess-name, .guess-p { font-size: 34px; }
  .bar-row { grid-template-columns: 120px minmax(0, 1fr) 44px; }
}
@media (prefers-reduced-motion: reduce) {
  .bar i, .est-bar .fill, .est-bar .tick { transition: none; }
}
```

- [ ] **Step 6: Build and check it live**

Run: `cd web && npm test && ./node_modules/.bin/vite build`, then

```bash
P=/private/tmp/claude-501/-Users-brodyk-work-drone-predictor/5fb2ff8b-d40a-4f1f-83df-78935023fbc3/scratchpad/proto
CDP_PROFILE=$P/cdp-profile node $P/cdp_shots.mjs "http://localhost:5173/" $P/shots/task8 1440x900 5000 'button.chip-btn'
CDP_PROFILE=$P/cdp-profile node $P/cdp_shots.mjs "http://localhost:5173/" $P/shots/task8-phone 390x844 5000 'button.chip-btn'
```
Expected (`task8-5000ms.png`): "Racing / FPV quad 54%" hero, four class bars, why-lines, warning line; side panel with 10 rows, rotor speed "11.8k or 7.85k rpm". Phone shot: no horizontal overflow, header buttons wrap. In a real browser, set *Prop size* to 5″ and play the Racing sample again: the guess should jump to ≥ 90%.

- [ ] **Step 7: Commit**

```bash
git add web/src
git commit -m "Show the live class guess and interval estimates"
```

---

### Task 9: Docs, full verification, push

**Files:**
- Modify: `web/README.md`, `README.md:51-53,75`, `PLAN.md`

**Interfaces:** none.

- [ ] **Step 1: `web/README.md`** — replace the intro paragraph with:

```markdown
A Vite + React frontend for Augur's live identification. Drop a recording (WAV, MP3
or OGG — decoded in the browser), play one of the 8 s simulated samples, or record
from the mic. Every 600 ms the latest 3 s are sent to `POST /identify`; the page
shows the best-guess drone class with its probability and the physical estimates as
90% intervals that update live. The guess names a *class*, not a model, and can be
fooled by other tonal sounds (fans, mains hum) — see `docs/design/2026-10-14-live-identify.md`.
```

and in its Run section replace `./.venv/bin/augur serve` with `./.venv/bin/augur serve   # or: ./.venv/bin/uvicorn augur.api:app --reload --reload-dir src`, remove the `make_demo_sample.py` sentence, and add `npm test   # node:test for src/lib` under `npm run dev`.

- [ ] **Step 2: `README.md`** — replace lines 51-53 with:

```markdown
A web UI (Vite + React) lives in [`web/`](web/): drop a recording, pick a sample or
record from the mic, and watch the drone-class guess and the interval estimates
update live. Run `augur serve`, then `npm install && npm run dev` in `web/`.
```

and change line 75's endpoint list to ``CLI · HTTP API (`/predict`, `/identify`, `/reference`, `/health`) · property-based round-trip suite``.

- [ ] **Step 3: `PLAN.md`** — set `_Last updated: 2026-10-14_`; replace the **Web UI** bullet with a **Live identification** bullet:

```markdown
- **Live identification** (`fusion/identify.py`, `POST /identify`, `web/`): hover-physics
  evidence per (class, blade count) from the `classes.yaml` mass/prop bands, mixed into
  class-weighted estimates; asserted prop size / motor count add their likelihood. On 48
  simulated drones it is right 54% of the time (chance 25%) with mean confidence 50% —
  calibrated, but military is nearly unidentifiable from audio (its envelope overlaps
  every class). The web UI is a single live screen (file, sample or mic → class guess +
  interval estimates with sparklines). Frame diagonal and disk loading now carry
  motor-count uncertainty instead of using the modal count.
```

add to **Next steps**:

```markdown
4. A drone-vs-not detector trained on real clips (DroneAudioDataset): physics alone can't
   tell a drone from a fan or mains hum, so the live mic can false-alarm.
5. `/predict` still assumes 2 blades when blade count isn't given; reuse identify's
   blade marginalization there.
```

and update the test count in "Where this stands" to the number from Step 4.

- [ ] **Step 4: Full verification**

Run:
```bash
./.venv/bin/pytest tests/ -q
./.venv/bin/ruff check src/ tests/
./.venv/bin/augur predict --audio web/public/templates/racing.wav --verbal data/demo/spec.json | head -5
(cd web && npm test && ./node_modules/.bin/vite build)
```
Expected: 164 passed; ruff clean; CLI prints the report as before; 9 node tests pass; build succeeds.

- [ ] **Step 5: Commit and push**

```bash
git add README.md PLAN.md web/README.md
git commit -m "Document live identification"
git push -u origin feature/live-identify
```
Expected: branch pushed. Ask the user whether to open a PR into `main`.
