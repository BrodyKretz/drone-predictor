"""Flight-state classification from audio.

Hover is detected from RPM steadiness. Naively tracking the per-frame argmax over
a wide band is fragile: it flips between closely-spaced per-motor peaks and noise
bins, faking a "climb". Instead we lock onto the detected blade-pass fundamental
and track the peak only in a narrow band around it, using a robust (IQR-based)
spread measure. This matters because sum(T)=m*g is valid ONLY in hover.
"""

from __future__ import annotations

import numpy as np

from augur.audio import rpm, spectral
from augur.types import FlightState


def classify_state(samples: np.ndarray, sample_rate: int,
                   steady_threshold: float = 0.04, nperseg: int = 8192) -> FlightState:
    """Classify flight state from the stability of the blade-pass fundamental.

    steady_threshold is the max robust coefficient of variation (IQR/median) of
    the fundamental track for the clip to count as steady hover. Per-motor control
    spread (a few percent) stays under this; a throttle ramp/climb exceeds it.
    """
    freqs, spec_db = spectral.average_spectrum(samples, sample_rate, nperseg=nperseg)
    f0, conf, _ = rpm.detect_fundamental(freqs, spec_db)
    if f0 <= 0 or conf < 0.3:
        return FlightState.unknown

    fmin, fmax = 0.85 * f0, 1.15 * f0
    times, track = spectral.dominant_freq_track(samples, sample_rate, fmin=fmin,
                                                 fmax=fmax, nperseg=nperseg)
    if track.size < 3:
        return FlightState.unknown

    median_f = float(np.median(track))
    if median_f <= 0:
        return FlightState.unknown
    q25, q75 = np.percentile(track, [25, 75])
    robust_cov = float((q75 - q25) / median_f)

    if robust_cov <= steady_threshold:
        return FlightState.hover

    # A sustained monotonic drift in the fundamental is a throttle ramp (climb).
    slope = np.polyfit(np.arange(track.size), track, 1)[0]
    rel_slope = abs(slope) * track.size / median_f
    if rel_slope > steady_threshold:
        return FlightState.climb
    return FlightState.mixed
