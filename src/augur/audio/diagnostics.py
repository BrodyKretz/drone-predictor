"""Display-side acoustics for the UI: a peak-preserving downsampled spectrum and
the harmonic comb. Nothing here changes an estimate."""

from __future__ import annotations

import numpy as np

from augur.audio import rpm, spectral
from augur.types import AcousticsDiagnostics, FlightStateEvidence, Harmonic

SPECTRUM_DISPLAY_BINS = 256
SPECTRUM_DB_FLOOR = -80.0
# Peak-relative dB would stretch a spectrum peaking this near digital silence to full scale.
SILENT_PEAK_DB = spectral.SILENCE_DB + 20.0


def max_pool_spectrum(freqs: np.ndarray, spec_db: np.ndarray, f_max_hz: float,
                      bins: int = SPECTRUM_DISPLAY_BINS) -> tuple[list[float], list[float]]:
    """Crop to [0, f_max_hz] and keep each group's loudest bin at its true
    frequency, so harmonic peaks survive downsampling. dB is relative to the
    peak (max 0) and floored at SPECTRUM_DB_FLOOR; silence, or a spectrum with a
    non-finite peak, is all floor."""
    if freqs.size != spec_db.size:
        raise ValueError(f"freqs and spec_db differ in length: {freqs.size} vs {spec_db.size}")
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    window = freqs <= f_max_hz
    if not window.any():
        raise ValueError(f"no spectrum bin at or below f_max_hz={f_max_hz}")
    f, d = freqs[window], spec_db[window]
    if f.size > bins:
        groups = np.array_split(np.arange(f.size), bins)
        idx = np.array([g[np.argmax(d[g])] for g in groups])
        f, d = f[idx], d[idx]
    peak = d.max()
    if np.isfinite(peak) and peak > SILENT_PEAK_DB:
        d = np.maximum(d - peak, SPECTRUM_DB_FLOOR)
    else:
        d = np.full(f.size, SPECTRUM_DB_FLOOR)
    return [float(x) for x in f], [float(x) for x in d]


def build_acoustics(freqs: np.ndarray, spec_db: np.ndarray, fundamental_hz: float,
                    peaks: np.ndarray, comb_confidence: float,
                    evidence: FlightStateEvidence) -> AcousticsDiagnostics:
    """fundamental_hz <= 0 means no tone was found: no harmonics, and the spectrum
    covers the whole detection band. With a tone it covers the comb plus one
    harmonic of headroom."""
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
