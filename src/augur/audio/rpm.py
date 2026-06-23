"""RPM recovery from the acoustic spectrum.

Primary, high-confidence path: find the blade pass frequency (BPF) as the
fundamental of the strongest harmonic comb, then RPM = BPF * 60 / num_blades.

Per-motor spread: motors run at slightly offset RPM for control authority, which
splits the fundamental into a tight cluster of peaks; the cluster width gives the
spread fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from augur.physics import core
from augur.types import Distribution


@dataclass
class RPMEstimate:
    rpm: Distribution
    fundamental_hz: float
    blade_count_used: int
    rpm_spread_fraction: float
    confidence: float  # 0..1, from comb score
    peak_freqs_hz: np.ndarray


def _find_peaks_db(freqs: np.ndarray, spec_db: np.ndarray, fmin: float, fmax: float,
                   prominence: float = 6.0) -> np.ndarray:
    band = (freqs >= fmin) & (freqs <= fmax)
    idx, _ = find_peaks(spec_db[band], prominence=prominence)
    return freqs[band][idx]


def _comb_score(fundamental: float, peaks: np.ndarray, n_harmonics: int, tol_hz: float) -> float:
    """Fraction of the first n_harmonics that have a peak within tol_hz."""
    if fundamental <= 0:
        return 0.0
    hits = 0
    for h in range(1, n_harmonics + 1):
        target = fundamental * h
        if peaks.size and np.min(np.abs(peaks - target)) <= tol_hz:
            hits += 1
    return hits / n_harmonics


def detect_fundamental(freqs: np.ndarray, spec_db: np.ndarray, fmin: float = 50.0,
                       fmax: float = 5000.0, n_harmonics: int = 6) -> tuple[float, float, np.ndarray]:
    """Return (fundamental_hz, confidence, peak_freqs).

    Scores each detected peak as a candidate fundamental by how well an integer
    harmonic comb built on it explains the other peaks. The lowest-frequency
    member of the best comb is the BPF (guards against locking onto a harmonic).
    """
    peaks = _find_peaks_db(freqs, spec_db, fmin, fmax)
    if peaks.size == 0:
        return 0.0, 0.0, peaks

    bin_hz = float(freqs[1] - freqs[0])
    tol_hz = max(3.0 * bin_hz, 5.0)

    best_f, best_score = 0.0, -1.0
    # Candidate fundamentals: each peak, and each peak divided by small integers
    # (in case the true fundamental is weaker than a harmonic).
    candidates = set()
    for p in peaks:
        for div in (1, 2, 3):
            f0 = p / div
            if fmin <= f0 <= fmax:
                candidates.add(round(f0, 2))

    for f0 in sorted(candidates):
        score = _comb_score(f0, peaks, n_harmonics, tol_hz)
        # Prefer lower fundamentals at equal score (avoid harmonic lock).
        if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and f0 < best_f):
            best_f, best_score = f0, score

    return best_f, best_score, peaks


def estimate_rpm(freqs: np.ndarray, spec_db: np.ndarray, num_blades: int = 2,
                 fmin: float = 50.0, fmax: float = 5000.0, n_samples: int = 4000,
                 rng: np.random.Generator | None = None) -> RPMEstimate:
    """Estimate RPM as a Distribution.

    The fundamental is located to within a frequency bin; that resolution plus
    the comb confidence sets the posterior width. High confidence + fine bins ->
    a tight RPM distribution (the spec targets <2% error here).
    """
    rng = rng or np.random.default_rng()
    f0, conf, peaks = detect_fundamental(freqs, spec_db, fmin, fmax, n_harmonics=6)

    if f0 <= 0:
        # No tonal content found: return an uninformative wide distribution.
        wide = Distribution.from_uniform(1000.0, 30000.0, n_samples, unit="rpm")
        return RPMEstimate(rpm=wide, fundamental_hz=0.0, blade_count_used=num_blades,
                           rpm_spread_fraction=0.0, confidence=0.0, peak_freqs_hz=peaks)

    bin_hz = float(freqs[1] - freqs[0])
    # Frequency uncertainty: ~half a bin, widened when comb confidence is low.
    sigma_hz = 0.5 * bin_hz / max(conf, 0.2)
    f0_samples = rng.normal(f0, sigma_hz, size=n_samples)
    rpm_samples = core.rpm_from_bpf(f0_samples, num_blades)

    spread = estimate_per_motor_spread(peaks, f0)

    return RPMEstimate(
        rpm=Distribution(rpm_samples, unit="rpm"),
        fundamental_hz=f0,
        blade_count_used=num_blades,
        rpm_spread_fraction=spread,
        confidence=conf,
        peak_freqs_hz=peaks,
    )


def estimate_per_motor_spread(peaks: np.ndarray, fundamental: float, window_frac: float = 0.08) -> float:
    """Spread fraction from the cluster of peaks around the fundamental.

    Returns (max - min) / fundamental over peaks within +/- window_frac of f0."""
    if fundamental <= 0 or peaks.size == 0:
        return 0.0
    lo, hi = fundamental * (1 - window_frac), fundamental * (1 + window_frac)
    cluster = peaks[(peaks >= lo) & (peaks <= hi)]
    if cluster.size < 2:
        return 0.0
    return float((cluster.max() - cluster.min()) / fundamental)
