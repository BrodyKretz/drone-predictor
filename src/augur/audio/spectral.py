"""Spectral front-end: load audio, compute spectrogram, average spectrum.

Uses scipy.signal so the core pipeline needs no heavy audio dependency. librosa
is an optional extra for richer order-tracking later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float64 in [-1, 1]. Returns (samples, sample_rate)."""
    sr, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    # Normalise integer PCM to [-1, 1].
    if np.issubdtype(np.asarray(data).dtype, np.floating) and np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    return data, int(sr)


def spectrogram(samples: np.ndarray, sample_rate: int, nperseg: int = 8192, overlap: float = 0.5):
    """Magnitude spectrogram. Returns (freqs, times, magnitude[freq, time])."""
    noverlap = int(nperseg * overlap)
    freqs, times, zxx = sps.stft(samples, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, np.abs(zxx)


def average_spectrum(samples: np.ndarray, sample_rate: int, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged magnitude spectrum in dB. Returns (freqs, spectrum_db)."""
    freqs, _, mag = spectrogram(samples, sample_rate, nperseg=nperseg)
    avg = mag.mean(axis=1)
    return freqs, 20.0 * np.log10(avg + 1e-12)


def dominant_freq_track(samples: np.ndarray, sample_rate: int, fmin: float = 50.0,
                        fmax: float = 5000.0, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame dominant frequency within [fmin, fmax]. Returns (times, freq_track).

    Used for hover/maneuver classification: a steady track means steady RPM.
    """
    freqs, times, mag = spectrogram(samples, sample_rate, nperseg=nperseg)
    band = (freqs >= fmin) & (freqs <= fmax)
    fband = freqs[band]
    track = np.array([fband[np.argmax(mag[band, j])] for j in range(mag.shape[1])])
    return times, track
