"""Frozen waveform preprocessing for the public evaluation protocol."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .schemas import DEFAULT_SAMPLING_RATE_HZ, validate_waveform


def detrend_and_bandpass(
    waveform: np.ndarray,
    *,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    low_hz: float = 1.0,
    high_hz: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Demean, linearly detrend, and zero-phase Butterworth filter each component."""
    x = validate_waveform(waveform, sampling_rate_hz=sampling_rate_hz)
    if not 0.0 < low_hz < high_hz < sampling_rate_hz / 2.0:
        raise ValueError("filter corners must satisfy 0 < low < high < Nyquist")
    centered = signal.detrend(signal.detrend(x, axis=0, type="constant"), axis=0, type="linear")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, centered, axis=0).astype(np.float32)


def event_centered_reference(
    filtered_event: np.ndarray,
    *,
    p_index: int = 1000,
    event_duration_s: float = 20.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    taper_onset: bool = False,
    taper_duration_s: float = 0.5,
) -> np.ndarray:
    """Construct the evaluator-held target; the primary definition is untapered."""
    x = validate_waveform(filtered_event, sampling_rate_hz=sampling_rate_hz).copy()
    if p_index < 0 or p_index >= len(x):
        raise ValueError("p_index is outside the waveform")
    end = min(len(x), p_index + int(round(event_duration_s * sampling_rate_hz)))
    x[:p_index] = 0.0
    x[end:] = 0.0
    if taper_onset:
        n = min(end - p_index, int(round(taper_duration_s * sampling_rate_hz)))
        if n > 1:
            x[p_index : p_index + n] *= np.linspace(0.0, 1.0, n)[:, None]
    return x.astype(np.float32)
