"""Minimal identity adapter implementing the public method contract."""

from __future__ import annotations

import numpy as np


def denoise(waveform: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    if waveform.ndim != 2 or waveform.shape[1] != 3:
        raise ValueError("expected (n_samples, 3) in Z,N,E order")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    return np.asarray(waveform, dtype=np.float32).copy()
