"""Template adapter for a user-supplied three-component denoiser."""

from __future__ import annotations

import numpy as np


def denoise(waveform: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Return a denoised array with unchanged shape, alignment, and amplitude units."""
    x = np.asarray(waveform)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("waveform must have shape (n_samples, 3) in Z,N,E order")
    if not np.all(np.isfinite(x)):
        raise ValueError("waveform contains non-finite values")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    # Replace this identity operation with the method under evaluation.
    output = x.copy()

    if output.shape != x.shape or not np.all(np.isfinite(output)):
        raise ValueError("adapter output violates the public waveform contract")
    return np.asarray(output, dtype=np.float32)
