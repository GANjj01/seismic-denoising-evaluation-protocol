"""Array and manifest validation shared by the evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COMPONENT_ORDER = ("Z", "N", "E")
DEFAULT_SAMPLING_RATE_HZ = 100.0


@dataclass(frozen=True)
class WaveformContract:
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ
    component_order: tuple[str, str, str] = COMPONENT_ORDER
    dtype: str = "float32 or float64"
    shape: str = "(n_samples, 3)"
    finite_values_required: bool = True
    amplitude_scale: str = "physical input scale; no hidden normalization"


def validate_waveform(
    waveform: Any,
    *,
    name: str = "waveform",
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Return a floating 3C array after enforcing the public adapter contract."""
    if not np.isfinite(float(sampling_rate_hz)) or float(sampling_rate_hz) <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    array = np.asarray(waveform)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_samples, 3) in Z,N,E order")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if array.dtype.kind not in "fc":
        raise TypeError(f"{name} must use a floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.asarray(array, dtype=np.float64)


def require_same_shape(*arrays: np.ndarray) -> None:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"waveform shapes differ: {sorted(shapes)}")
