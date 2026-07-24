"""External real-event scoring without a clean or pseudo-clean reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import amplitude_ratio, apparent_snr_gain
from .schemas import DEFAULT_SAMPLING_RATE_HZ, require_same_shape, validate_waveform


def score_real_event(
    output: np.ndarray,
    raw_input: np.ndarray,
    *,
    p_index: int = 1000,
    trigger_delay_s: float | None = None,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> dict[str, Any]:
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    x = validate_waveform(raw_input, name="raw_input", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, x)
    return {
        "apparent_snr_gain_db": apparent_snr_gain(y, x, p_index=p_index, sampling_rate_hz=sampling_rate_hz),
        "amplitude_ratio_raw": amplitude_ratio(y, x, p_index=p_index, sampling_rate_hz=sampling_rate_hz),
        "trigger_delay_s": trigger_delay_s,
    }
