"""Controlled reference-based mixture construction and scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import (
    amplitude_ratio,
    background_suppression,
    clean_snr_gain,
    normalized_cross_correlation,
    rms,
)
from .schemas import DEFAULT_SAMPLING_RATE_HZ, require_same_shape, validate_waveform


def inject_at_target_snr(
    continuous_noise: np.ndarray,
    event: np.ndarray,
    *,
    onset_s: float,
    target_snr_db: float,
    signal_duration_s: float = 10.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale noise and inject an event without an onset taper."""
    noise = validate_waveform(continuous_noise, name="continuous_noise", sampling_rate_hz=sampling_rate_hz)
    event_array = validate_waveform(event, name="event", sampling_rate_hz=sampling_rate_hz)
    onset = int(round(onset_s * sampling_rate_hz))
    if onset < 0 or onset + len(event_array) > len(noise):
        raise ValueError("event injection exceeds the continuous-noise window")
    signal_n = int(round(signal_duration_s * sampling_rate_hz))
    event_rms = rms(event_array[:signal_n])
    noise_rms = rms(noise[onset : onset + signal_n])
    scale = event_rms / ((10.0 ** (target_snr_db / 20.0)) * (noise_rms + 1e-12))
    clean = np.zeros_like(noise)
    clean[onset : onset + len(event_array)] = event_array
    mixture = clean + noise * scale
    return mixture.astype(np.float32), clean.astype(np.float32), float(scale)


def score_controlled_case(
    output: np.ndarray,
    noisy_input: np.ndarray,
    clean: np.ndarray,
    *,
    hidden_onset_s: float,
    event_crop_start_s: float,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> dict[str, Any]:
    """Return the frozen controlled-mixture report-card metrics for one case."""
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    x = validate_waveform(noisy_input, name="noisy_input", sampling_rate_hz=sampling_rate_hz)
    c = validate_waveform(clean, name="clean", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, x, c)
    crop_start = int(round(event_crop_start_s * sampling_rate_hz))
    crop_stop = crop_start + int(round(35.0 * sampling_rate_hz))
    if crop_start < 0 or crop_stop > len(y):
        raise ValueError("event-centered score crop exceeds the continuous waveform")
    y_crop, x_crop, c_crop = y[crop_start:crop_stop], x[crop_start:crop_stop], c[crop_start:crop_stop]
    ncc = normalized_cross_correlation(y_crop, c_crop, sampling_rate_hz=sampling_rate_hz)
    return {
        "clean_snr_gain_db": clean_snr_gain(y_crop, x_crop, c_crop, sampling_rate_hz=sampling_rate_hz),
        "amplitude_ratio_clean": amplitude_ratio(y_crop, c_crop, sampling_rate_hz=sampling_rate_hz),
        "waveform_correlation_z": ncc.correlation,
        "delay_s": ncc.delay_s,
        "background_suppression_db": background_suppression(
            y,
            x,
            hidden_onset_s=hidden_onset_s,
            sampling_rate_hz=sampling_rate_hz,
        ),
    }
