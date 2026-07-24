from __future__ import annotations

import math

import numpy as np
import pytest

from blindspot_eval_protocol.metrics import (
    amplitude_ratio,
    background_suppression,
    clean_snr_gain,
    covariance_shape_distance,
    normalized_cross_correlation,
)
from blindspot_eval_protocol.schemas import validate_waveform


def synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    clean = np.zeros((3500, 3), dtype=np.float64)
    time = np.arange(1000) / 100.0
    clean[1000:2000] = np.stack(
        [
            np.sin(2 * np.pi * 3 * time),
            0.5 * np.sin(2 * np.pi * 4 * time),
            0.25 * np.cos(2 * np.pi * 5 * time),
        ],
        axis=1,
    )
    return clean, clean + rng.normal(0.0, 0.2, clean.shape)


def test_identity_metrics_are_zero() -> None:
    clean, noisy = synthetic_case()
    assert clean_snr_gain(noisy, noisy, clean) == pytest.approx(0.0, abs=1e-12)
    long = np.random.default_rng(1).normal(size=(9000, 3))
    assert background_suppression(long, long, hidden_onset_s=30.0) == pytest.approx(0.0, abs=1e-10)


def test_amplitude_scaling_and_known_lag() -> None:
    clean, _ = synthetic_case()
    assert amplitude_ratio(0.5 * clean, clean) == pytest.approx(0.5, abs=1e-10)
    shifted = np.zeros_like(clean)
    shifted[7:, 0] = clean[:-7, 0]
    result = normalized_cross_correlation(shifted, clean)
    assert result.correlation > 0.99
    assert abs(result.delay_s) == pytest.approx(0.07)


def test_recovery_probe_is_monotone() -> None:
    clean, noisy = synthetic_case()
    gains = [
        clean_snr_gain(clean + (1.0 - fraction) * (noisy - clean), noisy, clean)
        for fraction in (0.0, 0.25, 0.5, 0.75)
    ]
    assert gains == sorted(gains)
    assert gains[0] == pytest.approx(0.0, abs=1e-12)


def test_covariance_distance_direction() -> None:
    clean, noisy = synthetic_case()
    clean_window = clean[1000:2000]
    assert covariance_shape_distance(clean_window, clean_window) == pytest.approx(0.0, abs=1e-12)
    assert covariance_shape_distance(noisy[1000:2000], clean_window) > 0.0


def test_invalid_waveforms_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_waveform(np.zeros((10, 2), dtype=float))
    invalid = np.zeros((10, 3), dtype=float)
    invalid[0, 0] = math.nan
    with pytest.raises(ValueError):
        validate_waveform(invalid)
