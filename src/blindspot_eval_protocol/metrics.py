"""Current metric definitions used by the frozen report cards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .schemas import DEFAULT_SAMPLING_RATE_HZ, require_same_shape, validate_waveform


EPS = 1e-12


def rms(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _window(p_index: int, duration_s: float, sampling_rate_hz: float) -> slice:
    return slice(p_index, p_index + int(round(duration_s * sampling_rate_hz)))


def absolute_clean_snr(
    output: np.ndarray,
    clean: np.ndarray,
    *,
    p_index: int = 1000,
    duration_s: float = 10.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> float:
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    c = validate_waveform(clean, name="clean", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, c)
    window = _window(p_index, duration_s, sampling_rate_hz)
    if window.stop > len(y):
        raise ValueError("clean-SNR window exceeds waveform length")
    return float(20.0 * np.log10(rms(c[window]) / (rms(y[window] - c[window]) + EPS)))


def clean_snr_gain(
    output: np.ndarray,
    noisy_input: np.ndarray,
    clean: np.ndarray,
    **kwargs: float,
) -> float:
    """Three-component clean-SNR(output) minus clean-SNR(noisy input)."""
    return absolute_clean_snr(output, clean, **kwargs) - absolute_clean_snr(noisy_input, clean, **kwargs)


def amplitude_ratio(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    p_index: int = 1000,
    duration_s: float = 2.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    quantile: float = 0.95,
) -> float:
    """Z-component post-P amplitude ratio based on absolute-amplitude quantiles."""
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    r = validate_waveform(reference, name="reference", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, r)
    window = _window(p_index, duration_s, sampling_rate_hz)
    if window.stop > len(y):
        raise ValueError("amplitude window exceeds waveform length")
    numerator = np.quantile(np.abs(y[window, 0]), quantile)
    denominator = np.quantile(np.abs(r[window, 0]), quantile)
    return float(numerator / (denominator + EPS))


@dataclass(frozen=True)
class CorrelationResult:
    correlation: float
    delay_s: float


def normalized_cross_correlation(
    output: np.ndarray,
    clean: np.ndarray,
    *,
    component: int = 0,
    p_index: int = 1000,
    duration_s: float = 10.0,
    max_lag_s: float = 1.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> CorrelationResult:
    """Best mean-removed NCC and lag within the frozen plus/minus-one-second search."""
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    c = validate_waveform(clean, name="clean", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, c)
    if component not in (0, 1, 2):
        raise ValueError("component must be 0 (Z), 1 (N), or 2 (E)")
    window = _window(p_index, duration_s, sampling_rate_hz)
    if window.stop > len(y):
        raise ValueError("correlation window exceeds waveform length")
    a = y[window, component] - np.mean(y[window, component])
    b = c[window, component] - np.mean(c[window, component])
    full = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    keep = np.abs(lags) <= int(round(max_lag_s * sampling_rate_hz))
    selected = np.flatnonzero(keep)
    best = selected[int(np.argmax(full[keep]))]
    lag_samples = int(lags[best])
    denominator = np.linalg.norm(a) * np.linalg.norm(b) + EPS
    return CorrelationResult(float(full[best] / denominator), float(lag_samples / sampling_rate_hz))


def background_suppression(
    output: np.ndarray,
    noisy_input: np.ndarray,
    *,
    hidden_onset_s: float,
    event_duration_s: float = 20.0,
    exclusion_margin_s: float = 2.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> float:
    """Three-component RMS suppression outside the frozen event-exclusion region."""
    y = validate_waveform(output, name="output", sampling_rate_hz=sampling_rate_hz)
    x = validate_waveform(noisy_input, name="noisy_input", sampling_rate_hz=sampling_rate_hz)
    require_same_shape(y, x)
    mask = np.ones(len(y), dtype=bool)
    start = max(0, int(round((hidden_onset_s - exclusion_margin_s) * sampling_rate_hz)))
    stop = min(len(y), int(round((hidden_onset_s + event_duration_s + exclusion_margin_s) * sampling_rate_hz)))
    mask[start:stop] = False
    if not np.any(mask):
        raise ValueError("background mask is empty")
    return float(20.0 * np.log10(rms(x[mask]) / (rms(y[mask]) + EPS)))


def apparent_snr(
    waveform: np.ndarray,
    *,
    p_index: int = 1000,
    signal_duration_s: float = 10.0,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> float:
    x = validate_waveform(waveform, sampling_rate_hz=sampling_rate_hz)
    stop = p_index + int(round(signal_duration_s * sampling_rate_hz))
    if p_index <= 0 or stop > len(x):
        raise ValueError("apparent-SNR windows exceed waveform length")
    return float(20.0 * np.log10(rms(x[p_index:stop, 0]) / (rms(x[:p_index, 0]) + EPS)))


def apparent_snr_gain(output: np.ndarray, raw_input: np.ndarray, **kwargs: float) -> float:
    return apparent_snr(output, **kwargs) - apparent_snr(raw_input, **kwargs)


def covariance_shape(window: np.ndarray) -> np.ndarray:
    x = validate_waveform(window)
    demeaned = x - np.mean(x, axis=0, keepdims=True)
    covariance = demeaned.T @ demeaned / len(demeaned)
    return covariance / max(float(np.trace(covariance)), EPS)


def covariance_shape_distance(output_window: np.ndarray, clean_window: np.ndarray) -> float:
    require_same_shape(output_window, clean_window)
    return float(np.linalg.norm(covariance_shape(output_window) - covariance_shape(clean_window), ord="fro"))


@dataclass(frozen=True)
class PolarizationAttributes:
    rectilinearity: float
    planarity: float
    principal_axis: np.ndarray


def polarization_attributes(window: np.ndarray) -> PolarizationAttributes:
    x = validate_waveform(window)
    demeaned = x - np.mean(x, axis=0, keepdims=True)
    covariance = demeaned.T @ demeaned / len(demeaned)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    l1, l2, l3 = np.maximum(values[order], 0.0)
    axis = vectors[:, order[0]]
    rectilinearity = 1.0 - (l2 + l3) / (2.0 * l1 + EPS)
    planarity = 1.0 - 2.0 * l3 / (l1 + l2 + EPS)
    return PolarizationAttributes(float(rectilinearity), float(planarity), axis)


def principal_axis_angle_deg(output_window: np.ndarray, clean_window: np.ndarray) -> float:
    a = polarization_attributes(output_window).principal_axis
    b = polarization_attributes(clean_window).principal_axis
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(math.degrees(math.acos(cosine)))
