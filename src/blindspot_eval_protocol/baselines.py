"""Protocol reference baselines and diagnostics."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .preprocessing import detrend_and_bandpass
from .schemas import DEFAULT_SAMPLING_RATE_HZ, validate_waveform


def identity(waveform: np.ndarray, sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ) -> np.ndarray:
    return validate_waveform(waveform, sampling_rate_hz=sampling_rate_hz).astype(np.float32, copy=True)


def bandpass(waveform: np.ndarray, sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ) -> np.ndarray:
    return detrend_and_bandpass(waveform, sampling_rate_hz=sampling_rate_hz)


def adversarial_scale(waveform: np.ndarray, factor: float = 0.01) -> np.ndarray:
    return (validate_waveform(waveform) * float(factor)).astype(np.float32)


def adversarial_shrink(waveform: np.ndarray, threshold_multiplier: float = 3.0) -> np.ndarray:
    x = validate_waveform(waveform)
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    threshold = threshold_multiplier * mad
    return (np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)).astype(np.float32)


def _spectral_wiener(x: np.ndarray, noise_psd: np.ndarray, frequencies: np.ndarray, fs: float) -> np.ndarray:
    output = np.zeros_like(x, dtype=np.float32)
    for component in range(3):
        f_total, total_psd = signal.welch(x[:, component], fs=fs, nperseg=min(512, len(x)))
        estimate = np.interp(f_total, frequencies, noise_psd[:, component])
        gain = np.maximum(1.0 - estimate / (total_psd + 1e-12), 0.0)
        fft_frequencies = np.fft.rfftfreq(len(x), 1.0 / fs)
        output[:, component] = np.fft.irfft(
            np.fft.rfft(x[:, component]) * np.interp(fft_frequencies, f_total, gain),
            n=len(x),
        ).astype(np.float32)
    return output


def wiener_blind(
    waveform: np.ndarray,
    *,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    frame_s: float = 4.0,
    hop_s: float = 2.0,
    low_energy_fraction: float = 0.30,
) -> np.ndarray:
    """Deployable blind Wiener baseline using low-energy frames from the input."""
    x = validate_waveform(waveform, sampling_rate_hz=sampling_rate_hz)
    frame = int(round(frame_s * sampling_rate_hz))
    hop = int(round(hop_s * sampling_rate_hz))
    starts = list(range(0, len(x) - frame + 1, hop))
    if len(starts) < 3:
        raise ValueError("waveform is too short for blind Wiener frame selection")
    frames = np.stack([x[start : start + frame] for start in starts])
    energy = np.mean(frames * frames, axis=(1, 2))
    count = max(3, int(math.ceil(low_energy_fraction * len(frames))))
    spectra = []
    for item in frames[np.argsort(energy)[:count]]:
        component_spectra = []
        for component in range(3):
            frequencies, psd = signal.welch(item[:, component], fs=sampling_rate_hz, nperseg=min(256, frame))
            component_spectra.append(psd)
        spectra.append(np.stack(component_spectra, axis=1))
    return _spectral_wiener(x, np.median(np.stack(spectra), axis=0), frequencies, sampling_rate_hz)


def idealized_wiener_upper_bound(
    waveform: np.ndarray,
    exact_noise_realization: np.ndarray,
    *,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Source-aware upper-bound diagnostic; this is not a deployable baseline."""
    x = validate_waveform(waveform, sampling_rate_hz=sampling_rate_hz)
    noise = validate_waveform(exact_noise_realization, sampling_rate_hz=sampling_rate_hz)
    if x.shape != noise.shape:
        raise ValueError("waveform and exact noise realization must have the same shape")
    spectra = []
    for component in range(3):
        frequencies, psd = signal.welch(noise[:, component], fs=sampling_rate_hz, nperseg=min(512, len(noise)))
        spectra.append(psd)
    return _spectral_wiener(x, np.stack(spectra, axis=1), frequencies, sampling_rate_hz)
