"""Unambiguous controlled reference-based case construction and scoring."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ControlledCaseSpecification:
    """Sample-index definition of one frozen 90 s controlled case."""

    sampling_rate_hz: float
    hidden_onset_sample: int
    event_reference_onset_sample: int = 1000
    injection_duration_samples: int = 2000
    signal_rms_duration_samples: int = 1000
    scoring_pre_onset_samples: int = 1000
    scoring_duration_samples: int = 3500
    continuous_duration_samples: int = 9000

    @classmethod
    def from_seconds(
        cls,
        hidden_onset_s: float,
        *,
        sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    ) -> "ControlledCaseSpecification":
        """Use round-to-nearest sample, matching frozen hundredth-second onsets."""
        return cls(
            sampling_rate_hz=float(sampling_rate_hz),
            hidden_onset_sample=int(round(float(hidden_onset_s) * float(sampling_rate_hz))),
            event_reference_onset_sample=int(round(10.0 * sampling_rate_hz)),
            injection_duration_samples=int(round(20.0 * sampling_rate_hz)),
            signal_rms_duration_samples=int(round(10.0 * sampling_rate_hz)),
            scoring_pre_onset_samples=int(round(10.0 * sampling_rate_hz)),
            scoring_duration_samples=int(round(35.0 * sampling_rate_hz)),
            continuous_duration_samples=int(round(90.0 * sampling_rate_hz)),
        )

    @property
    def injection_slice(self) -> slice:
        return slice(
            self.hidden_onset_sample,
            self.hidden_onset_sample + self.injection_duration_samples,
        )

    @property
    def event_segment_slice(self) -> slice:
        return slice(
            self.event_reference_onset_sample,
            self.event_reference_onset_sample + self.injection_duration_samples,
        )

    @property
    def event_rms_slice(self) -> slice:
        return slice(0, self.signal_rms_duration_samples)

    @property
    def noise_rms_slice(self) -> slice:
        return slice(
            self.hidden_onset_sample,
            self.hidden_onset_sample + self.signal_rms_duration_samples,
        )

    @property
    def scoring_slice(self) -> slice:
        start = self.hidden_onset_sample - self.scoring_pre_onset_samples
        return slice(start, start + self.scoring_duration_samples)

    def validate_event_reference(self, event_reference_length: int) -> None:
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be finite and positive")
        if self.hidden_onset_sample < 0:
            raise ValueError("hidden onset cannot be negative")
        if self.event_segment_slice.stop > event_reference_length:
            raise ValueError("P-to-P+20 s event segment exceeds event reference")

    def validate_continuous_case(self, continuous_noise_length: int) -> None:
        if self.injection_slice.stop > continuous_noise_length:
            raise ValueError("event injection exceeds continuous noise")
        if self.noise_rms_slice.stop > continuous_noise_length:
            raise ValueError("noise RMS window exceeds continuous noise")
        if self.scoring_slice.start < 0 or self.scoring_slice.stop > continuous_noise_length:
            raise ValueError("35 s scoring crop exceeds continuous case")
        if continuous_noise_length != self.continuous_duration_samples:
            raise ValueError(
                f"continuous noise must contain exactly {self.continuous_duration_samples} samples"
            )

    def validate(self, event_reference_length: int, continuous_noise_length: int) -> None:
        self.validate_event_reference(event_reference_length)
        self.validate_continuous_case(continuous_noise_length)


def extract_event_segment(
    event_reference: np.ndarray,
    specification: ControlledCaseSpecification,
) -> np.ndarray:
    reference = validate_waveform(
        event_reference,
        name="event_reference",
        sampling_rate_hz=specification.sampling_rate_hz,
    )
    specification.validate_event_reference(len(reference))
    return reference[specification.event_segment_slice].copy()


def inject_event_reference_at_target_snr(
    continuous_noise: np.ndarray,
    event_reference: np.ndarray,
    *,
    specification: ControlledCaseSpecification,
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Inject P through P+20 s after scaling the entire 90 s noise trace.

    Event RMS uses P through P+10 s of the extracted event segment. Noise RMS
    uses the 10 s continuous-noise interval beginning at the hidden P onset.
    """
    noise = validate_waveform(
        continuous_noise,
        name="continuous_noise",
        sampling_rate_hz=specification.sampling_rate_hz,
    )
    reference = validate_waveform(
        event_reference,
        name="event_reference",
        sampling_rate_hz=specification.sampling_rate_hz,
    )
    specification.validate(len(reference), len(noise))
    event_segment = reference[specification.event_segment_slice]
    event_rms = rms(event_segment[specification.event_rms_slice])
    noise_rms = rms(noise[specification.noise_rms_slice])
    if event_rms <= 0.0:
        raise ValueError("event RMS is zero in the frozen P-to-P+10 s window")
    if noise_rms <= 0.0:
        raise ValueError("noise RMS is zero in the hidden-onset 10 s window")
    scale = event_rms / ((10.0 ** (float(target_snr_db) / 20.0)) * noise_rms)
    noise_reference = noise * scale
    clean_reference = np.zeros_like(noise_reference)
    clean_reference[specification.injection_slice] = event_segment
    mixture = clean_reference + noise_reference
    return (
        mixture.astype(np.float32),
        clean_reference.astype(np.float32),
        noise_reference.astype(np.float32),
        float(scale),
    )


def achieved_target_snr_db(
    noise_reference: np.ndarray,
    clean_reference: np.ndarray,
    specification: ControlledCaseSpecification,
) -> float:
    noise = validate_waveform(noise_reference, name="noise_reference")
    clean = validate_waveform(clean_reference, name="clean_reference")
    require_same_shape(noise, clean)
    event_window = slice(
        specification.hidden_onset_sample,
        specification.hidden_onset_sample + specification.signal_rms_duration_samples,
    )
    return float(20.0 * np.log10(rms(clean[event_window]) / rms(noise[event_window])))


def score_controlled_case(
    output: np.ndarray,
    noisy_input: np.ndarray,
    clean_reference: np.ndarray,
    *,
    specification: ControlledCaseSpecification,
) -> dict[str, Any]:
    """Return primary report-card metrics from an already reconstructed case."""
    y = validate_waveform(output, name="output", sampling_rate_hz=specification.sampling_rate_hz)
    x = validate_waveform(noisy_input, name="noisy_input", sampling_rate_hz=specification.sampling_rate_hz)
    c = validate_waveform(clean_reference, name="clean_reference", sampling_rate_hz=specification.sampling_rate_hz)
    require_same_shape(y, x, c)
    specification.validate_continuous_case(len(y))
    score_slice = specification.scoring_slice
    y_crop, x_crop, c_crop = y[score_slice], x[score_slice], c[score_slice]
    ncc = normalized_cross_correlation(
        y_crop,
        c_crop,
        sampling_rate_hz=specification.sampling_rate_hz,
    )
    return {
        "clean_snr_gain_db": clean_snr_gain(
            y_crop,
            x_crop,
            c_crop,
            sampling_rate_hz=specification.sampling_rate_hz,
        ),
        "amplitude_ratio_clean": amplitude_ratio(
            y_crop,
            c_crop,
            sampling_rate_hz=specification.sampling_rate_hz,
        ),
        "waveform_correlation_z": ncc.correlation,
        "delay_s": ncc.delay_s,
        "background_suppression_db": background_suppression(
            y,
            x,
            hidden_onset_s=specification.hidden_onset_sample / specification.sampling_rate_hz,
            sampling_rate_hz=specification.sampling_rate_hz,
        ),
    }
