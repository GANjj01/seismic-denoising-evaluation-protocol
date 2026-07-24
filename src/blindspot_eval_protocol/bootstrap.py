"""Station-level bootstrap implementations used by released contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_stations: int
    n_matched_cases: int
    replicates: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return bool(self.ci_low > 0.0 or self.ci_high < 0.0)


def percentile_interval(samples: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    low, high = np.quantile(np.asarray(samples, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def paired_station_bootstrap(
    station_differences: Iterable[float],
    *,
    n_matched_cases: int,
    replicates: int = 20000,
    seed: int = 20260611,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    values = np.asarray(list(station_differences), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite station differences")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, values.size), replace=True).mean(axis=1)
    low, high = percentile_interval(samples, confidence_level)
    return BootstrapResult(float(values.mean()), low, high, int(values.size), int(n_matched_cases), replicates, seed)


def independent_station_bootstrap(
    group_a_station_means: Iterable[float],
    group_b_station_means: Iterable[float],
    *,
    replicates: int = 20000,
    seed: int = 20260611,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    a = np.asarray(list(group_a_station_means), dtype=float)
    b = np.asarray(list(group_b_station_means), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        raise ValueError("both station groups require finite values")
    rng = np.random.default_rng(seed)
    a_draws = rng.choice(a, size=(replicates, a.size), replace=True).mean(axis=1)
    b_draws = rng.choice(b, size=(replicates, b.size), replace=True).mean(axis=1)
    low, high = percentile_interval(a_draws - b_draws, confidence_level)
    return BootstrapResult(float(a.mean() - b.mean()), low, high, min(len(a), len(b)), len(a) + len(b), replicates, seed)


def reference_level_bootstrap(
    reference_values: Iterable[float],
    *,
    replicates: int = 1000,
    seed: int = 20260611,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    values = np.asarray(list(reference_values), dtype=float)
    values = values[np.isfinite(values)]
    return paired_station_bootstrap(
        values,
        n_matched_cases=len(values),
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )
