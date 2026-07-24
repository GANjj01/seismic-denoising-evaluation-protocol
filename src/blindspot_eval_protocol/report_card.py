"""Deterministic aggregation of released per-case metric rows."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


CONTROLLED_METRICS = (
    "clean_snr_gain_db",
    "amplitude_ratio_clean",
    "waveform_correlation_z",
    "delay_s",
    "background_suppression_db",
)


def _finite(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def aggregate_controlled_report_card(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate by method using frozen mean/median conventions."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(dict(row))
    output = []
    for method in sorted(grouped):
        items = grouped[method]
        clean = _finite(items, "clean_snr_gain_db")
        amplitude = _finite(items, "amplitude_ratio_clean")
        corr = _finite(items, "waveform_correlation_z")
        delay = _finite(items, "delay_s")
        background = _finite(items, "background_suppression_db")
        output.append(
            {
                "method": method,
                "n_cases": len(items),
                "clean_snr_gain_db_mean": float(clean.mean()),
                "amplitude_ratio_clean_median": float(np.median(amplitude)),
                "waveform_correlation_z_mean": float(corr.mean()),
                "delay_s_mean": float(delay.mean()),
                "background_suppression_db_mean": float(background.mean()),
            }
        )
    return output


def aggregate_external_report_card(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(dict(row))
    output = []
    for method in sorted(grouped):
        items = grouped[method]
        output.append(
            {
                "method": method,
                "n_cases": len(items),
                "apparent_snr_gain_db_mean": float(_finite(items, "apparent_snr_gain_db").mean()),
                "amplitude_ratio_raw_mean": float(_finite(items, "amplitude_ratio_raw").mean()),
                "trigger_delay_s_mean": float(_finite(items, "trigger_delay_s").mean()),
            }
        )
    return output
