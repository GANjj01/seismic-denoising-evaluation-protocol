"""Recompute the released no-taper E3 station-domain sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True

METRICS = ("output_vs_clean_snr", "amp_ratio_clean", "corr_z", "background_suppression_db")
COVARIATES = (
    "target_snr_db",
    "log10_noise_rms_injection_10s",
    "dominant_freq_hz",
    "spectral_slope_1_20",
    "log10_high_low_bandpower_ratio",
    "frame_rms_cv_4s",
)
STATION_COVARIATES = (
    "noise_rms_full_mean",
    "dominant_freq_hz_mean",
    "spectral_slope_1_20_mean",
    "frame_rms_cv_4s_mean",
)
SEED = 20260611
UNADJUSTED_REPLICATES = 20000
SENSITIVITY_REPLICATES = 1000
MATCHING_CALIPER = 2.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def stable_seed(method: str, metric: str, suffix: str) -> int:
    text = f"{method}|{metric}|{suffix}"
    return SEED + sum((index + 1) * ord(character) for index, character in enumerate(text))


def direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def zero_excluded(low: float, high: float) -> bool:
    return bool(low > 0.0 or high < 0.0)


def station_means(rows: list[dict[str, str]], method: str, metric: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            value = finite(row[metric])
            if math.isfinite(value):
                grouped[(row["group"], row["station_noise"])].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def unadjusted(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        for metric in METRICS:
            values = station_means(rows, method, metric)
            a = np.asarray([value for (group, _), value in values.items() if group == "A"])
            b = np.asarray([value for (group, _), value in values.items() if group == "B"])
            rng = np.random.default_rng(SEED)
            draws = np.empty(UNADJUSTED_REPLICATES, dtype=float)
            for index in range(UNADJUSTED_REPLICATES):
                draws[index] = float(
                    rng.choice(a, size=len(a), replace=True).mean()
                    - rng.choice(b, size=len(b), replace=True).mean()
                )
            estimate = float(a.mean() - b.mean())
            low, high = (float(item) for item in np.quantile(draws, [0.025, 0.975]))
            n_a = sum(1 for row in rows if row["method"] == method and row["group"] == "A")
            n_b = sum(1 for row in rows if row["method"] == method and row["group"] == "B")
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_A": n_a,
                    "n_B": n_b,
                    "n_station_A": len(a),
                    "n_station_B": len(b),
                    "station_mean_A": float(a.mean()),
                    "station_mean_B": float(b.mean()),
                    "unadjusted_A_minus_B": estimate,
                    "unadjusted_ci95_low": low,
                    "unadjusted_ci95_high": high,
                    "unadjusted_replicates": UNADJUSTED_REPLICATES,
                    "unadjusted_seed": SEED,
                }
            )
    return output


def design(rows: list[dict[str, str]]) -> np.ndarray:
    columns = [np.ones(len(rows)), np.asarray([1.0 if row["group"] == "A" else 0.0 for row in rows])]
    for name in COVARIATES:
        values = np.asarray([finite(row[name]) for row in rows])
        columns.append((values - np.nanmean(values)) / (np.nanstd(values) + 1e-12))
    return np.column_stack(columns)


def ols_group_effect(rows: list[dict[str, str]], metric: str) -> float:
    y = np.asarray([finite(row[metric]) for row in rows])
    x = design(rows)
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    beta, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    return float(beta[1])


def regression(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in subset:
            grouped[(row["group"], row["station_noise"])].append(row)
        stations = {
            group: sorted(station for current, station in grouped if current == group)
            for group in ("A", "B")
        }
        for metric in METRICS:
            estimate = ols_group_effect(subset, metric)
            rng = np.random.default_rng(stable_seed(method, metric, "regression"))
            x_full = design(subset)
            y_full = np.asarray([finite(row[metric]) for row in subset])
            blocks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
            for key in grouped:
                indices = [
                    index
                    for index, row in enumerate(subset)
                    if (row["group"], row["station_noise"]) == key
                    and np.isfinite(y_full[index])
                    and np.all(np.isfinite(x_full[index]))
                ]
                blocks[key] = (x_full[indices], y_full[indices])
            draws = []
            for _ in range(SENSITIVITY_REPLICATES):
                selected_x = []
                selected_y = []
                for group in ("A", "B"):
                    for station in rng.choice(stations[group], size=len(stations[group]), replace=True):
                        x_block, y_block = blocks[(group, str(station))]
                        selected_x.append(x_block)
                        selected_y.append(y_block)
                beta, *_ = np.linalg.lstsq(
                    np.concatenate(selected_x, axis=0),
                    np.concatenate(selected_y, axis=0),
                    rcond=None,
                )
                draws.append(float(beta[1]))
            low, high = (float(item) for item in np.quantile(draws, [0.025, 0.975]))
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "regression_adjusted_A_minus_B": estimate,
                    "regression_ci95_low": low,
                    "regression_ci95_high": high,
                    "regression_replicates": SENSITIVITY_REPLICATES,
                    "regression_seed": stable_seed(method, metric, "regression"),
                }
            )
    return output


def station_balance(rows: list[dict[str, str]], included: set[tuple[str, str]] | None = None) -> dict[str, dict[str, float]]:
    if included is not None:
        rows = [row for row in rows if (row["group"], row["station_noise"]) in included]
    output = {}
    for name in STATION_COVARIATES:
        a = np.asarray([finite(row[name]) for row in rows if row["group"] == "A"])
        b = np.asarray([finite(row[name]) for row in rows if row["group"] == "B"])
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        output[name] = {
            "mean_A": float(a.mean()),
            "mean_B": float(b.mean()),
            "median_A": float(np.median(a)),
            "median_B": float(np.median(b)),
            "smd": float((a.mean() - b.mean()) / (pooled + 1e-12)),
            "n_A": len(a),
            "n_B": len(b),
        }
    return output


def match_stations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    raw = {
        name: np.asarray([finite(row[name]) for row in rows])
        for name in STATION_COVARIATES
    }
    means = {name: float(values.mean()) for name, values in raw.items()}
    standard_deviations = {name: float(values.std()) + 1e-12 for name, values in raw.items()}
    vectors = {
        (row["group"], row["station_noise"]): np.asarray(
            [(finite(row[name]) - means[name]) / standard_deviations[name] for name in STATION_COVARIATES]
        )
        for row in rows
    }
    a_keys = sorted(key for key in vectors if key[0] == "A")
    b_available = {key for key in vectors if key[0] == "B"}
    pairs = []
    for a_key in a_keys:
        candidates = sorted(
            ((float(np.linalg.norm(vectors[a_key] - vectors[b_key])), b_key) for b_key in b_available),
            key=lambda item: (item[0], item[1][1]),
        )
        if candidates and candidates[0][0] <= MATCHING_CALIPER:
            distance, b_key = candidates[0]
            b_available.remove(b_key)
            pairs.append(
                {
                    "pair_id": len(pairs) + 1,
                    "station_A": a_key[1],
                    "station_B": b_key[1],
                    "distance": distance,
                    "caliper": MATCHING_CALIPER,
                    "replacement": False,
                    "tie_break": "lowest_distance_then_station_name",
                }
            )
    return pairs


def matched(rows: list[dict[str, str]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        for metric in METRICS:
            means = station_means(rows, method, metric)
            differences = np.asarray(
                [means[("A", pair["station_A"])] - means[("B", pair["station_B"])] for pair in pairs]
            )
            seed = stable_seed(method, metric, "matched")
            rng = np.random.default_rng(seed)
            draws = rng.choice(
                differences,
                size=(SENSITIVITY_REPLICATES, len(differences)),
                replace=True,
            ).mean(axis=1)
            low, high = (float(item) for item in np.quantile(draws, [0.025, 0.975]))
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "matched_A_minus_B": float(differences.mean()),
                    "matched_ci95_low": low,
                    "matched_ci95_high": high,
                    "n_pairs": len(differences),
                    "matched_replicates": SENSITIVITY_REPLICATES,
                    "matched_seed": seed,
                }
            )
    return output


def recompute(package_root: Path) -> dict[str, list[dict[str, Any]]]:
    root = package_root / "data" / "results" / "e3"
    cases = read_csv(root / "case_outcomes_and_covariates.csv")
    station_covariates = read_csv(root / "station_covariates.csv")
    unadjusted_rows = unadjusted(cases)
    regression_rows = regression(cases)
    pairs = match_stations(station_covariates)
    matched_rows = matched(cases, pairs)

    unadjusted_lookup = {(row["method"], row["metric"]): row for row in unadjusted_rows}
    regression_lookup = {(row["method"], row["metric"]): row for row in regression_rows}
    matched_lookup = {(row["method"], row["metric"]): row for row in matched_rows}
    adjusted = []
    directions = []
    for key in sorted(unadjusted_lookup):
        u = unadjusted_lookup[key]
        r = regression_lookup[key]
        m = matched_lookup[key]
        adjusted.append({**u, **r, **m})
        direction_values = [
            direction(float(u["unadjusted_A_minus_B"])),
            direction(float(r["regression_adjusted_A_minus_B"])),
            direction(float(m["matched_A_minus_B"])),
        ]
        exclusions = [
            zero_excluded(float(u["unadjusted_ci95_low"]), float(u["unadjusted_ci95_high"])),
            zero_excluded(float(r["regression_ci95_low"]), float(r["regression_ci95_high"])),
            zero_excluded(float(m["matched_ci95_low"]), float(m["matched_ci95_high"])),
        ]
        directions.append(
            {
                "method": key[0],
                "metric": key[1],
                "unadjusted_direction": direction_values[0],
                "regression_direction": direction_values[1],
                "matched_direction": direction_values[2],
                "direction_consistent": len(set(direction_values)) == 1,
                "zero_exclusion_consistent": len(set(exclusions)) == 1,
                "interpretation_grade": "qualified_consistent" if len(set(direction_values)) == 1 else "suggestive",
            }
        )

    before = station_balance(station_covariates)
    included = {("A", pair["station_A"]) for pair in pairs} | {("B", pair["station_B"]) for pair in pairs}
    after = station_balance(station_covariates, included)
    balance_rows = []
    for name in STATION_COVARIATES:
        balance_rows.append(
            {
                "covariate": name,
                "mean_A_before": before[name]["mean_A"],
                "mean_B_before": before[name]["mean_B"],
                "median_A_before": before[name]["median_A"],
                "median_B_before": before[name]["median_B"],
                "before_smd": before[name]["smd"],
                "mean_A_after": after[name]["mean_A"],
                "mean_B_after": after[name]["mean_B"],
                "median_A_after": after[name]["median_A"],
                "median_B_after": after[name]["median_B"],
                "after_smd": after[name]["smd"],
                "absolute_smd_reduction": abs(before[name]["smd"]) - abs(after[name]["smd"]),
                "residual_absolute_smd": abs(after[name]["smd"]),
                "n_station_A_before": before[name]["n_A"],
                "n_station_B_before": before[name]["n_B"],
                "n_station_A_after": after[name]["n_A"],
                "n_station_B_after": after[name]["n_B"],
                "n_pairs_after": len(pairs),
            }
        )
    return {
        "matching_pairs.csv": pairs,
        "adjusted_contrasts.csv": adjusted,
        "balance_diagnostics.csv": balance_rows,
        "direction_consistency.csv": directions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, rows in recompute(args.package_root.resolve()).items():
        write_csv(args.output_dir.resolve() / name, rows)
    print(f"PASS: E3 outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
