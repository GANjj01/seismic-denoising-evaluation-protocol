"""Recompute the released descriptive paired-seed E5 contrasts."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

METRICS = (
    "clean_snr_gain_db",
    "amplitude_ratio_clean",
    "waveform_correlation_z",
    "background_suppression_db",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recompute(package_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = package_root / "data" / "results" / "e5"
    selected = read_csv(root / "selected_checkpoints.csv")
    per_case = read_csv(root / "per_case_metrics.csv")
    by_method_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_case:
        for metric in METRICS:
            by_method_metric[(row["method"], metric)].append(float(row[metric]))

    contrasts: list[dict[str, Any]] = []
    for selected_row in selected:
        seed = selected_row["seed"]
        left = selected_row["lambda0_method"]
        right = selected_row["lambda05_method"]
        for metric in METRICS:
            left_mean = statistics.fmean(by_method_metric[(left, metric)])
            right_mean = statistics.fmean(by_method_metric[(right, metric)])
            contrasts.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "lambda0_method": left,
                    "lambda05_method": right,
                    "lambda0_mean": left_mean,
                    "lambda05_mean": right_mean,
                    "difference_lambda05_minus_lambda0": right_mean - left_mean,
                }
            )

    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        values = [float(row["difference_lambda05_minus_lambda0"]) for row in contrasts if row["metric"] == metric]
        signs = {"positive" if value > 0 else "negative" if value < 0 else "zero" for value in values}
        summary.append(
            {
                "metric": metric,
                "n_paired_seeds": len(values),
                "mean_difference": statistics.fmean(values),
                "median_difference": statistics.median(values),
                "minimum_difference": min(values),
                "maximum_difference": max(values),
                "direction_consistent": len(signs) == 1,
            }
        )
    return contrasts, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contrasts, summary = recompute(args.package_root.resolve())
    write_csv(args.output_dir.resolve() / "paired_seed_contrasts.csv", contrasts)
    write_csv(args.output_dir.resolve() / "seed_summary.csv", summary)
    print(f"PASS: E5 outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
