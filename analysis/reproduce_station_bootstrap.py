"""Recompute primary paired station-bootstrap contrasts from released cases."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True

METRICS = {
    "clean_snr_gain": "clean_snr_gain_db",
    "correlation_difference": "waveform_correlation_z",
    "background_suppression_difference": "background_suppression_db",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_station_values(
    rows: list[dict[str, str]],
    method: str,
    metric: str,
    reference: str = "Identity",
) -> tuple[np.ndarray, int]:
    cases: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        cases[row["case_id"]][row["method"]] = row
    by_station: dict[str, list[float]] = defaultdict(list)
    n_pairs = 0
    for methods in cases.values():
        if method not in methods or reference not in methods:
            continue
        try:
            left = float(methods[method][metric])
            right = float(methods[reference][metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        by_station[methods[method]["station_template"]].append(left - right)
        n_pairs += 1
    values = np.asarray([np.mean(by_station[key]) for key in sorted(by_station)], dtype=float)
    return values, n_pairs


def recompute(package_root: Path) -> list[dict[str, Any]]:
    source = package_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from blindspot_eval_protocol.bootstrap import paired_station_bootstrap  # pylint: disable=import-outside-toplevel

    rows = read_csv(package_root / "data" / "results" / "report_cards" / "controlled_mixture_per_case_metrics.csv")
    methods = sorted({row["method"] for row in rows if row["method"] != "Identity"})
    output = []
    for method in methods:
        for metric_name, source_metric in METRICS.items():
            values, n_pairs = paired_station_values(rows, method, source_metric)
            result = paired_station_bootstrap(values, n_matched_cases=n_pairs)
            output.append(
                {
                    "method": method,
                    "reference_method": "Identity",
                    "metric": metric_name,
                    "source_metric": source_metric,
                    "n_stations": result.n_stations,
                    "n_matched_cases": result.n_matched_cases,
                    "estimate": result.estimate,
                    "ci95_low": result.ci_low,
                    "ci95_high": result.ci_high,
                    "ci_excludes_zero": result.excludes_zero,
                    "replicates": result.replicates,
                    "seed": result.seed,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = recompute(args.package_root.resolve())
    destination = args.output_dir.resolve() / "station_bootstrap_contrasts.csv"
    write_csv(destination, output)
    print(f"PASS: station bootstrap written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
