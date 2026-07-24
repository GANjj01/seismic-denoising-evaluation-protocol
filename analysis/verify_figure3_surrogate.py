"""Verify consistency of the released Figure 3 numerical surrogate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify(package_root: Path) -> dict[str, Any]:
    root = package_root / "data" / "results" / "figure3"
    observed = read_csv(root / "observed_acf.csv")
    pointwise = read_csv(root / "pointwise_envelope.csv")
    simultaneous = read_csv(root / "simultaneous_studentized_envelope.csv")
    maximum = json.loads((root / "maximum_absolute_acf_summary.json").read_text(encoding="utf-8"))
    global_summary = json.loads((root / "global_deviation_summary.json").read_text(encoding="utf-8"))
    surrogate = json.loads((root / "surrogate_summary.json").read_text(encoding="utf-8"))

    if len(observed) != 501 or len(pointwise) != 501 or len(simultaneous) != 500:
        raise AssertionError("unexpected lag-grid length")
    observed_by_lag = {row["lag_seconds"]: float(row["station_median_acf"]) for row in observed}
    pointwise_by_lag = {row["lag_seconds"]: row for row in pointwise}
    exceedances = 0
    for lag, value in observed_by_lag.items():
        row = pointwise_by_lag[lag]
        if value < float(row["lower_2p5"]) or value > float(row["upper_97p5"]):
            exceedances += 1
    nonzero = {lag: abs(value) for lag, value in observed_by_lag.items() if float(lag) > 0.0}
    max_lag = max(nonzero, key=nonzero.get)
    max_value = nonzero[max_lag]

    sim_exceedances = 0
    max_deviation = (-math.inf, "")
    for row in simultaneous:
        lag = row["lag_seconds"]
        value = observed_by_lag[lag]
        lower = float(row["lower_simultaneous_95"])
        upper = float(row["upper_simultaneous_95"])
        sim_exceedances += int(value < lower or value > upper)
        deviation = abs(value - float(row["null_mean_acf"])) / float(row["null_sd_acf"])
        if deviation > max_deviation[0]:
            max_deviation = (deviation, lag)

    checks = {
        "replicates_match_config": surrogate["replicates"] == 10000,
        "seed_match_config": surrogate["seed"] == 20260725,
        "pointwise_exceedances_match": exceedances == surrogate["pointwise_exceedance_count"],
        "maximum_absolute_acf_match": math.isclose(max_value, maximum["observed_maximum_absolute_acf"], abs_tol=5e-10),
        "maximum_absolute_lag_match": math.isclose(float(max_lag), maximum["observed_maximum_absolute_acf_lag_seconds"], abs_tol=5e-12),
        "simultaneous_exceedances_match": sim_exceedances == global_summary["simultaneous_band_exceedance_count"],
        # The released envelope CSV is rounded to 10 decimals; the JSON retains
        # full precision from the Monte Carlo computation.
        "global_statistic_match": math.isclose(max_deviation[0], global_summary["observed_global_statistic"], rel_tol=0, abs_tol=1e-5),
        "global_lag_match": math.isclose(float(max_deviation[1]), global_summary["maximum_deviation_lag_seconds"], abs_tol=5e-12),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Figure 3 consistency failures: {failed}")
    return {
        "status": "PASS",
        "checks": checks,
        "pointwise_exceedance_count": exceedances,
        "simultaneous_band_exceedance_count": sim_exceedances,
        "observed_maximum_absolute_acf": max_value,
        "observed_global_statistic": max_deviation[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package_root.resolve())
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve() / "figure3_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: Figure 3 numerical verification written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
