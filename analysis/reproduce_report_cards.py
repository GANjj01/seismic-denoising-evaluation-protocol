"""Recompute released report cards from package-local per-case CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def _load_package(package_root: Path) -> None:
    source = package_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recompute(package_root: Path) -> dict[str, list[dict[str, Any]]]:
    _load_package(package_root)
    from blindspot_eval_protocol.report_card import (  # pylint: disable=import-outside-toplevel
        aggregate_controlled_report_card,
        aggregate_external_report_card,
    )

    result_root = package_root / "data" / "results" / "report_cards"
    controlled = aggregate_controlled_report_card(
        read_csv(result_root / "controlled_mixture_per_case_metrics.csv")
    )
    external = aggregate_external_report_card(
        read_csv(result_root / "external_real_event_per_case_metrics.csv")
    )
    return {"controlled_mixture_report_card.csv": controlled, "external_real_event_report_card.csv": external}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, rows in recompute(args.package_root.resolve()).items():
        write_csv(args.output_dir.resolve() / name, rows)
    print(f"PASS: report cards written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
