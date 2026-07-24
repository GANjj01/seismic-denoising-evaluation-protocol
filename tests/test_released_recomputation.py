from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(row[column]) for column in columns): row for row in rows}


def assert_rows_close(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    keys: tuple[str, ...],
    tolerance: float = 1e-10,
) -> None:
    left, right = keyed(expected, keys), keyed(actual, keys)
    assert left.keys() == right.keys()
    for key in left:
        for column in set(left[key]).intersection(right[key]):
            if str(left[key][column]).lower() == str(right[key][column]).lower():
                continue
            try:
                assert math.isclose(float(left[key][column]), float(right[key][column]), rel_tol=0, abs_tol=tolerance)
            except (TypeError, ValueError):
                assert left[key][column] == right[key][column]


def test_report_cards_recompute() -> None:
    module = load_module("analysis/reproduce_report_cards.py", "test_report_cards_module")
    actual = module.recompute(ROOT)
    assert_rows_close(
        read_csv("data/results/report_cards/controlled_mixture_report_card.csv"),
        actual["controlled_mixture_report_card.csv"],
        ("method",),
    )
    assert_rows_close(
        read_csv("data/results/report_cards/external_real_event_report_card.csv"),
        actual["external_real_event_report_card.csv"],
        ("method",),
    )


def test_station_bootstrap_recomputes() -> None:
    module = load_module("analysis/reproduce_station_bootstrap.py", "test_bootstrap_module")
    assert_rows_close(
        read_csv("data/results/report_cards/station_bootstrap_contrasts.csv"),
        module.recompute(ROOT),
        ("method", "metric"),
    )


def test_e3_point_ols_matching_and_smd() -> None:
    module = load_module("analysis/reproduce_e3_station_domain.py", "test_e3_module")
    result_root = ROOT / "data" / "results" / "e3"
    cases = module.read_csv(result_root / "case_outcomes_and_covariates.csv")
    stations = module.read_csv(result_root / "station_covariates.csv")
    pairs = module.match_stations(stations)
    assert_rows_close(read_csv("data/results/e3/matching_pairs.csv"), pairs, ("pair_id",))
    before = module.station_balance(stations)
    frozen_balance = keyed(read_csv("data/results/e3/balance_diagnostics.csv"), ("covariate",))
    for covariate, values in before.items():
        assert values["smd"] == pytest.approx(float(frozen_balance[(covariate,)]["before_smd"]), abs=1e-10)
    frozen = keyed(read_csv("data/results/e3/adjusted_contrasts.csv"), ("method", "metric"))
    for method in sorted({row["method"] for row in cases}):
        subset = [row for row in cases if row["method"] == method]
        for metric in module.METRICS:
            estimate = module.ols_group_effect(subset, metric)
            assert estimate == pytest.approx(float(frozen[(method, metric)]["regression_adjusted_A_minus_B"]), abs=1e-10)


def test_e5_recomputes() -> None:
    module = load_module("analysis/reproduce_e5_multiseed.py", "test_e5_module")
    contrasts, summary = module.recompute(ROOT)
    assert_rows_close(read_csv("data/results/e5/paired_seed_contrasts.csv"), contrasts, ("seed", "metric"))
    assert_rows_close(read_csv("data/results/e5/seed_summary.csv"), summary, ("metric",))


def test_figure3_consistency() -> None:
    module = load_module("analysis/verify_figure3_surrogate.py", "test_figure3_module")
    result = module.verify(ROOT)
    assert result["status"] == "PASS"
