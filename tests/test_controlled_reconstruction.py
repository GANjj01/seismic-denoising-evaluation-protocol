from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blindspot_eval_protocol.controlled_mixture import (
    ControlledCaseSpecification,
    achieved_target_snr_db,
    extract_event_segment,
    inject_event_reference_at_target_snr,
    score_controlled_case,
)


def load_script(relative: str, name: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECONSTRUCTOR = load_script("scripts/reconstruct_controlled_cases.py", "test_reconstructor")


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic() -> tuple[np.ndarray, np.ndarray, ControlledCaseSpecification]:
    rng = np.random.default_rng(19)
    noise = rng.normal(0, 1, (9000, 3)).astype(np.float32)
    event = np.zeros((3500, 3), dtype=np.float32)
    event[1000:3000] = rng.normal(0, 2, (2000, 3))
    return noise, event, ControlledCaseSpecification.from_seconds(30.0)


def test_manifest_times_durations_and_multivalue_lengths() -> None:
    cases = read_csv("data/manifests/controlled_mixture_cases.csv")
    requests = read_csv("data/manifests/controlled_mixture_requests.csv")
    assert len(cases) == 816
    assert len(requests) == 3264
    for row in requests:
        duration = (
            datetime.fromisoformat(row["end_time"].replace("Z", "+00:00"))
            - datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        ).total_seconds()
        assert duration >= float(row["expected_duration_s"])
    for row in cases:
        lengths = {
            len(row[name].split(";"))
            for name in (
                "noise_network", "noise_station", "noise_location",
                "noise_channel_pattern", "noise_start_time", "noise_end_time",
                "noise_provider", "noise_fdsn_source", "noise_source_id",
            )
        }
        assert lengths == {3}


def test_normalized_requests_round_trip_and_three_chunk_rule() -> None:
    cases = read_csv("data/manifests/controlled_mixture_cases.csv")
    requests = read_csv("data/manifests/controlled_mixture_requests.csv")
    RECONSTRUCTOR.validate_controlled_manifests(cases, requests)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in requests:
        grouped.setdefault(row["case_id"], []).append(row)
    for case in cases:
        rows = sorted(grouped[case["case_id"]], key=lambda row: int(row["sequence_index"]))
        assert case["noise_source_id"].split(";") == [row["source_window_id"] for row in rows[1:]]
        assert [int(row["used_samples"]) for row in rows[1:]] == [3500, 3500, 2000]
        assert [int(row["output_sample_start"]) for row in rows[1:]] == [0, 3500, 7000]
        assert [int(row["output_sample_stop"]) for row in rows[1:]] == [3500, 7000, 9000]


def test_hidden_onset_event_segment_target_snr_and_identity() -> None:
    noise, event, specification = synthetic()
    assert specification.hidden_onset_sample == 3000
    assert specification.event_segment_slice == slice(1000, 3000)
    assert specification.scoring_slice == slice(2000, 5500)
    assert np.array_equal(extract_event_segment(event, specification), event[1000:3000])
    mixture, clean, noise_reference, _ = inject_event_reference_at_target_snr(
        noise, event, specification=specification, target_snr_db=-5.0
    )
    assert clean.shape == mixture.shape == noise_reference.shape == (9000, 3)
    assert np.all(clean[:3000] == 0) and np.all(clean[5000:] == 0)
    assert achieved_target_snr_db(noise_reference, clean, specification) == pytest.approx(-5.0, abs=1e-5)
    metrics = score_controlled_case(mixture, mixture, clean, specification=specification)
    assert metrics["clean_snr_gain_db"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["background_suppression_db"] == pytest.approx(0.0, abs=1e-10)


def test_boundaries_raise_instead_of_truncating() -> None:
    noise, event, _ = synthetic()
    with pytest.raises(ValueError, match="scoring crop"):
        inject_event_reference_at_target_snr(
            noise,
            event,
            specification=ControlledCaseSpecification.from_seconds(5.0),
            target_snr_db=0.0,
        )
    with pytest.raises(ValueError, match="injection"):
        inject_event_reference_at_target_snr(
            noise,
            event,
            specification=ControlledCaseSpecification.from_seconds(80.0),
            target_snr_db=0.0,
        )


def test_deterministic_npz_hash_and_schema(tmp_path: Path) -> None:
    arrays = {
        "mixture": np.arange(60, dtype=np.float32).reshape(20, 3),
        "clean_reference": np.zeros((20, 3), dtype=np.float32),
        "noise_reference": np.ones((20, 3), dtype=np.float32),
        "sampling_rate_hz": np.asarray(100.0),
        "component_order": np.asarray(["Z", "N", "E"]),
        "hidden_onset_sample": np.asarray(10),
        "scoring_window_samples": np.asarray([0, 20]),
        "case_id": np.asarray("toy"),
    }
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    RECONSTRUCTOR.write_deterministic_npz(first, arrays)
    RECONSTRUCTOR.write_deterministic_npz(second, arrays)
    assert sha256(first) == sha256(second)
    with np.load(first, allow_pickle=False) as payload:
        assert set(arrays) == set(payload.files)


def test_dry_run_does_not_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "output"
    rows = RECONSTRUCTOR.reconstruct(
        ROOT,
        ROOT / "data/manifests/controlled_mixture_cases.csv",
        tmp_path / "absent_waveforms",
        output,
        limit=1,
        dry_run=True,
    )
    assert rows == []
    assert not output.exists()
    assert '"case_id"' in capsys.readouterr().out


def test_invalid_timestamp_is_rejected() -> None:
    cases = read_csv("data/manifests/controlled_mixture_cases.csv")[:1]
    requests = [
        dict(row)
        for row in read_csv("data/manifests/controlled_mixture_requests.csv")
        if row["case_id"] == cases[0]["case_id"]
    ]
    requests[1]["end_time"] = requests[1]["start_time"]
    with pytest.raises(ValueError, match="shorter"):
        RECONSTRUCTOR.validate_controlled_manifests(cases, requests)
