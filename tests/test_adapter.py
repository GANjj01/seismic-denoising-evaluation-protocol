from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative: str, name: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_adapter_end_to_end(tmp_path: Path) -> None:
    source = ROOT / "src"
    import sys
    sys.path.insert(0, str(source))
    from blindspot_eval_protocol.controlled_mixture import (
        ControlledCaseSpecification,
        inject_event_reference_at_target_snr,
    )

    reconstructor = load_script("scripts/reconstruct_controlled_cases.py", "adapter_reconstructor")
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 0.1, (9000, 3)).astype(np.float32)
    event = np.zeros((3500, 3), dtype=np.float32)
    time = np.arange(2000) / 100.0
    event[1000:3000] = np.stack(
        [
            np.sin(2 * np.pi * 3 * time),
            0.5 * np.sin(2 * np.pi * 4 * time),
            0.25 * np.cos(2 * np.pi * 5 * time),
        ],
        axis=1,
    )
    specification = ControlledCaseSpecification.from_seconds(30.0)
    mixture, clean, noise_reference, _ = inject_event_reference_at_target_snr(
        noise, event, specification=specification, target_snr_db=0.0
    )
    reconstructor.write_deterministic_npz(
        cases_dir / "toy.npz",
        {
            "case_id": np.asarray("toy"),
            "mixture": mixture,
            "clean_reference": clean,
            "noise_reference": noise_reference,
            "sampling_rate_hz": np.asarray(100.0),
            "component_order": np.asarray(["Z", "N", "E"]),
            "hidden_onset_sample": np.asarray(3000),
            "scoring_window_samples": np.asarray([2000, 5500]),
            "target_snr_db": np.asarray(0.0),
        },
    )
    manifest = tmp_path / "reconstruction_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "station_event", "station_noise", "target_snr_db"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "toy",
                "station_event": "TEST",
                "station_noise": "NOISE",
                "target_snr_db": "0",
            }
        )
    rows, report = load_script("scripts/evaluate_method.py", "test_evaluator").evaluate(
        ROOT,
        ROOT / "examples" / "identity_adapter.py",
        manifest,
        cases_dir,
        tmp_path / "output",
        method_name="Identity",
    )
    assert len(rows) == len(report) == 1
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["clean_snr_gain_db"]) == pytest.approx(0.0, abs=1e-10)
    assert float(rows[0]["background_suppression_db"]) == pytest.approx(0.0, abs=1e-10)
