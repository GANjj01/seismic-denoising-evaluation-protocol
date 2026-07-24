from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator() -> Any:
    path = ROOT / "scripts" / "evaluate_method.py"
    spec = importlib.util.spec_from_file_location("test_evaluator_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_adapter_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "cases"
    input_dir.mkdir()
    rng = np.random.default_rng(6)
    clean = np.zeros((9000, 3), dtype=np.float32)
    time = np.arange(2000) / 100.0
    clean[3000:5000] = np.stack(
        [
            np.sin(2 * np.pi * 3 * time),
            0.5 * np.sin(2 * np.pi * 4 * time),
            0.25 * np.cos(2 * np.pi * 5 * time),
        ],
        axis=1,
    )
    mixture = clean + rng.normal(0.0, 0.1, clean.shape).astype(np.float32)
    np.savez(input_dir / "toy.npz", mixture=mixture, clean=clean)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "station", "noise_station", "target_snr_db", "hidden_onset_s", "sampling_rate_hz"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "toy",
                "station": "TEST",
                "noise_station": "NOISE",
                "target_snr_db": "0",
                "hidden_onset_s": "30",
                "sampling_rate_hz": "100",
            }
        )

    rows, report = load_evaluator().evaluate(
        ROOT,
        ROOT / "examples" / "identity_adapter.py",
        manifest,
        tmp_path / "output",
        input_dir=input_dir,
        method_name="Identity",
    )
    assert len(rows) == 1
    assert len(report) == 1
    assert float(rows[0]["clean_snr_gain_db"]) == pytest.approx(0.0, abs=1e-10)
    assert float(rows[0]["background_suppression_db"]) == pytest.approx(0.0, abs=1e-10)
