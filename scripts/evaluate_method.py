"""Evaluate a Python adapter on reconstructed controlled-mixture cases."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.dont_write_bytecode = True


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter(path: Path) -> Callable[[np.ndarray, float], np.ndarray]:
    function = getattr(load_module(path, "public_method_adapter"), "denoise", None)
    if not callable(function):
        raise ValueError("adapter must define denoise(waveform, sampling_rate_hz)")
    return function


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_case(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".npz":
        raise ValueError("reconstructed cases must be NPZ artifacts")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "mixture", "clean_reference", "noise_reference", "sampling_rate_hz",
            "hidden_onset_sample", "component_order", "scoring_window_samples",
        }
        if missing := required - set(payload.files):
            raise ValueError(f"{path} lacks required arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def evaluate(
    package_root: Path,
    adapter_path: Path,
    reconstruction_manifest: Path,
    cases_dir: Path,
    output_dir: Path,
    *,
    method_name: str = "custom_method",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = package_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from blindspot_eval_protocol.controlled_mixture import (  # pylint: disable=import-outside-toplevel
        ControlledCaseSpecification,
        score_controlled_case,
    )
    from blindspot_eval_protocol.report_card import (  # pylint: disable=import-outside-toplevel
        aggregate_controlled_report_card,
    )
    from blindspot_eval_protocol.schemas import validate_waveform  # pylint: disable=import-outside-toplevel

    denoise = load_adapter(adapter_path)
    rows = read_csv(reconstruction_manifest)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result: dict[str, Any] = {
            "case_id": row["case_id"],
            "method": method_name,
            "station_event": row["station_event"],
            "station_noise": row["station_noise"],
            "target_snr_db": row["target_snr_db"],
            "clean_snr_gain_db": "",
            "amplitude_ratio_clean": "",
            "waveform_correlation_z": "",
            "delay_s": "",
            "background_suppression_db": "",
            "status": "error",
            "error_message": "",
        }
        try:
            path = cases_dir / f"{row['case_id']}.npz"
            payload = load_case(path)
            sampling_rate = float(payload["sampling_rate_hz"])
            mixture = validate_waveform(payload["mixture"], name="mixture", sampling_rate_hz=sampling_rate)
            clean = validate_waveform(
                payload["clean_reference"], name="clean_reference", sampling_rate_hz=sampling_rate
            )
            output = validate_waveform(
                denoise(mixture.astype(np.float32, copy=True), sampling_rate),
                name="adapter output",
                sampling_rate_hz=sampling_rate,
            )
            if output.shape != mixture.shape:
                raise ValueError(f"adapter changed shape from {mixture.shape} to {output.shape}")
            specification = ControlledCaseSpecification(
                sampling_rate_hz=sampling_rate,
                hidden_onset_sample=int(payload["hidden_onset_sample"]),
            )
            result.update(score_controlled_case(output, mixture, clean, specification=specification))
            result["status"] = "ok"
        except Exception as exc:  # keep a complete case ledger
            result["error_message"] = str(exc)
        output_rows.append(result)
    successful = [row for row in output_rows if row["status"] == "ok"]
    report_card = aggregate_controlled_report_card(successful)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_case_metrics.csv", output_rows)
    if report_card:
        write_csv(output_dir / "report_card.csv", report_card)
    (output_dir / "adapter_contract.json").write_text(
        json.dumps(
            {
                "adapter": adapter_path.name,
                "method_name": method_name,
                "input_shape": "(9000, 3)",
                "output_shape": "same as input",
                "component_order": ["Z", "N", "E"],
                "sampling_rate_hz": "stored in each reconstructed artifact",
                "method_visible_fields": ["mixture", "sampling_rate_hz"],
                "evaluator_held_fields": [
                    "clean_reference", "noise_reference", "hidden_onset_sample",
                    "target_snr_db",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_rows, report_card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases-dir", type=Path)
    parser.add_argument("--reconstruction-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--waveform-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="custom_method")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package_root = args.package_root.resolve()

    if args.cases_dir and args.reconstruction_manifest:
        cases_dir = args.cases_dir.resolve()
        reconstruction_manifest = args.reconstruction_manifest.resolve()
    elif args.manifest and args.waveform_root:
        reconstructed = args.output_dir.resolve() / "reconstructed_cases"
        reconstructor = load_module(
            package_root / "scripts" / "reconstruct_controlled_cases.py",
            "evaluation_reconstructor",
        )
        reconstructor.reconstruct(
            package_root,
            args.manifest.resolve(),
            args.waveform_root.resolve(),
            reconstructed,
        )
        cases_dir = reconstructed / "cases"
        reconstruction_manifest = reconstructed / "reconstruction_manifest.csv"
    else:
        parser.error(
            "provide either --cases-dir with --reconstruction-manifest, or "
            "--manifest with --waveform-root"
        )
    rows, _ = evaluate(
        package_root,
        args.adapter.resolve(),
        reconstruction_manifest,
        cases_dir,
        args.output_dir.resolve(),
        method_name=args.method_name,
    )
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"PASS: evaluated={len(rows)} failed={failed} output={args.output_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
