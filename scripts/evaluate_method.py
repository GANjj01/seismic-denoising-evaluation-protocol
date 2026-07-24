"""Evaluate a Python adapter on locally reconstructed controlled-mixture cases."""

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


def load_adapter(path: Path) -> Callable[[np.ndarray, float], np.ndarray]:
    spec = importlib.util.spec_from_file_location("public_method_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "denoise", None)
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


def load_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() != ".npz":
        raise ValueError("reconstructed cases must be external NPZ files")
    with np.load(path, allow_pickle=False) as payload:
        if "mixture" not in payload or "clean" not in payload:
            raise ValueError(f"{path} must contain mixture and clean arrays")
        return np.asarray(payload["mixture"]), np.asarray(payload["clean"])


def evaluate(
    package_root: Path,
    adapter_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    input_dir: Path | None = None,
    method_name: str = "custom_method",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = package_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from blindspot_eval_protocol.controlled_mixture import score_controlled_case  # pylint: disable=import-outside-toplevel
    from blindspot_eval_protocol.report_card import aggregate_controlled_report_card  # pylint: disable=import-outside-toplevel
    from blindspot_eval_protocol.schemas import validate_waveform  # pylint: disable=import-outside-toplevel

    denoise = load_adapter(adapter_path)
    rows = read_csv(manifest_path)
    base = input_dir or manifest_path.parent / "reconstructed_cases"
    output_rows = []
    for row in rows:
        case_path = Path(row.get("input_path", "")) if row.get("input_path") else base / f"{row['case_id']}.npz"
        mixture, clean = load_case(case_path)
        sampling_rate = float(row["sampling_rate_hz"])
        mixture = validate_waveform(mixture, name="mixture", sampling_rate_hz=sampling_rate)
        clean = validate_waveform(clean, name="clean", sampling_rate_hz=sampling_rate)
        output = validate_waveform(
            denoise(mixture.astype(np.float32, copy=True), sampling_rate),
            name="adapter output",
            sampling_rate_hz=sampling_rate,
        )
        if output.shape != mixture.shape:
            raise ValueError(f"adapter changed shape for case {row['case_id']}")
        metrics = score_controlled_case(
            output,
            mixture,
            clean,
            hidden_onset_s=float(row["hidden_onset_s"]),
            event_crop_start_s=float(row["hidden_onset_s"]) - 10.0,
            sampling_rate_hz=sampling_rate,
        )
        output_rows.append(
            {
                "case_id": row["case_id"],
                "station_template": row["station"],
                "station_noise": row["noise_station"].split(";")[0],
                "target_snr_db": row["target_snr_db"],
                "method": method_name,
                **metrics,
            }
        )
    report_card = aggregate_controlled_report_card(output_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_case_metrics.csv", output_rows)
    write_csv(output_dir / "report_card.csv", report_card)
    (output_dir / "adapter_contract.json").write_text(
        json.dumps(
            {
                "adapter": adapter_path.name,
                "method_name": method_name,
                "input_shape": "(n_samples, 3)",
                "output_shape": "same as input",
                "component_order": ["Z", "N", "E"],
                "sampling_rate_hz": "from manifest",
                "dtype": "floating",
                "finite_values": True,
                "amplitude_scale": "unchanged physical input scale unless method documentation states otherwise",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_rows, report_card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="custom_method")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    evaluate(
        args.package_root.resolve(),
        args.adapter.resolve(),
        args.manifest.resolve(),
        args.output_dir.resolve(),
        input_dir=args.input_dir.resolve() if args.input_dir else None,
        method_name=args.method_name,
    )
    print(f"PASS: evaluation outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
