"""Reconstruct controlled reference-based cases from manifest-defined waveforms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.dont_write_bytecode = True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def validate_controlled_manifests(
    cases: Iterable[dict[str, str]],
    requests: Iterable[dict[str, str]],
) -> None:
    cases, requests = list(cases), list(requests)
    case_ids = [row["case_id"] for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("controlled case_id values must be unique")
    pairing_ids = [row["deterministic_pairing_id"] for row in cases]
    if len(pairing_ids) != len(set(pairing_ids)):
        raise ValueError("deterministic_pairing_id values must be unique")
    request_ids = [row["request_id"] for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id values must be unique")
    known_case_ids = set(case_ids)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in requests:
        required_text = (
            "request_id", "role", "network", "station", "channel", "start_time",
            "end_time", "provider", "fdsn_source", "construction_role",
            "source_window_id", "duplicate_reason",
        )
        if any(not row.get(name, "").strip() for name in required_text):
            raise ValueError(f"request {row.get('request_id', '<missing>')} has an empty required field")
        if row["case_id"] not in known_case_ids:
            raise ValueError(f"request references unknown case_id {row['case_id']}")
        start, end = parse_utc(row["start_time"]), parse_utc(row["end_time"])
        if (end - start).total_seconds() + 1e-6 < float(row["expected_duration_s"]):
            raise ValueError(f"request {row['request_id']} is shorter than expected_duration_s")
        grouped.setdefault(row["case_id"], []).append(row)
    for case in cases:
        multivalue_names = (
            "noise_network", "noise_station", "noise_location",
            "noise_channel_pattern", "noise_start_time", "noise_end_time",
            "noise_provider", "noise_fdsn_source", "noise_source_id",
        )
        values = {name: case[name].split(";") for name in multivalue_names}
        if {len(items) for items in values.values()} != {3}:
            raise ValueError(f"case {case['case_id']} noise multivalue lengths are not all three")
        if any(not item.strip() for items in values.values() for item in items):
            raise ValueError(f"case {case['case_id']} has an empty noise multivalue element")
        rows = sorted(grouped.get(case["case_id"], []), key=lambda row: int(row["sequence_index"]))
        if len(rows) != 4:
            raise ValueError(f"case {case['case_id']} must have one event and three noise requests")
        if [(row["role"], int(row["sequence_index"])) for row in rows] != [
            ("event", 0), ("noise", 1), ("noise", 2), ("noise", 3)
        ]:
            raise ValueError(f"case {case['case_id']} has invalid request roles or order")
        if [row["construction_role"] for row in rows] != [
            "evaluator_event_reference",
            "primary_noise_chunk",
            "continuation_noise_chunk",
            "continuation_noise_chunk_truncated",
        ]:
            raise ValueError(f"case {case['case_id']} has invalid construction roles")
        expected = [(3500, 0, 3500), (3500, 3500, 7000), (2000, 7000, 9000)]
        observed = [
            (int(row["used_samples"]), int(row["output_sample_start"]), int(row["output_sample_stop"]))
            for row in rows[1:]
        ]
        if observed != expected:
            raise ValueError(f"case {case['case_id']} does not implement the frozen 3500+3500+2000 rule")
        if rows[0]["station"] != case["station"]:
            raise ValueError(f"case {case['case_id']} event station does not round-trip")
        noise_stations = case["noise_station"].split(";")
        if noise_stations != [row["station"] for row in rows[1:]]:
            raise ValueError(f"case {case['case_id']} noise station sequence does not round-trip")


def _npy_bytes(value: Any) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)
    return handle.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))


def locate_waveform(root: Path, source_window_id: str) -> Path:
    for suffix in (".mseed", ".miniseed"):
        direct = root / f"{source_window_id}{suffix}"
        if direct.is_file():
            return direct
    rejected_npz = root / f"{source_window_id}.npz"
    if rejected_npz.is_file():
        raise ValueError(
            f"source-waveform NPZ is not supported: {rejected_npz}; "
            "provide MiniSEED (.mseed or .miniseed)"
        )
    matches = [
        path for path in root.rglob(f"{source_window_id}.*")
        if path.suffix.lower() in {".mseed", ".miniseed"}
    ]
    rejected = [
        path for path in root.rglob(f"{source_window_id}.npz")
        if path.is_file()
    ]
    if not matches and rejected:
        raise ValueError(
            f"source-waveform NPZ is not supported: {rejected[0]}; "
            "provide MiniSEED (.mseed or .miniseed)"
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one local waveform for {source_window_id}; found {len(matches)}"
        )
    return matches[0]


def _fit_length(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) >= length:
        return array[:length].astype(np.float32)
    output = np.zeros((length, 3), dtype=np.float32)
    output[: len(array)] = array
    return output


def load_three_component(path: Path, sampling_rate_hz: float, length: int) -> np.ndarray:
    """Load Z/N/E, reproducing the historical trim/pad behavior."""
    if path.suffix.lower() == ".npz":
        raise ValueError(
            f"source-waveform NPZ is not supported: {path}; "
            "provide MiniSEED (.mseed or .miniseed)"
        )
    if path.suffix.lower() not in {".mseed", ".miniseed"}:
        raise ValueError(f"unsupported source-waveform format: {path.suffix or '<none>'}")

    try:
        from obspy import read
    except Exception as exc:
        raise RuntimeError("ObsPy is required to read MiniSEED files") from exc
    stream = read(str(path))
    stream.detrend("demean")
    stream.detrend("linear")
    if any(not np.isclose(trace.stats.sampling_rate, sampling_rate_hz) for trace in stream):
        stream.resample(sampling_rate_hz)
    stream.merge(method=1, fill_value=0)
    selected = []
    for alternatives in (("Z",), ("N", "1"), ("E", "2")):
        candidates = [trace.copy() for trace in stream if str(trace.stats.channel)[-1:] in alternatives]
        if not candidates:
            raise ValueError(f"{path} lacks component alternatives {alternatives}")
        selected.append(candidates[0])
    common_start = max(trace.stats.starttime for trace in selected)
    common_end = min(trace.stats.endtime for trace in selected)
    if common_end >= common_start:
        for trace in selected:
            trace.trim(common_start, common_end, pad=True, fill_value=0)
    data = np.stack([np.asarray(trace.data, dtype=np.float64) for trace in selected], axis=1)
    return _fit_length(data, length)


def load_fetcher(package_root: Path) -> Any:
    path = package_root / "scripts" / "fetch_fdsn_windows.py"
    spec = importlib.util.spec_from_file_location("controlled_fetcher", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load fetch helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct(
    package_root: Path,
    manifest_path: Path,
    waveform_root: Path,
    output_dir: Path,
    *,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    download_missing: bool = False,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    source = package_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from blindspot_eval_protocol.controlled_mixture import (  # pylint: disable=import-outside-toplevel
        ControlledCaseSpecification,
        achieved_target_snr_db,
        inject_event_reference_at_target_snr,
    )
    from blindspot_eval_protocol.preprocessing import (  # pylint: disable=import-outside-toplevel
        bandpass_filter,
        event_centered_reference,
    )

    request_path = manifest_path.with_name("controlled_mixture_requests.csv")
    cases, requests = read_csv(manifest_path), read_csv(request_path)
    validate_controlled_manifests(cases, requests)
    if case_ids:
        missing = case_ids - {row["case_id"] for row in cases}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
        cases = [row for row in cases if row["case_id"] in case_ids]
    if limit is not None:
        cases = cases[:limit]
    selected_ids = {row["case_id"] for row in cases}
    selected_requests = [row for row in requests if row["case_id"] in selected_ids]

    if download_missing and not dry_run:
        unique_missing = []
        seen = set()
        for row in selected_requests:
            source_id = row["source_window_id"]
            if source_id in seen:
                continue
            seen.add(source_id)
            try:
                locate_waveform(waveform_root, source_id)
            except FileNotFoundError:
                unique_missing.append(row)
        if unique_missing:
            temporary_manifest = output_dir / "_missing_requests.csv"
            write_csv(temporary_manifest, unique_missing)
            load_fetcher(package_root).run(temporary_manifest, waveform_root, provider, False)
            temporary_manifest.unlink()

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in selected_requests:
        grouped.setdefault(row["case_id"], []).append(row)
    config_path = package_root / "configs" / "controlled_mixture.yml"
    preprocessing_path = package_root / "configs" / "preprocessing.yml"
    config_hash = sha256_file(config_path)
    preprocessing_hash = sha256_file(preprocessing_path)
    output_rows: list[dict[str, Any]] = []

    for case in cases:
        ordered = sorted(grouped[case["case_id"]], key=lambda row: int(row["sequence_index"]))
        plan = {
            "case_id": case["case_id"],
            "event_source_id": ordered[0]["source_window_id"],
            "noise_source_ids": [row["source_window_id"] for row in ordered[1:]],
            "output_path": str(output_dir / "cases" / f"{case['case_id']}.npz"),
        }
        if dry_run:
            print(json.dumps(plan, sort_keys=True))
            continue
        paths = [locate_waveform(waveform_root, row["source_window_id"]) for row in ordered]
        sampling_rate = float(case["sampling_rate_hz"])
        event_raw = load_three_component(paths[0], sampling_rate, 3500)
        event_reference = event_centered_reference(
            bandpass_filter(event_raw, sampling_rate_hz=sampling_rate),
            p_index=1000,
            event_duration_s=20.0,
            sampling_rate_hz=sampling_rate,
            taper_onset=False,
        )
        noise_chunks = [
            bandpass_filter(
                load_three_component(path, sampling_rate, int(row["required_samples"])),
                sampling_rate_hz=sampling_rate,
            )[: int(row["used_samples"])]
            for path, row in zip(paths[1:], ordered[1:])
        ]
        continuous_noise = np.concatenate(noise_chunks, axis=0)
        continuous_noise -= np.median(continuous_noise, axis=0, keepdims=True)
        specification = ControlledCaseSpecification.from_seconds(
            float(case["hidden_onset_s"]), sampling_rate_hz=sampling_rate
        )
        mixture, clean, noise, scale = inject_event_reference_at_target_snr(
            continuous_noise,
            event_reference,
            specification=specification,
            target_snr_db=float(case["target_snr_db"]),
        )
        artifact_path = output_dir / "cases" / f"{case['case_id']}.npz"
        if artifact_path.exists() and not overwrite:
            raise FileExistsError(f"{artifact_path} exists; pass --overwrite to replace it")
        write_deterministic_npz(
            artifact_path,
            {
                "case_id": np.asarray(case["case_id"]),
                "clean_reference": clean,
                "component_order": np.asarray(["Z", "N", "E"]),
                "hidden_onset_sample": np.asarray(specification.hidden_onset_sample),
                "mixture": mixture,
                "noise_reference": noise,
                "sampling_rate_hz": np.asarray(sampling_rate),
                "scoring_window_samples": np.asarray(
                    [specification.scoring_slice.start, specification.scoring_slice.stop]
                ),
                "target_snr_db": np.asarray(float(case["target_snr_db"])),
            },
        )
        source_hashes = ";".join(sha256_file(path) for path in paths)
        output_rows.append(
            {
                "case_id": case["case_id"],
                "artifact_path": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "n_samples": len(mixture),
                "component_order": "Z;N;E",
                "station_event": case["station"],
                "station_noise": ordered[1]["station"],
                "target_snr_db": case["target_snr_db"],
                "hidden_onset_s": case["hidden_onset_s"],
                "hidden_onset_sample": specification.hidden_onset_sample,
                "sampling_rate_hz": sampling_rate,
                "event_request_id": ordered[0]["request_id"],
                "noise_request_ids": ";".join(row["request_id"] for row in ordered[1:]),
                "source_window_ids": ";".join(row["source_window_id"] for row in ordered),
                "source_sha256": source_hashes,
                "noise_scale": scale,
                "achieved_target_snr_db": achieved_target_snr_db(noise, clean, specification),
                "preprocessing_config_sha256": preprocessing_hash,
                "construction_config_sha256": config_hash,
                "manifest_sha256": sha256_file(manifest_path),
                "request_manifest_sha256": sha256_file(request_path),
            }
        )
    if not dry_run:
        write_csv(output_dir / "reconstruction_manifest.csv", output_rows)
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_count": len(output_rows),
            "component_order": ["Z", "N", "E"],
            "noise_construction": "ordered 3500 + 3500 + first 2000 samples",
            "event_reference": "untapered P through P+20 s",
            "config_sha256": config_hash,
            "preprocessing_sha256": preprocessing_hash,
            "manifest_sha256": sha256_file(manifest_path),
            "request_manifest_sha256": sha256_file(request_path),
            "script_sha256": sha256_file(Path(__file__)),
        }
        (output_dir / "reconstruction_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--provider")
    args = parser.parse_args()
    rows = reconstruct(
        args.package_root.resolve(),
        args.manifest.resolve(),
        args.waveform_root.resolve(),
        args.output_dir.resolve(),
        case_ids=set(args.case_id) or None,
        limit=args.limit,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        download_missing=args.download_missing,
        provider=args.provider,
    )
    print(f"PASS: reconstructed={len(rows)} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
