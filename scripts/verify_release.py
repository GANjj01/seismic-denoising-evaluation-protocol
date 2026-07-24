"""Offline, non-mutating verification for the curated public release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True
for variable in (
    "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import yaml

BANNED_DIRECTORIES = {
    ".git",
    "manuscript",
    "submitted_original",
    "diff",
    "response",
    "cover",
    "revision_workspace",
    "release_staging",
    "public_release_staging",
    "audit",
    "contact_sheets",
    "screenshots",
    "submission",
    "journal_submission",
    "review",
    "reviewers",
    "internal",
    "private",
    "legacy_scripts",
    "releases",
    "generated_figures",
    "manuscript_figures",
    "figures",
    "training",
    "diagnostics",
    "environments",
}
BANNED_NAME_FRAGMENTS = (
    "manuscript_diff",
    "response_to_reviewers",
    "revision_cover_letter",
    "supplementary_material",
    "marked_manuscript",
    "clean_manuscript",
    "latexdiff",
    "elsarticle",
    "final_location_anchor",
    "submission_readiness",
    "cover_letter",
    "reviewer_response",
)
BANNED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".mseed",
    ".miniseed",
    ".sac",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pyc",
}
BANNED_DOIS = tuple(
    "10.5281/" + "zenodo." + suffix
    for suffix in ("20681569", "21533969")
)
PUBLIC_TITLE = (
    "A Reproducible Evaluation Protocol for "
    + "Self-Supervised Three-Component Blind-Spot Seismic Denoising"
)

CSV_SCHEMAS = {
    "data/splits/train_validation_stations.csv": {"network", "station", "location", "channel_pattern", "provider", "fdsn_source", "split_role"},
    "data/splits/development_stations.csv": {"network", "station", "location", "channel_pattern", "provider", "fdsn_source", "split_role"},
    "data/splits/final_stations.csv": {"network", "station", "location", "channel_pattern", "provider", "fdsn_source", "split_role"},
    "data/manifests/controlled_mixture_cases.csv": {
        "case_id", "network", "station", "location", "channel_pattern", "start_time", "end_time",
        "event_noise_role", "target_snr_db", "hidden_onset_s", "sampling_rate_hz",
        "preprocessing_config_id", "deterministic_pairing_id", "provider", "fdsn_source",
        "noise_station", "noise_start_time", "noise_end_time", "noise_provider", "noise_fdsn_source",
    },
    "data/manifests/controlled_mixture_requests.csv": {
        "case_id", "request_id", "role", "sequence_index", "network", "station",
        "location", "channel", "start_time", "end_time", "provider", "fdsn_source",
        "expected_duration_s", "construction_role", "source_window_id",
        "required_samples", "used_samples", "output_sample_start", "output_sample_stop",
    },
    "data/manifests/external_real_event_cases.csv": {
        "case_id", "network", "station", "location", "channel_pattern", "start_time", "end_time",
        "event_noise_role", "p_time_s", "sampling_rate_hz", "preprocessing_config_id",
        "deterministic_pairing_id", "provider", "fdsn_source",
    },
    "data/results/report_cards/controlled_mixture_per_case_metrics.csv": {
        "case_id", "station_template", "station_noise", "method", "method_role",
        "clean_snr_gain_db", "amplitude_ratio_clean", "waveform_correlation_z",
        "delay_s", "background_suppression_db",
    },
    "data/results/report_cards/controlled_mixture_report_card.csv": {
        "method", "n_cases", "clean_snr_gain_db_mean", "amplitude_ratio_clean_median",
        "waveform_correlation_z_mean", "delay_s_mean", "background_suppression_db_mean",
    },
    "data/results/report_cards/external_real_event_per_case_metrics.csv": {
        "case_id", "station", "method", "method_role", "apparent_snr_gain_db",
        "amplitude_ratio_raw", "trigger_delay_s",
    },
    "data/results/report_cards/external_real_event_report_card.csv": {
        "method", "n_cases", "apparent_snr_gain_db_mean", "amplitude_ratio_raw_mean", "trigger_delay_s_mean",
    },
    "data/results/report_cards/station_bootstrap_contrasts.csv": {
        "method", "reference_method", "metric", "source_metric", "n_stations",
        "n_matched_cases", "estimate", "ci95_low", "ci95_high", "ci_excludes_zero",
        "replicates", "seed",
    },
    "data/results/e3/case_outcomes_and_covariates.csv": {
        "case_id", "group", "station_template", "station_noise", "target_snr_db", "method",
        "output_vs_clean_snr", "amp_ratio_clean", "corr_z", "background_suppression_db",
        "log10_noise_rms_injection_10s", "dominant_freq_hz", "spectral_slope_1_20",
        "log10_high_low_bandpower_ratio", "frame_rms_cv_4s",
    },
    "data/results/e3/station_outcomes.csv": {"group", "station_noise", "method", "metric", "station_mean"},
    "data/results/e3/station_covariates.csv": {"group", "station_noise", "n_cases", "noise_rms_full_mean"},
    "data/results/e3/matching_pairs.csv": {"pair_id", "station_A", "station_B", "distance", "caliper", "replacement", "tie_break"},
    "data/results/e3/adjusted_contrasts.csv": {
        "method", "metric", "unadjusted_A_minus_B", "regression_adjusted_A_minus_B", "matched_A_minus_B",
    },
    "data/results/e3/balance_diagnostics.csv": {"covariate", "before_smd", "after_smd", "n_pairs_after"},
    "data/results/e3/direction_consistency.csv": {
        "method", "metric", "direction_consistent", "zero_exclusion_consistent", "interpretation_grade",
    },
    "data/results/e5/per_case_metrics.csv": {
        "case_id", "station_template", "station_noise", "target_snr_db", "method",
        "clean_snr_gain_db", "amplitude_ratio_clean", "waveform_correlation_z", "background_suppression_db",
    },
    "data/results/e5/selected_checkpoints.csv": {
        "seed", "lambda0_method", "lambda05_method", "lambda0_epoch", "lambda05_epoch",
    },
    "data/results/e5/paired_seed_contrasts.csv": {
        "seed", "metric", "lambda0_method", "lambda05_method", "lambda0_mean", "lambda05_mean",
        "difference_lambda05_minus_lambda0",
    },
    "data/results/e5/seed_summary.csv": {
        "metric", "n_paired_seeds", "mean_difference", "median_difference",
        "minimum_difference", "maximum_difference", "direction_consistent",
    },
    "data/results/figure3/observed_acf.csv": {"lag_seconds", "station_median_acf", "station_q25_acf", "station_q75_acf"},
    "data/results/figure3/pointwise_envelope.csv": {"lag_seconds", "lower_2p5", "upper_97p5"},
    "data/results/figure3/simultaneous_studentized_envelope.csv": {
        "lag_seconds", "null_mean_acf", "null_sd_acf", "lower_simultaneous_95", "upper_simultaneous_95",
    },
}


def files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_rows(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    key_columns: tuple[str, ...],
    *,
    tolerance: float = 1e-10,
) -> None:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
        return {tuple(str(row[column]) for column in key_columns): row for row in rows}

    left, right = keyed(expected), keyed(actual)
    if left.keys() != right.keys():
        raise AssertionError(f"key mismatch for {key_columns}")
    for key in left:
        common = set(left[key]).intersection(right[key])
        for column in common:
            a, b = left[key][column], right[key][column]
            if str(a).lower() == str(b).lower():
                continue
            try:
                if math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance):
                    continue
            except (TypeError, ValueError):
                pass
            raise AssertionError(f"mismatch at {key} {column}: {a!r} != {b!r}")


def check_schemas(root: Path) -> None:
    for relative, required in CSV_SCHEMAS.items():
        path = root / relative
        if not path.exists():
            raise AssertionError(f"missing required CSV: {relative}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            fieldnames = set(csv.DictReader(handle).fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise AssertionError(f"{relative} missing columns {sorted(missing)}")
    for path in sorted((root / "configs").glob("*.yml")):
        if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict):
            raise AssertionError(f"invalid YAML mapping: {path.name}")
    reconstructor = load_module(
        root / "scripts" / "reconstruct_controlled_cases.py",
        "verify_manifest_semantics",
    )
    reconstructor.validate_controlled_manifests(
        cases := read_csv(root / "data" / "manifests" / "controlled_mixture_cases.csv"),
        requests := read_csv(root / "data" / "manifests" / "controlled_mixture_requests.csv"),
    )
    request_groups: dict[str, list[dict[str, str]]] = {}
    for row in requests:
        request_groups.setdefault(row["case_id"], []).append(row)
    case_lookup = {row["case_id"]: row for row in cases}
    result_rows = read_csv(
        root / "data" / "results" / "report_cards" / "controlled_mixture_per_case_metrics.csv"
    )
    result_ids = {row["case_id"] for row in result_rows}
    if result_ids != set(case_lookup):
        raise AssertionError("released per-case result IDs do not match controlled manifest")
    results_by_case: dict[str, list[dict[str, str]]] = {}
    for row in result_rows:
        results_by_case.setdefault(row["case_id"], []).append(row)
    for case_id, case in case_lookup.items():
        first_noise = min(
            (row for row in request_groups[case_id] if row["role"] == "noise"),
            key=lambda row: int(row["sequence_index"]),
        )
        matching = results_by_case[case_id]
        if any(row["station_template"] != case["station"] for row in matching):
            raise AssertionError(f"event station mismatch for released case {case_id}")
        if any(row["station_noise"] != first_noise["station"] for row in matching):
            raise AssertionError(f"noise audit station mismatch for released case {case_id}")


def check_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.exists():
        raise AssertionError("missing root SHA256SUMS.txt")
    entries = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in files(root)
        if path != checksum_path
    }
    if set(entries) != expected_paths:
        raise AssertionError("SHA256SUMS.txt path set differs from public tree")
    for relative, expected in entries.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def check_forbidden_content(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        lowered_parts = {part.lower() for part in relative.parts}
        if path.is_dir() and lowered_parts.intersection(BANNED_DIRECTORIES):
            raise AssertionError(f"forbidden directory: {relative}")
        if path.is_file():
            lower_name = path.name.lower()
            if any(fragment in lower_name for fragment in BANNED_NAME_FRAGMENTS):
                raise AssertionError(f"forbidden file name: {relative}")
            if path.suffix.lower() in BANNED_EXTENSIONS:
                raise AssertionError(f"forbidden extension: {relative}")
            if re.match(r"seismic-denoising-eval-protocol_v.*\.zip(?:\.sha256)?$", lower_name):
                raise AssertionError(f"release archive is inside Git tree: {relative}")

    text_extensions = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml", ".toml", ".cff", ""}
    title_hits = []
    for path in files(root):
        if path.suffix.lower() not in text_extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        lowered = text.lower()
        if any(doi in lowered for doi in BANNED_DOIS):
            raise AssertionError(f"deleted DOI found in {relative}")
        if re.search(r"[A-Za-z]:[\\/](?:Users|home)[\\/]", text) or re.search(r"/(?:home|Users)/[^/\s]+/", text):
            raise AssertionError(f"workstation absolute path found in {relative}")
        if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text):
            raise AssertionError(f"secret-like token found in {relative}")
        submission_identifier = "aiig" + "-d-26-" + "00176"
        if submission_identifier in lowered:
            raise AssertionError(f"submission identifier found in {relative}")
        if PUBLIC_TITLE in text:
            title_hits.append(relative)
    disallowed_title_hits = [
        path for path in title_hits
        if path not in {"CITATION.cff", "README.md", "docs/STUDY_MAPPING.md"}
    ]
    if disallowed_title_hits:
        raise AssertionError(f"article title found outside allowed study mapping files: {disallowed_title_hits}")

    for archive in root.rglob("*.zip"):
        raise AssertionError(f"archive embedded in public tree: {archive.relative_to(root)}")


def check_report_cards(root: Path) -> None:
    module = load_module(root / "analysis" / "reproduce_report_cards.py", "verify_report_cards")
    outputs = module.recompute(root)
    result_root = root / "data" / "results" / "report_cards"
    compare_rows(read_csv(result_root / "controlled_mixture_report_card.csv"), outputs["controlled_mixture_report_card.csv"], ("method",))
    compare_rows(read_csv(result_root / "external_real_event_report_card.csv"), outputs["external_real_event_report_card.csv"], ("method",))


def check_bootstrap(root: Path) -> None:
    module = load_module(root / "analysis" / "reproduce_station_bootstrap.py", "verify_bootstrap")
    actual = module.recompute(root)
    expected = read_csv(root / "data" / "results" / "report_cards" / "station_bootstrap_contrasts.csv")
    compare_rows(expected, actual, ("method", "metric"))


def check_e3(root: Path) -> None:
    module = load_module(root / "analysis" / "reproduce_e3_station_domain.py", "verify_e3")
    actual = module.recompute(root)
    result_root = root / "data" / "results" / "e3"
    keys = {
        "matching_pairs.csv": ("pair_id",),
        "adjusted_contrasts.csv": ("method", "metric"),
        "balance_diagnostics.csv": ("covariate",),
        "direction_consistency.csv": ("method", "metric"),
    }
    for name, key in keys.items():
        compare_rows(read_csv(result_root / name), actual[name], key)


def check_e3_quick(root: Path) -> None:
    module = load_module(root / "analysis" / "reproduce_e3_station_domain.py", "verify_e3_quick")
    result_root = root / "data" / "results" / "e3"
    cases = read_csv(result_root / "case_outcomes_and_covariates.csv")
    stations = read_csv(result_root / "station_covariates.csv")
    compare_rows(
        read_csv(result_root / "matching_pairs.csv"),
        module.match_stations(stations),
        ("pair_id",),
    )
    frozen = {
        (row["method"], row["metric"]): row
        for row in read_csv(result_root / "adjusted_contrasts.csv")
    }
    for method in sorted({row["method"] for row in cases}):
        subset = [row for row in cases if row["method"] == method]
        for metric in module.METRICS:
            estimate = module.ols_group_effect(subset, metric)
            expected = float(frozen[(method, metric)]["regression_adjusted_A_minus_B"])
            if not math.isclose(estimate, expected, rel_tol=0, abs_tol=1e-10):
                raise AssertionError(f"E3 quick OLS mismatch for {method} {metric}")


def check_e5(root: Path) -> None:
    module = load_module(root / "analysis" / "reproduce_e5_multiseed.py", "verify_e5")
    contrasts, summary = module.recompute(root)
    result_root = root / "data" / "results" / "e5"
    compare_rows(read_csv(result_root / "paired_seed_contrasts.csv"), contrasts, ("seed", "metric"))
    compare_rows(read_csv(result_root / "seed_summary.csv"), summary, ("metric",))


def check_figure3(root: Path) -> None:
    module = load_module(root / "analysis" / "verify_figure3_surrogate.py", "verify_figure3")
    module.verify(root)


def check_metric_units(root: Path) -> None:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from blindspot_eval_protocol.metrics import (  # pylint: disable=import-outside-toplevel
        amplitude_ratio,
        background_suppression,
        clean_snr_gain,
        covariance_shape_distance,
        normalized_cross_correlation,
    )
    from blindspot_eval_protocol.schemas import validate_waveform  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(7)
    clean = np.zeros((3500, 3), dtype=np.float64)
    time = np.arange(1000) / 100.0
    clean[1000:2000, 0] = np.sin(2 * np.pi * 4 * time)
    clean[1000:2000, 1] = 0.5 * np.sin(2 * np.pi * 6 * time)
    clean[1000:2000, 2] = 0.25 * np.cos(2 * np.pi * 5 * time)
    noisy = clean + rng.normal(0.0, 0.2, clean.shape)
    if abs(clean_snr_gain(noisy, noisy, clean)) > 1e-12:
        raise AssertionError("identity clean-SNR gain is not zero")
    if not math.isclose(amplitude_ratio(0.5 * clean, clean), 0.5, abs_tol=1e-10):
        raise AssertionError("amplitude scaling test failed")
    shifted = np.zeros_like(clean)
    shifted[5:, 0] = clean[:-5, 0]
    corr = normalized_cross_correlation(shifted, clean)
    if corr.correlation < 0.99 or not math.isclose(abs(corr.delay_s), 0.05, abs_tol=1e-12):
        raise AssertionError("known-lag NCC test failed")
    if covariance_shape_distance(clean[1000:2000], clean[1000:2000]) > 1e-12:
        raise AssertionError("covariance identity distance is not zero")
    recovered_half = clean + 0.5 * (noisy - clean)
    if clean_snr_gain(recovered_half, noisy, clean) <= 0:
        raise AssertionError("controlled recovery does not improve clean-SNR")
    long_noisy = rng.normal(size=(9000, 3))
    if abs(background_suppression(long_noisy, long_noisy, hidden_onset_s=30.0)) > 1e-10:
        raise AssertionError("identity background suppression is not zero")
    try:
        validate_waveform(np.zeros((10, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid shape was accepted")
    invalid = np.zeros((10, 3), dtype=float)
    invalid[0, 0] = np.nan
    try:
        validate_waveform(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN waveform was accepted")


def check_identity_release(root: Path) -> None:
    rows = [
        row for row in read_csv(root / "data" / "results" / "report_cards" / "controlled_mixture_per_case_metrics.csv")
        if row["method"] == "Identity"
    ]
    if len(rows) != 816:
        raise AssertionError("Identity row count is not 816")
    if max(abs(float(row["clean_snr_gain_db"])) for row in rows) > 1e-6:
        raise AssertionError("Identity clean-SNR gains exceed numerical tolerance")
    if max(abs(float(row["background_suppression_db"])) for row in rows) > 1e-9:
        raise AssertionError("Identity background suppression exceeds numerical tolerance")


def check_adapter(root: Path) -> None:
    environment = dict(os.environ)
    try:
        completed = subprocess.run(
            [
                sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider",
                str(root / "tests"),
            ],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("unit tests timed out after 120 seconds") from exc
    if completed.returncode:
        raise AssertionError(
            "unit tests failed:\n" + completed.stdout[-4000:] + completed.stderr[-4000:]
        )


def check_fdsn_dry_run(root: Path) -> None:
    fetcher = load_module(root / "scripts" / "fetch_fdsn_windows.py", "verify_fdsn")
    rows = read_csv(root / "data" / "manifests" / "controlled_mixture_requests.csv")[:8]
    requests = fetcher.requests_from_rows(rows)
    if not requests or len(requests) > len(rows):
        raise AssertionError("FDSN dry-run request expansion failed")
    required = {"network", "station", "location", "channel_pattern", "start_time", "end_time", "provider", "fdsn_source"}
    if any(required - set(request) for request in requests):
        raise AssertionError("FDSN request lacks explicit acquisition fields")


def check_no_cache(root: Path) -> None:
    hits = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix.lower() == ".pyc"
    ]
    if hits:
        raise AssertionError(f"Python cache artifacts found: {hits}")


def verify(root: Path, *, full: bool = False) -> list[tuple[str, str]]:
    start_hash = tree_hash(root)
    checks: list[tuple[str, Callable[[Path], None]]] = [
        ("01 csv_and_yaml_schemas", check_schemas),
        ("02 root_checksums", check_checksums),
        ("03_11 forbidden_content_and_archives", check_forbidden_content),
        ("12 report_card_recomputation", check_report_cards),
        ("13 station_bootstrap_recomputation", check_bootstrap),
        ("14 e3_quick_ols_matching_smd", check_e3_quick),
        ("15 e5_paired_seed_recomputation", check_e5),
        ("16 figure3_consistency", check_figure3),
        ("17 synthetic_metric_units", check_metric_units),
        ("18 identity_release_results", check_identity_release),
        ("19 example_adapter", check_adapter),
        ("19b fdsn_dry_run", check_fdsn_dry_run),
        ("22 no_python_cache", check_no_cache),
    ]
    if full:
        checks.insert(6, ("14b e3_full_1000_replicate_recomputation", check_e3))
    results = []
    for name, function in checks:
        started = time.perf_counter()
        function(root)
        results.append((name, f"PASS {time.perf_counter() - started:.3f}s"))
    end_hash = tree_hash(root)
    if start_hash != end_hash:
        raise AssertionError("verifier mutated the package tree")
    results.append(("20 repeatability_external_two_run_requirement", "PASS_THIS_RUN"))
    results.append(("21 tree_hash_unchanged", f"PASS {start_hash}"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    started = time.perf_counter()
    for name, result in verify(root, full=args.full):
        print(f"{name}: {result}")
    print(f"VERIFY_RELEASE: PASS mode={'full' if args.full else 'quick'} total={time.perf_counter() - started:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
