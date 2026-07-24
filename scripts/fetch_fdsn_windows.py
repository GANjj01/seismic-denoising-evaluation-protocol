"""Fetch manifest-defined waveform windows from FDSN, or print a dry run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _normalized_requests(
    rows: Iterable[dict[str, str]], provider_override: str | None
) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = row["source_window_id"]
        request = {
            "source_window_id": source_id,
            "network": row["network"],
            "station": row["station"],
            "location": row["location"],
            "channel_pattern": row["channel"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "event_noise_role": row["role"],
            "provider": provider_override or row["provider"],
            "fdsn_source": row["fdsn_source"],
        }
        if source_id in unique and unique[source_id] != request:
            raise ValueError(f"source_window_id {source_id} maps to inconsistent requests")
        unique[source_id] = request
    return [unique[key] for key in sorted(unique)]


def requests_from_rows(
    rows: Iterable[dict[str, str]], provider_override: str | None = None
) -> list[dict[str, str]]:
    rows = list(rows)
    if rows and "source_window_id" in rows[0] and "role" in rows[0]:
        return _normalized_requests(rows, provider_override)

    requests: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        common = {
            "source_window_id": row.get("event_source_id", ""),
            "network": row["network"],
            "station": row["station"],
            "location": row["location"],
            "channel_pattern": row["channel_pattern"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "event_noise_role": row["event_noise_role"],
            "provider": provider_override or row["provider"],
            "fdsn_source": row["fdsn_source"],
        }
        key = tuple(common[name] for name in ("network", "station", "location", "channel_pattern", "start_time", "end_time"))
        requests[key] = common
        if row.get("noise_station"):
            names = ("network", "station", "location", "channel_pattern", "start_time", "end_time", "provider", "fdsn_source")
            columns = {name: split_values(row[f"noise_{name}"]) for name in names}
            source_ids = split_values(row.get("noise_source_id", ""))
            lengths = {len(values) for values in columns.values()}
            if len(lengths) != 1:
                raise ValueError(f"noise request columns have unequal lengths for case {row['case_id']}")
            for index in range(next(iter(lengths))):
                item = {
                    "source_window_id": source_ids[index] if source_ids else "",
                    "network": columns["network"][index],
                    "station": columns["station"][index],
                    "location": columns["location"][index],
                    "channel_pattern": columns["channel_pattern"][index],
                    "start_time": columns["start_time"][index],
                    "end_time": columns["end_time"][index],
                    "event_noise_role": "noise",
                    "provider": provider_override or columns["provider"][index],
                    "fdsn_source": columns["fdsn_source"][index],
                }
                key = tuple(item[name] for name in ("network", "station", "location", "channel_pattern", "start_time", "end_time"))
                requests[key] = item
    return [requests[key] for key in sorted(requests)]


def output_name(request: dict[str, str]) -> str:
    if request.get("source_window_id"):
        return f"{request['source_window_id']}.mseed"
    compact = request["start_time"].replace("-", "").replace(":", "").replace(".", "").replace("Z", "")
    return f"{request['network']}.{request['station']}.{request['location']}.{request['event_noise_role']}.{compact}.mseed"


def run(manifest: Path, output_dir: Path, provider: str | None, dry_run: bool) -> list[dict[str, Any]]:
    requests = requests_from_rows(read_csv(manifest), provider)
    records: list[dict[str, Any]] = []
    if not dry_run:
        try:
            from obspy import UTCDateTime
            from obspy.clients.fdsn import Client
        except Exception as exc:
            raise SystemExit(f"ObsPy is required for downloads: {exc}") from exc
        output_dir.mkdir(parents=True, exist_ok=True)
        clients: dict[str, Any] = {}

    for request in requests:
        destination = output_dir / output_name(request)
        record = {**request, "output_path": str(destination)}
        records.append(record)
        print(json.dumps(record, sort_keys=True))
        if not dry_run:
            client = clients.setdefault(request["provider"], Client(request["provider"]))
            stream = client.get_waveforms(
                request["network"],
                request["station"],
                request["location"],
                request["channel_pattern"],
                UTCDateTime(request["start_time"]),
                UTCDateTime(request["end_time"]),
            )
            stream.write(str(destination), format="MSEED")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Raw waveforms are not redistributed. FDSN availability may change, "
            "waveform-level reconstruction depends on external services, and provider terms apply."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", help="Override each manifest provider")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = run(args.manifest.resolve(), args.output_dir.resolve(), args.provider, args.dry_run)
    print(f"PASS: {len(records)} unique request(s); downloaded={not args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
