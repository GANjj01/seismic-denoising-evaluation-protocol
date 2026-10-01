# Seismic Denoising Evaluation Protocol

This repository provides an executable evaluation protocol, frozen
machine-readable manifests, derived numerical results, and offline
verification tools for three-component seismic denoising.


## Capability Scope

**Fully offline**

- validate released CSV/JSON, schemas, checksums, and privacy boundaries;
- recompute report cards and station-bootstrap summaries;
- verify E3/E5 and released Figure 3 numerical artifacts; and
- run synthetic reconstruction, metric, and adapter tests.

**Requires externally acquired waveforms**

- reconstruct protocol-conformant controlled cases; and
- evaluate a new denoising method on those reconstructed cases.

Version 1.0.7 provides external real-event result tables and underlying metric
functions for audit, but not a complete batch CLI for new external real events.

**Not distributed**

- raw MiniSEED or other source waveforms;
- third-party model weights or training projects;
- journal submission and review files; and
- rendered publication figures.

FDSN availability may change and provider terms continue to apply. The
reconstruction pathway is protocol-conformant; it is not claimed to be a
byte-identical replay of unpublished historical case arrays.

## Quick Start

Install the base evaluation package:

```bash
python -m pip install .
```

Install test dependencies for offline verification:

```bash
python -m pip install ".[test]"
```

Install FDSN/MiniSEED support for waveform download and reconstruction:

```bash
python -m pip install ".[fetch]"
```

For development with both optional groups:

```bash
python -m pip install -e ".[test,fetch]"
```

Run the normal offline verifier:

```bash
python -B scripts/verify_release.py --package-root . --quick
```

Run the additional full E3 recomputation:

```bash
python -B scripts/verify_release.py --package-root . --full
```

Inspect the deduplicated FDSN acquisition plan without network access:

```bash
python -B scripts/fetch_fdsn_windows.py \
  --manifest data/manifests/controlled_mixture_requests.csv \
  --output-dir <OUTPUT_DIR> \
  --dry-run
```

Reconstruct controlled cases from locally supplied waveforms:

```bash
python -B scripts/reconstruct_controlled_cases.py \
  --package-root . \
  --manifest data/manifests/controlled_mixture_cases.csv \
  --waveform-root <WAVEFORM_ROOT> \
  --output-dir <OUTPUT_DIR>
```

Source waveforms must be MiniSEED (`.mseed` or `.miniseed`). Source-waveform
NPZ input is intentionally rejected. The reconstructor still writes standardized
reconstructed-case NPZ artifacts, which are the NPZ inputs accepted by
`evaluate_method.py`.

Evaluate an adapter that defines `denoise(waveform, sampling_rate_hz)`:

```bash
python -B scripts/evaluate_method.py \
  --adapter examples/identity_adapter.py \
  --cases-dir <CASES_DIR> \
  --reconstruction-manifest <RECONSTRUCTION_MANIFEST> \
  --output-dir <OUTPUT_DIR>
```

The evaluator passes only the `(9000, 3)` `Z,N,E` mixture and sampling rate to
the method. The adapter must preserve shape, alignment, component order, and
physical amplitude scale. Clean/noise references, hidden onset, and target SNR
remain evaluator-held.

## Recompute Released Numbers

Each analysis writes only to a user-selected output directory:

```bash
python -B analysis/reproduce_report_cards.py --package-root . --output-dir <OUTPUT_DIR>
python -B analysis/reproduce_station_bootstrap.py --package-root . --output-dir <OUTPUT_DIR>
python -B analysis/reproduce_e3_station_domain.py --package-root . --output-dir <OUTPUT_DIR>
python -B analysis/reproduce_e5_multiseed.py --package-root . --output-dir <OUTPUT_DIR>
python -B analysis/verify_figure3_surrogate.py --package-root . --output-dir <OUTPUT_DIR>
```

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the construction and scoring
rules, and [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for public
schemas.

## Baseline Roles

The idealized Wiener implementation is a source-aware upper-bound diagnostic,
not a deployable baseline. DeepDenoiser and CovNorm rows are released evaluation
results only; third-party weights and training projects are not distributed.

## License and Citation

Source code is MIT licensed. Released derived metrics and manifests use
`LICENSE-DATA` (CC BY 4.0). External waveforms remain under provider terms.
Software citation metadata for version 1.0.7 are in `CITATION.cff`; no DOI or
release date is asserted before formal publication of this public repository.
