# Seismic Denoising Evaluation Protocol

This repository provides evaluation code, frozen configurations,
machine-readable case manifests, derived numerical results, and verification
tools for three-component seismic denoising.

Journal manuscripts, supplementary manuscripts, response letters, cover
letters, and publication-layout figures are not distributed.

## Purpose

The package supports two practical workflows:

1. evaluate a new denoising method under a fixed three-component contract; and
2. recompute released report cards and diagnostic E3/E5 results from CSV/JSON.

It is an analysis and reproducibility repository, not a model-training project
or a publication archive.

## Waveform Contract

Adapters receive one NumPy array with shape `(n_samples, 3)`, floating dtype,
finite values, columns ordered `Z, N, E`, and sampling rate supplied explicitly
(100 Hz for the frozen cases). They must return the same shape, sample alignment,
component order, and physical amplitude scale. Hidden normalization or
case-dependent rescaling must be disclosed by the adapter.

## Scoring Tracks

- **Controlled mixtures:** evaluator-held untapered-onset references support
  clean-SNR gain, amplitude ratio, waveform correlation, background suppression,
  covariance shape, and polarization diagnostics.
- **External real events:** apparent-SNR, raw-input amplitude ratio, and trigger
  delay are reported without a clean reference.
- **E3/E5 diagnostics:** station-domain adjustment/matching and descriptive
  paired-seed contrasts are kept separate from the primary report card.

The idealized Wiener implementation is a **source-aware upper-bound diagnostic**,
not a deployable baseline. DeepDenoiser and CovNorm rows are released as
evaluation results only; third-party weights and training projects are not
distributed.

## Quick Verification

From the repository root:

```bash
python -B scripts/verify_release.py --package-root .
python -B -m pytest -p no:cacheprovider
```

The verifier is offline, does not write into the package, and checks schemas,
checksums, forbidden content, released recomputations, and adapter behavior.

## Evaluate a New Method

Use an adapter that defines `denoise(waveform, sampling_rate_hz)`:

```bash
python -B scripts/evaluate_method.py \
  --adapter examples/identity_adapter.py \
  --manifest data/manifests/controlled_mixture_cases.csv \
  --input-dir /path/to/reconstructed_cases \
  --output-dir /path/to/evaluation_output
```

Waveform evaluation requires locally reconstructed case arrays. The released
CSV report cards can be recomputed without raw waveforms.

## Fetch External Waveforms

Raw waveforms are not redistributed. Inspect explicit requests without network
access:

```bash
python -B scripts/fetch_fdsn_windows.py \
  --manifest data/manifests/external_real_event_cases.csv \
  --output-dir /path/to/waveforms \
  --dry-run
```

FDSN availability may change, waveform-level reconstruction depends on external
services, and the original provider terms remain applicable.

## Recompute Released Numbers

Each analysis writes only to a user-selected output directory:

```bash
python -B analysis/reproduce_report_cards.py --package-root . --output-dir /tmp/report_cards
python -B analysis/reproduce_station_bootstrap.py --package-root . --output-dir /tmp/bootstrap
python -B analysis/reproduce_e3_station_domain.py --package-root . --output-dir /tmp/e3
python -B analysis/reproduce_e5_multiseed.py --package-root . --output-dir /tmp/e5
python -B analysis/verify_figure3_surrogate.py --package-root . --output-dir /tmp/figure3
```

Protocol details are in [docs/PROTOCOL.md](docs/PROTOCOL.md), and every public
table is defined in [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Not Distributed

The repository excludes raw MiniSEED, tensors, model weights, training logs,
publication source, review correspondence, publication-layout figures, and
historical release archives.

## License

Source code is MIT licensed. Released derived metrics and manifests are covered
by `LICENSE-DATA` (CC BY 4.0). External waveforms remain under their original
provider terms.

## Citation

Citation metadata for version 1.0.6 are in `CITATION.cff`. A DOI is intentionally
omitted until a future archival record is created from the reviewed public
release.
