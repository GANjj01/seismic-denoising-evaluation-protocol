# v1.0.7 Curated Public Protocol Release

This release is a clean allowlist export for applying and auditing the
three-component seismic-denoising evaluation protocol.

It includes:

- current metric, preprocessing, controlled-case construction, and adapter APIs;
- frozen configurations, station splits, valid FDSN requests, and normalized
  controlled-case construction manifests;
- an external-waveform reconstruction CLI and standardized reconstructed-case
  NPZ outputs;
- controlled-mixture and external real-event report-card data;
- no-taper E3 station-domain and E5 paired-seed recomputation inputs;
- verification of released Figure 3 numerical artifacts; and
- quick/full offline verification plus synthetic tests.

It excludes raw waveforms, tensors, model weights, training logs, publication
sources, review correspondence, rendered figures, and historical release
archives. No archival DOI is assigned in this package.

## Correction relative to v1.0.6

Corrected the revision-stage untapered-onset N2V-sync report-card and
station-bootstrap artifacts to use the frozen development-selected epoch-10
checkpoint rather than best.pt (epoch 19). The originally submitted
tapered-onset results and other released analyses were unaffected.

Affected released files:

- `data/results/report_cards/controlled_mixture_per_case_metrics.csv`
  (the 816 `N2V-sync` rows)
- `data/results/report_cards/controlled_mixture_report_card.csv`
  (the `N2V-sync` row)
- `data/results/report_cards/station_bootstrap_contrasts.csv`
  (the three `N2V-sync` vs `Identity` contrasts)

Corrected N2V-sync versus Identity, station-level paired bootstrap,
B = 20000, seed 20260611, 34 stations, 816 matched cases:

| Metric | v1.0.6 | v1.0.7 |
| --- | --- | --- |
| clean-SNR gain (dB) | +0.200 [+0.138, +0.265] | +0.018 [-0.067, +0.111] |
| waveform-correlation difference | -0.0026 [-0.0042, -0.0011] | -0.0028 [-0.0047, -0.0009] |
| background-suppression difference (dB) | +0.318 [+0.154, +0.521] | +0.141 [-0.067, +0.397] |

The summary tables remain fully derivable from the released per-case data with
the packaged `analysis/reproduce_report_cards.py` and
`analysis/reproduce_station_bootstrap.py`.
