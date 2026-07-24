# Data Dictionary

All text tables are UTF-8 CSV with a header row. Boolean values are serialized
as `true` or `false`; missing numeric values are empty. Times are UTC ISO 8601.

## Splits

### `data/splits/train_validation_stations.csv`
### `data/splits/development_stations.csv`
### `data/splits/final_stations.csv`

`network`, `station`, `location`, `channel_pattern`, `provider`, `fdsn_source`,
and `split_role` identify the frozen station sets and acquisition endpoint.

## Case Manifests

### `data/manifests/controlled_mixture_cases.csv`

One row per controlled mixture. `case_id`, event request fields, noise request
fields, `target_snr_db`, `hidden_onset_s`, `sampling_rate_hz`,
`preprocessing_config_id`, and `deterministic_pairing_id` define the case.
Semicolon-separated noise fields preserve the ordered source-window sequence.
The fields `noise_used_samples` and `noise_output_sample_start/stop` make the
3,500 + 3,500 + 2,000-sample construction explicit.

### `data/manifests/controlled_mixture_requests.csv`

Normalized one-request-per-row representation. `request_id` is unique per case;
`source_window_id` identifies a deduplicated acquisition window. `role`,
`sequence_index`, `construction_role`, `required_samples`, `used_samples`, and
`output_sample_start/stop` specify the event request and three noise chunks.
Repeated source windows across target-SNR siblings are intentional.

## User-Generated Reconstruction Outputs

`scripts/reconstruct_controlled_cases.py` writes outside the package. Each case
NPZ contains `mixture`, `clean_reference`, `noise_reference`,
`sampling_rate_hz`, `component_order`, `hidden_onset_sample`,
`scoring_window_samples`, `target_snr_db`, and `case_id`.

Its `reconstruction_manifest.csv` records case/artifact hashes, sample count,
component order, event and noise request IDs, source-window IDs and hashes,
target SNR, achieved SNR, and preprocessing/construction/manifest hashes.

### `data/manifests/external_real_event_cases.csv`

One row per event window. Request columns explicitly contain network, station,
location, channel pattern, start/end time, provider, and FDSN source. `p_time_s`
and `sampling_rate_hz` define scoring alignment.

## Report Cards

### `data/results/report_cards/controlled_mixture_per_case_metrics.csv`

Identifiers and method output metrics: `clean_snr_gain_db`,
`amplitude_ratio_clean`, `waveform_correlation_z`, `delay_s`, and
`background_suppression_db`.

### `data/results/report_cards/controlled_mixture_report_card.csv`

Deterministic method aggregation with case count, clean-SNR mean, clean
amplitude-ratio median, waveform-correlation mean, delay mean, and background
suppression mean.

### `data/results/report_cards/external_real_event_per_case_metrics.csv`

Real-event method rows with `apparent_snr_gain_db`, `amplitude_ratio_raw`,
`trigger_delay_s`, and three PSD-band diagnostics.

### `data/results/report_cards/external_real_event_report_card.csv`

Method-level case means for apparent-SNR gain, raw-input amplitude ratio, and
trigger delay.

### `data/results/report_cards/station_bootstrap_contrasts.csv`

Paired method-minus-Identity station-bootstrap estimates, percentile 95%
intervals, zero-exclusion indicator, replicate count, and seed.

## E3 Station Domain

### `data/results/e3/case_outcomes_and_covariates.csv`

Package-local no-taper case outcomes joined to the six frozen adjustment
covariates. It is the sufficient public input for E3 OLS and station bootstrap.

### `data/results/e3/station_outcomes.csv`

Equal-weight station means by group, method, and metric.

### `data/results/e3/station_covariates.csv`

Station-level noise summaries used for greedy matching and balance checks.

### `data/results/e3/matching_pairs.csv`

Frozen group-A/group-B station pairs, standardized Euclidean distance, caliper,
replacement policy, and tie-break rule.

### `data/results/e3/adjusted_contrasts.csv`

Unadjusted, OLS-adjusted, and matched A-minus-B estimates with frozen bootstrap
intervals and design metadata.

### `data/results/e3/balance_diagnostics.csv`

Pre/post matching group summaries, standardized mean differences, and pair
counts.

### `data/results/e3/direction_consistency.csv`

Direction and zero-exclusion consistency across the three E3 sensitivity
layers.

## E5 Multi-Seed

### `data/results/e5/per_case_metrics.csv`

Case-level no-taper outcomes for the six selected lambda/seed checkpoints.

### `data/results/e5/selected_checkpoints.csv`

Paired seed, method identifier, and selected epoch for lambda 0 and lambda 0.5.

### `data/results/e5/paired_seed_contrasts.csv`

Within-seed means and lambda-0.5-minus-lambda-0 differences for four metrics.

### `data/results/e5/seed_summary.csv`

Descriptive mean, median, range, and sign consistency across three paired seeds.

## Figure 3 Numerical Artifacts

### `data/results/figure3/observed_acf.csv`

Lag in seconds and observed station-median/q25/q75 autocorrelation.

### `data/results/figure3/pointwise_envelope.csv`

Lag-wise 2.5% and 97.5% filtered-white-surrogate quantiles.

### `data/results/figure3/simultaneous_studentized_envelope.csv`

Lag-wise null mean/SD and simultaneous 95% studentized bounds.

### `data/results/figure3/maximum_absolute_acf_summary.json`

Observed maximum absolute nonzero-lag ACF and its maximum-absolute null
threshold.

### `data/results/figure3/global_deviation_summary.json`

Studentized q95, observed global statistic, maximum-deviation lag, and
simultaneous exceedance count.

### `data/results/figure3/surrogate_summary.json`

Monte Carlo replicate/seed metadata and high-level exceedance counts.
