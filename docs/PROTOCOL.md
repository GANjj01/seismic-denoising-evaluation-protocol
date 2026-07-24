# Evaluation Protocol

## Preprocessing

Waveforms have shape `(n_samples, 3)` and component order `Z,N,E`. Frozen cases
use 100 Hz. Each source window is demeaned and linearly detrended during loading,
then filtered separately with a fourth-order 1-20 Hz zero-phase Butterworth
band-pass.

An event source window has 3,500 samples with P at sample 1,000. The
evaluator-held, untapered reference is formed by zeroing samples before P and
after P+20 s. The 0.5 s onset ramp belongs only to the released target-sensitivity
diagnostic and is not used by the primary reconstruction.

## Controlled Case Construction

Each case uses one event request and exactly three ordered noise requests from
`controlled_mixture_requests.csv`. Each noise request is loaded and filtered as
a 3,500-sample chunk before concatenation. The first 3,500 samples of chunk 1,
the first 3,500 of chunk 2, and the first 2,000 of chunk 3 fill output ranges
`[0,3500)`, `[3500,7000)`, and `[7000,9000)`. There is no gap, overlap, or
crossfade. A component-wise median is removed after concatenation.

`hidden_onset_s` is the P onset in the full 9,000-sample mixture. Seconds are
converted with `round(seconds * sampling_rate)`. The injected event segment is
explicitly extracted from event-reference samples `[1000,3000)`.

Target-SNR scaling uses three-component RMS:

- event RMS: the first 1,000 samples of the extracted event, P to P+10 s;
- noise RMS: 1,000 continuous-noise samples beginning at the hidden onset; and
- scale: `event_rms / (10**(target_snr_db/20) * noise_rms)`.

The entire 90 s noise trace is scaled, and the 20 s event is inserted at the
hidden onset. The scoring crop begins 10 s before onset and contains 3,500
samples. Any source, injection, RMS, or scoring window that exceeds its array
raises an error; it is never silently truncated.

Pseudocode:

```text
event = bandpass(load(event_request, 3500))
reference = zero_outside(event, P=1000, P_plus_20=3000)
chunks = [bandpass(load(noise_request_i, 3500)) for i in 1..3]
noise = concatenate(chunks[0], chunks[1], chunks[2][:2000])
noise = noise - component_median(noise)
scale = rms(reference[P:P+10s]) / (10^(target_snr/20) * rms(noise[onset:onset+10s]))
mixture = scale * noise
mixture[onset:onset+20s] += reference[P:P+20s]
score_crop = mixture[onset-10s:onset-10s+35s]
```

Source-window requests can be shared across target-SNR sibling cases. The
released `station_noise` is the station of the first ordered noise chunk and is
an audit grouping label; all three chunks participate in construction.

## Station-Disjoint Design

Training/internal-validation, development, and final station lists are frozen
in `data/splits/`. Final-set stations are disjoint from the other roles. No
checkpoint selection, hyperparameter tuning, normalization fitting, threshold
fitting, or method adaptation may use final-set outcomes.

## Controlled Scoring

- **Clean-SNR gain:** output minus noisy-input three-component error SNR over
  P to P+10 s against the evaluator-held reference.
- **Amplitude ratio:** output/reference Z-component q95 absolute amplitude over
  P to P+2 s.
- **Waveform correlation and delay:** best mean-removed Z-component NCC over
  P to P+10 s, with lag restricted to plus or minus 1 s.
- **Background suppression:** `20 log10(rms(input background) /
  rms(output background))`, excluding 2 s before onset through 2 s after the
  injected event.

Covariance and polarization functions are available as optional diagnostics;
the primary `evaluate_method.py` output does not claim to compute them.

## External Real Events

External real-event scoring has no clean or pseudo-clean target. Apparent SNR
uses Z-component P-to-P+10 s RMS against pre-P RMS, and is interpreted jointly
with raw-input amplitude ratio and trigger delay. Version 1.0.6 distributes
released CSV results and metric functions for this track, but no end-to-end
batch evaluator for new external-event waveforms.

## Aggregation and Uncertainty

The controlled report card uses case means except for clean-reference amplitude
ratio, which uses the median. Primary paired comparisons use matched complete
cases, within-station means, and equal station weights. The frozen interval is
a two-sided 95% percentile interval with 20,000 replicates and seed 20260611.

E3 independently resamples different station groups. Its OLS and matching
analyses are sensitivity layers, not causal estimates. E5 is a descriptive
paired comparison across three seeds, not an inferential population estimate.
These diagnostics are not folded into a primary cross-method ranking.

## Baseline Roles and Interpretation

Identity is the noisy-input baseline. Band-pass is a signal-processing
reference. Adversarial scale/shrink controls expose metric failure modes.
Wiener-blind estimates noise from low-energy input frames. Idealized Wiener is
source-aware and is only an upper-bound diagnostic.

Target-dependent results are conditional on the evaluator-held reference.
Apparent SNR and background suppression can reward attenuation, so they must be
read with amplitude and waveform-fidelity metrics. Results from different
supervision regimes or diagnostic roles are not a unified ranking.
