# Evaluation Protocol

## Preprocessing

Waveforms have shape `(n_samples, 3)` with columns `Z, N, E`. Frozen cases use
100 Hz sampling. Each component is demeaned, linearly detrended, and filtered
with a fourth-order 1-20 Hz Butterworth band-pass applied by zero-phase SOS
filtering.

Event-centered windows are 35 s long with P at 10 s (`p_index=1000`). The
evaluator-held reference is formed from the filtered event by zeroing samples
before P and after P+20 s. The primary reference applies no onset taper. A
0.5 s post-P ramp belongs only to the frozen target-sensitivity comparison.

## Controlled Reference-Based Mixtures

An event segment is inserted into a 90 s continuous-noise trace at a hidden
onset. Noise is scaled from the three-component RMS over the first 10 s of the
event to target SNR levels -5, 0, and +5 dB. Methods receive only the mixture.
They do not receive the hidden onset, target SNR, clean reference, or exact
noise realization.

Background scoring excludes the interval from 2 s before onset through 2 s
after the 20 s injected event. Target-dependent metrics use a 35 s crop with P
at 10 s.

## External Real Events

External real-event scoring has no clean or pseudo-clean target. Apparent SNR
compares Z-component RMS over P to P+10 s with pre-P RMS. It is reported jointly
with Z-component two-second q95 amplitude ratio and trigger delay. Apparent SNR
alone is not evidence of waveform recovery.

## Metric Definitions

- **Clean-SNR gain:** three-component RMS-error SNR over P to P+10 s against the
  evaluator-held reference, minus the same SNR for the noisy input.
- **Amplitude ratio:** ratio of Z-component q95 absolute amplitude over P to
  P+2 s. The reference is pseudo-clean for controlled mixtures and raw input for
  real events.
- **Waveform correlation:** best mean-removed Z-component NCC over P to P+10 s
  with lag restricted to plus or minus 1 s; the maximizing lag is the delay.
- **Background suppression:** `20 log10(rms(input background) /
  rms(output background))` across all three components.
- **Covariance-shape distance:** Frobenius distance between trace-normalized
  three-component covariance matrices over P to P+10 s.
- **Polarization:** covariance eigenvalue rectilinearity and planarity, plus
  principal-axis angle with eigenvector sign ambiguity removed.

## Aggregation and Uncertainty

The report card uses case means except for clean-reference amplitude ratio,
which uses the median. Primary paired comparisons are matched complete cases,
averaged within station, then bootstrapped with equal station weights. The
frozen primary interval is a two-sided 95% percentile interval with 20,000
replicates and seed 20260611.

E3 compares different station groups and therefore independently resamples
group-A and group-B stations. Its OLS and matching analyses are sensitivity
layers, not causal estimates. E5 is a descriptive paired comparison across
three seeds and is not an inferential population estimate.

## Baseline Roles

Identity checks that a method improves on passing the input through. Band-pass
is a simple signal-processing reference. Adversarial scale/shrink controls
expose metric failure modes. Wiener-blind estimates noise from low-energy input
frames. Idealized Wiener is source-aware and is used only as an upper-bound
diagnostic.

## Final-Set Restrictions

No checkpoint selection, hyperparameter tuning, normalization fitting,
threshold fitting, or method adaptation may use final-set outcomes. Any method
evaluated with this package must freeze its adapter and configuration before
final scoring.

## Interpretation Limits

Target-dependent results are conditional on the specified evaluator-held
reference. Apparent SNR and background suppression can reward attenuation, so
they must be interpreted with amplitude and waveform-fidelity metrics. Results
from different supervision regimes or diagnostic roles are not a unified
method ranking.
