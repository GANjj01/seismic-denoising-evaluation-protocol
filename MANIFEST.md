# Public Export Manifest

Version 1.0.7 was assembled by allowlist from frozen numerical and protocol
sources. No old Git metadata or release archive was copied.

| Old path category | New path | Keep/Delete | Reason |
|---|---|---|---|
| `src/` metric helpers | `src/blindspot_eval_protocol/` | Keep/rewrite | Current formulas and explicit 3C contract |
| `configs/` | `configs/` | Keep/rewrite | Portable frozen settings |
| `splits/` and `indices/` | `data/splits/`, `data/manifests/` | Keep/normalize | Explicit FDSN reconstruction fields |
| Historical case constructor | `scripts/reconstruct_controlled_cases.py` | Keep/rewrite | Auditable manifest-to-case pathway |
| Historical evaluator | `scripts/evaluate_method.py` | Keep/rewrite | Public adapter contract and report-card metrics |
| `per_case_metrics/` | `data/results/report_cards/` | Keep/normalize | Machine-readable recomputation input |
| E3 closure tables | `data/results/e3/` | Keep/normalize | OLS, matching, SMD, and direction audit |
| E5 reconstruction tables | `data/results/e5/` | Keep/normalize | Descriptive paired-seed audit |
| Figure 3 numerical tables | `data/results/figure3/` | Keep/rename | Numerical verification without rendered figures |
| Baseline scripts | `src/blindspot_eval_protocol/baselines.py` | Keep/rewrite | Reference implementations and role labels |
| Training logs and checkpoints | none | Delete | Not required to apply or audit the protocol |
| Legacy scripts and historical releases | none | Delete | Superseded and outside the allowlist |
| Publication and review materials | none | Delete | Outside repository purpose |
| Rendered figures and renderers | none | Delete | Numerical CSV/JSON are the public carrier |
| Duplicate licenses | `LICENSE`, `LICENSE-DATA` | Consolidate | MIT code and CC BY 4.0 derived data |

Root `SHA256SUMS.txt` covers every tracked regular file except itself. The
offline verifier checks the allowlist boundary, manifest semantics, synthetic
construction, adapter behavior, and released recomputations.
