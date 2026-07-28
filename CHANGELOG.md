# Changelog

All notable changes to `tsdml` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-28

First public release. Implements Ciganovic, D'Amario and Tancioni (2026),
*Double Machine Learning for Time Series*, The Econometrics Journal
(arXiv:2603.10999).

### Added

**Estimators**
- `ReverseCrossFitting` — RCF-DML for the partially linear model: deterministic
  block cross-fitting with time-reversed auxiliary samples and no buffer
  deletion, fold-average point estimate (eq. 2.5) and HAC inference from the
  stacked score sequence (Theorem 2.1).
- `NLOCrossFitting` — the Neighbors-Left-Out benchmark of Semenova et al. (2023).
- `DMLLocalProjections` — horizon-specific dynamic effects with automatic
  cumulation of differenced outcomes (eq. 5.18) and policy-residual caching
  across horizons.

**Tuning**
- `Calibrator` with the `goldilocks_zone` criterion of Section 3, plus the
  standard `rmse` criterion for comparison; per-fold validation blocks carved
  from the auxiliary sample so tuning stays measurable with respect to
  `F_aux,k`.
- `goldilocks_select` as a standalone function on any RMSE profile.

**Folds and inference**
- `reverse_cf_folds`, `nlo_folds`, `fold_direction`, `BlockStructure`.
- `sample_use_rcf` / `sample_use_nlo` — the closed forms of Section 2.1.
- `hac_lrv` with Bartlett, Quadratic Spectral, Parzen and EWC estimators.
- `compute_hac_bandwidth` with the paper's `min(h+1, 24)` default plus Andrews
  (1991), Newey-West (1994), power rules and the LLSW (2018) rules.
- `fixed_b_critical_value` — Kiefer-Vogelsang (2005) for Bartlett and the exact
  Student-t fixed-b distribution for EWC.

**Data**
- `load_macroprudential` — the quarterly Italian dataset of Section 5,
  2005Q2-2024Q4, 37 series.
- `macroprudential_spec` — the paper's exact specification, ready to unpack.
- `DataProcessor` — transformation codes, fast/slow contemporaneous timing,
  lags, leads and fold-divisible truncation.
- `simulate_svar` — the recursive SVAR of eq. (4.13), in both the mis-specified
  benchmark and the correctly specified ordering.
- `simulate_plr` — a transparent partially linear design, with
  `resid_persistence` to break conditional stability on purpose.

**Diagnostics**
- `boundary_leakage_test` — do adjacent blocks predict main-block residuals?
- `residual_autocorrelation_test` — Ljung-Box on cross-fitted residuals.
- `diagnose` — both, on a fitted estimator.

**Output**
- `plot_irf`, `plot_irf_panel` (Figure 2), `plot_irf_comparison`,
  `plot_block_structure` (Figure 1), `plot_goldilocks_profile`,
  `plot_sample_use`, `plot_residuals`.
- `irf_table`, `estimation_table`, `calibration_table`, `sample_use_table`,
  `to_latex` (booktabs), `to_markdown`.
- `use_journal_style` — serif, spine-trimmed, colourblind-safe, editable-PDF
  styling applied by every plotting function.

**Documentation**
- `README.md` with a full user guide.
- `docs/STEP_BY_STEP_GUIDE.md` — a thirteen-step annotated walkthrough from raw
  CSV to submitted figure.
- Three runnable examples under `examples/`.

### Verified

Checked bit-for-bit against the authors' replication package on the paper's own
data and specification: data-preparation arrays, estimation index, stage-one
residuals, the point estimate and standard error, the Goldilocks-selected
penalties for all six folds and both equations, and the scaled impulse responses
for all five outcomes and nine horizons. Regression tests in
`tests/test_paper_replication.py` pin these values.
