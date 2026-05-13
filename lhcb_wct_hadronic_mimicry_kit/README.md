# LHCb WCT Hadronic Mimicry Test Kit

This directory contains stress tests for a specific failure mode: a WCT/log-periodic ansatz might falsely identify structured charm-tail pseudo-data as a target log-periodic signal.

The tests here are **synthetic-injection and pseudo-experiment checks**. They are meant to support the main analysis by testing whether charm-only Breit-Wigner-like residuals can mimic the target WCT pattern.

Run commands from this directory unless you pass explicit paths:

```bash
cd lhcb_wct_hadronic_mimicry_kit
```

Several driver scripts call sibling scripts by relative filename, so running them from the repository root can fail.

---

## Install

Core packages:

```bash
pip install numpy pandas scipy matplotlib uproot awkward
```

`uproot` and `awkward` are only needed for the optional ROOT/ntuple-loading skeleton. The injection, scanner, strong test, and sweep use standard scientific Python packages.

---

## Main scripts

| Script | Purpose |
|---|---|
| `07_inject_charm_mimicry.py` | Generate synthetic charm-only residuals in linear q2 using J/psi and psi(2S) Breit-Wigner-like tails. |
| `08_scan_wct_on_injections.py` | Fit constant, charm-tail, and bounded WCT/log-periodic models; run Monte Carlo pseudo-experiments; report false-positive rates. |
| `09_locked_k_cross_channel.py` | Test a fixed k value across independent CSV channels without floating k. |
| `10_unbinned_likelihood_skeleton.py` | Experimental skeleton for future ntuple-level unbinned likelihood work. |
| `11_global_pvalue_look_elsewhere.py` | Convert local scan improvements into global look-elsewhere p-values using pseudo-experiments. |
| `12_run_strong_mimicry_test.py` | Main strong mimicry test: generate charm-only data, scan WCT-vs-charm, run MC injections, and write PASS/FAIL report. |
| `13_sweep_mimicry_strength.py` | Robustness sweep over charm-tail strength and noise. |
| `run_demo.py` | Small demo: one fake charm dataset plus a short MC scan. |

---

## Expected CSV format

Minimum columns for real, binned, or synthetic input:

```csv
q2,y,sigma
```

where:

- `q2` is dilepton invariant mass squared in GeV^2,
- `y` is the residual or observable value,
- `sigma` is the uncertainty.

Optional columns:

- `channel`
- `bin_low`
- `bin_high`

---

## Quick start

Generate one fake charm-only dataset:

```bash
python 07_inject_charm_mimicry.py --out fake_charm.csv --n-bins 16 --seed 1
```

Run the WCT scan on that injection:

```bash
python 08_scan_wct_on_injections.py --input fake_charm.csv --n-mc 500 --out-dir mimicry_results
```

Run the small end-to-end demo:

```bash
python run_demo.py
```

The demo writes:

```text
fake_charm.csv
demo_results/fit_comparison.csv
demo_results/mc_false_positive_rows.csv
demo_results/mc_false_positive_summary.json
demo_results/scan_stats.json
```

---

## Strong mimicry test

Run the main strong test:

```bash
python 12_run_strong_mimicry_test.py --n-mc 1000 --target-k 11.7 --out-dir strong_mimicry_results
```

Windows shortcut:

```bat
RUN_STRONG_TEST.bat
```

Main outputs:

```text
strong_mimicry_results/STRONG_MIMICRY_REPORT.md
strong_mimicry_results/fake_charm.csv
strong_mimicry_results/fit_comparison.csv
strong_mimicry_results/mc_false_positive_rows.csv
strong_mimicry_results/mc_false_positive_summary.json
strong_mimicry_results/scan_stats.json
```

Default pass criteria:

```text
false_positive_rate_vs_charm <= 0.01
false_positive_rate_vs_charm_and_k <= 0.002
wct_preferred_over_charm_by_bic_rate <= 0.05
```

These thresholds can be changed intentionally with:

```bash
python 12_run_strong_mimicry_test.py \
  --max-fp-charm 0.01 \
  --max-fp-charm-k 0.002 \
  --max-bic-rate 0.05
```

---

## Sweep test

Run a robustness sweep across charm-tail strength and noise:

```bash
python 13_sweep_mimicry_strength.py --n-mc 200 --target-k 11.7 --out-dir mimicry_sweep_results
```

Windows shortcut:

```bat
RUN_SWEEP_TEST.bat
```

Default grid:

```text
charm_scale = 0.15, 0.25, 0.35, 0.50, 0.75
sigma       = 0.10, 0.15, 0.20
```

Override the grid with:

```bash
python 13_sweep_mimicry_strength.py \
  --charm-scales 0.15 0.25 0.35 \
  --sigmas 0.10 0.15 \
  --n-mc 200
```

Main table:

```text
mimicry_sweep_results/mimicry_sweep_summary.csv
```

Each sweep cell also writes a full strong-test result directory.

---

## Locked-k and global-pvalue utilities

Run a locked-k test across several channel CSVs:

```bash
python 09_locked_k_cross_channel.py --k 11.7 --inputs B0Kst.csv BpK.csv BsPhi.csv --out locked_k_results.csv
```

Compute a global look-elsewhere p-value from an input CSV and observed local improvement:

```bash
python 11_global_pvalue_look_elsewhere.py \
  --input fake_charm.csv \
  --observed-delta 9.0 \
  --n-mc 1000 \
  --k-min 2.0 \
  --k-max 25.0 \
  --out global_pvalue.json
```

---

## Optional unbinned skeleton

The unbinned likelihood script is a skeleton for future ntuple-level work. It can load either CSV or ROOT inputs:

```bash
python 10_unbinned_likelihood_skeleton.py --csv fake_charm.csv
python 10_unbinned_likelihood_skeleton.py --root data/job0.root --tree DecayTree
```

Treat this as scaffolding, not a paper-grade likelihood implementation.


## Scanner note

The scanner uses bounded optimization for polishing so that the reported best `k` stays inside the requested scan range. This fixes an earlier demo behavior where an unconstrained Nelder-Mead polish could report a best `k` outside the bounded differential-evolution range.