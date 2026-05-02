# LHCb WCT Hadronic Mimicry Test Kit

Purpose: test whether a WCT log-periodic ansatz falsely "discovers" structure in Standard-Model-only charm-tail pseudo-data.

## Main scripts

1. `07_inject_charm_mimicry.py`
   - Generates synthetic charm-loop residuals in linear q^2.
   - Uses Breit-Wigner tails for J/psi and psi(2S).
   - Saves pseudo-data to CSV.

2. `08_scan_wct_on_injections.py`
   - Fits the 5-parameter WCT log-periodic model to injected fake data.
   - Compares against constant-shift and charm-tail models.
   - Runs Monte Carlo injections and reports false-positive rate.

3. `09_locked_k_cross_channel.py`
   - Tests a fixed k value across independent CSV channels.
   - Does not float k.
   - Fits only amplitude/phase/envelope unless locked.

4. `10_unbinned_likelihood_skeleton.py`
   - Skeleton for future ntuple-level unbinned extended likelihood.
   - Includes optional uproot loading hooks.

5. `11_global_pvalue_look_elsewhere.py`
   - Converts local scan improvements into global p-values using pseudo-experiments.

## Expected CSV format for real or binned input

Minimum columns:

```csv
q2,y,sigma
```

where:
- `q2` is dilepton invariant mass squared in GeV^2,
- `y` is residual/observable value,
- `sigma` is uncertainty.

Optional:
- `channel`
- `bin_low`
- `bin_high`

## Example usage

Generate one fake charm-only dataset:

```bash
python 07_inject_charm_mimicry.py --out fake_charm.csv --n-bins 16 --seed 1
```

Run WCT scan on injections:

```bash
python 08_scan_wct_on_injections.py --input fake_charm.csv --n-mc 500 --out-dir mimicry_results
```

Run locked-k test on several channel CSVs:

```bash
python 09_locked_k_cross_channel.py --k 11.7 --inputs B0Kst.csv BpK.csv BsPhi.csv --out locked_k_results.csv
```

Unbinned skeleton:

```bash
python 10_unbinned_likelihood_skeleton.py --root data/job0.root --tree DecayTree
```

## Interpretation

Fail condition:
- WCT repeatedly fits charm-only pseudo-data with significant delta chi2 at the same k region.

Pass condition:
- WCT has poor fit or non-significant improvement on fake charm-only data, while retaining improvement on real data.


## v2 fixed scanner note

The original demo could report a best `k` outside the requested range because the polishing step used unconstrained Nelder-Mead after a bounded differential-evolution search.

v2 fixes this by using bounded L-BFGS-B polishing.

The main null-test statistic is now WCT versus the charm model, not WCT versus a constant model:

```text
false_positive_rate_vs_charm
false_positive_rate_vs_charm_and_k
wct_preferred_over_charm_by_aic_rate
wct_preferred_over_charm_by_bic_rate
```

Why: charm-only fake data is intentionally non-constant, so WCT beating a constant null is not evidence of a false discovery. The relevant question is whether WCT beats the charm-tail null.


## Strong test runner

Run the next serious test:

```bash
python 12_run_strong_mimicry_test.py --n-mc 1000 --target-k 11.7 --out-dir strong_mimicry_results
```

Windows shortcut:

```bat
RUN_STRONG_TEST.bat
```

Main report:

```text
strong_mimicry_results/STRONG_MIMICRY_REPORT.md
```

Pass criteria:

```text
false_positive_rate_vs_charm <= 0.01
false_positive_rate_vs_charm_and_k <= 0.002
wct_preferred_over_charm_by_bic_rate <= 0.05
```

## Sweep test

Run a robustness sweep across charm-tail strength and noise:

```bash
python 13_sweep_mimicry_strength.py --n-mc 200 --target-k 11.7 --out-dir mimicry_sweep_results
```

Windows shortcut:

```bat
RUN_SWEEP_TEST.bat
```

Main table:

```text
mimicry_sweep_results/mimicry_sweep_summary.csv
```
