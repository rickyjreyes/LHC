# Methodology and parity status

This document records what the R reproduction does, which quantities were
**executed and numerically compared** against the committed Python outputs, and
which were implemented but not runnable in the current environment.

## Environment

* R 4.3.3, x86_64 Ubuntu 24.04.
* Python 3.11 + numpy 2.4.6 + scipy 1.17.1 (used **only** to generate the
  parity fixtures `R/fixtures/*`, never for analysis logic).
* The LHCb open-data ntuples are served from an OAuth-gated CERN endpoint and
  are **not present**; the raw event-level stages therefore cannot be executed
  end-to-end here.

## What is committed and therefore verifiable without the ntuples

The Python stage-28 and stage-29 outputs commit **per-bin inputs**:

* `outputs_sideband_subtracted/sideband_subtracted_bins.csv`
  (ell, N_signal, N_Blow, N_Bhigh, R_subtracted, variance)
* `outputs_charm_trimmed_control/charm_trimmed_bins_*.csv` (region, ell, counts)
  and `charm_trimmed_sideband_bins.csv`

Because the entire WLS scan / integer / comb / well / triplet engine for these
stages is a deterministic function of those committed inputs, the R engine can
be run on them and compared **exactly** to the committed Python scan/comb/
triplet/verdict outputs.

## Executed parity results

Run `Rscript R/compare_python_r.R`. Summary (all PASS):

| Stage | Quantity | Max diff | Tolerance |
|---|---|---|---|
| domain | Delta_ell_A, k(n=10/15/20) | 0 / <1e-13 | 1e-12 / 1e-10 |
| 09d engine | SciPy `gaussian_kde` density (fixture) | ~1e-16 | 1e-12 |
| 09d engine | polar L-BFGS-B `D_base` (fixture) | ~4e-11 | 1e-6 (rel) |
| 09d engine | two-mode `DeltaD` at k2=20 (fixture) | ~2e-9 | 1e-6 (rel) |
| 09d engine | projected-Newton null scan (fixture) | exact idx; ~1e-9 | 1e-9 |
| 28 | continuous scan (1301 pts) | ~1e-13 | 1e-5 |
| 28 | best grid index / best k | exact / ~0 | exact / 1e-9 |
| 28 | integer scan, comb, wells (13), triplets (220) | <1e-12 | 1e-5 |
| 28 | verdict deltaChi2 (best/kref/n15/comb) | 0–6e-14 | 1e-5 |
| 28 | alpha | ~3e-16 | 1e-8 |
| 29 | 3 raw-count region scans + comb + triplet | 0–4e-12 | 1e-5 |
| 29 | sideband-subtracted (var=max(residual,1) quirk) | 0 | 1e-5 |

The stage-28 and stage-29 regression targets in the task statement are
reproduced (e.g. best scan k = 8.78, DeltaChi2 = 5.3035493319; alpha =
0.28495897903372835; locked (10,15,20) DeltaChi2 = 4.141641179068927;
charm signal best DeltaChi2 = 352.8910426902785).

## Engine validation (fixtures)

The Poisson stages (09d/12/13/16/25) cannot be run on the real spectrum without
the ntuples, but their numerical cores are validated against Python-generated
fixtures on identical data:

* `R/fixtures/kde_reference_scale_*.json` — scipy `gaussian_kde` at scales 1.0,
  1.5; R matches to ~1e-16.
* `R/fixtures/poisson_reference.json` — exact CPU polar L-BFGS-B base/two-mode
  fits and the projected-Newton null engine on a 43-bin synthetic spectrum; R
  matches the deviances/`DeltaD`/best-index to <=1e-6 (rel) and the
  projected-Newton scan exactly.

These are the same algorithms the real stages use, so once `data/` is populated
the real-data regression targets (D_base = 1255.9044965985288, best k2 = 23.08,
DeltaD = 150.90386012713225, etc.) are expected to follow within the same
tolerances.

## Null distributions

Identical RNG draws are not required across languages. The R nulls are
reproducible within R (`RNGkind("L'Ecuyer-CMRG")` + fixed seed) and compared by
mean/median/p95/p99/max, observed tail count and verdict.

## Two null engines (09d)

* `--null-engine python-compatible`: projected-Newton, 10 iterations, ridge
  1e-8, eta clip [-0.2, 0.2], radial projection of both modes, full k2 scan —
  reproduces the Python GPU bootstrap path on CPU.
* `--null-engine exact`: refit each null with the bounded polar optimizer.

These are never mixed into one p-value.
