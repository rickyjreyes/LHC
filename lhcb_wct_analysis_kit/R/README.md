# R computational reproduction

An independent **R** implementation of the yield-side LHCb / WCT workflow for the
open-data `B0 -> (K*(892)0 -> K+ pi-) mu+ mu-` candidate spectrum. All physics
selections, q2 reconstruction, transformations, fits, scans, bootstraps,
statistics and verdicts run in R. The original Python analysis scripts are never
called from R; the committed Python outputs are used only as regression
references.

> **This is a computational reproduction in a second language, not independent
> experimental corroboration.** The same open data analysed twice does not add
> experimental evidence. The smooth empirical baseline used in stages 09d/13/16
> is **not** a full Standard Model amplitude analysis with official
> efficiencies, acceptance, backgrounds, covariance, form factors or hadronic
> uncertainties. Sideband and charm controls are reported even when they weaken
> the main interpretation.

## 1. Supported R version

R 4.3.3 (developed/tested). Any R >= 4.1 should work.

## 2. Dependency installation

```r
install.packages(c("jsonlite", "digest", "optparse", "testthat"))
# optional, only for ROOT intake / parquet caching:
install.packages(c("reticulate", "arrow"))
reticulate::py_install(c("uproot", "numpy", "awkward"))
```

System packages (Ubuntu): `apt-get install -y r-base-core build-essential libuv1-dev`.
See `renv.lock` for exact versions.

## 3. ROOT-input setup

Place the LHCb open-data files under `data/` (`data/*.dvntuple.root`,
`data/*.root`); see `../DOWNLOAD.md`. The files are served from an
**OAuth-gated** CERN endpoint and are **not** committed. ROOT reading uses
`reticulate` + Python `uproot` strictly as an I/O adapter.

## 4. Parquet-cache setup

```bash
Rscript R/inspect_root.R --export --out data_cache/events.parquet
```

This exports only the selected branches once; subsequent stages can then read
`--input-format parquet --input data_cache/events.parquet` without reticulate.

## 5. Single-stage commands

```bash
# Controls — reproducible NOW from committed per-bin inputs (no ntuples needed):
Rscript R/sideband_subtracted.R --bins ../outputs_sideband_subtracted/sideband_subtracted_bins.csv --outdir outputs_sideband_subtracted_r
Rscript R/charm_trimmed_control.R --committed-dir outputs_charm_trimmed_control --outdir outputs_charm_trimmed_control_r

# Data-gated Poisson stages (need event-level q2):
Rscript R/two_mode_kde_polar.R   --input-format root --data-dir data --n-null 5000
Rscript R/integer_winding_scan.R --q2-csv data_cache/q2.csv --n-null 2000
Rscript R/koide_comb_scan.R      --q2-csv data_cache/q2.csv
Rscript R/wct_vs_smooth_likelihood.R --q2-csv data_cache/q2.csv --n-null 1000
Rscript R/veto_window_covariance.R   --q2-csv data_cache/q2.csv
```

## 6. Fast preview command

```bash
Rscript R/run_all.R --fast --continue-on-error
```

## 7. Full paper-grade command

```bash
Rscript R/run_all.R --full --controls --n-null 5000 --seed 12345 --continue-on-error
```

## 8. Control-test command

```bash
Rscript R/run_all.R --only 28,29 --continue-on-error
```

## 9. Test command

```bash
Rscript tests/testthat.R
```

## 10. Python-versus-R parity command

```bash
Rscript R/compare_python_r.R --out parity_report.csv
```

## 11. Known RNG differences

Python uses `numpy.random.default_rng` (PCG64); R uses
`RNGkind("L'Ecuyer-CMRG")`. Identical null draws are **not** reproduced.
Null comparisons use summary statistics (mean/median/p95/p99/max), observed
tail count and the resulting verdict — never raw draw equality. A deterministic
null fixture (`R/fixtures/poisson_reference.json`) lets the Python and R fit
engines be tested on identical null data.

## 12. Stage-specific baseline differences

* **09d** histograms in **linear** q2 (60 bins); KDE trained on event-level q2
  outside the widened vetoes; PSI2S upper veto edge 14.5.
* **13/16** histogram in **log** q2 (240 bins); KDE built from histogram centers
  repeated by integer bin counts.
* See `KNOWN_DIFFERENCES.md` for the full list.

## 13. Stage-13 coefficient-bound warning

Stage 13 constrains **each sine/cosine coefficient** independently to
`[-0.1, 0.1]`. This is **not** a radial-amplitude cap, so `sqrt(a^2+b^2)` can
exceed 0.1 while every coefficient is within bounds. The output reports both
`coefficient_bound_active` and `radial_amplitude_above_0p1` and emits a warning
when this happens. `--amplitude-bound radial` enables a separate **corrected
audit mode** that writes to its own directory and never replaces parity outputs.

## 14. Reproduction, not corroboration

See the banner above: the R port on the same data is a computational
reproduction, not independent experimental confirmation.

## 15. Smooth null is not a full SM analysis

See the banner above: the empirical smooth baseline is not a complete Standard
Model / QFT prediction.

## 16. Controls must be reported

Sideband (28) and charm-trimmed (29) controls are reported in full even when
they weaken the main yield-side interpretation. Stage 28 currently shows
**non-survival** (best scan p ~= 0.82), and stage 29 shows strong structure in
the B sidebands — both reproduced exactly by the R port.
