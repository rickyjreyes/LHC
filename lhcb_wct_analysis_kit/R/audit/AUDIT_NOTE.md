# Internal Audit Note — LHCb / WCT Statistical-Audit Suite

Date: 2026-06-22
Scope: pre-implementation audit of the existing repository, written before the
new `R/audit/` modules were authored. This note records *what already exists*,
*what each stage actually did*, *what data each replayable artifact carries*,
and *where conventions diverge*. It is the reference the audit modules are
built against. It makes **no** scientific claim of its own.

> Interpretation boundary (applies to the whole suite): the Python pipeline and
> its R reproduction analyze the **same** LHCb open data. Running both is a
> cross-language *computational reproduction*, **not** an independent
> experimental replication. The "smooth"/KDE baseline is **not** a full
> Standard Model amplitude analysis (no official acceptance, efficiency,
> detector response, form factors, hadronic uncertainties, or covariance).

---

## 1. Analyses already implemented in R (canonical reproduction)

Under `R/` the following canonical reproductions exist and must keep
reproducing the committed Python outputs:

| R file | Mirrors Python | Needs ROOT? | Notes |
|---|---|---|---|
| `lhcb_domain.R` | `config.py` | no | single source of truth for q2 domain, vetoes, active intervals, `Delta_ell_A`, `n<->k` map |
| `lhcb_io.R` | `lhcb_utils.py` | I/O only | ROOT/parquet adapters + committed-artifact readers |
| `lhcb_kde.R` | KDE in 09d/12/13 | yes | KDE baseline conventions |
| `lhcb_poisson.R` | polar bounded-Poisson | yes | L-BFGS-B polar fit + null engine |
| `lhcb_wls.R` | WLS in 28/29 | no | weighted least squares for sideband residuals |
| `two_mode_kde_polar.R` | `09d_*` | yes | two-mode KDE bounded-Poisson scan |
| `integer_winding_scan.R` | `12_*` | yes | active-domain integer winding |
| `koide_comb_scan.R` | `13_*` | yes | Koide/trig comb |
| `wct_vs_smooth_likelihood.R` | `16_*` | yes | smooth-null vs locked comb |
| `veto_window_covariance.R` | `25_*` | yes | veto covariance / active-domain invariance |
| `sideband_subtracted.R` | `28_*` | **no** | replayable from committed per-bin CSV |
| `charm_trimmed_control.R` | `29_*` | **no** | replayable from committed per-bin CSV |
| `compare_python_r.R` | — | no | parity tool (writes `parity_report.csv`) |

The new audit modules **wrap** these; they do not modify them.

## 2. Which analyses require the OAuth-gated ROOT ntuples

Event-level recomputation (re-binning, re-training KDE, re-fitting the full
pipeline per replicate) requires the six ROOT files under `data/`:

```
00382466_0000000{1,2,3}_1.dvntuple.root
00382467_0000000{1,2,3}_1.dvntuple.root
tree = B0_KstMuMu/DecayTree
q2   = derived from muon four-vectors
B0_M, Kst_892_0_M
```

Stages whose *full-pipeline* audit (full-pipeline null, injection–recovery,
KDE/bandwidth re-estimation, file/run holdout) needs raw events: **09d, 12, 13,
16, 25**. These run only in `--mode full`. In `--mode replay` the suite
summarizes the committed scan/null tables and clearly labels the event-level
pieces as `UNAVAILABLE_NO_EVENT_DATA`.

## 3. Which analyses are reproducible from committed per-bin CSV only

Fully replayable without ROOT:

- **Stage 28** — `outputs_sideband_subtracted/`: per-bin residuals
  (`*_bins.csv`), full k-scan (`*_scan.csv`), integer scan, comb tests, wells,
  triplets, and `*_summary.json` (counts + alpha + verdict).
- **Stage 29** — `outputs_charm_trimmed_control/`: per-region per-bin CSVs for
  `signal_B_signal_Kst`, `B_low_sideband_Kst_signal`,
  `B_high_sideband_Kst_signal`, plus the sideband-subtracted charm-trimmed
  region, scans, combs, integer scans, and `*_summary.json`.

Replayable at the *summary/scan/null* level (parameter values, deltaD/deltaChi2
curves, null tail summaries) for **09d, 12, 13, 16, 25** via their committed
`*_summary.json`, `*_scan*.csv`, `*_null*.csv`, `*_summary.csv`.

## 4. Numerical conventions that differ by stage (do NOT homogenize)

| Stage | Histogram coord | Bins | Baseline / KDE | Amplitude bound | Null |
|---|---|---|---|---|---|
| 09d | **linear q2** | 60 (43 retained after mask) | event-level KDE, bw scale 1.50 canonical, trained outside widened vetoes | **radial** A1,A2 ≤ 0.10 | local scan-max Poisson, 5000 |
| 12 | linear q2 (reuses 09d) | 60 | KDE bw scales {0.5,0.75,1.0,1.25,1.5} | radial A2 ≤ 0.10 | integer-scan-max, 5000 |
| 13 | **log q2** | 240 | KDE = hist centers repeated by counts | **coefficient-wise** a,b ∈ [−0.1,0.1] (NOT radial ≤0.1) | model-scan-max, 5000 |
| 16 | log q2 | 240 | smooth empirical null + nuisance k1 | locked combs | bootstrap H0, 5000 |
| 25 | linear q2 | — | per-veto recomputed active support & `Delta_ell_A` | A_MAX 0.05 | scan, 1501 pts, k∈[6,36] |
| 28 | log q2 active | 240 | **WLS** sideband-subtracted residual | none (WLS) | Gaussian, k∈[6,32], 1301 pts, 1000 null |
| 29 | log q2 active | 240 | WLS per region | none (WLS) | per region, 500 null |

Key fixed constants: `k1 = 7.61054`, reference `k2 = 19.5296`,
`Delta_ell_A = 4.780150335923678`, `alpha (28) = 0.28495897903372835`.

### Known statistical issues to preserve-in-parity + correct-in-audit
1. **Stage 13 coefficient-wise vs radial amplitude bounds.** Coefficient bounds
   `|a|,|b| ≤ 0.1` allow radial `sqrt(a^2+b^2)` up to ~0.14 (e.g. committed
   `A_minus = 0.1359`, `A0 = 0.1326`). Parity keeps coefficient bounds; a
   separate corrected audit re-imposes radial ≤ 0.1.
2. **Stage 28/29 sideband normalization `alpha` treated as known.**
   `Var(R) = N_sig + alpha^2 N_side` conditions on a fixed alpha estimated from
   the same data. Corrected audit re-estimates alpha per bootstrap replicate.
3. **Baseline conditioned-on, not re-estimated.** 09d's parity null holds the
   KDE baseline fixed; a full-pipeline null re-trains it per replicate.
4. **Model comparison after selection.** AIC/BIC reported at a *selected*
   frequency/comb do not by themselves pay for the search; the family
   correction does.
5. **Stage 29 preserved variance quirk** (sideband variance behavior) kept in
   parity; corrected-variance audit written to a separate directory.

## 5. Parity outputs vs corrected-audit outputs

- Parity outputs live in the **existing** `outputs_*` and `outputs_*_r`
  directories and must never be overwritten.
- All new audit artifacts go to **new** roots:
  `outputs_statistical_audit_r/`, `tables_r/statistical_audit/`,
  `figures_r/statistical_audit/`, `reports/rendered/`.
- Corrected modes are labeled `mode = corrected_audit` in every table and never
  share a file with `mode = parity`.

## 6. Statistical searches performed in each stage (for family correction)

| Stage | Search dimension(s) | Count |
|---|---|---|
| 09d | k2 over [18,24] | 601 grid pts (+ fixed ref) |
| 12 | integer n 10..22 × 5 bandwidths | 65 |
| 13 | Q/comb model family × 5 bandwidths | many (see `koide_comb_summary.csv`) |
| 16 | 3 locked comb models × 5 bandwidths | 15 |
| 25 | 6 veto schemes × k∈[6,36] (1501) | 6×1501 |
| 28 | k∈[6,32] (1301) + 13 integers + 2 combs | 1316 |
| 29 | 3 regions × (1301 k + integers + combs) | ~3×1316 |

The multiple-testing correction operates on the **registry**, not on the
strongest result only.

## 7. Where the same data were reused across analyses (non-independence)

- 09d, 12, 13, 16 all derive from the **same** selected event sample
  (298,801 after mass cuts) and the same active intervals — they are **not**
  independent looks.
- 25 reuses the same events with different veto windows.
- 28 and 29 share the signal window and overlapping sideband definitions; the
  sideband-subtracted residual is a **linear combination** of the same counts —
  the audit must not invent independence between them.
- Stage 29's three regions overlap in reconstruction/background; the
  signal-specificity statistic is computed with bootstrap, not assumed-iid.

## 8. Headline committed values the audit guards against drift (regression targets)

- 09d: `D_base = 1255.9044965985288`; best `k2 = 23.08`,
  `DeltaD_add = 150.90386012713225`, `A2 = 0.10` (bound active);
  reference `k2 = 19.5296`, `DeltaD_add = 70.0029667147544`; local scan-max
  empirical p = 1/5001.
- 12: best `n = 20`, `k = 26.28865146755044`, `DeltaD = 135.5123713507652`;
  `n15` `k = 19.716488600662828`, `DeltaD = 58.25363341553543`.
  Branch switches n=10 (low bw) → n=20 (high bw): **model sensitivity**.
- 13 (bw 1): Q=2/3 `DeltaD = 373.077171046471`,
  `A_minus=0.13595411505586577`, `A0=0.1326216068906595`,
  `A_plus=0.09094541951333819`; **best canonical model is Q=4/9**
  `DeltaD = 457.4416529560142`.
- 28: `alpha = 0.28495897903372835`; best `k=8.78`,
  `DeltaChi2 = 5.303549331940928`, scan-max p = 0.8161838161838162;
  ref-k `DeltaChi2 = 0.4468052357436818`, p = 0.8221778221778222;
  n15 `DeltaChi2 = 1.2124631626656992`, p = 0.5754245754245755;
  comb 10/15/20 `DeltaChi2 = 4.141641179068927`, p = 0.6853146853146853.
  **Reference mode / n15 / comb do NOT survive** → weakens signal specificity.
- 29: charm-trimmed B sidebands show **stronger** structure than the signal
  window (low SB best `DeltaChi2 ≈ 956`, high SB `≈ 1045`, signal `≈ 353`).
  **Not signal-specific.**

## 9. Execution-environment caveat for this build

The build environment had R 4.3.3 with core CRAN packages
(jsonlite, dplyr, tidyr, purrr, readr, stringr, tibble, digest, scales,
ggplot2, optparse, testthat) but **no** ROOT/event data, and **no** optional
reporting packages (gt, patchwork, viridis, arrow, reticulate, quarto, ragg,
svglite). Therefore:
- Only **replay mode** was executed here; full event-level audits were
  implemented but not run (no ntuples).
- Optional packages degrade gracefully; the report falls back to a Markdown/HTML
  artifact when quarto is absent.
- No statement in the suite claims a final-resolution (10k-null) simulation ran
  unless it actually did.
