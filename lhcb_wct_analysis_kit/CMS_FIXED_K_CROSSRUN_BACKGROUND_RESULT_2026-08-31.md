# CMS fixed-k cross-run background-discrimination result — 2026-08-31

## Scope

This record documents the output of the prespecified post-unblinding protocol

```text
CMS_FIXED_K_CROSSRUN_BACKGROUND_TEST_PLAN_2026-08-31.md
```

implemented by

```text
34_cms_fixed_k_crossrun_background_test.py
```

The exact CMS-mapped LHCb frequency was held fixed throughout:

```text
k = 3.512912912912913
```

Frequency scanning was disabled.

The two request-48 run groups were treated as separate samples:

```text
A = 00382466
B = 00382467
```

For each Chebyshev log-rate background degree 2 through 6, the free fixed-k phase fitted in one run group was transferred to the other run group. The target run then refit its own Chebyshev background at the same degree plus a nonnegative amplitude at the transferred phase. The direction was then reversed.

The paired empirical null repeated the complete procedure in every pseudoexperiment: regenerate both run groups under their fitted smooth nulls, refit both null backgrounds, re-estimate both training phases, transfer those phases, and refit both target backgrounds.

This remains exploratory because request-48 was already unblinded before stage 34.

---

## Observed results

| Chebyshev degree | phi_A (rad) | phi_B (rad) | phase difference | q(B | phi_A) | q(A | phi_B) | q_joint | paired-null exceedances | add-one p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | -1.40069030 | -1.27076761 | -7.444 deg | 74.49833292 | 99.85854540 | 174.35687832 | 0 / 1000 | 0.000999000999 |
| 3 | -2.32369045 | -2.48422810 | +9.198 deg | 46.93747503 | 67.38162384 | 114.31909887 | 0 / 1000 | 0.000999000999 |
| 4 | -2.92535074 | -3.02438629 | +5.674 deg | 82.40091252 | 110.70044702 | 193.10135953 | 0 / 1000 | 0.000999000999 |
| 5 | +1.91303058 | +1.90791643 | +0.293 deg | 84.95375449 | 105.73014172 | 190.68389621 | 0 / 1000 | 0.000999000999 |
| 6 | +1.20777887 | +0.31586893 | +51.103 deg | 26.25984310 | 25.51326029 | 51.77310339 | 0 / 1000 | 0.000999000999 |

All five prespecified background degrees produced zero exceedances in 1,000 paired refit-null pseudoexperiments.

The add-one value 1/1001 is a Monte Carlo resolution statement, not an estimate that the physical p-value is exactly 0.000999. For zero exceedances in 1,000 trials, an exact one-sided binomial 95% upper bound on the unresolved exceedance probability is approximately 0.00299.

---

## Main result

The fixed CMS-mapped frequency retains cross-run predictive value after the target smooth background is refit independently.

For degrees 2 through 5, the phase learned independently in each run is also reasonably coherent, with absolute phase differences from approximately 0.3 to 9.2 degrees.

Degree 6 is the critical stress test. It produces a much larger phase difference:

```text
|Delta phi| = 51.103 deg
```

so degree 6 does **not** support a claim of precise phase replication.

However, even at degree 6, the phase learned in A improves B and the phase learned in B improves A:

```text
q(B | phi_A) = 26.25984310
q(A | phi_B) = 25.51326029
q_joint      = 51.77310339
```

and none of 1,000 paired refit-null pseudoexperiments reached the observed joint statistic.

Therefore the degree-6 result is more accurately described as:

> **The exact fixed-k component remains cross-run predictive under a degree-6 refitted smooth background, but the exact fitted phase becomes background-model dependent.**

That distinction should be preserved.

---

## Relation to stage 33

Stage 33 found that the combined fixed-k score was strongly stable across:

- both run groups individually;
- KDE bandwidth scale 0.75 through 2.50;
- 48, 60, 72, 90, and 120 bins;
- nominal, narrowed, widened, and shifted charmonium vetoes;
- Chebyshev background degrees 2 through 6.

The weakest stage-33 fixed-baseline result was degree 6:

```text
q = 12.928710
empirical add-one p = 0.00209979
```

Stage 34 addresses that background-family vulnerability more directly by using the two run groups for cross-prediction and by refitting the target background at every directional test and in every paired pseudoexperiment.

The degree-6 cross-run statistic remains large under that stronger procedure.

---

## What this supports

Within the implemented exploratory LHCb analysis, the following statement is supported:

> **A component at the exact CMS-mapped log frequency is reproducibly present across the two request-48 run groups and retains predictive value after independently refitting flexible Chebyshev smooth backgrounds through degree 6.**

This is stronger than the stage-32 fixed-baseline diagnostic alone and substantially weakens the explanation that the response is only a consequence of one run group, one KDE bandwidth, one binning, one veto configuration, or a low-order smooth-background choice.

---

## What this does not establish

This result does not establish:

- a prospective independent replication;
- a detector/systematics-calibrated physical significance;
- phase universality across CMS and LHCb;
- exact phase stability within LHCb at high background flexibility;
- an inclusive-dimuon replication equivalent to the CMS observable;
- WCT causation;
- a discovery-grade p-value or sigma value.

The paired null is a parametric analysis-model calibration. It does not generate detector, trigger, reconstruction, acceptance, or Standard Model modeling systematics.

No CMS and LHCb p-values or Z values should be combined without a separately frozen joint statistic and joint null.

---

## Next targeted test

The current paired-null resolution at the most conservative prespecified background degree (degree 6) is only 1,000 trials.

The next test should therefore deepen **degree 6 only**, without changing the observed statistic or model:

```text
k fixed = 3.512912912912913
Chebyshev degree = 6
same A -> B and B -> A phase-transfer statistic
same refit procedure
10,000 paired pseudoexperiments
```

Degree 6 is selected for deeper calibration because it was the weakest / most flexible prespecified background stress test, not because it produced the smallest p-value.

A zero-exceedance outcome in 10,000 trials would still only establish a Monte Carlo tail bound for this analysis model; it would not be a physical discovery significance.
