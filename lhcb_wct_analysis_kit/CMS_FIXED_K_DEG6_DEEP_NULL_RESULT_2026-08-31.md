# CMS fixed-k degree-6 deep paired-null result — 2026-08-31

## Scope

This record documents the deep paired-null calibration defined in

```text
CMS_FIXED_K_DEG6_DEEP_NULL_PLAN_2026-08-31.md
```

and implemented by

```text
35_cms_fixed_k_degree6_deep_null.py
```

The test is the same degree-6 cross-run phase-transfer statistic used in stage 34. The only change is the null depth: 10,000 complete paired pseudoexperiments instead of 1,000.

The exact CMS-mapped LHCb frequency remained frozen at

```text
k = 3.512912912912913
```

Frequency scanning remained disabled.

This result remains exploratory and post-unblinding because the request-48 data were already inspected before stages 34 and 35.

---

## Observed statistic

Run groups:

```text
A = 00382466
B = 00382467
```

Background:

```text
Chebyshev log-rate degree = 6
```

Observed cross-run free-phase fits:

```text
phi_A = 1.2077788654 rad
phi_B = 0.3158689334 rad
phase difference = 51.102675 deg
```

Directional transferred-phase tests:

```text
q(B | phi_A) = 26.2598430982
q(A | phi_B) = 25.5132602920
q_joint      = 51.7731033902
```

The degree-6 fitted phase is therefore not stable enough to support precise phase replication. The relevant question is narrower: does the fixed-k component learned in one run retain predictive value in the other after independently refitting a flexible degree-6 smooth background?

---

## Deep paired-null calibration

The paired null repeated the complete stage-34 procedure in every pseudoexperiment:

1. generate independent pseudo spectra for both run groups from their fitted degree-6 smooth nulls;
2. refit the degree-6 null background in each pseudo spectrum;
3. fit the free fixed-k phase in each pseudo training run;
4. transfer that phase to the opposite pseudo run;
5. refit the opposite run's degree-6 background plus a nonnegative amplitude at the transferred phase;
6. repeat in both directions and sum the directional likelihood-ratio statistics.

Result:

```text
paired pseudoexperiments = 10000
exceedances              = 0
add-one p                 = 1 / 10001
                          = 9.99900009999e-05
```

Because there were zero exceedances, the exact one-sided binomial upper bounds on the unresolved exceedance probability are

```text
95% upper bound = 0.0002995283598
99% upper bound = 0.0004604109969
```

These are bounds for the implemented parametric analysis-model null only. They are not detector-, reconstruction-, acceptance-, or physics-systematics-calibrated physical p-values.

---

## Main result

The most conservative prespecified smooth-background stress test remains positive under deeper calibration:

> **At the exact CMS-mapped frequency, the two-way cross-run phase-transfer statistic remains larger than all 10,000 degree-6 paired refit-null pseudoexperiments. The exact one-sided 95% upper bound on the unresolved exceedance probability under this analysis-model null is approximately 2.995e-4.**

This substantially strengthens the conclusion from stage 34 that the fixed-k cross-run predictive component is not trivially removed by independently refitting a flexible degree-6 Chebyshev smooth background.

It does **not** establish exact phase universality. The observed degree-6 phase difference is approximately 51.1 degrees.

---

## Relation to stages 31–34

The result chain is now:

```text
Stage 31:
  exact CMS frequency + exact mapped CMS phase + positive sign
  -> FAIL in the LHCb request-48 exclusive B0 -> K*0 mu+ mu- spectrum

Stage 32:
  exact CMS frequency fixed, phase free
  -> strong exploratory response

Stage 33:
  fixed-k robustness across run groups, KDE bandwidths, binning,
  veto perturbations, Chebyshev degrees 2..6, and refit nulls
  -> response survives broadly, degree 6 is weakest background stress test

Stage 34:
  cross-run phase prediction with independently refitted Chebyshev backgrounds
  -> all degrees 2..6 show 0/1000 paired-null exceedances

Stage 35:
  degree-6-only deep calibration, same statistic/model, 10000 paired nulls
  -> 0/10000 exceedances
  -> exact one-sided 95% upper bound ~= 2.995e-4
```

The strongest defensible summary is therefore:

> **An exploratory component at the exact CMS-mapped log frequency is reproducibly present across the two LHCb request-48 run groups and retains cross-run predictive value after independently refitting flexible smooth backgrounds through Chebyshev degree 6. The degree-6 paired analysis-model null gives 0/10,000 exceedances, while the fitted phase remains background-model dependent.**

---

## What this does not establish

This result does not establish:

- a prospective independent replication;
- an inclusive-dimuon replication equivalent to the CMS observable;
- a detector/systematics-calibrated physical significance;
- exact phase stability at high background flexibility;
- CMS/LHCb phase universality;
- a globally calibrated discovery p-value or sigma;
- WCT causation.

The request-48 sample has been repeatedly analyzed in this repository, so no further reanalysis of the same events can restore blind/prospective status.

No CMS and LHCb p-values or Z values should be combined without a separately frozen joint statistic and a justified joint null.

---

## Next scientific priority

Further deepening of the same request-48 null is lower value than acquiring new information.

The next high-value test should be frozen before inspecting a genuinely new sample or more directly comparable observable. Priority order:

1. untouched LHCb data or a new LHCb channel, with the exact fixed-k statistic frozen before opening the sample;
2. preferably an inclusive opposite-sign dimuon spectrum rather than the exclusive B0 -> K*0 mu+ mu- candidate spectrum;
3. highest-value cross-detector test: ATLAS inclusive dimuon data with the CMS frequency frozen before inspection.

The current request-48 result should now be treated as a completed exploratory robustness chain rather than repeatedly re-optimized for smaller same-data p-values.
